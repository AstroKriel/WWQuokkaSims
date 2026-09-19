## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import typing

from collections import abc as collections_abc

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes

## local
from .. import read_fields

##
## === DATA STRUCTURES
##


@dataclasses.dataclass(frozen=True)
class _LeafBox:
    """One box (any AMR level), alongside its own `child_mask`."""

    box: typing.Any
    child_mask: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        box_resolution = tuple(int(num_cells) for num_cells in self.box.ActiveDimensions)
        if self.child_mask.shape != box_resolution:
            raise ValueError(
                f"`child_mask` shape {self.child_mask.shape} does not match the box's own"
                f" resolution {box_resolution}.",
            )


##
## === INTERNAL HELPERS
##


def _ensure_isotropic_cell(
    *,
    cell_widths: numpy.ndarray,
) -> None:
    """Raise if a box's cells are not isotropic; a single native `dx` is only well-defined then."""
    if not numpy.allclose(
            cell_widths,
            cell_widths[0],
            rtol=1e-6,
            atol=0.0,
    ):
        raise ValueError(
            f"box has anisotropic cell widths ({cell_widths}); a single native dx is only"
            " well-defined for isotropic cells.",
        )


def ensure_consistent_leaf_ordering(
    *,
    reference_leaves: read_fields.AMRLeaves,
    other_leaves: read_fields.AMRLeaves,
) -> None:
    """
    Raise if two independently-loaded `AMRLeaves` do not share the same per-cell ordering.

    Composing multiple `AMRLeaves` elementwise (eg. `internal_energy = total_energy -
    kinetic_energy - magnetic_energy`) is only valid if `values[i]` refers to the same
    physical cell in each: `_iterate_amr_leaf_boxes` walks `yt_dataset.index.grids` in a
    fixed order for a given open dataset, so independent reads agree, but that is an
    assumption about yt's behaviour, not something the type system enforces.
    """
    if not numpy.array_equal(reference_leaves.positions, other_leaves.positions):
        raise ValueError(
            "leaf cells from independent AMRLeaves reads are not consistently ordered;"
            " cannot combine their `values` elementwise.",
        )


##
## === PUBLIC FUNCTIONS
##


def _iterate_amr_leaf_boxes(
    *,
    yt_dataset: typing.Any,
) -> collections_abc.Iterator[_LeafBox]:
    """
    Yield every box (a yt `AMRGridPatch`, any AMR level) alongside its own `child_mask`.

    Unlike `uniform_resolution.level0_boxes.iterate_amr_level_0_boxes`, this walks every level; a
    box's `child_mask` is `True` for cells not covered by a finer box, so masking each
    box's data with it and concatenating across boxes gives every leaf cell in the
    hierarchy exactly once, each still at its own native resolution. Callers pull
    whichever field they need directly off each yielded box (`box[field_key][child_mask]`),
    alongside that box's own cell width (`box.dds`) for the matching native `dx`.
    """
    num_boxes_read = 0
    for box in yt_dataset.index.grids:
        yield _LeafBox(
            box=box,
            child_mask=box.child_mask,
        )
        num_boxes_read += 1
    if num_boxes_read == 0:
        raise ValueError("no boxes were found in this snapshot.")


def load_amr_leaves(
    *,
    yt_dataset: typing.Any,
    field_key: read_fields.FieldKey,
) -> read_fields.AMRLeaves:
    """
    Read one scalar field at every leaf cell across the full AMR hierarchy.

    Unlike `whole_domain`/`uniform_resolution.chunked_domain`, this never resamples onto one
    uniform resolution: each returned value keeps the native cell width and physical position
    of the box (any AMR level) it was read from, and cells covered by a finer box are
    excluded via that box's own `child_mask`. The result is flat and unordered, not
    indexed by domain position -- callers that need per-cell statistics pool it directly
    (e.g. a PDF), not a spatial map.
    """
    axis_names = [read_fields.BOXLIB_3D_AXES_LABELS[axis] for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER]
    grouped_box_values: list[numpy.ndarray] = []
    grouped_box_cell_width: list[numpy.ndarray] = []
    grouped_box_positions: list[numpy.ndarray] = []
    for leaf_box in _iterate_amr_leaf_boxes(yt_dataset=yt_dataset):
        if not leaf_box.child_mask.any():
            continue
        box_cell_widths = numpy.asarray(leaf_box.box.dds, dtype=numpy.float64)
        _ensure_isotropic_cell(cell_widths=box_cell_widths)
        box_values = numpy.asarray(leaf_box.box[field_key], dtype=numpy.float64)[leaf_box.child_mask]
        box_positions = numpy.stack(
            [
                numpy.asarray(leaf_box.box[("index", axis_name)], dtype=numpy.float64)[leaf_box.child_mask]
                for axis_name in axis_names
            ],
            axis=-1,
        )
        box_cell_width = numpy.full(box_values.shape, box_cell_widths[0], dtype=numpy.float64)
        grouped_box_values.append(box_values)
        grouped_box_cell_width.append(box_cell_width)
        grouped_box_positions.append(box_positions)
    if len(grouped_box_values) == 0:
        raise ValueError(f"no leaf cells were found for {field_key} in this snapshot.")
    return read_fields.AMRLeaves(
        values=numpy.concatenate(grouped_box_values),
        cell_width=numpy.concatenate(grouped_box_cell_width),
        positions=numpy.concatenate(grouped_box_positions),
    )


def load_amr_leaves_of_derived_field(
    *,
    yt_dataset: typing.Any,
    field_keys: tuple[read_fields.FieldKey, ...],
    derive_fn: collections_abc.Callable[[numpy.ndarray], numpy.ndarray],
) -> read_fields.AMRLeaves:
    """
    Read a derived scalar field at every leaf cell, combining `field_keys` pointwise via
    `derive_fn` before masking to each box's own leaf cells. See `load_amr_leaves` for the
    single, already-stored-field case this generalises.

    `derive_fn(raw_box_farray) -> derived_box_farray` receives `field_keys` stacked along
    axis 0 for one box's own (unmasked) cells, and must return a same-shaped scalar array.
    `derive_fn` must be pointwise (no spatial derivative): neighbouring leaf cells can sit
    at different native resolutions, so there is no well-defined stencil between them.
    """
    axis_names = [read_fields.BOXLIB_3D_AXES_LABELS[axis] for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER]
    grouped_box_values: list[numpy.ndarray] = []
    grouped_box_cell_width: list[numpy.ndarray] = []
    grouped_box_positions: list[numpy.ndarray] = []
    for leaf_box in _iterate_amr_leaf_boxes(yt_dataset=yt_dataset):
        if not leaf_box.child_mask.any():
            continue
        box_cell_widths = numpy.asarray(leaf_box.box.dds, dtype=numpy.float64)
        _ensure_isotropic_cell(cell_widths=box_cell_widths)
        raw_box_farray = numpy.stack(
            [numpy.asarray(leaf_box.box[field_key], dtype=numpy.float64) for field_key in field_keys],
            axis=0,
        )
        derived_box_farray = derive_fn(raw_box_farray)
        box_values = derived_box_farray[leaf_box.child_mask]
        box_positions = numpy.stack(
            [
                numpy.asarray(leaf_box.box[("index", axis_name)], dtype=numpy.float64)[leaf_box.child_mask]
                for axis_name in axis_names
            ],
            axis=-1,
        )
        box_cell_width = numpy.full(box_values.shape, box_cell_widths[0], dtype=numpy.float64)
        grouped_box_values.append(box_values)
        grouped_box_cell_width.append(box_cell_width)
        grouped_box_positions.append(box_positions)
    if len(grouped_box_values) == 0:
        raise ValueError(f"no leaf cells were found for {field_keys} in this snapshot.")
    return read_fields.AMRLeaves(
        values=numpy.concatenate(grouped_box_values),
        cell_width=numpy.concatenate(grouped_box_cell_width),
        positions=numpy.concatenate(grouped_box_positions),
    )


## } MODULE
