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
from jormi.ww_arrays.farrays_3d import difference_sarrays
from jormi.ww_fields import cartesian_axes

## local
## direct-name import, not the usual module import: this file is reachable from
## `_snapshot_fields/__init__.py` (via `derive_magnetic_fields.py`/`derive_mhd_fields.py`),
## so `from ..._snapshot_fields import read_fields` would need that package fully resolved
## while it is still mid-import -- a real circular dependency
from . import level0_boxes
from ..._snapshot_fields.read_fields import FieldKey

##
## === DATA STRUCTURES
##


@dataclasses.dataclass(frozen=True)
class ExpandedFArray:
    """One box's expanded, ghost-zone-padded array, alongside where its own
    (non-expanded) cells belong in a composited output array."""

    farray: numpy.ndarray
    cell_range: tuple[slice, slice, slice]

    def __post_init__(
        self,
    ) -> None:
        num_spatial_axes = len(self.cell_range)
        if self.farray.ndim != num_spatial_axes + 1:
            raise ValueError(
                f"`farray` must have {num_spatial_axes + 1} dims (one leading component"
                f" axis plus {num_spatial_axes} spatial axes); got {self.farray.ndim}.",
            )
        implied_resolution = tuple(axis_range.stop - axis_range.start for axis_range in self.cell_range)
        trailing_shape = self.farray.shape[-num_spatial_axes:]
        if any(trailing_dim < resolution
               for (trailing_dim, resolution) in zip(trailing_shape, implied_resolution)):
            raise ValueError(
                f"`farray`'s trailing shape {trailing_shape} is smaller than `cell_range`'s"
                f" own resolution {implied_resolution}; an expanded array cannot be smaller"
                " than the region it expands.",
            )


##
## === PUBLIC FUNCTIONS
##


def compute_num_extra_cells(
    *,
    grad_order: int,
) -> int:
    """Extra cells needed per side for a centered `grad_order` finite difference."""
    if grad_order not in difference_sarrays.GRAD_FN_LOOKUP:
        valid_grad_orders = tuple(difference_sarrays.GRAD_FN_LOOKUP.keys())
        raise ValueError(f"`grad_order` must be one of {valid_grad_orders}; got {grad_order}.")
    return grad_order // 2


def trim_expanded_box(
    *,
    expanded_farray: numpy.ndarray,
    num_extra_cells: int,
) -> numpy.ndarray:
    """
    Drop the outer `num_extra_cells` cells from every spatial (trailing 3) axis.

    Works for any number of leading component axes (0 for a scalar, 1 for a vector, ...):
    those are left untouched. Apply this to whatever a box-local computation returns,
    since only the outer `num_extra_cells` layer used the expanded (neighbor) data.
    """
    return expanded_farray[
        ...,
        num_extra_cells:-num_extra_cells,
        num_extra_cells:-num_extra_cells,
        num_extra_cells:-num_extra_cells,
    ]


def iterate_expanded_vfield_boxes(
    *,
    yt_dataset: typing.Any,
    vfield_key_lookup: dict[cartesian_axes.CartesianAxis_3D, FieldKey],
    num_extra_cells: int,
) -> collections_abc.Iterator[ExpandedFArray]:
    """
    For each amr_level=0 box, yield an expanded raw vector-field block (its own cells
    plus `num_extra_cells` of correctly-stitched, periodic-boundary-aware neighbor data)
    and the domain-index slices its own (non-expanded) cells belong to.

    Only reads raw field values via yt's `retrieve_ghost_zones`; no derivative or other
    computation happens here. The caller is responsible for applying whatever
    `num_extra_cells`-consistent local computation it needs to the expanded block,
    trimming it back down with `trim_expanded_box` before placing a result at the
    yielded slices.

    Callers must have already called `yt_dataset.force_periodicity()` if the domain is
    periodic; `retrieve_ghost_zones` otherwise refuses to read past a domain edge.
    """
    field_keys = tuple(vfield_key_lookup[comp_axis] for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER)
    for level0_box in level0_boxes.iterate_amr_level_0_boxes(yt_dataset=yt_dataset):
        expanded_view = level0_box.box.retrieve_ghost_zones(num_extra_cells, list(field_keys))
        expanded_varray = numpy.stack(
            [numpy.asarray(expanded_view[field_key], dtype=numpy.float64) for field_key in field_keys],
            axis=0,
        )
        yield ExpandedFArray(
            farray=expanded_varray,
            cell_range=level0_box.cell_range,
        )


## } MODULE
