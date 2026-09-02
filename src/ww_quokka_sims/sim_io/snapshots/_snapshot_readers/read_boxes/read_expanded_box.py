## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from collections.abc import Iterator
from typing import Any

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes

## local
## direct-name import, not the usual module import: this file is reachable from
## `_snapshot_fields/__init__.py` (via `derive_magnetic_fields.py`/`derive_mhd_fields.py`),
## so `from ..._snapshot_fields import read_fields` would need that package fully resolved
## while it is still mid-import -- a real circular dependency
from . import base_reader
from ..._snapshot_fields.read_fields import FieldKey

##
## === PUBLIC FUNCTIONS
##

_VALID_GRAD_ORDERS = (2, 4, 6)


def compute_num_extra_cells(
    grad_order: int,
) -> int:
    """Extra cells needed per side for a centered `grad_order` finite difference."""
    if grad_order not in _VALID_GRAD_ORDERS:
        raise ValueError(f"`grad_order` must be one of {_VALID_GRAD_ORDERS}; got {grad_order}.")
    return grad_order // 2


def trim_expanded_box(
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


def load_expanded_vfield_boxes(
    *,
    yt_dataset: Any,
    vfield_key_lookup: dict[cartesian_axes.CartesianAxis_3D, FieldKey],
    num_extra_cells: int,
) -> Iterator[tuple[numpy.ndarray, tuple[slice, slice, slice]]]:
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
    field_keys = tuple(vfield_key_lookup[axis] for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER)
    for grid_box, placement_slices in base_reader.extract_amr_level_0_boxes(yt_dataset):
        expanded_box = grid_box.retrieve_ghost_zones(num_extra_cells, list(field_keys))
        expanded_varray = numpy.stack(
            [numpy.asarray(expanded_box[field_key], dtype=numpy.float64) for field_key in field_keys],
            axis=0,
        )
        yield expanded_varray, placement_slices


## } MODULE
