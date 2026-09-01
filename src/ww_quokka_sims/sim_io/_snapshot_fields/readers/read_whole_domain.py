## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy

## local
from .. import read_fields

##
## === PUBLIC FUNCTIONS
##


def initialize_whole_domain_grid(
    *,
    yt_dataset: Any,
    amr_level: int,
) -> Any:
    """
    Return a yt covering grid spanning the whole domain at `amr_level`'s resolution.

    For `amr_level > 0`, this is the composite of the finest data available up to and
    including `amr_level` (coarser regions are filled by interpolating from the highest
    level that does cover them); `amr_level=0` always reads the base level directly.
    """
    refinement_ratio = int(yt_dataset.refine_by)
    num_cells = yt_dataset.domain_dimensions * (refinement_ratio**amr_level)
    return yt_dataset.covering_grid(
        level=amr_level,
        left_edge=yt_dataset.domain_left_edge,
        dims=num_cells,
    )


def load_sarray(
    *,
    whole_domain_grid: Any,
    field_key: read_fields.FieldKey,
) -> numpy.ndarray:
    """Read one scalar field out of an already-built whole-domain grid as a plain 3D array."""
    sarray_3d = numpy.asarray(whole_domain_grid[field_key], dtype=numpy.float64)
    if sarray_3d.ndim != 3:
        raise ValueError(f"expected a 3D array for {field_key}; got shape {sarray_3d.shape}.")
    return numpy.ascontiguousarray(sarray_3d)


## } MODULE
