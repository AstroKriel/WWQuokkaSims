## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy

## local
from . import base_reader
from ... import read_fields

##
## === PUBLIC FUNCTIONS
##


def load_sarray(
    *,
    yt_dataset: Any,
    field_key: read_fields.FieldKey,
) -> numpy.ndarray:
    """
    Read one scalar field at amr_level=0 by reading each box's own cells individually
    and placing them directly into their slice of the output array.

    Unlike `read_whole_domain`, this never materializes yt's own whole-domain buffer,
    only ever holding one box (small) plus the output array (the size of the field
    itself) in memory at once. See https://github.com/yt-project/yt/issues/3958 for the
    documented ~6x memory overhead `covering_grid` carries on top of the output array's
    own size.
    """
    resolution = tuple(int(num_cells) for num_cells in yt_dataset.domain_dimensions)
    ## NaN-filled rather than numpy.empty: uninitialized memory could be finite garbage,
    ## which would hide a box that never got written (e.g. from an indexing bug)
    sarray_3d = numpy.full(resolution, numpy.nan, dtype=numpy.float64)
    for grid_box, placement_slices in base_reader.extract_amr_level_0_boxes(yt_dataset):
        sarray_3d[placement_slices] = numpy.asarray(grid_box[field_key], dtype=numpy.float64)
    if numpy.isnan(sarray_3d).any():
        raise ValueError(
            "some cells were never written by any amr_level=0 box; the boxes do not"
            " fully tile the domain (or the placement indices above are wrong).",
        )
    return numpy.ascontiguousarray(sarray_3d)


## } MODULE
