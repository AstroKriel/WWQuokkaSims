## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from collections.abc import Iterator
from typing import Any

## third-party
import numpy

##
## === PUBLIC FUNCTIONS
##


def extract_amr_level_0_boxes(
    yt_dataset: Any,
) -> Iterator[tuple[Any, tuple[slice, slice, slice]]]:
    """
    Yield each amr_level=0 box (a yt `AMRGridPatch`) alongside the domain-index slices
    its own cells belong to, computed from its left edge and the domain's cell grid.

    Only amr_level=0 boxes: at that level they tile the domain with no overlap and need
    no cross-level compositing. This only handles which boxes exist and where each one
    belongs in the full domain; callers pull whatever field data they need directly off
    each yielded box (`box[field_key]` for its own cells, `box.retrieve_ghost_zones(...)`
    for an expanded box).
    """
    domain_left_edge = numpy.array([float(value) for value in yt_dataset.domain_left_edge])
    domain_right_edge = numpy.array([float(value) for value in yt_dataset.domain_right_edge])
    resolution = numpy.array([int(num_cells) for num_cells in yt_dataset.domain_dimensions])
    cell_widths = (domain_right_edge - domain_left_edge) / resolution

    num_boxes_read = 0
    for grid_box in yt_dataset.index.grids:
        if int(grid_box.Level) != 0:
            continue
        box_left_edge = numpy.array([float(value) for value in grid_box.LeftEdge])
        start_index_float = (box_left_edge - domain_left_edge) / cell_widths
        start_index = numpy.round(start_index_float).astype(int)
        if numpy.max(numpy.abs(start_index_float - start_index)) > 1e-3:
            raise ValueError(
                f"box left edge {box_left_edge} does not align to the domain's cell"
                f" grid (cell_widths={cell_widths}); computed fractional start index"
                f" {start_index_float}.",
            )
        end_index = start_index + numpy.array(grid_box.ActiveDimensions)
        placement_slices = (
            slice(start_index[0], end_index[0]),
            slice(start_index[1], end_index[1]),
            slice(start_index[2], end_index[2]),
        )
        yield grid_box, placement_slices
        num_boxes_read += 1
    if num_boxes_read == 0:
        raise ValueError("no amr_level=0 boxes were found in this snapshot.")


## } MODULE
