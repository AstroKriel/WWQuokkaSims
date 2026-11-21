## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import typing

## third-party
import numpy

## local
from ww_quokka_sims.sim_io.snapshots import amr_box_layout

##
## === PUBLIC FUNCTIONS
##


def read_amr_boxes(
    *,
    yt_dataset: typing.Any,
) -> list[amr_box_layout.AMRBox]:
    """
    Read every box on every AMR level from the yt index, as cell indices on the box's own level.

    Each box's start index is recovered from its left edge and its level's cell width, and
    must land on a whole cell; a box that does not is a sign the snapshot metadata and the
    assumed refinement ratio disagree, so it raises rather than rounding silently.
    """
    domain_left_edge = numpy.array([float(_value) for _value in yt_dataset.domain_left_edge])
    domain_right_edge = numpy.array([float(_value) for _value in yt_dataset.domain_right_edge])
    base_resolution = numpy.array([int(_num_cells) for _num_cells in yt_dataset.domain_dimensions])
    refinement_ratio = int(yt_dataset.refine_by)
    amr_boxes = []
    for yt_box in yt_dataset.index.grids:
        amr_level = int(yt_box.Level)
        level_resolution = base_resolution * refinement_ratio**amr_level
        level_cell_widths = (domain_right_edge - domain_left_edge) / level_resolution
        box_left_edge = numpy.array([float(_value) for _value in yt_box.LeftEdge])
        start_index_float = (box_left_edge - domain_left_edge) / level_cell_widths
        start_index = numpy.round(start_index_float).astype(int)
        if numpy.max(numpy.abs(start_index_float - start_index)) > 1e-3:
            raise ValueError(
                f"level-{amr_level} box left edge {box_left_edge} does not align to its level's"
                f" cell grid (cell_widths={level_cell_widths}); computed fractional start index"
                f" {start_index_float}.",
            )
        end_index = start_index + numpy.array([int(_num_cells) for _num_cells in yt_box.ActiveDimensions])
        amr_box = amr_box_layout.AMRBox(
            amr_level=amr_level,
            start_index=(
                int(start_index[0]),
                int(start_index[1]),
                int(start_index[2]),
            ),
            end_index=(
                int(end_index[0]),
                int(end_index[1]),
                int(end_index[2]),
            ),
        )
        amr_boxes.append(amr_box)
    if not amr_boxes:
        raise ValueError("the yt index lists no boxes.")
    return amr_boxes


## } MODULE
