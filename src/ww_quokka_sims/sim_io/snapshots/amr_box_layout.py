## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import pathlib

from typing import TypeAlias

## personal
from jormi.ww_io import json_io

## local
from ww_quokka_sims.sim_io.snapshots import find_snapshots

##
## === TYPE ALIASES
##

CellIndex_3D: TypeAlias = tuple[int, int, int]
Resolution_3D: TypeAlias = tuple[int, int, int]
Bounds_3D: TypeAlias = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

##
## === DATA STRUCTURES
##


@dataclasses.dataclass(frozen=True)
class AMRBox:
    """
    One AMR box, as the cells it spans on its own level.

    `start_index` is the first cell and `end_index` is one past the last cell, counted on
    the box's own `amr_level` grid, so `end_index - start_index` is the box's resolution.
    """

    amr_level: int
    start_index: CellIndex_3D
    end_index: CellIndex_3D

    def __post_init__(
        self,
    ) -> None:
        if self.amr_level < 0:
            raise ValueError(f"`amr_level` must be non-negative; got {self.amr_level}.")
        for axis_index in range(3):
            if self.end_index[axis_index] <= self.start_index[axis_index]:
                raise ValueError(
                    f"box spans no cells along axis {axis_index}:"
                    f" start_index={self.start_index}, end_index={self.end_index}.",
                )

    @property
    def resolution(
        self,
    ) -> Resolution_3D:
        return (
            self.end_index[0] - self.start_index[0],
            self.end_index[1] - self.start_index[1],
            self.end_index[2] - self.start_index[2],
        )


@dataclasses.dataclass(frozen=True)
class AMRBoxLayout:
    """Where every AMR box sits in one snapshot, with the metadata needed to place it in the domain."""

    sim_time: float
    step_index: find_snapshots.StepIndex
    refinement_ratio: int
    base_resolution: Resolution_3D
    domain_bounds: Bounds_3D
    boxes: list[AMRBox]

    def __post_init__(
        self,
    ) -> None:
        if self.refinement_ratio < 2:
            raise ValueError(f"`refinement_ratio` must be at least 2; got {self.refinement_ratio}.")
        for box in self.boxes:
            level_resolution = self.get_level_resolution(amr_level=box.amr_level)
            for axis_index in range(3):
                box_is_inside = (box.start_index[axis_index]
                                 >= 0) and (box.end_index[axis_index] <= level_resolution[axis_index])
                if not box_is_inside:
                    raise ValueError(
                        f"box {box} extends outside the level-{box.amr_level} domain"
                        f" of {level_resolution} cells.",
                    )

    @property
    def max_amr_level(
        self,
    ) -> int:
        return max(_box.amr_level for _box in self.boxes)

    def get_level_resolution(
        self,
        *,
        amr_level: int,
    ) -> Resolution_3D:
        scale = self.refinement_ratio**amr_level
        return (
            self.base_resolution[0] * scale,
            self.base_resolution[1] * scale,
            self.base_resolution[2] * scale,
        )

    def get_cell_widths(
        self,
        *,
        amr_level: int,
    ) -> tuple[float, float, float]:
        level_resolution = self.get_level_resolution(amr_level=amr_level)
        return (
            (self.domain_bounds[0][1] - self.domain_bounds[0][0]) / level_resolution[0],
            (self.domain_bounds[1][1] - self.domain_bounds[1][0]) / level_resolution[1],
            (self.domain_bounds[2][1] - self.domain_bounds[2][0]) / level_resolution[2],
        )

    def get_boxes_on_level(
        self,
        *,
        amr_level: int,
    ) -> list[AMRBox]:
        return [_box for _box in self.boxes if _box.amr_level == amr_level]

    def get_box_bounds(
        self,
        box: AMRBox,
    ) -> Bounds_3D:
        """Return the physical extent of `box` along each axis."""
        cell_widths = self.get_cell_widths(amr_level=box.amr_level)
        axis_bounds = [
            (
                self.domain_bounds[_axis_index][0] + box.start_index[_axis_index] * cell_widths[_axis_index],
                self.domain_bounds[_axis_index][0] + box.end_index[_axis_index] * cell_widths[_axis_index],
            ) for _axis_index in range(3)
        ]
        return (axis_bounds[0], axis_bounds[1], axis_bounds[2])

    def get_boxes_crossing_plane(
        self,
        *,
        axis_index: int,
        coordinate: float,
    ) -> list[AMRBox]:
        """Return the boxes whose extent along `axis_index` contains `coordinate`, lower edge inclusive."""
        crossing_boxes = []
        for box in self.boxes:
            lower_edge, upper_edge = self.get_box_bounds(box)[axis_index]
            if (lower_edge <= coordinate) and (coordinate < upper_edge):
                crossing_boxes.append(box)
        return crossing_boxes

    def save_to_file(
        self,
        *,
        file_path: pathlib.Path,
    ) -> None:
        output_dict = {
            "sim_time":
            self.sim_time,
            "step_index":
            self.step_index.value,
            "refinement_ratio":
            self.refinement_ratio,
            "base_resolution":
            list(self.base_resolution),
            "domain_bounds": [list(_axis_bounds) for _axis_bounds in self.domain_bounds],
            "boxes": [
                {
                    "amr_level": _box.amr_level,
                    "start_index": list(_box.start_index),
                    "end_index": list(_box.end_index),
                } for _box in self.boxes
            ],
        }
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict=output_dict,
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        *,
        file_path: pathlib.Path,
    ) -> "AMRBoxLayout":
        input_dict = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        base_resolution = input_dict["base_resolution"]
        domain_bounds = input_dict["domain_bounds"]
        boxes = [
            AMRBox(
                amr_level=int(_box_dict["amr_level"]),
                start_index=(
                    int(_box_dict["start_index"][0]),
                    int(_box_dict["start_index"][1]),
                    int(_box_dict["start_index"][2]),
                ),
                end_index=(
                    int(_box_dict["end_index"][0]),
                    int(_box_dict["end_index"][1]),
                    int(_box_dict["end_index"][2]),
                ),
            ) for _box_dict in input_dict["boxes"]
        ]
        return cls(
            sim_time=float(input_dict["sim_time"]),
            step_index=find_snapshots.StepIndex.from_value(step_index_value=int(input_dict["step_index"])),
            refinement_ratio=int(input_dict["refinement_ratio"]),
            base_resolution=(
                int(base_resolution[0]),
                int(base_resolution[1]),
                int(base_resolution[2]),
            ),
            domain_bounds=(
                (float(domain_bounds[0][0]), float(domain_bounds[0][1])),
                (float(domain_bounds[1][0]), float(domain_bounds[1][1])),
                (float(domain_bounds[2][0]), float(domain_bounds[2][1])),
            ),
            boxes=boxes,
        )


## } MODULE
