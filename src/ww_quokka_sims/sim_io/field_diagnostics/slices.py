## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

##
## === SLICED FIELD
##

AxisBounds = tuple[tuple[float, float], tuple[float, float]]  # ((xmin, xmax), (ymin, ymax))


@dataclass(frozen=True)
class SlicedField:
    """A single 2D slice, self-contained enough to plot without the raw snapshot or uniform_domain."""

    sarray_2d: numpy.ndarray
    axis_bounds: AxisBounds
    min_value: float
    max_value: float
    comp_label: str
    step_time: float
    step_index: int
    amr_level: int = 0

    def save_to_file(
        self,
        file_path: Path,
    ) -> None:
        numpy.savez(
            file_path,
            sarray_2d=self.sarray_2d,
            axis_bounds=numpy.array(self.axis_bounds),
            comp_label=self.comp_label,
            min_value=self.min_value,
            max_value=self.max_value,
            step_time=self.step_time,
            step_index=self.step_index,
            amr_level=self.amr_level,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: Path,
    ) -> "SlicedField":
        with numpy.load(file_path) as npz:
            saved_bounds = npz["axis_bounds"]
            axis_bounds: AxisBounds = (
                (float(saved_bounds[0][0]), float(saved_bounds[0][1])),
                (float(saved_bounds[1][0]), float(saved_bounds[1][1])),
            )
            return cls(
                sarray_2d=npz["sarray_2d"],
                axis_bounds=axis_bounds,
                min_value=float(npz["min_value"]),
                max_value=float(npz["max_value"]),
                comp_label=str(npz["comp_label"]),
                step_time=float(npz["step_time"]),
                step_index=int(npz["step_index"]),
                amr_level=int(npz["amr_level"]),
            )


## } MODULE
