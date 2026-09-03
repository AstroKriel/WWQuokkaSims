## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import json_io
from jormi.ww_validation import validate_arrays, validate_types

##
## === TIME POINT
##


@dataclass(frozen=True)
class TimePoint:
    sim_time: float
    latex_label: str
    value: float

    def save_to_file(
        self,
        file_path: Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "sim_time": self.sim_time,
                "latex_label": self.latex_label,
                "value": self.value,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: Path,
    ) -> "TimePoint":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={"sim_time", "latex_label", "value"},
            param_name="<TimePoint JSON>",
        )
        return cls(
            sim_time=float(data["sim_time"]),
            latex_label=data["latex_label"],
            value=float(data["value"]),
        )


##
## === TIME SERIES
##


@dataclass(frozen=True)
class TimeSeries:
    """An in-memory collection of `TimePoint`s, one per snapshot; assembled by loading however many
    of the underlying per-snapshot files already exist, not itself saved as one file.
    """

    points: list[TimePoint]

    @property
    def num_points(
        self,
    ) -> int:
        return len(self.points)

    @property
    def latex_label(
        self,
    ) -> str:
        return self.points[0].latex_label

    def get_sorted_arrays(
        self,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if not self.points:
            return (
                numpy.asarray([], dtype=float),
                numpy.asarray([], dtype=float),
            )
        sorted_points = sorted(self.points, key=lambda point: point.sim_time)
        time_array = validate_arrays.as_1d([point.sim_time for point in sorted_points])
        values_array = validate_arrays.as_1d([point.value for point in sorted_points])
        return (
            time_array,
            values_array,
        )


## } MODULE
