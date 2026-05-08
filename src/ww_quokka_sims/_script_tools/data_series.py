## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import (
    field_operators,
    field_models,
)
from jormi.ww_fns import parallel_dispatch
from jormi.ww_validation import (
    validate_arrays,
    validate_types,
)

## local
from ww_quokka_sims.sim_io import load_snapshot

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class FieldArgs:
    snapshot_dir: Path
    field_name: str
    field_loader: Callable


@dataclass(frozen=True)
class DataPoint:
    sim_time: float
    vi_value: float


@dataclass(frozen=True)
class DataSeries:
    points: list[DataPoint]

    @property
    def num_points(
        self,
    ) -> int:
        return len(self.points)

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
        values_array = validate_arrays.as_1d([point.vi_value for point in sorted_points])
        return (
            time_array,
            values_array,
        )


##
## === FIELD PROCESSING
##


@final
class LoadDataSeries:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        field_name: str,
        field_loader: Callable,
        use_parallel: bool = True,
    ):
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        self.snapshot_dirs = sorted(snapshot_dirs)
        self.field_name = field_name
        self.field_loader = field_loader
        self.use_parallel = bool(use_parallel)

    @staticmethod
    def load_snapshot(
        field_args: FieldArgs,
    ) -> DataPoint:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=field_args.snapshot_dir,
                verbose=False,
        ) as snapshot:
            sfield_3d = field_args.field_loader(snapshot)
        if not isinstance(sfield_3d, field_models.ScalarField_3D):
            raise TypeError(
                f"Expected ScalarField_3D from `{field_args.field_loader.__name__}`, got {type(sfield_3d).__name__}.",
            )
        sim_time = sfield_3d.sim_time
        if (sim_time is None) or (not numpy.isfinite(sim_time)):
            raise ValueError(f"Invalid sim_time for field: {sim_time!r}")
        vi_value = field_operators.compute_sfield_volume_integral(sfield_3d=sfield_3d)
        return DataPoint(
            sim_time=float(sim_time),
            vi_value=float(vi_value),
        )

    def run(
        self,
    ) -> DataSeries:
        grouped_field_args: list[FieldArgs] = [
            FieldArgs(
                snapshot_dir=Path(snapshot_dir),
                field_name=self.field_name,
                field_loader=self.field_loader,
            ) for snapshot_dir in self.snapshot_dirs
        ]
        if not grouped_field_args:
            return DataSeries(points=[])
        ## load each snapshot in parallel if the series is large enough to justify it, else serial
        if self.use_parallel and (len(grouped_field_args) > 5):
            data_points: list[DataPoint] = parallel_dispatch.run_in_parallel(
                worker_fn=LoadDataSeries.load_snapshot,
                grouped_args=grouped_field_args,
                timeout_seconds=120,
                show_progress=True,
                enable_plotting=True,
            )
        else:
            data_points = [
                LoadDataSeries.load_snapshot(field_args=field_args) for field_args in grouped_field_args
            ]
        return DataSeries(points=data_points)


## } MODULE
