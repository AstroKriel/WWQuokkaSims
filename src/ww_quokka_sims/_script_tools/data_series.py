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
from jormi.ww_io import json_io
from jormi.ww_validation import (
    validate_arrays,
    validate_types,
)

## local
from ww_quokka_sims.sim_io.snapshots import load_snapshot

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
    latex_label: str
    vi_value: float


@dataclass(frozen=True)
class DataSeries:
    points: list[DataPoint]

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
        data_dir: Path | None = None,
        overwrite: bool = False,
        cache_key: str | None = None,
    ):
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        self.snapshot_dirs = sorted(snapshot_dirs)
        self.field_name = field_name
        self.field_loader = field_loader
        self.use_parallel = bool(use_parallel)
        self.data_dir = data_dir
        self.overwrite = bool(overwrite)
        self.cache_key = cache_key if cache_key is not None else field_name

    def _cache_path(
        self,
    ) -> Path | None:
        """Resume cache, keyed by snapshot directory name; separate from whatever final output
        format the calling script writes via its own --save-data (this is purely to avoid
        recomputing a snapshot's volume integral on a rerun, not a user-facing artifact).

        `cache_key` defaults to `field_name`, but callers loading the same field from more than
        one input directory into the same `data_dir` (eg. a two-run comparison) must pass a
        distinct `cache_key` per loader, else they'd silently overwrite each other's cache.
        """
        if self.data_dir is None:
            return None
        return self.data_dir / f"{self.cache_key}-vi_series_cache.json"

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
                f"expected ScalarField_3D from `{field_args.field_loader.__name__}`, got {type(sfield_3d).__name__}.",
            )
        sim_time = sfield_3d.sim_time
        if (sim_time is None) or (not numpy.isfinite(sim_time)):
            raise ValueError(f"invalid sim_time for field: {sim_time!r}.")
        vi_value = field_operators.compute_sfield_volume_integral(sfield_3d=sfield_3d)
        return DataPoint(
            sim_time=float(sim_time),
            latex_label=sfield_3d.latex_label,
            vi_value=float(vi_value),
        )

    def run(
        self,
    ) -> DataSeries:
        cache_path = self._cache_path()
        cached_entries: dict = {}
        if (cache_path is not None) and (not self.overwrite) and cache_path.exists():
            cached_entries = json_io.read_json_file_into_dict(
                file_path=cache_path,
                verbose=False,
            )
        output_dict = dict(cached_entries)

        data_points: list[DataPoint] = []
        pending_field_args: list[FieldArgs] = []
        for snapshot_dir in self.snapshot_dirs:
            snapshot_dir = Path(snapshot_dir)
            cached = cached_entries.get(snapshot_dir.name)
            if cached is not None:
                data_points.append(
                    DataPoint(
                        sim_time=cached["sim_time"],
                        latex_label=cached["latex_label"],
                        vi_value=cached["vi_value"],
                    ),
                )
                continue
            pending_field_args.append(
                FieldArgs(
                    snapshot_dir=snapshot_dir,
                    field_name=self.field_name,
                    field_loader=self.field_loader,
                ),
            )
        if not pending_field_args:
            return DataSeries(points=data_points)

        def write_cache() -> None:
            if cache_path is None:
                return
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            json_io.save_dict_to_json_file(
                file_path=cache_path,
                input_dict=output_dict,
                overwrite=True,
                verbose=False,
            )

        ## load each pending snapshot in parallel if there are enough to justify it, else serial.
        ## the serial path caches after every snapshot; the parallel path can only cache once the
        ## whole batch returns, since jormi's parallel_dispatch has no per-task completion hook
        if self.use_parallel and (len(pending_field_args) > 5):
            new_points: list[DataPoint] = parallel_dispatch.run_in_parallel(
                worker_fn=LoadDataSeries.load_snapshot,
                grouped_args=pending_field_args,
                timeout_seconds=120,
                show_progress=True,
                enable_plotting=True,
            )
            for field_args, data_point in zip(pending_field_args, new_points):
                output_dict[field_args.snapshot_dir.name] = {
                    "sim_time": data_point.sim_time,
                    "latex_label": data_point.latex_label,
                    "vi_value": data_point.vi_value,
                }
            write_cache()
            data_points.extend(new_points)
        else:
            for field_args in pending_field_args:
                data_point = LoadDataSeries.load_snapshot(field_args=field_args)
                output_dict[field_args.snapshot_dir.name] = {
                    "sim_time": data_point.sim_time,
                    "latex_label": data_point.latex_label,
                    "vi_value": data_point.vi_value,
                }
                write_cache()
                data_points.append(data_point)
        return DataSeries(points=data_points)


## } MODULE
