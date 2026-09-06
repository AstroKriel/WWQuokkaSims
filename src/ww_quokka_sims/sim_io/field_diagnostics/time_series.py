## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import pathlib
import typing

from collections import abc as collections_abc

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import annotate_panel, manage_figure
from jormi.ww_validation import validate_arrays, validate_types

## local
from ww_quokka_sims.sim_io.snapshots import field_registry, load_snapshot

##
## === TIME POINT
##


@dataclasses.dataclass(frozen=True)
class TimePoint:
    sim_time: float
    value: float
    latex_label: str

    def save_to_file(
        self,
        file_path: pathlib.Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "sim_time": self.sim_time,
                "value": self.value,
                "latex_label": self.latex_label,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: pathlib.Path,
    ) -> "TimePoint":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            param_name="<TimePoint JSON>",
            required_keys={
                "sim_time",
                "value",
                "latex_label",
            },
        )
        return cls(
            sim_time=float(data["sim_time"]),
            value=float(data["value"]),
            latex_label=data["latex_label"],
        )


##
## === TIME SERIES
##


@dataclasses.dataclass(frozen=True)
class TimeSeries:
    """An in-memory collection of `TimePoint`s, one per snapshot; assembled by loading however many
    of the underlying per-snapshot files already exist, not itself saved as one file.
    """

    time_points: list[TimePoint]

    @property
    def num_time_points(
        self,
    ) -> int:
        return len(self.time_points)

    @property
    def latex_label(
        self,
    ) -> str:
        return self.time_points[0].latex_label

    def get_sorted_time_points(
        self,
    ) -> list[TimePoint]:
        return sorted(
            self.time_points,
            key=lambda time_point: time_point.sim_time,
        )


##
## === STATISTIC
##


@dataclasses.dataclass(frozen=True)
class FieldStatistic:
    name: str
    _compute_fn: collections_abc.Callable[[field_models.ScalarField_3D], float]

    def compute_statistic(
        self,
        field_3d: field_models.ScalarField_3D,
    ) -> float:
        return self._compute_fn(field_3d)


##
## === GENERATE TIME SERIES
##


@dataclasses.dataclass(frozen=True)
class TimePointArgs:
    snapshot_dir: pathlib.Path
    registered_field: field_registry.RegisteredField
    field_statistic: FieldStatistic
    amr_level: int = 0
    cache_file_path: pathlib.Path | None = None


@typing.final
class GenerateTimeSeries:

    def __init__(
        self,
        *,
        snapshot_dirs: list[pathlib.Path],
        registered_field: field_registry.RegisteredField,
        field_statistic: FieldStatistic,
        data_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        save_data: bool,
        save_figure: bool,
        num_workers: int | None = None,
        overwrite: bool = False,
        amr_level: int = 0,
        apply_log10_plot: bool = False,
    ):
        self.snapshot_dirs = sorted(snapshot_dirs)
        self.registered_field = registered_field
        self.field_statistic = field_statistic
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.save_data = save_data
        self.save_figure = save_figure
        self.num_workers = num_workers
        self.overwrite = overwrite
        self.amr_level = amr_level
        self.apply_log10_plot = apply_log10_plot

    def _get_cache_file_path(
        self,
        *,
        snapshot_dir: pathlib.Path,
    ) -> pathlib.Path:
        """Per-snapshot resume-cache path, hidden under `.cache/` so it is never mistaken for real output."""
        return self.data_dir / ".cache" / "time_series" / self.field_statistic.name / f"{self.registered_field.name}-{snapshot_dir.name}.json"

    @staticmethod
    def _compute_time_point(
        time_point_args: TimePointArgs,
    ) -> TimePoint:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=time_point_args.snapshot_dir,
                verbose=False,
        ) as quokka_snapshot:
            field_3d = time_point_args.registered_field.load(
                quokka_snapshot=quokka_snapshot,
                amr_level=time_point_args.amr_level,
            )
        assert isinstance(field_3d, field_models.ScalarField_3D)
        statistic = time_point_args.field_statistic.compute_statistic(field_3d)
        sim_time = field_3d.sim_time
        validate_types.ensure_finite_float(
            param=sim_time,
            param_name="sim_time",
        )
        assert sim_time is not None
        time_point = TimePoint(
            sim_time=float(sim_time),
            value=float(statistic),
            latex_label=field_3d.latex_label,
        )
        if time_point_args.cache_file_path is not None:
            manage_io.create_directory(
                directory=time_point_args.cache_file_path.parent,
                verbose=False,
            )
            time_point.save_to_file(file_path=time_point_args.cache_file_path)
        return time_point

    def _compute_time_series(
        self,
    ) -> TimeSeries:
        time_points: list[TimePoint] = []
        time_series_args: list[TimePointArgs] = []
        for snapshot_dir in self.snapshot_dirs:
            snapshot_dir = pathlib.Path(snapshot_dir)
            cache_file_path = self._get_cache_file_path(snapshot_dir=snapshot_dir)
            if not (self.overwrite) and cache_file_path.exists():
                time_point = TimePoint.load_from_file(file_path=cache_file_path)
                time_points.append(time_point)
            else:
                time_point_args = TimePointArgs(
                    snapshot_dir=snapshot_dir,
                    registered_field=self.registered_field,
                    field_statistic=self.field_statistic,
                    amr_level=self.amr_level,
                    cache_file_path=cache_file_path,
                )
                time_series_args.append(time_point_args)
        if (self.num_workers != 1) and (len(time_series_args) > 5):
            new_time_points: list[TimePoint] = parallel_dispatch.run_in_parallel(
                worker_fn=GenerateTimeSeries._compute_time_point,
                grouped_args=time_series_args,
                num_workers=self.num_workers,
                timeout_seconds=120,
                show_progress=True,
                enable_plotting=True,
            )
            time_points.extend(new_time_points)
        else:
            for time_point_args in time_series_args:
                time_point = GenerateTimeSeries._compute_time_point(time_point_args=time_point_args)
                time_points.append(time_point)
        return TimeSeries(time_points=time_points)

    @staticmethod
    def _as_arrays(
        *,
        time_points: list[TimePoint],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if not (time_points):
            return (
                numpy.asarray([], dtype=float),
                numpy.asarray([], dtype=float),
            )
        time_array = validate_arrays.as_1d([time_point.sim_time for time_point in time_points])
        values_array = validate_arrays.as_1d([time_point.value for time_point in time_points])
        return (
            time_array,
            values_array,
        )

    def _save(
        self,
        *,
        time_series: TimeSeries,
    ) -> None:
        manage_io.create_directory(
            directory=self.data_dir,
            verbose=False,
        )
        time_array, values_array = self._as_arrays(time_points=time_series.get_sorted_time_points())
        json_io.save_dict_to_json_file(
            file_path=self.data_dir / f"{self.registered_field.name}-{self.field_statistic.name}-time_series.json",
            input_dict={
                "sim_times": time_array,
                "values": values_array,
                "latex_label": time_series.latex_label,
            },
            overwrite=True,
            verbose=False,
        )

    def _plot(
        self,
        *,
        time_series: TimeSeries,
    ) -> None:
        fig, ax = manage_figure.create_figure()
        time_array, values_array = self._as_arrays(time_points=time_series.get_sorted_time_points())
        if time_array.size == 0:
            annotate_panel.add_text(
                panel=ax,
                x_pos_fraction=0.5,
                y_pos_fraction=0.5,
                label="no data",
                x_alignment="center",
                y_alignment="center",
            )
            return
        ylabel = f"${time_series.latex_label}$"
        fig_name = f"{self.registered_field.name}-{self.field_statistic.name}-time_series.png"
        if self.apply_log10_plot:
            if self.registered_field.expected_properties.is_strictly_positive:
                values_array = compute_array_stats.compute_safe_log10(values_array)
            else:
                values_array = compute_array_stats.compute_safe_log10(numpy.abs(values_array))
            ylabel = rf"$\log_{{10}}\big({time_series.latex_label}\big)$"
            fig_name = f"log10_{fig_name}"
        ax.plot(
            time_array,
            values_array,
            color="black",
            marker="o",
            ms=6,
            ls="-",
            lw=1.5,
        )
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        fig_path = self.figures_dir / fig_name
        manage_figure.save_figure(
            figure=fig,
            figure_path=fig_path,
            verbose=True,
        )

    def run(
        self,
    ) -> None:
        time_series = self._compute_time_series()
        if self.save_data:
            self._save(time_series=time_series)
        if self.save_figure:
            self._plot(time_series=time_series)


## } MODULE
