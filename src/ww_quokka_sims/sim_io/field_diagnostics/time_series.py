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
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import json_io
from jormi.ww_plots import annotate_panel, manage_figure
from jormi.ww_validation import validate_arrays, validate_types

## local
from ww_quokka_sims.sim_io.snapshots import load_snapshot

##
## === TIME POINT
##


@dataclass(frozen=True)
class TimePoint:
    sim_time: float
    value: float
    latex_label: str

    def save_to_file(
        self,
        file_path: Path,
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
        file_path: Path,
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


@dataclass(frozen=True)
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
## === GENERATE TIME SERIES
##


@dataclass(frozen=True)
class ResolvedFieldArgs:
    snapshot_dir: Path
    field_name: str
    field_loader: Callable
    statistic_fn: Callable
    amr_level: int = 0
    cache_file_path: Path | None = None


@final
class GenerateTimeSeries:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        field_name: str,
        field_loader: Callable,
        statistic_name: str,
        statistic_fn: Callable[[field_models.ScalarField_3D], float],
        data_dir: Path,
        figures_dir: Path,
        save_data: bool,
        save_figure: bool,
        num_workers: int | None = None,
        overwrite: bool = False,
        amr_level: int = 0,
        apply_log10_plot: bool = False,
    ):
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        self.snapshot_dirs = sorted(snapshot_dirs)
        self.field_name = field_name
        self.field_loader = field_loader
        self.statistic_name = statistic_name
        self.statistic_fn = statistic_fn
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
        snapshot_dir: Path,
    ) -> Path:
        """Per-snapshot resume-cache path, hidden under `.cache/` so it is never mistaken for real output."""
        return self.data_dir / ".cache" / "time_series" / self.statistic_name / f"{self.field_name}-{snapshot_dir.name}.json"

    @staticmethod
    def _compute_time_point(
        field_args: ResolvedFieldArgs,
    ) -> TimePoint:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=field_args.snapshot_dir,
                verbose=False,
        ) as snapshot:
            sfield_3d = field_args.field_loader(snapshot, amr_level=field_args.amr_level)
        if not isinstance(sfield_3d, field_models.ScalarField_3D):
            raise TypeError(
                f"expected ScalarField_3D from `{field_args.field_loader.__name__}`, got {type(sfield_3d).__name__}.",
            )
        sim_time = sfield_3d.sim_time
        if (sim_time is None) or (not numpy.isfinite(sim_time)):
            raise ValueError(f"invalid sim_time for field: {sim_time!r}.")
        value = field_args.statistic_fn(sfield_3d)
        time_point = TimePoint(
            sim_time=float(sim_time),
            value=float(value),
            latex_label=sfield_3d.latex_label,
        )
        if field_args.cache_file_path is not None:
            field_args.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            time_point.save_to_file(field_args.cache_file_path)
        return time_point

    def _compute_time_series(
        self,
    ) -> TimeSeries:
        time_points: list[TimePoint] = []
        pending_field_args: list[ResolvedFieldArgs] = []
        for snapshot_dir in self.snapshot_dirs:
            snapshot_dir = Path(snapshot_dir)
            cache_file_path = self._get_cache_file_path(snapshot_dir)
            if (not self.overwrite) and cache_file_path.exists():
                time_points.append(TimePoint.load_from_file(cache_file_path))
                continue
            pending_field_args.append(
                ResolvedFieldArgs(
                    snapshot_dir=snapshot_dir,
                    field_name=self.field_name,
                    field_loader=self.field_loader,
                    statistic_fn=self.statistic_fn,
                    amr_level=self.amr_level,
                    cache_file_path=cache_file_path,
                ),
            )
        if not pending_field_args:
            return TimeSeries(time_points=time_points)
        if (self.num_workers != 1) and (len(pending_field_args) > 5):
            new_time_points: list[TimePoint] = parallel_dispatch.run_in_parallel(
                worker_fn=GenerateTimeSeries._compute_time_point,
                grouped_args=pending_field_args,
                num_workers=self.num_workers,
                timeout_seconds=120,
                show_progress=True,
                enable_plotting=True,
            )
            time_points.extend(new_time_points)
        else:
            for field_args in pending_field_args:
                time_points.append(GenerateTimeSeries._compute_time_point(field_args=field_args))
        return TimeSeries(time_points=time_points)

    @staticmethod
    def _as_arrays(
        sorted_time_points: list[TimePoint],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if not sorted_time_points:
            return (
                numpy.asarray([], dtype=float),
                numpy.asarray([], dtype=float),
            )
        time_array = validate_arrays.as_1d([time_point.sim_time for time_point in sorted_time_points])
        values_array = validate_arrays.as_1d([time_point.value for time_point in sorted_time_points])
        return (
            time_array,
            values_array,
        )

    def _save(
        self,
        *,
        time_series: TimeSeries,
    ) -> None:
        self.data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        time_array, values_array = self._as_arrays(time_series.get_sorted_time_points())
        json_io.save_dict_to_json_file(
            file_path=self.data_dir / f"{self.field_name}-{self.statistic_name}-time_series.json",
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
        time_array, values_array = self._as_arrays(time_series.get_sorted_time_points())
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
        plot_values = values_array
        ylabel = f"${time_series.latex_label}$"
        fig_name = f"{self.field_name}-{self.statistic_name}-time_series.png"
        if self.apply_log10_plot:
            plot_values = compute_array_stats.compute_safe_log10(numpy.abs(values_array))
            ylabel = rf"$\log_{{10}}\big({time_series.latex_label}\big)$"
            fig_name = f"log10_{self.field_name}-{self.statistic_name}-time_series.png"
        ax.plot(
            time_array,
            plot_values,
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
