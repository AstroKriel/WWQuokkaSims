## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields.fields_3d import (
    field_models,
    field_operators,
)
from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import json_io, manage_log
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._scripts.snapshot_tools import (
    cli,
    field_registry,
)
from ww_quokka_sims.sim_io.field_diagnostics import time_series
from ww_quokka_sims.sim_io.snapshots import load_snapshot

##
## === STATISTICS
##

_STATISTIC_LOOKUP: dict[str, Callable[[field_models.ScalarField_3D], float]] = {
    "total": field_operators.compute_sfield_volume_integral,
    "rms": field_operators.compute_sfield_rms,
}

##
## === TIME SERIES
##


@dataclass(frozen=True)
class ResolvedFieldArgs:
    snapshot_dir: Path
    field_name: str
    field_loader: Callable
    statistic_fn: Callable
    cache_file_path: Path | None = None
    amr_level: int = 0


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

    def _cache_file_path(
        self,
        snapshot_dir: Path,
    ) -> Path:
        """Per-snapshot resume-cache path, hidden under `.cache/` so it is never mistaken for real output."""
        return self.data_dir / ".cache" / "time_series" / self.statistic_name / f"{self.field_name}-{snapshot_dir.name}.json"

    @staticmethod
    def _compute_snapshot_point(
        field_args: ResolvedFieldArgs,
    ) -> time_series.TimePoint:
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
        statistic_value = field_args.statistic_fn(sfield_3d)
        data_point = time_series.TimePoint(
            sim_time=float(sim_time),
            latex_label=sfield_3d.latex_label,
            value=float(statistic_value),
        )
        if field_args.cache_file_path is not None:
            field_args.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            data_point.save_to_file(field_args.cache_file_path)
        return data_point

    def _compute_field_series(
        self,
    ) -> time_series.TimeSeries:
        data_points: list[time_series.TimePoint] = []
        pending_field_args: list[ResolvedFieldArgs] = []
        for snapshot_dir in self.snapshot_dirs:
            snapshot_dir = Path(snapshot_dir)
            cache_file_path = self._cache_file_path(snapshot_dir)
            if (not self.overwrite) and cache_file_path.exists():
                data_points.append(time_series.TimePoint.load_from_file(cache_file_path))
                continue
            pending_field_args.append(
                ResolvedFieldArgs(
                    snapshot_dir=snapshot_dir,
                    field_name=self.field_name,
                    field_loader=self.field_loader,
                    statistic_fn=self.statistic_fn,
                    cache_file_path=cache_file_path,
                    amr_level=self.amr_level,
                ),
            )
        if not pending_field_args:
            return time_series.TimeSeries(points=data_points)

        if (self.num_workers != 1) and (len(pending_field_args) > 5):
            new_points: list[time_series.TimePoint] = parallel_dispatch.run_in_parallel(
                worker_fn=GenerateTimeSeries._compute_snapshot_point,
                grouped_args=pending_field_args,
                num_workers=self.num_workers,
                timeout_seconds=120,
                show_progress=True,
                enable_plotting=True,
            )
            data_points.extend(new_points)
        else:
            for field_args in pending_field_args:
                data_points.append(GenerateTimeSeries._compute_snapshot_point(field_args=field_args))
        return time_series.TimeSeries(points=data_points)

    def _save_series(
        self,
        *,
        field_series: time_series.TimeSeries,
    ) -> None:
        self.data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        time_array, values_array = field_series.get_sorted_arrays()
        json_io.save_dict_to_json_file(
            file_path=self.data_dir / f"{self.field_name}-{self.statistic_name}-time_series.json",
            input_dict={
                "sim_times": time_array,
                "values": values_array,
            },
            overwrite=True,
            verbose=False,
        )

    def run(
        self,
    ) -> None:
        field_series = self._compute_field_series()
        if self.save_data:
            self._save_series(field_series=field_series)
        if not self.save_figure:
            return
        fig, ax = manage_figure.create_figure()
        time_array, values_array = field_series.get_sorted_arrays()
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
        ylabel = f"${field_series.latex_label}$"
        fig_name = f"{self.field_name}-{self.statistic_name}-time_series.png"
        if self.apply_log10_plot:
            plot_values = compute_array_stats.compute_safe_log10(numpy.abs(values_array))
            ylabel = rf"$\log_{{10}}\big({field_series.latex_label}\big)$"
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


##
## === DIAGNOSTIC PIPELINE
##


@final
class DiagnosticPipeline:

    def __init__(
        self,
        *,
        snapshot_args: cli.SnapshotArgs,
        field_args: cli.FieldArgs,
        diagnostic_output_args: cli.DiagnosticOutputArgs,
        statistic_name: str,
        num_workers: int | None = None,
        apply_log10_plot: bool = False,
    ):
        field_registry.validate_fields(
            field_names=field_args.fields,
            allowed_types=(field_models.ScalarField_3D, ),
        )
        if statistic_name not in _STATISTIC_LOOKUP:
            raise ValueError(f"unknown statistic `{statistic_name}`; expected one of {sorted(_STATISTIC_LOOKUP)}.")
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_args.fields)
        self.amr_level = field_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args
        self.statistic_name = statistic_name
        self.num_workers = num_workers
        self.apply_log10_plot = apply_log10_plot

    def _pipeline(
        self,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        statistic_fn = _STATISTIC_LOOKUP[self.statistic_name]
        for field_name in self.fields_to_plot:
            registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
            generate_time_series = GenerateTimeSeries(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                field_name=field_name,
                field_loader=registered_field.loader,
                statistic_name=self.statistic_name,
                statistic_fn=statistic_fn,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                num_workers=self.num_workers,
                overwrite=self.diagnostic_output_args.overwrite,
                amr_level=self.amr_level,
                apply_log10_plot=self.apply_log10_plot,
            )
            generate_time_series.run()

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.diagnostic_output_args,
            allow_index_width=False,
        )
        if resolved_inputs is not None:
            self._pipeline(resolved_inputs)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    parser = argparse.ArgumentParser(
        description="Generate a time-evolving statistic of Quokka snapshots.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_write=True,
                allow_figures=True,
                allow_parallel=True,
            ),
        ],
    )
    parser.add_argument(
        "--apply-log10-plot",
        action="store_true",
        default=False,
        help="Apply log10(|field|) to the plotted field (does not affect the saved `.json` datasets).",
    )
    parser.add_argument(
        "--statistic",
        type=str,
        choices=sorted(_STATISTIC_LOOKUP.keys()),
        default="total",
        help="Statistic applied to each snapshot's field before building its time series.",
    )
    user_args = parser.parse_args()
    diagnostic_pipeline = DiagnosticPipeline(
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args),
        field_args=cli.FieldArgs.from_user_args(user_args),
        diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args),
        statistic_name=user_args.statistic,
        num_workers=user_args.num_workers,
        apply_log10_plot=user_args.apply_log10_plot,
    )
    diagnostic_pipeline.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
