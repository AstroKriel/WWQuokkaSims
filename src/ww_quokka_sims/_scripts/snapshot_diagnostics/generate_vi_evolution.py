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
## === FIELD PROCESSING
##


@dataclass(frozen=True)
class ResolvedFieldArgs:
    snapshot_dir: Path
    field_name: str
    field_loader: Callable
    cache_file_path: Path | None = None
    amr_level: int = 0


@final
class LoadTimeSeries:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        field_name: str,
        field_loader: Callable,
        num_workers: int | None = None,
        data_dir: Path | None = None,
        overwrite: bool = False,
        amr_level: int = 0,
    ):
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        self.snapshot_dirs = sorted(snapshot_dirs)
        self.field_name = field_name
        self.field_loader = field_loader
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.amr_level = amr_level

    def _cache_file_path(
        self,
        snapshot_dir: Path,
    ) -> Path | None:
        """Resume cache, one file per snapshot; separate from whatever final output format the
        calling script writes via its own --save-data (this is purely to avoid recomputing a
        snapshot's volume integral on a rerun, not a user-facing artifact). Each snapshot's file
        is independent, so a crash never risks a previously-completed snapshot's result.
        """
        if self.data_dir is None:
            return None
        return self.data_dir / f"{self.field_name}-{snapshot_dir.name}.json"

    @staticmethod
    def load_snapshot(
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
        vi_value = field_operators.compute_sfield_volume_integral(sfield_3d=sfield_3d)
        data_point = time_series.TimePoint(
            sim_time=float(sim_time),
            latex_label=sfield_3d.latex_label,
            value=float(vi_value),
        )
        if field_args.cache_file_path is not None:
            ## saved by the worker itself (not the orchestrating run() below), so this holds
            ## whether the snapshot was dispatched serially or in a parallel worker process;
            ## each snapshot's result is persisted the moment it's computed, never batched
            field_args.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            data_point.save_to_file(field_args.cache_file_path)
        return data_point

    def run(
        self,
    ) -> time_series.TimeSeries:
        data_points: list[time_series.TimePoint] = []
        pending_field_args: list[ResolvedFieldArgs] = []
        for snapshot_dir in self.snapshot_dirs:
            snapshot_dir = Path(snapshot_dir)
            cache_file_path = self._cache_file_path(snapshot_dir)
            if (cache_file_path is not None) and (not self.overwrite) and cache_file_path.exists():
                data_points.append(time_series.TimePoint.load_from_file(cache_file_path))
                continue
            pending_field_args.append(
                ResolvedFieldArgs(
                    snapshot_dir=snapshot_dir,
                    field_name=self.field_name,
                    field_loader=self.field_loader,
                    cache_file_path=cache_file_path,
                    amr_level=self.amr_level,
                ),
            )
        if not pending_field_args:
            return time_series.TimeSeries(points=data_points)

        ## load each pending snapshot in parallel if there are enough to justify it, else serial;
        ## either way, `load_snapshot` above persists each result itself as it completes
        if (self.num_workers != 1) and (len(pending_field_args) > 5):
            new_points: list[time_series.TimePoint] = parallel_dispatch.run_in_parallel(
                worker_fn=LoadTimeSeries.load_snapshot,
                grouped_args=pending_field_args,
                num_workers=self.num_workers,
                timeout_seconds=120,
                show_progress=True,
                enable_plotting=True,
            )
            data_points.extend(new_points)
        else:
            for field_args in pending_field_args:
                data_points.append(LoadTimeSeries.load_snapshot(field_args=field_args))
        return time_series.TimeSeries(points=data_points)


##
## === FIGURE RENDERING
##


@final
class GenerateTimeSeries:

    def __init__(
        self,
        *,
        data_dir: Path,
        figures_dir: Path,
        field_name: str,
        save_data: bool,
        save_figure: bool,
        apply_log10_plot: bool = False,
    ):
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.save_data = save_data
        self.save_figure = save_figure
        self.apply_log10_plot = apply_log10_plot

    def _save_series(
        self,
        *,
        vi_series: time_series.TimeSeries,
        data_dir: Path,
    ) -> None:
        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        time_array, values_array = vi_series.get_sorted_arrays()
        json_io.save_dict_to_json_file(
            file_path=data_dir / f"{self.field_name}-vi_evolution.json",
            input_dict={
                "sim_times": time_array,
                "vi_values": values_array,
            },
            overwrite=True,
            verbose=False,
        )

    def run(
        self,
        *,
        vi_series: time_series.TimeSeries,
    ) -> None:
        ## optionally write the time series data to JSON
        if self.save_data:
            self._save_series(
                vi_series=vi_series,
                data_dir=self.data_dir,
            )
        if not self.save_figure:
            return
        fig, ax = manage_figure.create_figure()
        time_array, values_array = vi_series.get_sorted_arrays()
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
        ylabel = f"${vi_series.latex_label}$"
        fig_name = f"{self.field_name}-time_evolution.png"
        if self.apply_log10_plot:
            plot_values = compute_array_stats.compute_safe_log10(numpy.abs(values_array))
            ylabel = rf"$\log_{{10}}\big({vi_series.latex_label}\big)$"
            fig_name = f"log10_{self.field_name}-time_evolution.png"
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
        num_workers: int | None = None,
        apply_log10_plot: bool = False,
    ):
        field_registry.validate_fields(
            field_names=field_args.fields,
            allowed_types=(field_models.ScalarField_3D, ),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_args.fields)
        self.amr_level = field_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args
        self.num_workers = num_workers
        self.apply_log10_plot = apply_log10_plot

    def _generate_fields(
        self,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        for field_name in self.fields_to_plot:
            registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
            loader = LoadTimeSeries(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                field_name=field_name,
                field_loader=registered_field.loader,
                num_workers=self.num_workers,
                data_dir=resolved_inputs.data_dir,
                overwrite=self.diagnostic_output_args.overwrite,
                amr_level=self.amr_level,
            )
            vi_series = loader.run()
            generate_time_series = GenerateTimeSeries(
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                field_name=field_name,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                apply_log10_plot=self.apply_log10_plot,
            )
            generate_time_series.run(vi_series=vi_series)

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.diagnostic_output_args,
            allow_index_width=False,
        )
        if resolved_inputs is not None:
            self._generate_fields(resolved_inputs)


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
    user_args = parser.parse_args()
    diagnostic_pipeline = DiagnosticPipeline(
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args),
        field_args=cli.FieldArgs.from_user_args(user_args),
        diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args),
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
