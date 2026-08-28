## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from dataclasses import dataclass
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import json_io, manage_log
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._script_tools import (
    cli,
    data_series,
    field_registry,
)
from ww_quokka_sims.sim_io import find_snapshots

##
## === FIGURE RENDERING
##


@final
class GenerateDataSeries:

    def __init__(
        self,
        *,
        data_dir: Path,
        figures_dir: Path,
        field_name: str,
        save_data: bool,
        save_figure: bool,
        apply_log10: bool = False,
    ):
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.save_data = save_data
        self.save_figure = save_figure
        self.apply_log10 = bool(apply_log10)

    def _save_series(
        self,
        *,
        vi_series: data_series.DataSeries,
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
        vi_series: data_series.DataSeries,
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
        if self.apply_log10:
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
## === SCRIPT INTERFACE
##


@dataclass(frozen=True)
class ResolvedInputs:
    snapshot_dirs: list[Path]
    data_dir: Path
    figures_dir: Path


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        snapshot_tag: str,
        fields_to_plot: list[str],
        save_data: bool,
        save_figure: bool,
        overwrite: bool = False,
        data_dir: Path | None = None,
        figures_dir: Path | None = None,
        use_parallel: bool = True,
        apply_log10: bool = False,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        cli.ensure_save_flag_selected(
            save_figure=save_figure,
            save_data=save_data,
        )
        field_registry.validate_fields(field_names=fields_to_plot)
        self.input_dir = Path(input_dir)
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = list(fields_to_plot)
        self.save_data = save_data
        self.save_figure = save_figure
        self.overwrite = bool(overwrite)
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.figures_dir = Path(figures_dir) if figures_dir is not None else None
        self.use_parallel = bool(use_parallel)
        self.apply_log10 = bool(apply_log10)

    def _resolve_inputs(
        self,
    ) -> ResolvedInputs | None:
        snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
            input_dir=self.input_dir,
            snapshot_tag=self.snapshot_tag,
        )
        if not snapshot_dirs:
            return None
        data_dir = cli.resolve_output_dir(
            output_dir=self.data_dir,
            default_dir=snapshot_dirs[0].parent,
        )
        figures_dir = cli.resolve_output_dir(
            output_dir=self.figures_dir,
            default_dir=data_dir,
        )
        return ResolvedInputs(
            snapshot_dirs=snapshot_dirs,
            data_dir=data_dir,
            figures_dir=figures_dir,
        )

    def _generate_fields(
        self,
        resolved_inputs: ResolvedInputs,
    ) -> None:
        for field_name in self.fields_to_plot:
            field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
            loader = data_series.LoadDataSeries(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                field_name=field_name,
                field_loader=field_meta.loader,
                use_parallel=self.use_parallel,
                data_dir=resolved_inputs.data_dir,
                overwrite=self.overwrite,
            )
            vi_series = loader.run()
            generate_data_series = GenerateDataSeries(
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                field_name=field_name,
                save_data=self.save_data,
                save_figure=self.save_figure,
                apply_log10=self.apply_log10,
            )
            generate_data_series.run(vi_series=vi_series)

    def run(
        self,
    ) -> None:
        resolved_inputs = self._resolve_inputs()
        if resolved_inputs is not None:
            self._generate_fields(resolved_inputs)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    parser = argparse.ArgumentParser(
        description="Generate volume-integrated field evolution from Quokka simulations: figures and/or extracted data.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_output=True,
            ),
        ],
    )
    parser.add_argument(
        "--log10",
        action="store_true",
        default=False,
        help="Apply log10(|field|) to the plotted data (does not affect saved JSON data).",
    )
    user_args = parser.parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        save_data=user_args.save_data,
        save_figure=user_args.save_figure,
        overwrite=user_args.overwrite,
        data_dir=user_args.data_dir,
        figures_dir=user_args.figures_dir,
        use_parallel=True,
        apply_log10=user_args.log10,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
