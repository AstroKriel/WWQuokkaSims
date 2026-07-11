## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from pathlib import Path
from typing import final

## personal
from jormi.ww_io import json_io, manage_log
from jormi.ww_plots import (
    annotate_axis,
    manage_plots,
    style_plots,
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
class RenderDataSeries:

    def __init__(
        self,
        *,
        extracted_dir: Path,
        figures_dir: Path,
        field_name: str,
        extract_data: bool,
    ):
        self.extracted_dir = extracted_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.extract_data = extract_data

    def _save_series(
        self,
        *,
        vi_series: data_series.DataSeries,
        extracted_dir: Path,
    ) -> None:
        extracted_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        time_array, values_array = vi_series.get_sorted_arrays()
        json_io.save_dict_to_json_file(
            file_path=extracted_dir / f"{self.field_name}-vi_evolution.json",
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
        if self.extract_data:
            self._save_series(
                vi_series=vi_series,
                extracted_dir=self.extracted_dir,
            )
        fig, ax = manage_plots.create_figure()
        time_array, values_array = vi_series.get_sorted_arrays()
        if time_array.size == 0:
            annotate_axis.add_text(
                ax=ax,
                x_pos=0.5,
                y_pos=0.5,
                label="no data",
                x_alignment="center",
                y_alignment="center",
            )
            return
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
        ax.set_ylabel(f"${vi_series.latex_label}$")
        fig_path = self.figures_dir / f"{self.field_name}-time_evolution.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


##
## === SCRIPT INTERFACE
##


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        snapshot_tag: str,
        fields_to_plot: list[str],
        extract_data: bool,
        extracted_dir: Path | None = None,
        figures_dir: Path | None = None,
        use_parallel: bool = True,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        field_registry.validate_fields(field_names=fields_to_plot)
        self.input_dir = Path(input_dir)
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = list(fields_to_plot)
        self.extract_data = extract_data
        self.extracted_dir = Path(extracted_dir) if extracted_dir is not None else None
        self.figures_dir = Path(figures_dir) if figures_dir is not None else None
        self.use_parallel = bool(use_parallel)

    def run(
        self,
    ) -> None:
        snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
            input_dir=self.input_dir,
            snapshot_tag=self.snapshot_tag,
        )
        if not snapshot_dirs:
            return
        extracted_dir = cli.resolve_output_dir(
            output_dir=self.extracted_dir,
            default_dir=snapshot_dirs[0].parent,
        )
        figures_dir = cli.resolve_output_dir(
            output_dir=self.figures_dir,
            default_dir=extracted_dir,
        )
        for field_name in self.fields_to_plot:
            field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
            loader = data_series.LoadDataSeries(
                snapshot_dirs=snapshot_dirs,
                field_name=field_name,
                field_loader=field_meta.loader,
                use_parallel=self.use_parallel,
            )
            vi_series = loader.run()
            render_data_series = RenderDataSeries(
                extracted_dir=extracted_dir,
                figures_dir=figures_dir,
                field_name=field_name,
                extract_data=self.extract_data,
            )
            render_data_series.run(vi_series=vi_series)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    user_args = argparse.ArgumentParser(
        description="Plot volume-integrated field evolution from Quokka simulations.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                produces_data=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        extract_data=user_args.save_data,
        extracted_dir=user_args.extracted_dir,
        figures_dir=user_args.figures_dir,
        use_parallel=True,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
