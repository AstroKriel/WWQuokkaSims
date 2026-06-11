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
        out_dir: Path,
        field_name: str,
        latex_label: str,
        extract_data: bool,
    ):
        self.out_dir = out_dir
        self.field_name = field_name
        self.latex_label = latex_label
        self.extract_data = extract_data

    @staticmethod
    def _annotate_fit(
        *,
        ax,
        slope_stat,
        intercept_stat,
        color: annotate_axis.ColorType = "black",
        x_pos: float = 0.95,
        y_pos: float = 0.95,
        x_alignment: str = "right",
        y_alignment: str = "top",
        fontsize: float = 12,
        with_box: bool = True,
    ) -> None:

        def _fmt(
            stat,
        ) -> str:
            return f"{stat.value:.3e}" + (
                f" +/- {stat.sigma:.1e}" if getattr(stat, "sigma", None) is not None else ""
            )

        label = f"slope: {_fmt(slope_stat)} intercept: {_fmt(intercept_stat)}"
        annotate_axis.add_text(
            ax=ax,
            x_pos=x_pos,
            y_pos=y_pos,
            label=label,
            x_alignment=x_alignment,
            y_alignment=y_alignment,
            text_size=fontsize,
            text_color=color,
            box_alpha=0.85 if with_box else 0.0,
            box_color="white",
        )

    def _save_series(
        self,
        *,
        vi_series: data_series.DataSeries,
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        time_array, values_array = vi_series.get_sorted_arrays()
        json_io.save_dict_to_json_file(
            file_path=out_dir / f"{self.field_name}-vi_evolution.json",
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
                out_dir=self.out_dir,
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
        ax.set_ylabel(f"${self.latex_label}$")
        fig_path = self.out_dir / f"{self.field_name}-time_evolution.png"
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
        out_dir: Path | None = None,
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
        self.out_dir = Path(out_dir) if out_dir is not None else None
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
        out_dir = self.out_dir if self.out_dir is not None else snapshot_dirs[0].parent
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
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
                out_dir=out_dir,
                field_name=field_name,
                latex_label=field_meta.latex_label,
                extract_data=self.extract_data,
            )
            render_data_series.run(vi_series=vi_series)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
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
        out_dir=user_args.out_dir,
        use_parallel=True,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
