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
from jormi.ww_data import (
    interpolate_series,
    series_types,
)
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_io import json_io, manage_log
from jormi.ww_plots import manage_figure, style_figure
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._script_tools import (
    cli,
    data_series,
    field_registry,
)
from ww_quokka_sims.sim_io.snapshots import find_snapshots

##
## === FIGURE RENDERING
##


@final
class GenerateComparisonPlot:

    def __init__(
        self,
        *,
        data_dir: Path,
        figures_dir: Path,
        field_name: str,
        label_dir_1: str,
        label_dir_2: str,
        save_data: bool,
        save_figure: bool,
        marker_dir_1: str = "o",
        marker_dir_2: str = "s",
    ):
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.label_dir_1 = str(label_dir_1)
        self.label_dir_2 = str(label_dir_2)
        self.save_data = save_data
        self.save_figure = save_figure
        self.marker_dir_1 = str(marker_dir_1)
        self.marker_dir_2 = str(marker_dir_2)

    def _save_comparison(
        self,
        *,
        t_array: numpy.ndarray,
        y_array: numpy.ndarray,
    ) -> None:
        self.data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_io.save_dict_to_json_file(
            file_path=self.data_dir / f"{self.field_name}-time_comparison.json",
            input_dict={
                "time": t_array,
                "frac_diff": y_array,
            },
            overwrite=True,
            verbose=False,
        )

    def run(
        self,
        *,
        vi_series_1: data_series.DataSeries,
        vi_series_2: data_series.DataSeries,
    ) -> None:
        """Plots (series_2 / series_1) - 1; series_1 is the reference denominator."""
        ## sort both series by time and check that neither is empty
        t_array_1, y_array_1 = vi_series_1.get_sorted_arrays()
        t_array_2, y_array_2 = vi_series_2.get_sorted_arrays()
        if (t_array_1.size == 0) and (t_array_2.size == 0):
            raise RuntimeError(
                "No data found for either directory.\n"
                f"dir_1 ({self.label_dir_1}): empty DataSeries\n"
                f"dir_2 ({self.label_dir_2}): empty DataSeries",
            )
        if t_array_1.size == 0:
            raise RuntimeError(
                "No data found for dir_1.\n"
                f"dir_1 ({self.label_dir_1}): empty DataSeries\n"
                f"dir_2 ({self.label_dir_2}): {t_array_2.size} points",
            )
        if t_array_2.size == 0:
            raise RuntimeError(
                "No data found for dir_2.\n"
                f"dir_1 ({self.label_dir_1}): {t_array_1.size} points\n"
                f"dir_2 ({self.label_dir_2}): empty DataSeries",
            )
        x1_min = float(t_array_1[0])
        x1_max = float(t_array_1[-1])
        x2_min = float(t_array_2[0])
        x2_max = float(t_array_2[-1])
        in_bounds_mask_1 = (x2_min <= t_array_1) & (t_array_1 <= x2_max)
        if not numpy.any(in_bounds_mask_1):
            raise RuntimeError(
                "There are no overlapping times for the comparison.\n"
                f"dir_1 ({self.label_dir_1}): x in [{x1_min}, {x1_max}]\n"
                f"dir_2 ({self.label_dir_2}): x in [{x2_min}, {x2_max}]",
            )
        ## interpolate series_2 onto the overlapping subset of series_1's time grid
        interp_result = interpolate_series.interpolate_1d(
            data_series=series_types.DataSeries(
                x_values=t_array_2,
                y_values=y_array_2,
            ),
            x_interp=t_array_1[in_bounds_mask_1],
            spline_order=3,
        )
        t_array_common = interp_result.x_values
        y_array_2_interp = interp_result.y_values
        if t_array_common.size == 0:
            raise RuntimeError(
                "No overlapping times remain after interpolation bounds handling.\n"
                f"dir_1 ({self.label_dir_1}): x in [{float(t_array_1[0])}, {float(t_array_1[-1])}]\n"
                f"dir_2 ({self.label_dir_2}): x in [{x2_min}, {x2_max}]",
            )
        y_array_1_common = y_array_1[in_bounds_mask_1]
        y_array_1_common = y_array_1_common[:t_array_common.size]
        if not numpy.all(numpy.isfinite(y_array_1_common)):
            raise RuntimeError(
                f"Non-finite values found in dir_1 ({self.label_dir_1}) on the comparison grid.",
            )
        if not numpy.all(numpy.isfinite(y_array_2_interp)):
            raise RuntimeError(
                f"Non-finite values found in interpolated dir_2 ({self.label_dir_2}) on the comparison grid.",
            )
        zero_mask = numpy.isclose(
            a=y_array_1_common,
            b=0.0,
            rtol=0.0,
            atol=0.0,
        )
        if numpy.any(zero_mask):
            raise RuntimeError(
                "Cannot compute fractional difference because dir_1 contains zeros on the comparison grid.\n"
                f"dir_1 ({self.label_dir_1}): {int(numpy.sum(zero_mask))} zero values in y_array",
            )
        ## fractional difference: (series_2 / series_1) - 1, with series_1 as the reference
        y_array_frac_diff = y_array_2_interp / y_array_1_common - 1.0
        ## optionally write the comparison data to JSON
        if self.save_data:
            self._save_comparison(
                t_array=t_array_common,
                y_array=y_array_frac_diff,
            )
        if not self.save_figure:
            return
        fig, ax = manage_figure.create_figure()
        ax.plot(
            t_array_common,
            y_array_frac_diff,
            color="black",
            marker=self.marker_dir_2,
            ms=6,
            ls="-",
            lw=1.5,
            label=f"{self.label_dir_2}/{self.label_dir_1} - 1",
        )
        ax.set_xlabel("time")
        ax.set_ylabel(f"${vi_series_1.latex_label}$ (frac. diff.)")
        fig_path = self.figures_dir / f"{self.field_name}-time_comparison.png"
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
    data_dir: Path
    figures_dir: Path
    snapshot_dirs_1: list[Path]
    snapshot_dirs_2: list[Path]


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        dir_1: Path,
        dir_2: Path,
        snapshot_tag: str,
        fields_to_plot: list[str],
        data_dir: Path,
        figures_dir: Path | None,
        save_data: bool,
        save_figure: bool,
        overwrite: bool = False,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        cli.ensure_save_flag_selected(
            save_figure=save_figure,
            save_data=save_data,
        )
        if not Path(dir_1).is_dir():
            raise ValueError(f"dir_1 does not exist: {dir_1}.")
        if not Path(dir_2).is_dir():
            raise ValueError(f"dir_2 does not exist: {dir_2}.")
        self.dir_1 = Path(dir_1)
        self.dir_2 = Path(dir_2)
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir) if figures_dir is not None else None
        field_registry.validate_fields(
            field_names=fields_to_plot,
            allowed_types=(field_models.ScalarField_3D, ),
        )
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = list(fields_to_plot)
        self.save_data = save_data
        self.save_figure = save_figure
        self.overwrite = bool(overwrite)

    def _resolve_inputs(
        self,
    ) -> ResolvedInputs:
        data_dir = cli.resolve_output_dir(
            output_dir=self.data_dir,
            default_dir=self.data_dir,
        )
        figures_dir = cli.resolve_output_dir(
            output_dir=self.figures_dir,
            default_dir=data_dir,
        )
        ## find snapshot dirs for each of the two sim roots, matched by snapshot_tag
        snapshot_dirs_1 = find_snapshots.resolve_snapshot_dirs(
            input_dir=self.dir_1,
            snapshot_tag=self.snapshot_tag,
            max_elems=100,
        )
        snapshot_dirs_2 = find_snapshots.resolve_snapshot_dirs(
            input_dir=self.dir_2,
            snapshot_tag=self.snapshot_tag,
            max_elems=100,
        )
        if not snapshot_dirs_1:
            raise RuntimeError(
                f"No snapshot directories resolved for dir_1: {self.dir_1} (tag={self.snapshot_tag!r})",
            )
        if not snapshot_dirs_2:
            raise RuntimeError(
                f"No snapshot directories resolved for dir_2: {self.dir_2} (tag={self.snapshot_tag!r})",
            )
        return ResolvedInputs(
            data_dir=data_dir,
            figures_dir=figures_dir,
            snapshot_dirs_1=snapshot_dirs_1,
            snapshot_dirs_2=snapshot_dirs_2,
        )

    def _generate_comparisons(
        self,
        resolved_inputs: ResolvedInputs,
    ) -> None:
        ## use the sim root directory names as labels in the plot legend
        label_dir_1 = self.dir_1.name
        label_dir_2 = self.dir_2.name
        for field_name in self.fields_to_plot:
            field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
            loader_1 = data_series.LoadDataSeries(
                snapshot_dirs=resolved_inputs.snapshot_dirs_1,
                field_name=field_name,
                field_loader=field_meta.loader,
                use_parallel=True,
                data_dir=resolved_inputs.data_dir,
                overwrite=self.overwrite,
                cache_key=f"{field_name}-{label_dir_1}",
            )
            loader_2 = data_series.LoadDataSeries(
                snapshot_dirs=resolved_inputs.snapshot_dirs_2,
                field_name=field_name,
                field_loader=field_meta.loader,
                use_parallel=True,
                data_dir=resolved_inputs.data_dir,
                overwrite=self.overwrite,
                cache_key=f"{field_name}-{label_dir_2}",
            )
            vi_series_1 = loader_1.run()
            vi_series_2 = loader_2.run()
            generate_comparison_plot = GenerateComparisonPlot(
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                field_name=field_name,
                label_dir_1=label_dir_1,
                label_dir_2=label_dir_2,
                save_data=self.save_data,
                save_figure=self.save_figure,
                marker_dir_1="o",
                marker_dir_2="s",
            )
            generate_comparison_plot.run(
                vi_series_1=vi_series_1,
                vi_series_2=vi_series_2,
            )

    def run(
        self,
    ) -> None:
        resolved_inputs = self._resolve_inputs()
        self._generate_comparisons(resolved_inputs)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    user_args = argparse.ArgumentParser(
        description="Compare volume-integrated field evolution between two Quokka simulations: figures and/or extracted data.",
        parents=[
            cli.base_parser(
                num_dirs=2,
                allow_vfields=False,
                allow_output=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        dir_1=user_args.input_dir_1,
        dir_2=user_args.input_dir_2,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        data_dir=user_args.data_dir,
        figures_dir=user_args.figures_dir,
        save_data=user_args.save_data,
        save_figure=user_args.save_figure,
        overwrite=user_args.overwrite,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
