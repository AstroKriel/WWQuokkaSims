## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
import argparse

from pathlib import Path

from jormi.ww_types import check_types
from jormi import ww_lists
from jormi.ww_io import manage_io
from jormi.ww_data import series_types, interpolate_series
from jormi.ww_plots import manage_plots

from plot_vi_evolution import DataSeries, LoadDataSeries

from ww_quokka_sims.sim_io import find_datasets

import utils

##
## === OPERATOR CLASSES
##


class RenderComparisonPlot:

    def __init__(
        self,
        *,
        fig_dir: Path,
        field_name: str,
        color: str,
        label_dir_1: str,
        label_dir_2: str,
        marker_dir_1: str = "o",
        marker_dir_2: str = "s",
    ):
        self.fig_dir = Path(fig_dir)
        self.field_name = field_name
        self.color = color
        self.label_dir_1 = str(label_dir_1)
        self.label_dir_2 = str(label_dir_2)
        self.marker_dir_1 = str(marker_dir_1)
        self.marker_dir_2 = str(marker_dir_2)

    def run(
        self,
        *,
        data_series_1: DataSeries,
        data_series_2: DataSeries,
    ) -> None:
        x_array_1, y_array_1 = data_series_1.get_sorted_arrays()
        x_array_2, y_array_2 = data_series_2.get_sorted_arrays()
        if (x_array_1.size == 0) and (x_array_2.size == 0):
            raise RuntimeError(
                "No data found for either directory.\n"
                f"dir_1 ({self.label_dir_1}): empty DataSeries\n"
                f"dir_2 ({self.label_dir_2}): empty DataSeries",
            )
        if x_array_1.size == 0:
            raise RuntimeError(
                "No data found for dir_1.\n"
                f"dir_1 ({self.label_dir_1}): empty DataSeries\n"
                f"dir_2 ({self.label_dir_2}): {x_array_2.size} points",
            )
        if x_array_2.size == 0:
            raise RuntimeError(
                "No data found for dir_2.\n"
                f"dir_1 ({self.label_dir_1}): {x_array_1.size} points\n"
                f"dir_2 ({self.label_dir_2}): empty DataSeries",
            )
        x1_min = float(x_array_1[0])
        x1_max = float(x_array_1[-1])
        x2_min = float(x_array_2[0])
        x2_max = float(x_array_2[-1])
        in_bounds_mask_1 = (x2_min <= x_array_1) & (x_array_1 <= x2_max)
        if not numpy.any(in_bounds_mask_1):
            raise RuntimeError(
                "There are no overlapping times for the comparison.\n"
                f"dir_1 ({self.label_dir_1}): x in [{x1_min}, {x1_max}]\n"
                f"dir_2 ({self.label_dir_2}): x in [{x2_min}, {x2_max}]",
            )
        interp_result = interpolate_series.interpolate_1d(
            data_series=series_types.DataSeries(
                x_values=x_array_2,
                y_values=y_array_2,
            ),
            x_interp=x_array_1[in_bounds_mask_1],
            spline_order=3,
        )
        x_array_common = interp_result.x_values
        y_array_2_interp = interp_result.y_values
        if x_array_common.size == 0:
            raise RuntimeError(
                "No overlapping times remain after interpolation bounds handling.\n"
                f"dir_1 ({self.label_dir_1}): x in [{float(x_array_1[0])}, {float(x_array_1[-1])}]\n"
                f"dir_2 ({self.label_dir_2}): x in [{x2_min}, {x2_max}]",
            )
        y_array_1_common = y_array_1[in_bounds_mask_1]
        y_array_1_common = y_array_1_common[:x_array_common.size]
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
        y_array_frac_diff = y_array_2_interp / y_array_1_common - 1.0
        fig, ax = manage_plots.create_figure()
        ax.plot(
            x_array_common,
            y_array_frac_diff,
            color=self.color,
            marker=self.marker_dir_2,
            ms=6,
            ls="-",
            lw=1.5,
            label=f"{self.label_dir_2}/{self.label_dir_1} - 1",
        )
        ax.set_xlabel("time")
        ax.set_ylabel(self.field_name + " (frac. diff.)")
        fig_path = self.fig_dir / f"{self.field_name}_time_comparison.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


class ScriptInterface:

    def __init__(
        self,
        *,
        dir_1: Path,
        dir_2: Path,
        dataset_tag: str,
        fields_to_plot: list[str],
        out_dir: Path,
    ):
        check_types.ensure_nonempty_string(
            param=dataset_tag,
            param_name="dataset_tag",
        )
        manage_io.does_directory_exist(
            directory=dir_1,
            raise_error=True,
        )
        manage_io.does_directory_exist(
            directory=dir_2,
            raise_error=True,
        )
        manage_io.does_directory_exist(
            directory=out_dir,
            raise_error=True,
        )
        self.dir_1 = Path(dir_1)
        self.dir_2 = Path(dir_2)
        self.fig_dir = Path(out_dir)
        valid_fields = set(utils.QUOKKA_FIELD_LOOKUP.keys())
        if (not fields_to_plot) or (not set(fields_to_plot).issubset(valid_fields)):
            raise ValueError(f"Provide one or more fields to plot (via -f) from: {sorted(valid_fields)}")
        self.dataset_tag = dataset_tag
        self.fields_to_plot = list(fields_to_plot)

    def run(
        self,
    ) -> None:
        dataset_dirs_1 = find_datasets.resolve_dataset_dirs(
            input_dir=self.dir_1,
            dataset_tag=self.dataset_tag,
            max_elems=100,
        )
        dataset_dirs_2 = find_datasets.resolve_dataset_dirs(
            input_dir=self.dir_2,
            dataset_tag=self.dataset_tag,
            max_elems=100,
        )
        if not dataset_dirs_1:
            raise RuntimeError(
                f"No dataset directories resolved for dir_1: {self.dir_1} (tag={self.dataset_tag!r})",
            )
        if not dataset_dirs_2:
            raise RuntimeError(
                f"No dataset directories resolved for dir_2: {self.dir_2} (tag={self.dataset_tag!r})",
            )
        label_dir_1 = self.dir_1.name
        label_dir_2 = self.dir_2.name
        for field_name in self.fields_to_plot:
            field_meta = utils.QUOKKA_FIELD_LOOKUP[field_name]
            load_data_series_1 = LoadDataSeries(
                dataset_dirs=dataset_dirs_1,
                field_name=field_name,
                field_loader=field_meta["loader"],
                use_parallel=True,
            )
            load_data_series_2 = LoadDataSeries(
                dataset_dirs=dataset_dirs_2,
                field_name=field_name,
                field_loader=field_meta["loader"],
                use_parallel=True,
            )
            data_series_1 = load_data_series_1.run()
            data_series_2 = load_data_series_2.run()
            render_comparison_plot = RenderComparisonPlot(
                fig_dir=self.fig_dir,
                field_name=field_name,
                color=field_meta["color"],
                label_dir_1=label_dir_1,
                label_dir_2=label_dir_2,
                marker_dir_1="o",
                marker_dir_2="s",
            )
            render_comparison_plot.run(
                data_series_1=data_series_1,
                data_series_2=data_series_2,
            )


##
## === ARGPARSE
##


def get_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-1",
        "-d1",
        type=lambda path: Path(path).expanduser().resolve(),
        required=True,
        help="First directory to compare.",
    )
    parser.add_argument(
        "--dir-2",
        "-d2",
        type=lambda path: Path(path).expanduser().resolve(),
        required=True,
        help="Second directory to compare.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        default="plt",
        help=
        "Dataset tag used to identify output directories (e.g., `plt` -> plt00010, plt00020). Default: `plt`.",
    )
    field_list = ww_lists.as_string(
        elems=sorted(
            utils.QUOKKA_FIELD_LOOKUP.keys(),
        ),
    )
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=None,
        help=f"List of (vector and/or scalar) fields to plot. Options: {field_list}",
    )
    parser.add_argument(
        "--out",
        type=lambda path: Path(path).expanduser().resolve(),
        required=True,
        help="Output directory for figures.",
    )
    return parser.parse_args()


##
## === PROGRAM MAIN
##


def main():
    user_args = get_user_args()
    script_interface = ScriptInterface(
        dir_1=user_args.dir_1,
        dir_2=user_args.dir_2,
        dataset_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        out_dir=user_args.out,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
