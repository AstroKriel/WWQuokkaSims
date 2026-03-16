## { SCRIPT

##
## === DEPENDENCIES
##

import argparse
import numpy

from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import json_io
from jormi.ww_types import check_types, check_arrays
from jormi.ww_plots import manage_plots, annotate_axis
from jormi.ww_fields.fields_3d import field_types, field_operators

from ww_quokka_sims.sim_io import load_dataset
import quokka_fields  # local utils
from ww_quokka_sims.sim_io import find_datasets


##
## === DATA CLASSES
##


@dataclass(frozen=True)
class FieldArgs:
    dataset_dir: Path
    field_name: str
    field_loader: Callable


@dataclass(frozen=True)
class DataPoint:
    sim_time: float
    vi_value: float


@dataclass(frozen=True)
class DataSeries:
    points: list[DataPoint]

    @property
    def num_points(
        self,
    ) -> int:
        return len(self.points)

    def get_sorted_arrays(
        self,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if not self.points:
            return (
                numpy.asarray([], dtype=float),
                numpy.asarray([], dtype=float),
            )
        sorted_points = sorted(self.points, key=lambda point: point.sim_time)
        time_array = check_arrays.as_1d([point.sim_time for point in sorted_points])
        values_array = check_arrays.as_1d([point.vi_value for point in sorted_points])
        return (
            time_array,
            values_array,
        )


##
## === COMPUTE
##


class LoadDataSeries:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        field_name: str,
        field_loader: Callable,
        use_parallel: bool = True,
    ):
        self.dataset_dirs = list(sorted(dataset_dirs))
        self.field_name = field_name
        self.field_loader = field_loader
        self.use_parallel = bool(use_parallel)

    @staticmethod
    def _load_snapshot(
        field_args: FieldArgs,
    ) -> DataPoint:
        with load_dataset.QuokkaDataset(dataset_dir=field_args.dataset_dir, verbose=False) as ds:
            sfield_3d = field_args.field_loader(ds)
        if not isinstance(sfield_3d, field_types.ScalarField_3D):
            raise TypeError(
                f"Expected ScalarField_3D from `{field_args.field_loader.__name__}`, got {type(sfield_3d).__name__}.",
            )
        sim_time = sfield_3d.sim_time
        if (sim_time is None) or (not numpy.isfinite(sim_time)):
            raise ValueError(f"Invalid sim_time for field: {sim_time!r}")
        vi_value = field_operators.compute_sfield_volume_integral(sfield_3d=sfield_3d)
        return DataPoint(
            sim_time=float(sim_time),
            vi_value=float(vi_value),
        )

    def run(
        self,
    ) -> DataSeries:
        grouped_field_args: list[FieldArgs] = [
            FieldArgs(
                dataset_dir=Path(dataset_dir),
                field_name=self.field_name,
                field_loader=self.field_loader,
            ) for dataset_dir in self.dataset_dirs
        ]
        if not grouped_field_args:
            return DataSeries(points=[])
        ## load each snapshot in parallel if the series is large enough to justify it, else serial
        if self.use_parallel and (len(grouped_field_args) > 5):
            data_points: list[DataPoint] = parallel_dispatch.run_in_parallel(
                worker_fn=LoadDataSeries._load_snapshot,
                grouped_args=grouped_field_args,
                timeout_seconds=120,
                show_progress=True,
                enable_plotting=True,
            )
        else:
            data_points = [
                LoadDataSeries._load_snapshot(field_args=field_args) for field_args in grouped_field_args
            ]
        return DataSeries(points=data_points)


##
## === RENDER
##


class RenderDataSeries:

    def __init__(
        self,
        *,
        out_dir: Path,
        field_name: str,
        extract_data: bool,
    ):
        self.out_dir = out_dir
        self.field_name = field_name
        self.extract_data = extract_data

    @staticmethod
    def _annotate_fit(
        *,
        ax,
        slope_stat,
        intercept_stat,
        color: str = "black",
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
        data_series: DataSeries,
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        time_array, values_array = data_series.get_sorted_arrays()
        json_io.save_dict_to_json_file(
            file_path=out_dir / f"{self.field_name}_vi_evolution.json",
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
        data_series: DataSeries,
    ) -> None:
        ## optionally write the time series data to JSON
        if self.extract_data:
            self._save_series(
                data_series=data_series,
                out_dir=self.out_dir,
            )
        fig, ax = manage_plots.create_figure()
        time_array, values_array = data_series.get_sorted_arrays()
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
        ax.set_ylabel(self.field_name)
        fig_path = self.out_dir / f"{self.field_name}_time_evolution.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


##
## === SCRIPT INTERFACE
##


class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        dataset_tag: str,
        fields_to_plot: list[str],
        extract_data: bool,
        use_parallel: bool = True,
    ):
        check_types.ensure_nonempty_string(
            param=dataset_tag,
            param_name="dataset_tag",
        )
        quokka_fields.validate_fields(field_names=fields_to_plot)
        self.input_dir = Path(input_dir)
        self.dataset_tag = dataset_tag
        self.fields_to_plot = list(fields_to_plot)
        self.extract_data = extract_data
        self.use_parallel = bool(use_parallel)

    def run(
        self,
    ) -> None:
        ## find all dataset dirs under input_dir whose names match dataset_tag, sorted by index
        dataset_dirs = find_datasets.resolve_dataset_dirs(
            input_dir=self.input_dir,
            dataset_tag=self.dataset_tag,
        )
        if not dataset_dirs:
            return
        ## output goes to the sim root (the shared parent of all dataset dirs)
        out_dir = dataset_dirs[0].parent
        ## load the volume-integral time series and render the evolution plot for each requested field
        for field_name in self.fields_to_plot:
            field_meta = quokka_fields.QUOKKA_FIELD_LOOKUP[field_name]
            load_data_series = LoadDataSeries(
                dataset_dirs=dataset_dirs,
                field_name=field_name,
                field_loader=field_meta["loader"],
                use_parallel=self.use_parallel,
            )
            data_series = load_data_series.run()
            render_data_series = RenderDataSeries(
                out_dir=out_dir,
                field_name=field_name,
                extract_data=self.extract_data,
            )
            render_data_series.run(data_series=data_series)


##
## === PROGRAM MAIN
##


def main():
    user_args = argparse.ArgumentParser(
        description="Plot volume-integrated field evolution from Quokka simulations.",
        parents=[
            quokka_fields.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_extract=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.dir,
        dataset_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        extract_data=user_args.extract,
        use_parallel=True,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
