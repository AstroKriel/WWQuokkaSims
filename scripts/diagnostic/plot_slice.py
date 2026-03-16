## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    domain_types,
    field_types,
)
from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import (
    manage_io,
    manage_log,
)
from jormi.ww_plots import (
    add_color,
    annotate_axis,
    manage_plots,
    plot_data,
)
from jormi.ww_types import check_types

## local
from ww_quokka_sims.sim_io import (
    find_datasets,
    load_dataset,
)
import quokka_fields

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class FieldArgs:
    field_name: str
    field_loader: Callable
    cmap_name: str


class WorkerArgs(NamedTuple):
    dataset_dir: str
    dataset_tag: str
    field_name: str
    field_loader: Callable
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...]
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...]
    cmap_name: str
    out_dir: str
    index_width: int
    extract_data: bool


@dataclass(frozen=True)
class Dataset:
    uniform_domain: domain_types.UniformDomain_3D
    field: field_types.ScalarField_3D | field_types.VectorField_3D

    @property
    def sim_time(
        self,
    ) -> float:
        sim_time = self.field.sim_time
        if (sim_time is None) or (not numpy.isfinite(sim_time)):
            msg = f"Invalid sim_time for field: {sim_time!r}"
            manage_log.log_error(msg)
            raise RuntimeError(msg)
        return float(sim_time)


@dataclass(frozen=True)
class FieldComp:
    data_3d: numpy.ndarray
    label: str


AxisBounds = tuple[tuple[float, float], tuple[float, float]]  # ((xmin, xmax), (ymin, ymax))


@dataclass(frozen=True)
class SlicedField:
    data_2d: numpy.ndarray
    label: str
    axis_bounds: AxisBounds


##
## === FIELD PROCESSING
##


def _parse_axes(
    *,
    axes: tuple[str, ...] | list[str] | None,
) -> tuple[cartesian_axes.CartesianAxis_3D, ...]:
    if axes is None:
        return tuple(cartesian_axes.DEFAULT_3D_AXES_ORDER)
    parsed_axes: list[cartesian_axes.CartesianAxis_3D] = []
    for axis_name in check_types.as_tuple(param=axes):
        try:
            parsed_axes.append(cartesian_axes.as_axis(axis=axis_name))
        except (TypeError, ValueError):
            raise ValueError("Provide one or more axes (via -a/-c) from: x_0, x_1, x_2")
    return tuple(parsed_axes)


def _axis_to_index(
    axis: cartesian_axes.CartesianAxis_3D,
) -> int:
    return cartesian_axes.get_axis_index(axis)


def get_slice_bounds(
    *,
    uniform_domain: domain_types.UniformDomain_3D,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> AxisBounds:
    (x0_min, x0_max), (x1_min, x1_max), (x2_min, x2_max) = uniform_domain.domain_bounds
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X2:
        return ((x0_min, x0_max), (x1_min, x1_max))
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X1:
        return ((x0_min, x0_max), (x2_min, x2_max))
    return (
        (x1_min, x1_max),
        (x2_min, x2_max),
    )


def get_slice_labels(
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> tuple[str, str]:
    axes_plane = [ax for ax in cartesian_axes.DEFAULT_3D_AXES_ORDER if ax != axis_to_slice]
    return (
        axes_plane[0].axis_label if "$" in axes_plane[0].axis_label else f"${axes_plane[0].axis_label}$",
        axes_plane[1].axis_label if "$" in axes_plane[1].axis_label else f"${axes_plane[1].axis_label}$",
    )


def slice_field(
    *,
    data_3d: numpy.ndarray,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
    uniform_domain: domain_types.UniformDomain_3D,
) -> SlicedField:
    num_cells_x0, num_cells_x1, num_cells_x2 = data_3d.shape
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X2:
        data_2d = data_3d[:, :, num_cells_x2 // 2]
    elif axis_to_slice == cartesian_axes.CartesianAxis_3D.X1:
        data_2d = data_3d[:, num_cells_x1 // 2, :]
    else:
        data_2d = data_3d[num_cells_x0 // 2, :, :]
    label_parts = [
        rf"{ax.axis_label}=L_{ax.axis_index}/2" if ax == axis_to_slice else ax.axis_label
        for ax in cartesian_axes.DEFAULT_3D_AXES_ORDER
    ]
    label = "$(" + ", ".join(label_parts) + ")$"
    axis_bounds = get_slice_bounds(
        uniform_domain=uniform_domain,
        axis_to_slice=axis_to_slice,
    )
    return SlicedField(
        data_2d=data_2d,
        label=label,
        axis_bounds=axis_bounds,
    )


##
## === FIGURE RENDERING
##


@dataclass(frozen=True)
class FieldPlotter:
    dataset_tag: str
    field_args: FieldArgs
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...]
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...]
    extract_data: bool

    @staticmethod
    def plot_slice(
        *,
        ax,
        sim_time: float,
        field_slice: SlicedField,
        label: str,
        cmap_name: str,
    ) -> None:
        min_value = float(numpy.nanmin(field_slice.data_2d))
        max_value = float(numpy.nanmax(field_slice.data_2d))
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=field_slice.data_2d,
            data_format="xy",
            axis_aspect_ratio="equal",
            axis_bounds=field_slice.axis_bounds,
            cbar_bounds=(min_value, max_value),
            palette_config=add_color.SequentialConfig(palette_name=cmap_name),
            add_cbar=True,
            cbar_label=label,
            cbar_side="right",
        )
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=0.95,
            x_alignment="center",
            y_alignment="top",
            label=f"min-value = {min_value:.2e}\nmax-value = {max_value:.2e}",
            text_size=16,
            box_alpha=0.5,
        )
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=0.5,
            x_alignment="center",
            y_alignment="center",
            label=rf"$t = {sim_time:.2f}$",
            text_size=16,
            box_alpha=0.5,
        )
        annotate_axis.add_text(
            ax=ax,
            x_pos=0.5,
            y_pos=0.05,
            x_alignment="center",
            y_alignment="bottom",
            label=field_slice.label,
            text_size=16,
            box_alpha=0.5,
        )

    def _load_dataset(
        self,
        *,
        dataset_dir: Path,
    ) -> Dataset:
        with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
            uniform_domain = ds.load_3d_uniform_domain()
            field = self.field_args.field_loader(ds)  # ScalarField_3D or VectorField_3D
        return Dataset(
            uniform_domain=uniform_domain,
            field=field,
        )

    def _get_field_comps(
        self,
        *,
        field: field_types.ScalarField_3D | field_types.VectorField_3D,
    ) -> list[FieldComp]:
        field_name = self.field_args.field_name
        if isinstance(field, field_types.ScalarField_3D):
            sarray_3d = field_types.extract_3d_sarray(
                sfield_3d=field,
                param_name=f"<{field_name}_sfield_3d>",
            )
            return [
                FieldComp(
                    data_3d=sarray_3d,
                    label=field_types.get_label(field),
                ),
            ]
        if isinstance(field, field_types.VectorField_3D):
            if not self.comps_to_plot:
                raise ValueError(
                    f"Vector field `{field_name}` requires at least one component to plot; none provided.",
                )
            varray_3d = field_types.extract_3d_varray(
                vfield_3d=field,
                param_name=f"<{field_name}_vfield_3d>",
            )
            return [
                FieldComp(
                    data_3d=varray_3d[_axis_to_index(comp_axis)],
                    label=field_types.get_vcomp_label(field, comp_axis),
                ) for comp_axis in self.comps_to_plot
            ]
        raise ValueError(f"{field_name} is an unrecognised field type.")

    def _plot_field_comps(
        self,
        *,
        axs_grid,
        field_comps: list[FieldComp],
        uniform_domain: domain_types.UniformDomain_3D,
        sim_time: float,
    ) -> None:
        for row_index, field_comp in enumerate(field_comps):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                field_slice = slice_field(
                    data_3d=field_comp.data_3d,
                    axis_to_slice=axis_to_slice,
                    uniform_domain=uniform_domain,
                )
                self.plot_slice(
                    ax=ax,
                    sim_time=sim_time,
                    field_slice=field_slice,
                    label=field_comp.label,
                    cmap_name=self.field_args.cmap_name,
                )

    def _label_axes(
        self,
        *,
        axs_grid,
    ) -> None:
        num_rows = len(axs_grid)
        for row_index in range(num_rows):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                x_label_string, y_label_string = get_slice_labels(axis_to_slice)
                if (num_rows == 1) or (row_index == num_rows - 1):
                    ax.set_xlabel(x_label_string)
                ax.set_ylabel(y_label_string)

    def _save_slices(
        self,
        *,
        field_comps: list[FieldComp],
        uniform_domain: domain_types.UniformDomain_3D,
        dataset_index: int,
        index_width: int,
        out_dir: Path,
    ) -> None:
        is_vfield = len(field_comps) > 1
        field_name = self.field_args.field_name
        padded_index = f"{dataset_index:0{index_width}d}"
        for comp_index, field_comp in enumerate(field_comps):
            for axis_to_slice in self.axes_to_slice:
                field_slice = slice_field(
                    data_3d=field_comp.data_3d,
                    axis_to_slice=axis_to_slice,
                    uniform_domain=uniform_domain,
                )
                axis_name = axis_to_slice.name.lower()
                comp_part = f"_comp{comp_index}" if is_vfield else ""
                file_name = f"{field_name}{comp_part}_slice={axis_name}_{padded_index}.npy"
                numpy.save(out_dir / file_name, field_slice.data_2d)

    def plot_dataset(
        self,
        *,
        dataset_dir: Path,
        out_dir: Path,
        index_width: int,
        verbose: bool,
    ) -> None:
        dataset = self._load_dataset(dataset_dir=dataset_dir)
        dataset_index = int(
            find_datasets.get_dataset_index_string(
                dataset_dir=dataset_dir,
                dataset_tag=self.dataset_tag,
            ),
        )
        field_comps = self._get_field_comps(field=dataset.field)
        if self.extract_data:
            self._save_slices(
                field_comps=field_comps,
                uniform_domain=dataset.uniform_domain,
                dataset_index=dataset_index,
                index_width=index_width,
                out_dir=out_dir,
            )
        fig, axs_grid = manage_plots.create_figure_grid(
            num_rows=len(field_comps),
            num_cols=len(self.axes_to_slice),
            x_spacing=0.75,
            y_spacing=0.25,
        )
        fig.subplots_adjust(right=0.82)
        self._plot_field_comps(
            axs_grid=axs_grid,
            field_comps=field_comps,
            uniform_domain=dataset.uniform_domain,
            sim_time=dataset.sim_time,
        )
        self._label_axes(axs_grid=axs_grid)
        field_name = self.field_args.field_name
        padded_index = f"{dataset_index:0{index_width}d}"
        fig_name = f"{field_name}_slice_{padded_index}.png"
        fig_path = out_dir / fig_name
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=verbose,
        )


def render_fields_in_serial(
    *,
    dataset_tag: str,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...],
    dataset_dirs: list[Path],
    out_dir: Path,
    index_width: int,
    extract_data: bool,
) -> None:
    for field_name in fields_to_plot:
        field_meta = quokka_fields.QUOKKA_FIELD_LOOKUP[field_name]
        field_args = FieldArgs(
            field_name=field_name,
            field_loader=field_meta["loader"],
            cmap_name=field_meta["cmap"],
        )
        field_plotter = FieldPlotter(
            dataset_tag=dataset_tag,
            field_args=field_args,
            comps_to_plot=comps_to_plot,
            axes_to_slice=axes_to_slice,
            extract_data=extract_data,
        )
        for dataset_dir in dataset_dirs:
            field_plotter.plot_dataset(
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                index_width=index_width,
                verbose=False,
            )


def _plot_dataset_worker(
    *user_args,
) -> None:
    worker_args = WorkerArgs(*user_args)
    field_args = FieldArgs(
        field_name=worker_args.field_name,
        field_loader=worker_args.field_loader,
        cmap_name=worker_args.cmap_name,
    )
    field_plotter = FieldPlotter(
        dataset_tag=worker_args.dataset_tag,
        field_args=field_args,
        comps_to_plot=worker_args.comps_to_plot,
        axes_to_slice=worker_args.axes_to_slice,
        extract_data=worker_args.extract_data,
    )
    field_plotter.plot_dataset(
        dataset_dir=Path(worker_args.dataset_dir),
        out_dir=Path(worker_args.out_dir),
        index_width=int(worker_args.index_width),
        verbose=False,
    )


def render_fields_in_parallel(
    *,
    dataset_tag: str,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...],
    dataset_dirs: list[Path],
    out_dir: Path,
    index_width: int,
    extract_data: bool,
) -> None:
    grouped_args: list[WorkerArgs] = []
    for field_name in fields_to_plot:
        field_meta = quokka_fields.QUOKKA_FIELD_LOOKUP[field_name]
        for dataset_dir in dataset_dirs:
            grouped_args.append(
                WorkerArgs(
                    dataset_dir=str(dataset_dir),
                    dataset_tag=dataset_tag,
                    field_name=field_name,
                    field_loader=field_meta["loader"],
                    comps_to_plot=comps_to_plot,
                    axes_to_slice=axes_to_slice,
                    cmap_name=field_meta["cmap"],
                    out_dir=str(out_dir),
                    index_width=index_width,
                    extract_data=extract_data,
                ),
            )
    parallel_dispatch.run_in_parallel(
        worker_fn=_plot_dataset_worker,
        grouped_args=grouped_args,
        timeout_seconds=120,
        show_progress=True,
        enable_plotting=True,
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
        fields_to_plot: tuple[str, ...] | list[str] | None,
        comps_to_plot: tuple[str, ...] | list[str] | None,
        axes_to_slice: tuple[str, ...] | list[str] | None,
        extract_data: bool,
        use_parallel: bool = True,
        animate_only: bool = False,
    ):
        check_types.ensure_nonempty_string(
            param=dataset_tag,
            param_name="dataset_tag",
        )
        valid_fields = set(quokka_fields.QUOKKA_FIELD_LOOKUP.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide one or more field to plot (via -f) from: {sorted(valid_fields)}")
        self.input_dir = Path(input_dir)
        self.dataset_tag = dataset_tag
        self.fields_to_plot = check_types.as_tuple(param=fields_to_plot)
        ## axis selection now uses CartesianAxis enums internally
        self.comps_to_plot = _parse_axes(axes=comps_to_plot)
        self.axes_to_slice = _parse_axes(axes=axes_to_slice)
        self.extract_data = extract_data
        self.use_parallel = bool(use_parallel)
        self.animate_only = bool(animate_only)

    def _animate_fields(
        self,
        *,
        out_dir: Path,
    ) -> None:
        for field_name in self.fields_to_plot:
            fig_paths = manage_io.ItemFilter(
                prefix=f"{field_name}_slice_",
                suffix=".png",
                include_folders=False,
                include_files=True,
            ).filter(directory=out_dir)
            if len(fig_paths) < 3:
                manage_log.log_hint(
                    text=(
                        f"Skipping animation for `{field_name}`: "
                        f"only found {len(fig_paths)} frame(s), but need at least 3."
                    ),
                )
                continue
            mp4_path = out_dir / f"{field_name}_slices.mp4"
            manage_plots.animate_pngs_to_mp4(
                frames_dir=out_dir,
                mp4_path=mp4_path,
                pattern=f"{field_name}_slice_*.png",
                fps=60,
                timeout_seconds=120,
            )

    def run(
        self,
    ) -> None:
        ## find all dataset dirs under input_dir whose names match dataset_tag, sorted by index
        dataset_dirs = find_datasets.resolve_dataset_dirs(
            input_dir=self.input_dir,
            dataset_tag=self.dataset_tag,
            max_elems=100,
        )
        if not dataset_dirs:
            return
        ## output goes to the sim root (the shared parent of all dataset dirs)
        out_dir = dataset_dirs[0].parent
        ## index_width is the zero-pad width derived from the total number of datasets found
        index_width = find_datasets.get_max_index_width(
            dataset_dirs=dataset_dirs,
            dataset_tag=self.dataset_tag,
        )
        ## render slice images; use parallel workers if the dataset count warrants it
        if not self.animate_only:
            if self.use_parallel and (len(dataset_dirs) > 5):
                render_fields_in_parallel(
                    dataset_tag=self.dataset_tag,
                    fields_to_plot=self.fields_to_plot,
                    comps_to_plot=self.comps_to_plot,
                    axes_to_slice=self.axes_to_slice,
                    dataset_dirs=dataset_dirs,
                    out_dir=out_dir,
                    index_width=index_width,
                    extract_data=self.extract_data,
                )
            else:
                render_fields_in_serial(
                    dataset_tag=self.dataset_tag,
                    fields_to_plot=self.fields_to_plot,
                    comps_to_plot=self.comps_to_plot,
                    axes_to_slice=self.axes_to_slice,
                    dataset_dirs=dataset_dirs,
                    out_dir=out_dir,
                    index_width=index_width,
                    extract_data=self.extract_data,
                )
        ## stitch rendered PNGs into an MP4 animation (no-op if animate flag is not set)
        self._animate_fields(out_dir=out_dir)


##
## === PROGRAM MAIN
##


def main():
    parser = argparse.ArgumentParser(
        description="Plot midplane slices of Quokka field components.",
        parents=[
            quokka_fields.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_extract=True,
            ),
        ],
    )
    parser.add_argument(
        "--animate-only",
        action="store_true",
        default=False,
        help="Skip rendering and go straight to animation (default: False).",
    )
    user_args = parser.parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.dir,
        dataset_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        comps_to_plot=user_args.comps,
        axes_to_slice=user_args.axes,
        extract_data=user_args.extract,
        animate_only=user_args.animate_only,
        use_parallel=True,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
