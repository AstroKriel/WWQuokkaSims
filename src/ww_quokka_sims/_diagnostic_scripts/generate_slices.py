## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    NamedTuple,
    final,
)

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
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
    style_plots,
)
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._script_tools import (
    cli,
    field_registry,
)
from ww_quokka_sims.sim_io import (
    find_snapshots,
    load_snapshot,
)

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class FieldArgs:
    field_name: str
    field_loader: Callable
    cmap_name: str
    amr_level: int = 0


class WorkerArgs(NamedTuple):
    """Flat, pickleable argument bundle passed to the parallel slice-render worker."""

    snapshot_dir: str
    snapshot_tag: str
    field_name: str
    field_loader: Callable
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...]
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...]
    cmap_name: str
    data_dir: str
    figures_dir: str
    index_width: int
    save_data: bool
    save_figure: bool
    overwrite: bool
    hide_annotations: bool
    amr_level: int = 0
    plot_log10: bool = False


@dataclass(frozen=True)
class SnapshotData:
    uniform_domain: domain_models.UniformDomain_3D
    field: field_models.ScalarField_3D | field_models.VectorField_3D

    @property
    def step_time(
        self,
    ) -> float:
        step_time = self.field.sim_time
        if (step_time is None) or (not numpy.isfinite(step_time)):
            msg = f"Invalid sim_time for field: {step_time!r}."
            manage_log.log_error(text=msg)
            raise RuntimeError(msg)
        return float(step_time)


@dataclass(frozen=True)
class FieldComp:
    sarray_3d: numpy.ndarray
    label: str
    comp_axis: cartesian_axes.CartesianAxis_3D | None = None


AxisBounds = tuple[tuple[float, float], tuple[float, float]]  # ((xmin, xmax), (ymin, ymax))


@dataclass(frozen=True)
class SlicedField:
    """A single 2D slice, self-contained enough to plot without the raw snapshot or uniform_domain."""

    sarray_2d: numpy.ndarray
    axis_bounds: AxisBounds
    min_value: float
    max_value: float


Row = tuple[str, dict[cartesian_axes.CartesianAxis_3D, SlicedField]]  # (comp_label, {axis: SlicedField})

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
    for axis_name in validate_types.as_tuple(param=axes):
        try:
            parsed_axes.append(
                cartesian_axes.as_axis(
                    axis=axis_name,
                ),
            )
        except (TypeError, ValueError):
            raise ValueError("Provide one or more axes (via -a/-c) from: x_0, x_1, x_2")
    return tuple(parsed_axes)


def _axis_to_index(
    axis: cartesian_axes.CartesianAxis_3D,
) -> int:
    return cartesian_axes.get_axis_index(axis)


def get_slice_bounds(
    *,
    uniform_domain: domain_models.UniformDomain_3D,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> AxisBounds:
    """Return physical bounds of the two plane axes (i.e. those not being sliced)."""
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


def get_slice_plane_label(
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> str:
    """Return the "which plane was sliced" annotation text; a pure function of `axis_to_slice` alone."""
    label_parts = [
        rf"{ax.axis_label}=L_{ax.axis_index}/2" if ax == axis_to_slice else ax.axis_label
        for ax in cartesian_axes.DEFAULT_3D_AXES_ORDER
    ]
    return "$(" + ", ".join(label_parts) + ")$"


def _compute_min_max(
    sarray_2d: numpy.ndarray,
) -> tuple[float, float]:
    return (
        float(numpy.nanmin(sarray_2d)),
        float(numpy.nanmax(sarray_2d)),
    )


def slice_field(
    *,
    sarray_3d: numpy.ndarray,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
    uniform_domain: domain_models.UniformDomain_3D,
) -> SlicedField:
    num_cells_x0, num_cells_x1, num_cells_x2 = sarray_3d.shape
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X2:
        sarray_2d = sarray_3d[:, :, num_cells_x2 // 2]
    elif axis_to_slice == cartesian_axes.CartesianAxis_3D.X1:
        sarray_2d = sarray_3d[:, num_cells_x1 // 2, :]
    else:
        sarray_2d = sarray_3d[num_cells_x0 // 2, :, :]
    axis_bounds = get_slice_bounds(
        uniform_domain=uniform_domain,
        axis_to_slice=axis_to_slice,
    )
    min_value, max_value = _compute_min_max(sarray_2d)
    return SlicedField(
        sarray_2d=sarray_2d,
        axis_bounds=axis_bounds,
        min_value=min_value,
        max_value=max_value,
    )


##
## === FIGURE RENDERING
##


@dataclass(frozen=True)
class GenerateFieldSlices:
    snapshot_tag: str
    field_args: FieldArgs
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...]
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...]
    save_data: bool
    save_figure: bool
    overwrite: bool = False
    hide_annotations: bool = False
    plot_log10: bool = False

    @staticmethod
    def plot_slice(
        *,
        ax: manage_plots.PlotAxis,
        step_time: float,
        field_slice: SlicedField,
        plane_label: str,
        comp_label: str,
        cmap_name: str,
        hide_annotations: bool = False,
    ) -> None:
        plot_data.plot_2d_array(
            ax=ax,
            array_2d=field_slice.sarray_2d,
            data_format="xy",
            axis_aspect_ratio="equal",
            axis_bounds=field_slice.axis_bounds,
            cbar_bounds=(field_slice.min_value, field_slice.max_value),
            palette_config=add_color.SequentialConfig(palette_name=cmap_name),
            add_cbar=True,
            cbar_label=comp_label,
            cbar_side="right",
        )
        if not hide_annotations:
            annotate_axis.add_text(
                ax=ax,
                x_pos=0.5,
                y_pos=0.95,
                x_alignment="center",
                y_alignment="top",
                label=f"min-value = {field_slice.min_value:.2e}\nmax-value = {field_slice.max_value:.2e}",
                text_size=16,
                box_alpha=0.5,
            )
            annotate_axis.add_text(
                ax=ax,
                x_pos=0.5,
                y_pos=0.5,
                x_alignment="center",
                y_alignment="center",
                label=rf"$t = {step_time:.2f}$",
                text_size=16,
                box_alpha=0.5,
            )
            annotate_axis.add_text(
                ax=ax,
                x_pos=0.5,
                y_pos=0.05,
                x_alignment="center",
                y_alignment="bottom",
                label=plane_label,
                text_size=16,
                box_alpha=0.5,
            )

    def _load_snapshot(
        self,
        *,
        snapshot_dir: Path,
    ) -> SnapshotData:
        amr_level = self.field_args.amr_level
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as snapshot:
            uniform_domain = snapshot.load_3d_uniform_domain(amr_level=amr_level)
            field = self.field_args.field_loader(
                snapshot,
                amr_level=amr_level,
            )  # ScalarField_3D or VectorField_3D
        return SnapshotData(
            uniform_domain=uniform_domain,
            field=field,
        )

    def _get_field_comps(
        self,
        *,
        field: field_models.ScalarField_3D | field_models.VectorField_3D,
    ) -> list[FieldComp]:
        field_name = self.field_args.field_name
        if isinstance(field, field_models.ScalarField_3D):
            sarray_3d = field_models.extract_3d_sarray(
                sfield_3d=field,
                param_name=f"<{field_name}_sfield_3d>",
            )
            return [
                FieldComp(
                    sarray_3d=sarray_3d,
                    label=field_models.get_label(field),
                ),
            ]
        if not self.comps_to_plot:
            raise ValueError(
                f"Vector field `{field_name}` requires at least one component to plot; none provided.",
            )
        varray_3d = field_models.extract_3d_varray(
            vfield_3d=field,
            param_name=f"<{field_name}_vfield_3d>",
        )
        return [
            FieldComp(
                sarray_3d=varray_3d[_axis_to_index(comp_axis)],
                label=field_models.get_vcomp_label(field, comp_axis=comp_axis),
                comp_axis=comp_axis,
            ) for comp_axis in self.comps_to_plot
        ]

    def _rows_from_field_comps(
        self,
        *,
        field_comps: list[FieldComp],
        uniform_domain: domain_models.UniformDomain_3D,
    ) -> list[Row]:
        return [
            (
                field_comp.label,
                {
                    axis_to_slice: slice_field(
                        sarray_3d=field_comp.sarray_3d,
                        axis_to_slice=axis_to_slice,
                        uniform_domain=uniform_domain,
                    )
                    for axis_to_slice in self.axes_to_slice
                },
            ) for field_comp in field_comps
        ]

    def _plot_rows(
        self,
        *,
        axs_grid: manage_plots.PlotAxesGrid,
        rows: list[Row],
        step_time: float,
    ) -> None:
        for row_index, (comp_label, sliced_by_axis) in enumerate(rows):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                self.plot_slice(
                    ax=ax,
                    step_time=step_time,
                    field_slice=sliced_by_axis[axis_to_slice],
                    plane_label=get_slice_plane_label(axis_to_slice),
                    comp_label=comp_label,
                    cmap_name=self.field_args.cmap_name,
                    hide_annotations=self.hide_annotations,
                )

    def _label_axes(
        self,
        *,
        axs_grid: manage_plots.PlotAxesGrid,
    ) -> None:
        num_rows = len(axs_grid)
        for row_index in range(num_rows):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                x_label_string, y_label_string = get_slice_labels(axis_to_slice)
                if (num_rows == 1) or (row_index == num_rows - 1):
                    ax.set_xlabel(x_label_string)
                ax.set_ylabel(y_label_string)

    def _data_file_name(
        self,
        *,
        comp_axis: cartesian_axes.CartesianAxis_3D | None,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        padded_index: str,
    ) -> str:
        field_name = self.field_args.field_name
        comp_part = f"-comp={comp_axis.axis_label}" if comp_axis is not None else ""
        return (
            f"{field_name}{comp_part}-slice={axis_to_slice.axis_label}-index={padded_index}"
            f"-amr_level={self.field_args.amr_level}.npz"
        )

    def _figure_file_name(
        self,
        *,
        padded_index: str,
    ) -> str:
        field_name = self.field_args.field_name
        plot_name = f"log10_{field_name}" if self.plot_log10 else field_name
        return f"{plot_name}-slice-index={padded_index}.png"

    def _find_saved_comp_axes(
        self,
        *,
        padded_index: str,
        data_dir: Path,
    ) -> list[cartesian_axes.CartesianAxis_3D | None] | None:
        """Return the comp identities of a complete saved dataset for this snapshot, without loading
        the raw field; `[None]` for a scalar field, `self.comps_to_plot` for a vector field, or `None`
        if neither is fully present on disk.
        """
        scalar_paths = [
            data_dir / self._data_file_name(comp_axis=None, axis_to_slice=axis_to_slice, padded_index=padded_index)
            for axis_to_slice in self.axes_to_slice
        ]
        if all(path.exists() for path in scalar_paths):
            return [None]
        vector_paths = [
            data_dir / self._data_file_name(comp_axis=comp_axis, axis_to_slice=axis_to_slice, padded_index=padded_index)
            for comp_axis in self.comps_to_plot
            for axis_to_slice in self.axes_to_slice
        ]
        if all(path.exists() for path in vector_paths):
            return list(self.comps_to_plot)
        return None

    def _save_field_comps(
        self,
        *,
        field_comps: list[FieldComp],
        uniform_domain: domain_models.UniformDomain_3D,
        step_time: float,
        step_index: int,
        padded_index: str,
        data_dir: Path,
    ) -> None:
        for field_comp in field_comps:
            for axis_to_slice in self.axes_to_slice:
                field_slice = slice_field(
                    sarray_3d=field_comp.sarray_3d,
                    axis_to_slice=axis_to_slice,
                    uniform_domain=uniform_domain,
                )
                file_name = self._data_file_name(
                    comp_axis=field_comp.comp_axis,
                    axis_to_slice=axis_to_slice,
                    padded_index=padded_index,
                )
                numpy.savez(
                    data_dir / file_name,
                    sarray_2d=field_slice.sarray_2d,
                    axis_bounds=numpy.array(field_slice.axis_bounds),
                    comp_label=field_comp.label,
                    min_value=field_slice.min_value,
                    max_value=field_slice.max_value,
                    step_time=step_time,
                    step_index=step_index,
                    amr_level=self.field_args.amr_level,
                )

    def _load_saved_rows(
        self,
        *,
        comp_axes: list[cartesian_axes.CartesianAxis_3D | None],
        padded_index: str,
        data_dir: Path,
    ) -> tuple[list[Row], float]:
        rows: list[Row] = []
        step_time: float | None = None
        for comp_axis in comp_axes:
            sliced_by_axis: dict[cartesian_axes.CartesianAxis_3D, SlicedField] = {}
            comp_label = ""
            for axis_to_slice in self.axes_to_slice:
                file_name = self._data_file_name(comp_axis=comp_axis, axis_to_slice=axis_to_slice, padded_index=padded_index)
                with numpy.load(data_dir / file_name) as npz:
                    saved_bounds = npz["axis_bounds"]
                    axis_bounds: AxisBounds = (
                        (float(saved_bounds[0][0]), float(saved_bounds[0][1])),
                        (float(saved_bounds[1][0]), float(saved_bounds[1][1])),
                    )
                    sliced_by_axis[axis_to_slice] = SlicedField(
                        sarray_2d=npz["sarray_2d"],
                        axis_bounds=axis_bounds,
                        min_value=float(npz["min_value"]),
                        max_value=float(npz["max_value"]),
                    )
                    comp_label = str(npz["comp_label"])
                    step_time = float(npz["step_time"])
            rows.append((comp_label, sliced_by_axis))
        assert step_time is not None
        return rows, step_time

    def _render_figure(
        self,
        *,
        rows: list[Row],
        step_time: float,
        step_index: int,
        padded_index: str,
        figures_dir: Path,
        verbose: bool,
    ) -> None:
        if self.plot_log10:
            log10_rows: list[Row] = []
            for comp_label, sliced_by_axis in rows:
                if all(numpy.all(field_slice.sarray_2d == 0) for field_slice in sliced_by_axis.values()):
                    continue
                log10_sliced_by_axis: dict[cartesian_axes.CartesianAxis_3D, SlicedField] = {}
                for axis_to_slice, field_slice in sliced_by_axis.items():
                    log10_sarray_2d = compute_array_stats.compute_safe_log10(numpy.abs(field_slice.sarray_2d))
                    min_value, max_value = _compute_min_max(log10_sarray_2d)
                    log10_sliced_by_axis[axis_to_slice] = SlicedField(
                        sarray_2d=log10_sarray_2d,
                        axis_bounds=field_slice.axis_bounds,
                        min_value=min_value,
                        max_value=max_value,
                    )
                log10_rows.append((rf"$\log_{{10}}({comp_label.strip('$')})$", log10_sliced_by_axis))
            rows = log10_rows
            if not rows:
                manage_log.log_hint(
                    text=(
                        f"Skipping `{self.field_args.field_name}` at snapshot {step_index}: "
                        f"all components are exactly zero, so there is no data to safely log10."
                    ),
                )
                return
        num_rows = len(rows)
        fig, axs_grid = manage_plots.create_figure_grid(
            num_rows=num_rows,
            num_cols=len(self.axes_to_slice),
            x_spacing=1.0,
            y_spacing=0.25,
        )
        fig.subplots_adjust(right=0.82)
        self._plot_rows(
            axs_grid=axs_grid,
            rows=rows,
            step_time=step_time,
        )
        self._label_axes(axs_grid=axs_grid)
        fig_path = figures_dir / self._figure_file_name(padded_index=padded_index)
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=verbose,
        )

    def generate_snapshot(
        self,
        *,
        snapshot_dir: Path,
        data_dir: Path,
        figures_dir: Path,
        index_width: int,
        verbose: bool,
    ) -> None:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            ),
        )
        padded_index = f"{step_index:0{index_width}d}"
        figure_path = figures_dir / self._figure_file_name(padded_index=padded_index)
        figure_needed = self.save_figure and (self.overwrite or not figure_path.exists())
        saved_comp_axes = self._find_saved_comp_axes(padded_index=padded_index, data_dir=data_dir)
        data_complete = saved_comp_axes is not None
        data_needed = self.save_data and (self.overwrite or not data_complete)

        if not data_needed and not figure_needed:
            return

        if figure_needed and not data_needed and data_complete:
            ## cheap path: reconstruct the figure from already-saved data, skip the raw snapshot entirely
            assert saved_comp_axes is not None
            manage_log.log_hint(
                text=(
                    f"`{self.field_args.field_name}` at snapshot {step_index}: "
                    f"building figure from saved data, skipping the raw snapshot."
                ),
            )
            rows, step_time = self._load_saved_rows(
                comp_axes=saved_comp_axes,
                padded_index=padded_index,
                data_dir=data_dir,
            )
            self._render_figure(
                rows=rows,
                step_time=step_time,
                step_index=step_index,
                padded_index=padded_index,
                figures_dir=figures_dir,
                verbose=verbose,
            )
            return

        ## need the raw snapshot: either the data itself needs (re)computing, or no saved data
        ## exists yet to reconstruct the figure from
        snapshot_data = self._load_snapshot(snapshot_dir=snapshot_dir)
        field_comps = self._get_field_comps(field=snapshot_data.field)
        if data_needed:
            self._save_field_comps(
                field_comps=field_comps,
                uniform_domain=snapshot_data.uniform_domain,
                step_time=snapshot_data.step_time,
                step_index=step_index,
                padded_index=padded_index,
                data_dir=data_dir,
            )
        if figure_needed:
            rows = self._rows_from_field_comps(
                field_comps=field_comps,
                uniform_domain=snapshot_data.uniform_domain,
            )
            self._render_figure(
                rows=rows,
                step_time=snapshot_data.step_time,
                step_index=step_index,
                padded_index=padded_index,
                figures_dir=figures_dir,
                verbose=verbose,
            )


def generate_fields_in_serial(
    *,
    snapshot_tag: str,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...],
    snapshot_dirs: list[Path],
    data_dir: Path,
    figures_dir: Path,
    index_width: int,
    save_data: bool,
    save_figure: bool,
    overwrite: bool = False,
    hide_annotations: bool = False,
    plot_log10: bool = False,
    amr_level: int = 0,
) -> None:
    for field_name in fields_to_plot:
        field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
        field_args = FieldArgs(
            field_name=field_name,
            field_loader=field_meta.loader,
            cmap_name=field_meta.cmap,
            amr_level=amr_level,
        )
        generate_field_slices = GenerateFieldSlices(
            snapshot_tag=snapshot_tag,
            field_args=field_args,
            comps_to_plot=comps_to_plot,
            axes_to_slice=axes_to_slice,
            save_data=save_data,
            save_figure=save_figure,
            overwrite=overwrite,
            hide_annotations=hide_annotations,
            plot_log10=plot_log10,
        )
        for snapshot_dir in snapshot_dirs:
            generate_field_slices.generate_snapshot(
                snapshot_dir=snapshot_dir,
                data_dir=data_dir,
                figures_dir=figures_dir,
                index_width=index_width,
                verbose=False,
            )


def _generate_snapshot_worker(
    *user_args,
) -> None:
    """Positional-only signature required so WorkerArgs elements survive multiprocessing pickling."""
    worker_args = WorkerArgs(*user_args)
    field_args = FieldArgs(
        field_name=worker_args.field_name,
        field_loader=worker_args.field_loader,
        cmap_name=worker_args.cmap_name,
        amr_level=worker_args.amr_level,
    )
    generate_field_slices = GenerateFieldSlices(
        snapshot_tag=worker_args.snapshot_tag,
        field_args=field_args,
        comps_to_plot=worker_args.comps_to_plot,
        axes_to_slice=worker_args.axes_to_slice,
        save_data=worker_args.save_data,
        save_figure=worker_args.save_figure,
        overwrite=worker_args.overwrite,
        hide_annotations=worker_args.hide_annotations,
        plot_log10=worker_args.plot_log10,
    )
    generate_field_slices.generate_snapshot(
        snapshot_dir=Path(worker_args.snapshot_dir),
        data_dir=Path(worker_args.data_dir),
        figures_dir=Path(worker_args.figures_dir),
        index_width=int(worker_args.index_width),
        verbose=False,
    )


def generate_fields_in_parallel(
    *,
    snapshot_tag: str,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...],
    snapshot_dirs: list[Path],
    data_dir: Path,
    figures_dir: Path,
    index_width: int,
    save_data: bool,
    save_figure: bool,
    overwrite: bool = False,
    hide_annotations: bool = False,
    plot_log10: bool = False,
    amr_level: int = 0,
    num_workers: int | None = None,
) -> None:
    grouped_args: list[WorkerArgs] = []
    for field_name in fields_to_plot:
        field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
        for snapshot_dir in snapshot_dirs:
            grouped_args.append(
                WorkerArgs(
                    snapshot_dir=str(snapshot_dir),
                    snapshot_tag=snapshot_tag,
                    field_name=field_name,
                    field_loader=field_meta.loader,
                    comps_to_plot=comps_to_plot,
                    axes_to_slice=axes_to_slice,
                    cmap_name=field_meta.cmap,
                    data_dir=str(data_dir),
                    figures_dir=str(figures_dir),
                    index_width=index_width,
                    save_data=save_data,
                    save_figure=save_figure,
                    overwrite=overwrite,
                    hide_annotations=hide_annotations,
                    amr_level=amr_level,
                    plot_log10=plot_log10,
                ),
            )
    parallel_dispatch.run_in_parallel(
        worker_fn=_generate_snapshot_worker,
        grouped_args=grouped_args,
        num_workers=num_workers,
        timeout_seconds=120,
        show_progress=True,
        enable_plotting=True,
    )


##
## === SCRIPT INTERFACE
##


@dataclass(frozen=True)
class ResolvedInputs:
    snapshot_dirs: list[Path]
    data_dir: Path
    figures_dir: Path
    index_width: int


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path | None,
        snapshot_tag: str,
        fields_to_plot: tuple[str, ...] | list[str] | None,
        comps_to_plot: tuple[str, ...] | list[str] | None,
        axes_to_slice: tuple[str, ...] | list[str] | None,
        amr_level: int = 0,
        save_data: bool,
        save_figure: bool,
        overwrite: bool = False,
        data_dir: Path | None = None,
        figures_dir: Path | None = None,
        num_workers: int | None = None,
        animate: bool = False,
        hide_annotations: bool = False,
        plot_log10: bool = False,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        if not animate:
            cli.ensure_save_flag_selected(
                save_figure=save_figure,
                save_data=save_data,
            )
        if (save_data or save_figure) and (input_dir is None):
            raise ValueError("`--input-dir` is required with `--save-data`/`--save-figure`.")
        valid_fields = set(
            field_registry.QUOKKA_FIELD_LOOKUP.keys(),
        )
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide one or more fields to plot (via -f) from: {sorted(valid_fields)}.")
        self.input_dir = Path(input_dir) if input_dir is not None else None
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = validate_types.as_tuple(param=fields_to_plot)
        self.comps_to_plot = _parse_axes(axes=comps_to_plot)
        self.axes_to_slice = _parse_axes(axes=axes_to_slice)
        self.amr_level = amr_level
        self.save_data = save_data
        self.save_figure = save_figure
        self.overwrite = bool(overwrite)
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.figures_dir = Path(figures_dir) if figures_dir is not None else None
        self.num_workers = num_workers
        self.animate = bool(animate)
        self.hide_annotations = bool(hide_annotations)
        self.plot_log10 = bool(plot_log10)

    def _animate_fields(
        self,
        *,
        figures_dir: Path,
    ) -> None:
        for field_name in self.fields_to_plot:
            plot_name = f"log10_{field_name}" if self.plot_log10 else field_name
            fig_paths = manage_io.filter_directory(
                figures_dir,
                prefix=f"{plot_name}-slice-index=",
                suffix=".png",
                include_folders=False,
            )
            if len(fig_paths) < 3:
                manage_log.log_hint(
                    text=(
                        f"Skipping animation for `{plot_name}`: "
                        f"only found {len(fig_paths)} frame(s), but need at least 3."
                    ),
                )
                continue
            mp4_path = figures_dir / f"{plot_name}-slices.mp4"
            manage_plots.animate_pngs_to_mp4(
                frames_dir=figures_dir,
                mp4_path=mp4_path,
                pattern=f"{plot_name}-slice-index=*.png",
                fps=60,
                timeout_seconds=120,
            )

    def _resolve_inputs(
        self,
    ) -> ResolvedInputs | None:
        ## input_dir is required whenever save_data/save_figure is set (checked in __init__)
        assert self.input_dir is not None
        snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
            input_dir=self.input_dir,
            snapshot_tag=self.snapshot_tag,
            max_elems=100,
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
        index_width = find_snapshots.get_max_index_width(
            snapshot_dirs=snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
        )
        return ResolvedInputs(
            snapshot_dirs=snapshot_dirs,
            data_dir=data_dir,
            figures_dir=figures_dir,
            index_width=index_width,
        )

    def _generate_fields(
        self,
        resolved_inputs: ResolvedInputs,
    ) -> None:
        if (self.num_workers != 1) and (len(resolved_inputs.snapshot_dirs) > 5):
            generate_fields_in_parallel(
                snapshot_tag=self.snapshot_tag,
                fields_to_plot=self.fields_to_plot,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                index_width=resolved_inputs.index_width,
                save_data=self.save_data,
                save_figure=self.save_figure,
                overwrite=self.overwrite,
                hide_annotations=self.hide_annotations,
                plot_log10=self.plot_log10,
                amr_level=self.amr_level,
                num_workers=self.num_workers,
            )
        else:
            generate_fields_in_serial(
                snapshot_tag=self.snapshot_tag,
                fields_to_plot=self.fields_to_plot,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                index_width=resolved_inputs.index_width,
                save_data=self.save_data,
                save_figure=self.save_figure,
                overwrite=self.overwrite,
                hide_annotations=self.hide_annotations,
                plot_log10=self.plot_log10,
                amr_level=self.amr_level,
            )

    def _resolve_animate_dir(
        self,
        figures_dir: Path | None,
    ) -> Path:
        default_figures_dir = self.data_dir if self.data_dir is not None else self.input_dir
        resolved_figures_dir = figures_dir if figures_dir is not None else default_figures_dir
        if resolved_figures_dir is None:
            raise ValueError("`--animate` needs `--figures-dir` (or `--data-dir`/`--input-dir`) to know where to look.")
        return resolved_figures_dir

    def run(
        self,
    ) -> None:
        figures_dir = self.figures_dir
        if self.save_data or self.save_figure:
            resolved_inputs = self._resolve_inputs()
            if resolved_inputs is not None:
                self._generate_fields(resolved_inputs)
                figures_dir = resolved_inputs.figures_dir
        if self.animate:
            self._animate_fields(figures_dir=self._resolve_animate_dir(figures_dir))


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    parser = argparse.ArgumentParser(
        description="Generate midplane slices of Quokka field components: figures and/or extracted data.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_slicing=True,
                allow_output=True,
                allow_parallel=True,
            ),
        ],
    )
    parser.add_argument(
        "--plot-log10",
        action="store_true",
        default=False,
        help="Apply log10(|field|) to the plotted field (does not affect saved NPZ slices).",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        default=False,
        help="Hide text annotations: min/max values, sim time, and field label (default: False).",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        default=False,
        help="Animate figures exist under --figures-dir into an MP4 (default: False).",
    )
    user_args = parser.parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        comps_to_plot=user_args.comps,
        axes_to_slice=user_args.axes,
        save_data=user_args.save_data,
        save_figure=user_args.save_figure,
        overwrite=user_args.overwrite,
        data_dir=user_args.data_dir,
        figures_dir=user_args.figures_dir,
        animate=user_args.animate,
        hide_annotations=user_args.no_annotations,
        num_workers=user_args.num_workers,
        plot_log10=user_args.plot_log10,
        amr_level=user_args.amr_level,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
