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
    annotate_panel,
    manage_figure,
    plot_data,
    style_figure,
)
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._scripts.snapshot_tools import (
    cli,
    field_registry,
)
from ww_quokka_sims.sim_io.field_diagnostics import slices
from ww_quokka_sims.sim_io.snapshots import (
    find_snapshots,
    load_snapshot,
)

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class ResolvedFieldArgs:
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
    apply_log10_plot: bool = False


@dataclass(frozen=True)
class SnapshotData:
    uniform_domain: domain_models.UniformDomain_3D
    field: field_models.AnyField_3D

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


Row = tuple[str, dict[cartesian_axes.CartesianAxis_3D, slices.SlicedField]]  # (comp_label, {axis: SlicedField})

##
## === FIELD PROCESSING
##


def _axis_to_index(
    axis: cartesian_axes.CartesianAxis_3D,
) -> int:
    return cartesian_axes.get_axis_index(axis)


def get_slice_bounds(
    *,
    uniform_domain: domain_models.UniformDomain_3D,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> slices.AxisBounds:
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
    comp_label: str,
    step_time: float,
    step_index: int,
    amr_level: int,
) -> slices.SlicedField:
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
    return slices.SlicedField(
        sarray_2d=sarray_2d,
        axis_bounds=axis_bounds,
        min_value=min_value,
        max_value=max_value,
        comp_label=comp_label,
        step_time=step_time,
        step_index=step_index,
        amr_level=amr_level,
    )


##
## === FIGURE RENDERING
##


@dataclass(frozen=True)
class GenerateFieldSlices:
    snapshot_tag: str
    field_args: ResolvedFieldArgs
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...]
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...]
    save_data: bool
    save_figure: bool
    overwrite: bool = False
    hide_annotations: bool = False
    apply_log10_plot: bool = False

    @staticmethod
    def plot_slice(
        *,
        ax: manage_figure.Panel,
        step_time: float,
        field_slice: slices.SlicedField,
        plane_label: str,
        comp_label: str,
        cmap_name: str,
        hide_annotations: bool = False,
    ) -> None:
        plot_data.plot_2d_array(
            panel=ax,
            array_2d=field_slice.sarray_2d,
            data_format="xy",
            data_aspect_ratio="equal",
            axis_ranges=field_slice.axis_bounds,
            colorbar_range=(field_slice.min_value, field_slice.max_value),
            palette_config=add_color.SequentialConfig(palette_name=cmap_name),
            add_colorbar=True,
            colorbar_label=comp_label,
            colorbar_side="right",
        )
        if not hide_annotations:
            annotate_panel.add_text(
                panel=ax,
                x_pos_fraction=0.5,
                y_pos_fraction=0.95,
                x_alignment="center",
                y_alignment="top",
                label=f"min-value = {field_slice.min_value:.2e}\nmax-value = {field_slice.max_value:.2e}",
                text_size_pt=16,
                box_alpha=0.5,
            )
            annotate_panel.add_text(
                panel=ax,
                x_pos_fraction=0.5,
                y_pos_fraction=0.5,
                x_alignment="center",
                y_alignment="center",
                label=rf"$t = {step_time:.2f}$",
                text_size_pt=16,
                box_alpha=0.5,
            )
            annotate_panel.add_text(
                panel=ax,
                x_pos_fraction=0.5,
                y_pos_fraction=0.05,
                x_alignment="center",
                y_alignment="bottom",
                label=plane_label,
                text_size_pt=16,
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
        field: field_models.AnyField_3D,
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
        if not isinstance(field, field_models.VectorField_3D):
            raise ValueError(f"{field_name} is an unrecognised field type.")
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
        step_time: float,
        step_index: int,
    ) -> list[Row]:
        return [
            (
                field_comp.label,
                {
                    axis_to_slice: slice_field(
                        sarray_3d=field_comp.sarray_3d,
                        axis_to_slice=axis_to_slice,
                        uniform_domain=uniform_domain,
                        comp_label=field_comp.label,
                        step_time=step_time,
                        step_index=step_index,
                        amr_level=self.field_args.amr_level,
                    )
                    for axis_to_slice in self.axes_to_slice
                },
            ) for field_comp in field_comps
        ]

    def _plot_rows(
        self,
        *,
        axs_grid: manage_figure.PanelGrid,
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
        axs_grid: manage_figure.PanelGrid,
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
        plot_name = f"log10_{field_name}" if self.apply_log10_plot else field_name
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
                    comp_label=field_comp.label,
                    step_time=step_time,
                    step_index=step_index,
                    amr_level=self.field_args.amr_level,
                )
                file_name = self._data_file_name(
                    comp_axis=field_comp.comp_axis,
                    axis_to_slice=axis_to_slice,
                    padded_index=padded_index,
                )
                field_slice.save_to_file(data_dir / file_name)

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
            sliced_by_axis: dict[cartesian_axes.CartesianAxis_3D, slices.SlicedField] = {}
            comp_label = ""
            for axis_to_slice in self.axes_to_slice:
                file_name = self._data_file_name(comp_axis=comp_axis, axis_to_slice=axis_to_slice, padded_index=padded_index)
                field_slice = slices.SlicedField.load_from_file(data_dir / file_name)
                sliced_by_axis[axis_to_slice] = field_slice
                comp_label = field_slice.comp_label
                step_time = field_slice.step_time
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
        if self.apply_log10_plot:
            log10_rows: list[Row] = []
            for comp_label, sliced_by_axis in rows:
                if all(numpy.all(field_slice.sarray_2d == 0) for field_slice in sliced_by_axis.values()):
                    continue
                log10_sliced_by_axis: dict[cartesian_axes.CartesianAxis_3D, slices.SlicedField] = {}
                for axis_to_slice, field_slice in sliced_by_axis.items():
                    log10_sarray_2d = compute_array_stats.compute_safe_log10(numpy.abs(field_slice.sarray_2d))
                    min_value, max_value = _compute_min_max(log10_sarray_2d)
                    log10_sliced_by_axis[axis_to_slice] = slices.SlicedField(
                        sarray_2d=log10_sarray_2d,
                        axis_bounds=field_slice.axis_bounds,
                        min_value=min_value,
                        max_value=max_value,
                        comp_label=field_slice.comp_label,
                        step_time=field_slice.step_time,
                        step_index=field_slice.step_index,
                        amr_level=field_slice.amr_level,
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
        fig, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=num_rows,
            num_panel_cols=len(self.axes_to_slice),
            panel_width_cm=8.0,
            panel_aspect_ratio=1.0,
            panel_row_gap_pt=20.0,
            panel_col_gap_pt=20.0,
        )
        self._plot_rows(
            axs_grid=axs_grid,
            rows=rows,
            step_time=step_time,
        )
        self._label_axes(axs_grid=axs_grid)
        fig_path = figures_dir / self._figure_file_name(padded_index=padded_index)
        manage_figure.save_figure(
            figure=fig,
            figure_path=fig_path,
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
                step_time=snapshot_data.step_time,
                step_index=step_index,
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
    apply_log10_plot: bool = False,
    amr_level: int = 0,
) -> None:
    for field_name in fields_to_plot:
        registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
        field_args = ResolvedFieldArgs(
            field_name=field_name,
            field_loader=registered_field.loader,
            cmap_name=registered_field.cmap,
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
            apply_log10_plot=apply_log10_plot,
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
    field_args = ResolvedFieldArgs(
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
        apply_log10_plot=worker_args.apply_log10_plot,
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
    apply_log10_plot: bool = False,
    amr_level: int = 0,
    num_workers: int | None = None,
) -> None:
    grouped_args: list[WorkerArgs] = []
    for field_name in fields_to_plot:
        registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
        for snapshot_dir in snapshot_dirs:
            grouped_args.append(
                WorkerArgs(
                    snapshot_dir=str(snapshot_dir),
                    snapshot_tag=snapshot_tag,
                    field_name=field_name,
                    field_loader=registered_field.loader,
                    comps_to_plot=comps_to_plot,
                    axes_to_slice=axes_to_slice,
                    cmap_name=registered_field.cmap,
                    data_dir=str(data_dir),
                    figures_dir=str(figures_dir),
                    index_width=index_width,
                    save_data=save_data,
                    save_figure=save_figure,
                    overwrite=overwrite,
                    hide_annotations=hide_annotations,
                    amr_level=amr_level,
                    apply_log10_plot=apply_log10_plot,
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
## === ANIMATION
##
## `--animate` stitches already-saved figures into an MP4; unlike the rest of this script, it
## needs no snapshot source at all, so it stays outside `DiagnosticPipeline`.
##


def _resolve_animate_figures_dir(
    *,
    figures_dir: Path | None,
    data_dir: Path | None,
    input_dir: Path | None,
) -> Path:
    resolved_figures_dir = figures_dir if figures_dir is not None else (data_dir if data_dir is not None else input_dir)
    if resolved_figures_dir is None:
        raise ValueError("`--animate` needs `--figures-dir` (or `--data-dir`/`--input-dir`) to know where to look.")
    return resolved_figures_dir


def _animate_saved_figures(
    *,
    figures_dir: Path,
    fields_to_plot: tuple[str, ...],
    apply_log10_plot: bool = False,
) -> None:
    for field_name in fields_to_plot:
        plot_name = f"log10_{field_name}" if apply_log10_plot else field_name
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
        video_path = figures_dir / f"{plot_name}-slices.mp4"
        manage_figure.animate_frames_to_video(
            frames_dir=figures_dir,
            video_path=video_path,
            pattern=f"{plot_name}-slice-index=*.png",
            frames_per_second=60,
            timeout_seconds=120,
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
        field_comp_axes_args: cli.FieldCompAxesArgs,
        diagnostic_output_args: cli.DiagnosticOutputArgs,
        num_workers: int | None = None,
        hide_annotations: bool = False,
        apply_log10_plot: bool = False,
    ):
        field_registry.validate_fields(
            field_names=field_comp_axes_args.fields,
            allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_comp_axes_args.fields)
        self.comps_to_plot = cli.parse_axes(axes=field_comp_axes_args.comps)
        self.axes_to_slice = cli.parse_axes(axes=field_comp_axes_args.axes)
        self.amr_level = field_comp_axes_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args
        self.num_workers = num_workers
        self.hide_annotations = hide_annotations
        self.apply_log10_plot = apply_log10_plot

    def _generate_fields(
        self,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        assert resolved_inputs.index_width is not None
        if (self.num_workers != 1) and (len(resolved_inputs.snapshot_dirs) > 5):
            generate_fields_in_parallel(
                snapshot_tag=self.snapshot_args.snapshot_tag,
                fields_to_plot=self.fields_to_plot,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                index_width=resolved_inputs.index_width,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                hide_annotations=self.hide_annotations,
                apply_log10_plot=self.apply_log10_plot,
                amr_level=self.amr_level,
                num_workers=self.num_workers,
            )
        else:
            generate_fields_in_serial(
                snapshot_tag=self.snapshot_args.snapshot_tag,
                fields_to_plot=self.fields_to_plot,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                index_width=resolved_inputs.index_width,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                hide_annotations=self.hide_annotations,
                apply_log10_plot=self.apply_log10_plot,
                amr_level=self.amr_level,
            )

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.diagnostic_output_args,
            max_elems=100,
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
        description="Generate midplane slices of Quokka snapshots.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_slicing=True,
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
        help="Apply log10(|field|) to the plotted field (does not affect the saved `.npz` data slices).",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        default=False,
        help="Hide metadata annotations: min/max values, sim time, and field label (default: False).",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        default=False,
        help="Animate figures that exist under --figures-dir into an MP4 (default: False).",
    )
    user_args = parser.parse_args()
    if not (user_args.save_data or user_args.save_figure or user_args.animate):
        raise ValueError("must pass `--save-figure`, `--save-data`, and/or `--animate`; none was given.")
    field_registry.validate_fields(
        field_names=user_args.fields,
        allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
    )
    if user_args.save_data or user_args.save_figure:
        if user_args.input_dir is None:
            raise ValueError("`--input-dir` is required with `--save-data`/`--save-figure`.")
        diagnostic_pipeline = DiagnosticPipeline(
            snapshot_args=cli.SnapshotArgs.from_user_args(user_args),
            field_comp_axes_args=cli.FieldCompAxesArgs.from_user_args(user_args),
            diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args),
            num_workers=user_args.num_workers,
            hide_annotations=user_args.no_annotations,
            apply_log10_plot=user_args.apply_log10_plot,
        )
        diagnostic_pipeline.run()
    if user_args.animate:
        figures_dir = _resolve_animate_figures_dir(
            figures_dir=user_args.figures_dir,
            data_dir=user_args.data_dir,
            input_dir=user_args.input_dir,
        )
        _animate_saved_figures(
            figures_dir=figures_dir,
            fields_to_plot=validate_types.as_tuple(param=user_args.fields),
            apply_log10_plot=user_args.apply_log10_plot,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
