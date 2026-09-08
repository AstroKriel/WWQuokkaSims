## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import pathlib
import typing

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import domain_models, field_models
from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, annotate_panel, latex_labels, manage_figure, plot_data

## local
from ww_quokka_sims.sim_io.field_diagnostics import field_palettes
from ww_quokka_sims.sim_io.snapshots import field_registry, find_snapshots, load_snapshot

##
## === SLICED FIELD
##

AxisBounds = tuple[tuple[float, float], tuple[float, float]]


@dataclasses.dataclass(frozen=True)
class FieldSlice:
    """A single 2D slice, self-contained enough to plot without the raw snapshot or uniform_domain."""

    sarray_2d: numpy.ndarray
    axis_bounds: AxisBounds
    min_value: float
    max_value: float
    comp_label: latex_labels.LatexLabel
    sim_time: float
    step_index: find_snapshots.StepIndex
    amr_level: int = 0

    def save_to_file(
        self,
        file_path: pathlib.Path,
    ) -> None:
        numpy.savez(
            file_path,
            sarray_2d=self.sarray_2d,
            axis_bounds=numpy.array(self.axis_bounds),
            comp_label=self.comp_label.content,
            min_value=self.min_value,
            max_value=self.max_value,
            sim_time=self.sim_time,
            step_index=self.step_index.value,
            amr_level=self.amr_level,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: pathlib.Path,
    ) -> "FieldSlice":
        with numpy.load(file_path) as npz:
            saved_bounds = npz["axis_bounds"]
            axis_bounds: AxisBounds = (
                (float(saved_bounds[0][0]), float(saved_bounds[0][1])),
                (float(saved_bounds[1][0]), float(saved_bounds[1][1])),
            )
            return cls(
                sarray_2d=npz["sarray_2d"],
                axis_bounds=axis_bounds,
                min_value=float(npz["min_value"]),
                max_value=float(npz["max_value"]),
                comp_label=latex_labels.LatexLabel(content=str(npz["comp_label"])),
                sim_time=float(npz["sim_time"]),
                step_index=find_snapshots.StepIndex.from_value(int(npz["step_index"])),
                amr_level=int(npz["amr_level"]),
            )


##
## === DATA CLASSES
##


@dataclasses.dataclass(frozen=True)
class ResolvedFieldArgs:
    registered_field: field_registry.RegisteredField
    amr_level: int = 0


class WorkerArgs(typing.NamedTuple):
    """Flat, pickleable argument bundle passed to the parallel slice-render worker."""

    snapshot_dir: str
    snapshot_tag: str
    registered_field: field_registry.RegisteredField
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...]
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...]
    data_dir: str
    figures_dir: str
    index_width: int
    save_data: bool
    save_figure: bool
    overwrite: bool
    hide_annotations: bool
    amr_level: int = 0
    apply_log10_plot: bool = False


@dataclasses.dataclass(frozen=True)
class SnapshotData:
    uniform_domain: domain_models.UniformDomain_3D
    field_3d: field_models.AnyField_3D

    @property
    def sim_time(
        self,
    ) -> float:
        sim_time = self.field_3d.sim_time
        if (sim_time is None) or (not numpy.isfinite(sim_time)):
            msg = f"Invalid sim_time for field: {sim_time!r}."
            manage_log.log_error(text=msg)
            raise RuntimeError(msg)
        return float(sim_time)


@dataclasses.dataclass(frozen=True)
class FieldComp:
    sarray_3d: numpy.ndarray
    label: latex_labels.LatexLabel
    comp_axis: cartesian_axes.CartesianAxis_3D | None = None


Row = tuple[latex_labels.LatexLabel, dict[cartesian_axes.CartesianAxis_3D, "FieldSlice"]]

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
) -> AxisBounds:
    """Return physical bounds of the two plane axes (i.e. those not being sliced)."""
    (x0_min, x0_max), (x1_min, x1_max), (x2_min, x2_max) = uniform_domain.domain_bounds
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X2:
        return ((x0_min, x0_max), (x1_min, x1_max))
    elif axis_to_slice == cartesian_axes.CartesianAxis_3D.X1:
        return ((x0_min, x0_max), (x2_min, x2_max))
    else:
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


def slice_3d_farray(
    *,
    farray_3d: numpy.ndarray,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
    uniform_domain: domain_models.UniformDomain_3D,
    comp_label: latex_labels.LatexLabel,
    sim_time: float,
    step_index: find_snapshots.StepIndex,
    amr_level: int,
) -> FieldSlice:
    num_cells_x0, num_cells_x1, num_cells_x2 = farray_3d.shape
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X2:
        sarray_2d = farray_3d[:, :, num_cells_x2 // 2]
    elif axis_to_slice == cartesian_axes.CartesianAxis_3D.X1:
        sarray_2d = farray_3d[:, num_cells_x1 // 2, :]
    else:
        sarray_2d = farray_3d[num_cells_x0 // 2, :, :]
    axis_bounds = get_slice_bounds(
        uniform_domain=uniform_domain,
        axis_to_slice=axis_to_slice,
    )
    min_value, max_value = _compute_min_max(sarray_2d)
    return FieldSlice(
        sarray_2d=sarray_2d,
        axis_bounds=axis_bounds,
        min_value=min_value,
        max_value=max_value,
        comp_label=comp_label,
        sim_time=sim_time,
        step_index=step_index,
        amr_level=amr_level,
    )


##
## === FIGURE RENDERING
##


@dataclasses.dataclass(frozen=True)
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
        sim_time: float,
        field_slice: FieldSlice,
        plane_label: str,
        comp_label: latex_labels.LatexLabel,
        palette_config: add_color.PaletteConfig,
        show_colorbar_label: bool = True,
        hide_annotations: bool = False,
    ) -> None:
        palette = plot_data.plot_2d_array(
            panel=ax,
            array_2d=field_slice.sarray_2d,
            data_format="xy",
            data_aspect_ratio="equal",
            axis_ranges=field_slice.axis_bounds,
            colorbar_range=(field_slice.min_value, field_slice.max_value),
            palette_config=palette_config,
            add_colorbar=False,
        )
        add_color.add_colorbar(
            panels=ax,
            palette=palette,
            ## every column in a row shares the same quantity, so only the rightmost one
            ## needs the label; the bar and its own tick values still belong on every column
            label=comp_label.label if show_colorbar_label else None,
            colorbar_side="right",
            colorbar_gap_pt=15.0,
            label_gap_pt=10.0,
        )
        if not hide_annotations:
            annotate_panel.add_text(
                panel=ax,
                x_pos_fraction=0.5,
                y_pos_fraction=0.95,
                x_alignment="center",
                y_alignment="top",
                label=f"min-value = {field_slice.min_value:.2e}\nmax-value = {field_slice.max_value:.2e}",
                box_alpha=0.5,
            )
            annotate_panel.add_text(
                panel=ax,
                x_pos_fraction=0.5,
                y_pos_fraction=0.5,
                x_alignment="center",
                y_alignment="center",
                label=rf"$t = {sim_time:.2f}$",
                box_alpha=0.5,
            )
            annotate_panel.add_text(
                panel=ax,
                x_pos_fraction=0.5,
                y_pos_fraction=0.05,
                x_alignment="center",
                y_alignment="bottom",
                label=plane_label,
                box_alpha=0.5,
            )

    def _load_snapshot(
        self,
        *,
        snapshot_dir: pathlib.Path,
    ) -> SnapshotData:
        amr_level = self.field_args.amr_level
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as quokka_snapshot:
            uniform_domain = quokka_snapshot.load_3d_uniform_domain(amr_level=amr_level)
            field_3d = self.field_args.registered_field.load(
                quokka_snapshot=quokka_snapshot,
                amr_level=amr_level,
            )
        return SnapshotData(
            uniform_domain=uniform_domain,
            field_3d=field_3d,
        )

    def _get_field_comps(
        self,
        *,
        field_3d: field_models.AnyField_3D,
    ) -> list[FieldComp]:
        field_name = self.field_args.registered_field.name
        if isinstance(field_3d, field_models.ScalarField_3D):
            sarray_3d = field_models.extract_3d_sarray(
                sfield_3d=field_3d,
                param_name=f"<{field_name}_sfield_3d>",
            )
            return [
                FieldComp(
                    sarray_3d=sarray_3d,
                    label=field_models.get_label(field_3d),
                ),
            ]
        elif isinstance(field_3d, field_models.VectorField_3D):
            if not self.comps_to_plot:
                raise ValueError(
                    f"Vector field `{field_name}` requires at least one component to plot; none provided.",
                )
            varray_3d = field_models.extract_3d_varray(
                vfield_3d=field_3d,
                param_name=f"<{field_name}_vfield_3d>",
            )
            return [
                FieldComp(
                    sarray_3d=varray_3d[_axis_to_index(comp_axis)],
                    label=field_models.get_vcomp_label(
                        vfield_3d=field_3d,
                        comp_axis=comp_axis,
                    ),
                    comp_axis=comp_axis,
                ) for comp_axis in self.comps_to_plot
            ]
        else:
            raise ValueError(f"{field_name} is an unrecognised field type.")

    def _rows_from_field_comps(
        self,
        *,
        field_comps: list[FieldComp],
        uniform_domain: domain_models.UniformDomain_3D,
        sim_time: float,
        step_index: find_snapshots.StepIndex,
    ) -> list[Row]:
        return [
            (
                field_comp.label,
                {
                    axis_to_slice:
                    slice_3d_farray(
                        farray_3d=field_comp.sarray_3d,
                        axis_to_slice=axis_to_slice,
                        uniform_domain=uniform_domain,
                        comp_label=field_comp.label,
                        sim_time=sim_time,
                        step_index=step_index,
                        amr_level=self.field_args.amr_level,
                    )
                    for axis_to_slice in self.axes_to_slice
                },
            )
            for field_comp in field_comps
        ]

    def _plot_rows(
        self,
        *,
        axs_grid: manage_figure.PanelGrid,
        rows: list[Row],
        sim_time: float,
    ) -> None:
        num_cols = len(self.axes_to_slice)
        expected_properties = self.field_args.registered_field.expected_properties
        pivot_value = expected_properties.pivot_value
        if self.apply_log10_plot:
            ## log10 of a strictly-positive field diverges around log10(1) = 0; log10 of a signed
            ## field is taken of its abs value (see above), which has no sign left to pivot around
            pivot_value = 0.0 if expected_properties.is_strictly_positive else None
        for row_index, (comp_label, sliced_by_axis) in enumerate(rows):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                field_slice = sliced_by_axis[axis_to_slice]
                self.plot_slice(
                    ax=ax,
                    sim_time=sim_time,
                    field_slice=field_slice,
                    plane_label=get_slice_plane_label(axis_to_slice),
                    comp_label=comp_label,
                    palette_config=add_color.resolve_continuous_palette_config(
                        pivot_value=pivot_value,
                        value_range=(field_slice.min_value, field_slice.max_value),
                        sequential_palette_name=field_palettes.SEQUENTIAL_PALETTE_NAME,
                        diverging_palette_name=field_palettes.DIVERGING_PALETTE_NAME,
                    ),
                    show_colorbar_label=col_index == num_cols - 1,
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

    def _get_data_file_name(
        self,
        *,
        comp_axis: cartesian_axes.CartesianAxis_3D | None,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        padded_step_index_string: str,
    ) -> str:
        field_name = self.field_args.registered_field.name
        comp_part = f"-comp={comp_axis.axis_label}" if comp_axis is not None else ""
        return (
            f"{field_name}{comp_part}-slice={axis_to_slice.axis_label}-index={padded_step_index_string}"
            f"-amr_level={self.field_args.amr_level}.npz"
        )

    def _get_figure_file_name(
        self,
        *,
        padded_step_index_string: str,
    ) -> str:
        field_name = self.field_args.registered_field.name
        plot_name = f"log10_{field_name}" if self.apply_log10_plot else field_name
        return f"{plot_name}-slice-index={padded_step_index_string}.png"

    def _find_saved_comp_axes(
        self,
        *,
        padded_step_index_string: str,
        data_dir: pathlib.Path,
    ) -> list[cartesian_axes.CartesianAxis_3D | None] | None:
        """Return the comp identities of a complete saved dataset for this snapshot, without loading
        the raw field; `[None]` for a scalar field, `self.comps_to_plot` for a vector field, or `None`
        if neither is fully present on disk.
        """
        sfield_paths = [
            data_dir / self._get_data_file_name(
                comp_axis=None,
                axis_to_slice=axis_to_slice,
                padded_step_index_string=padded_step_index_string,
            ) for axis_to_slice in self.axes_to_slice
        ]
        if all(sfield_path.exists() for sfield_path in sfield_paths):
            return [None]
        else:
            vfield_paths = [
                data_dir / self._get_data_file_name(
                    comp_axis=comp_axis,
                    axis_to_slice=axis_to_slice,
                    padded_step_index_string=padded_step_index_string,
                ) for comp_axis in self.comps_to_plot for axis_to_slice in self.axes_to_slice
            ]
            if all(vfield_path.exists() for vfield_path in vfield_paths):
                return list(self.comps_to_plot)
            else:
                return None

    def _save_field_comps(
        self,
        *,
        field_comps: list[FieldComp],
        uniform_domain: domain_models.UniformDomain_3D,
        sim_time: float,
        step_index: find_snapshots.StepIndex,
        padded_step_index_string: str,
        data_dir: pathlib.Path,
    ) -> None:
        for field_comp in field_comps:
            for axis_to_slice in self.axes_to_slice:
                field_slice = slice_3d_farray(
                    farray_3d=field_comp.sarray_3d,
                    axis_to_slice=axis_to_slice,
                    uniform_domain=uniform_domain,
                    comp_label=field_comp.label,
                    sim_time=sim_time,
                    step_index=step_index,
                    amr_level=self.field_args.amr_level,
                )
                data_file_name = self._get_data_file_name(
                    comp_axis=field_comp.comp_axis,
                    axis_to_slice=axis_to_slice,
                    padded_step_index_string=padded_step_index_string,
                )
                field_slice.save_to_file(data_dir / data_file_name)

    def _load_saved_rows(
        self,
        *,
        comp_axes: list[cartesian_axes.CartesianAxis_3D | None],
        padded_step_index_string: str,
        data_dir: pathlib.Path,
    ) -> tuple[list[Row], float]:
        rows: list[Row] = []
        sim_time: float | None = None
        for comp_axis in comp_axes:
            sliced_by_axis: dict[cartesian_axes.CartesianAxis_3D, FieldSlice] = {}
            comp_label: latex_labels.LatexLabel | None = None
            for axis_to_slice in self.axes_to_slice:
                data_file_name = self._get_data_file_name(
                    comp_axis=comp_axis,
                    axis_to_slice=axis_to_slice,
                    padded_step_index_string=padded_step_index_string,
                )
                field_slice = FieldSlice.load_from_file(data_dir / data_file_name)
                sliced_by_axis[axis_to_slice] = field_slice
                comp_label = field_slice.comp_label
                sim_time = field_slice.sim_time
            assert comp_label is not None
            rows.append((comp_label, sliced_by_axis))
        assert sim_time is not None
        return rows, sim_time

    def _render_figure(
        self,
        *,
        rows: list[Row],
        sim_time: float,
        step_index: find_snapshots.StepIndex,
        padded_step_index_string: str,
        figures_dir: pathlib.Path,
        verbose: bool,
    ) -> None:
        if self.apply_log10_plot:
            is_strictly_positive = self.field_args.registered_field.expected_properties.is_strictly_positive
            log10_rows: list[Row] = []
            for comp_label, sliced_by_axis in rows:
                if all(numpy.all(field_slice.sarray_2d == 0) for field_slice in sliced_by_axis.values()):
                    continue
                log10_sliced_by_axis: dict[cartesian_axes.CartesianAxis_3D, FieldSlice] = {}
                for axis_to_slice, field_slice in sliced_by_axis.items():
                    sarray_2d = field_slice.sarray_2d if is_strictly_positive else numpy.abs(
                        field_slice.sarray_2d,
                    )
                    log10_sarray_2d = compute_array_stats.compute_safe_log10(sarray_2d)
                    min_value, max_value = _compute_min_max(log10_sarray_2d)
                    log10_sliced_by_axis[axis_to_slice] = FieldSlice(
                        sarray_2d=log10_sarray_2d,
                        axis_bounds=field_slice.axis_bounds,
                        min_value=min_value,
                        max_value=max_value,
                        comp_label=field_slice.comp_label,
                        sim_time=field_slice.sim_time,
                        step_index=field_slice.step_index,
                        amr_level=field_slice.amr_level,
                    )
                log10_comp_label = latex_labels.LatexLabel(content=rf"\log_{{10}}({comp_label.content})")
                log10_rows.append((log10_comp_label, log10_sliced_by_axis))
            rows = log10_rows
            if not rows:
                manage_log.log_hint(
                    text=(
                        f"Skipping `{self.field_args.registered_field.name}` at snapshot {step_index.value}: "
                        f"all components are exactly zero, so there is no data to safely log10."
                    ),
                )
                return
        num_rows = len(rows)
        figure, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=num_rows,
            num_panel_cols=len(self.axes_to_slice),
            panel_width_cm=8.0,
            panel_aspect_ratio=1.0,
            panel_row_gap_pt=40.0,
            panel_col_gap_pt=100.0,
        )
        self._plot_rows(
            axs_grid=axs_grid,
            rows=rows,
            sim_time=sim_time,
        )
        self._label_axes(axs_grid=axs_grid)
        figure_path = figures_dir / self._get_figure_file_name(padded_step_index_string=padded_step_index_string)
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
            verbose=verbose,
        )

    def generate_snapshot_slices(
        self,
        *,
        snapshot_dir: pathlib.Path,
        data_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        index_width: int,
        verbose: bool,
    ) -> None:
        step_index = find_snapshots.get_step_index(
            snapshot_dir=snapshot_dir,
            snapshot_tag=self.snapshot_tag,
        )
        padded_step_index_string = step_index.get_padded_string(index_width=index_width)
        figure_path = figures_dir / self._get_figure_file_name(padded_step_index_string=padded_step_index_string)
        figure_needed = self.save_figure and (self.overwrite or not figure_path.exists())
        saved_comp_axes = self._find_saved_comp_axes(
            padded_step_index_string=padded_step_index_string,
            data_dir=data_dir,
        )
        data_complete = saved_comp_axes is not None
        data_needed = self.save_data and (self.overwrite or not data_complete)
        if data_needed or figure_needed:
            if figure_needed and not data_needed and data_complete:
                ## cheap path: reconstruct the figure from already-saved data, skip the raw snapshot entirely
                assert saved_comp_axes is not None
                manage_log.log_hint(
                    text=(
                        f"`{self.field_args.registered_field.name}` at snapshot {step_index.value}: "
                        f"building figure from saved data, skipping the raw snapshot."
                    ),
                )
                rows, sim_time = self._load_saved_rows(
                    comp_axes=saved_comp_axes,
                    padded_step_index_string=padded_step_index_string,
                    data_dir=data_dir,
                )
                self._render_figure(
                    rows=rows,
                    sim_time=sim_time,
                    step_index=step_index,
                    padded_step_index_string=padded_step_index_string,
                    figures_dir=figures_dir,
                    verbose=verbose,
                )
            else:
                snapshot_data = self._load_snapshot(snapshot_dir=snapshot_dir)
                field_comps = self._get_field_comps(field_3d=snapshot_data.field_3d)
                if data_needed:
                    self._save_field_comps(
                        field_comps=field_comps,
                        uniform_domain=snapshot_data.uniform_domain,
                        sim_time=snapshot_data.sim_time,
                        step_index=step_index,
                        padded_step_index_string=padded_step_index_string,
                        data_dir=data_dir,
                    )
                if figure_needed:
                    rows = self._rows_from_field_comps(
                        field_comps=field_comps,
                        uniform_domain=snapshot_data.uniform_domain,
                        sim_time=snapshot_data.sim_time,
                        step_index=step_index,
                    )
                    self._render_figure(
                        rows=rows,
                        sim_time=snapshot_data.sim_time,
                        step_index=step_index,
                        padded_step_index_string=padded_step_index_string,
                        figures_dir=figures_dir,
                        verbose=verbose,
                    )


def generate_fields_in_serial(
    *,
    snapshot_tag: str,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...],
    snapshot_dirs: list[pathlib.Path],
    data_dir: pathlib.Path,
    figures_dir: pathlib.Path,
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
            registered_field=registered_field,
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
            generate_field_slices.generate_snapshot_slices(
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
        registered_field=worker_args.registered_field,
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
    generate_field_slices.generate_snapshot_slices(
        snapshot_dir=pathlib.Path(worker_args.snapshot_dir),
        data_dir=pathlib.Path(worker_args.data_dir),
        figures_dir=pathlib.Path(worker_args.figures_dir),
        index_width=int(worker_args.index_width),
        verbose=False,
    )


def generate_fields_in_parallel(
    *,
    snapshot_tag: str,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...],
    snapshot_dirs: list[pathlib.Path],
    data_dir: pathlib.Path,
    figures_dir: pathlib.Path,
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
                    registered_field=registered_field,
                    comps_to_plot=comps_to_plot,
                    axes_to_slice=axes_to_slice,
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
## `--animate` stitches already-saved figures into an MP4; unlike the rest of this module, it
## needs no snapshot source at all.
##


def resolve_figures_dir_to_animate(
    *,
    figures_dir: pathlib.Path | None,
    data_dir: pathlib.Path | None,
    input_dir: pathlib.Path | None,
) -> pathlib.Path:
    if figures_dir is not None:
        resolved_figures_dir = figures_dir
    elif data_dir is not None:
        resolved_figures_dir = data_dir
    elif input_dir is not None:
        resolved_figures_dir = input_dir
    else:
        raise ValueError(
            "`--animate` also needs `--figures-dir` (or `--data-dir`/`--input-dir`) to know where to look.",
        )
    return resolved_figures_dir


def animate_saved_figures(
    *,
    figures_dir: pathlib.Path,
    fields_to_plot: tuple[str, ...],
    apply_log10_plot: bool = False,
) -> None:
    for field_name in fields_to_plot:
        plot_name = f"log10_{field_name}" if apply_log10_plot else field_name
        figure_prefix = f"{plot_name}-slice-index="
        figure_paths = manage_io.filter_directory(
            directory=figures_dir,
            prefix=figure_prefix,
            suffix=".png",
            include_folders=False,
        )
        if len(figure_paths) < 3:
            manage_log.log_hint(
                text=(
                    f"Skipping animation for `{plot_name}`: "
                    f"found {len(figure_paths)} frame(s), but need at least 3."
                ),
            )
            continue
        video_path = figures_dir / f"{plot_name}-slices.mp4"
        manage_figure.animate_frames_to_video(
            frames_dir=figures_dir,
            video_path=video_path,
            pattern=f"{figure_prefix}*.png",
            frames_per_second=60,
            timeout_seconds=120,
        )


## } MODULE
