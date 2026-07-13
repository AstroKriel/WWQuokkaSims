## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
)
from jormi.ww_io import manage_log
from jormi.ww_plots import (
    add_color,
    annotate_axis,
    manage_plots,
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
    profile_models,
)

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class CompProfile:
    step_time: float
    step_index: int
    comp_name: str
    comp_label: str
    axis_labels: list[cartesian_axes.AxisLike_3D]
    x_array_by_axis: list[numpy.ndarray]
    y_array_by_axis: list[numpy.ndarray]

    @property
    def num_axes(
        self,
    ) -> int:
        return len(self.axis_labels)

    def get_domain(
        self,
        *,
        axis_index: int,
    ) -> numpy.ndarray:
        return self.x_array_by_axis[axis_index]

    def get_values(
        self,
        *,
        axis_index: int,
    ) -> numpy.ndarray:
        return self.y_array_by_axis[axis_index]


##
## === FIELD PROCESSING
##


@final
class ComputeCompProfiles:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        snapshot_tag: str,
        field_name: str,
        field_loader: Callable,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...],
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.field_name = field_name
        self.field_loader = field_loader
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice
        self.amr_level = amr_level

    @staticmethod
    def _compute_cell_centers(
        *,
        uniform_domain_3d: domain_models.UniformDomain_3D,
        axis_to_slice: cartesian_axes.AxisLike_3D,
    ) -> numpy.ndarray:
        (x_min, _), (y_min, _), (z_min, _) = uniform_domain_3d.domain_bounds
        num_cells_x, num_cells_y, num_cells_z = uniform_domain_3d.resolution
        cell_width_x, cell_width_y, cell_width_z = uniform_domain_3d.cell_widths
        ax_idx = cartesian_axes.get_axis_index(axis_to_slice)
        if ax_idx == 0:
            return x_min + (numpy.arange(num_cells_x) + 0.5) * cell_width_x
        if ax_idx == 1:
            return y_min + (numpy.arange(num_cells_y) + 0.5) * cell_width_y
        if ax_idx == 2:
            return z_min + (numpy.arange(num_cells_z) + 0.5) * cell_width_z
        raise ValueError("axis must be one of: x_0, x_1, x_2")

    @staticmethod
    def _extract_1d_midplane_profile(
        *,
        data_3d: numpy.ndarray,
        axis_to_slice: cartesian_axes.AxisLike_3D,
    ) -> numpy.ndarray:
        num_cells_x, num_cells_y, num_cells_z = data_3d.shape
        slice_index_x = num_cells_x // 2
        slice_index_y = num_cells_y // 2
        slice_index_z = num_cells_z // 2
        ax_idx = cartesian_axes.get_axis_index(axis_to_slice)
        if ax_idx == 0:
            return data_3d[:, slice_index_y, slice_index_z]
        if ax_idx == 1:
            return data_3d[slice_index_x, :, slice_index_z]
        if ax_idx == 2:
            return data_3d[slice_index_x, slice_index_y, :]
        raise ValueError("axis must be one of: x_0, x_1, x_2")

    def _compute_scalar_profiles(
        self,
        *,
        field: field_models.ScalarField_3D,
        uniform_domain_3d: domain_models.UniformDomain_3D,
        step_index: int,
    ) -> list[CompProfile]:
        field_models.ensure_3d_sfield(field)
        step_time = field.sim_time
        assert step_time is not None
        axis_labels = list(self.axes_to_slice)
        x_array_by_axis: list[numpy.ndarray] = []
        y_array_by_axis: list[numpy.ndarray] = []
        for axis_to_slice in axis_labels:
            x_positions = ComputeCompProfiles._compute_cell_centers(
                uniform_domain_3d=uniform_domain_3d,
                axis_to_slice=axis_to_slice,
            )
            field_profile = ComputeCompProfiles._extract_1d_midplane_profile(
                data_3d=field.fdata.farray,
                axis_to_slice=axis_to_slice,
            )
            x_array_by_axis.append(x_positions)
            y_array_by_axis.append(field_profile)
        return [
            CompProfile(
                step_time=step_time,
                step_index=step_index,
                comp_name=self.field_name,
                axis_labels=axis_labels,
                comp_label=field_models.get_label(field),
                x_array_by_axis=x_array_by_axis,
                y_array_by_axis=y_array_by_axis,
            ),
        ]

    def _compute_vector_profiles(
        self,
        *,
        field: field_models.VectorField_3D,
        uniform_domain_3d: domain_models.UniformDomain_3D,
        step_index: int,
    ) -> list[CompProfile]:
        if len(self.comps_to_plot) == 0:
            raise ValueError(
                f"Vector field `{self.field_name}` requires at least one component to plot; none provided.",
            )
        field_models.ensure_3d_vfield(field)
        step_time = field.sim_time
        assert step_time is not None
        comp_names = sorted(self.comps_to_plot)
        axis_labels = list(self.axes_to_slice)
        comp_profiles: list[CompProfile] = []
        for comp_name in comp_names:
            comp_label = field_models.get_vcomp_label(field, comp_axis=comp_name)
            x_array_by_axis: list[numpy.ndarray] = []
            y_array_by_axis: list[numpy.ndarray] = []
            for axis_to_slice in axis_labels:
                x_positions = ComputeCompProfiles._compute_cell_centers(
                    uniform_domain_3d=uniform_domain_3d,
                    axis_to_slice=axis_to_slice,
                )
                comp_index = cartesian_axes.get_axis_index(comp_name)
                comp_data_3d = field.fdata.farray[comp_index]
                comp_profile = ComputeCompProfiles._extract_1d_midplane_profile(
                    data_3d=comp_data_3d,
                    axis_to_slice=axis_to_slice,
                )
                x_array_by_axis.append(x_positions)
                y_array_by_axis.append(comp_profile)
            comp_profiles.append(
                CompProfile(
                    step_time=step_time,
                    step_index=step_index,
                    comp_name=cartesian_axes.get_axis_label(comp_name),
                    axis_labels=axis_labels,
                    comp_label=comp_label,
                    x_array_by_axis=x_array_by_axis,
                    y_array_by_axis=y_array_by_axis,
                ),
            )
        return comp_profiles

    def run(
        self,
    ) -> dict[str, list[CompProfile]]:
        comp_profiles_lookup: dict[str, list[CompProfile]] = {}
        for snapshot_dir in self.snapshot_dirs:
            step_index = int(
                find_snapshots.get_step_index_string(
                    snapshot_dir=snapshot_dir,
                    snapshot_tag=self.snapshot_tag,
                ),
            )
            with load_snapshot.QuokkaSnapshot(
                    snapshot_dir=snapshot_dir,
                    verbose=False,
            ) as snapshot:
                uniform_domain_3d = snapshot.load_3d_uniform_domain(amr_level=self.amr_level)
                field = self.field_loader(snapshot, amr_level=self.amr_level)  # ScalarField or VectorField
            if isinstance(field, field_models.ScalarField_3D):
                comp_profiles = self._compute_scalar_profiles(
                    field=field,
                    uniform_domain_3d=uniform_domain_3d,
                    step_index=step_index,
                )
            elif isinstance(field, field_models.VectorField_3D):
                comp_profiles = self._compute_vector_profiles(
                    field=field,
                    uniform_domain_3d=uniform_domain_3d,
                    step_index=step_index,
                )
            else:
                raise ValueError(f"{self.field_name} is an unrecognised field type.")
            for comp_profile in comp_profiles:
                comp_label = comp_profile.comp_label
                if comp_label not in comp_profiles_lookup:
                    comp_profiles_lookup[comp_label] = []
                comp_profiles_lookup[comp_label].append(comp_profile)
        for comp_label in comp_profiles_lookup:
            comp_profiles_lookup[comp_label].sort(key=lambda item: item.step_time)
        return comp_profiles_lookup


##
## === FIGURE RENDERING
##


@final
class RenderCompProfiles:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        snapshot_tag: str,
        index_width: int,
        field_name: str,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...],
        field_loader: Callable,
        cmap_name: str,
        extracted_dir: Path,
        figures_dir: Path,
        extract_data: bool,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.index_width = index_width
        self.extracted_dir = extracted_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice
        self.field_loader = field_loader
        self.cmap_name = cmap_name
        self.extract_data = extract_data
        self.amr_level = amr_level

    def _save_comp_profiles(
        self,
        *,
        comp_profiles_lookup: dict[str, list[CompProfile]],
        extracted_dir: Path,
    ) -> None:
        extracted_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        comp_labels = list(comp_profiles_lookup.keys())
        num_snapshots = len(comp_profiles_lookup[comp_labels[0]])
        first_profile = comp_profiles_lookup[comp_labels[0]][0]
        is_scalar = first_profile.comp_name == self.field_name
        for position_index in range(num_snapshots):
            any_profile = comp_profiles_lookup[comp_labels[0]][position_index]
            step_time = any_profile.step_time
            step_index = any_profile.step_index
            index_tag = f"index={step_index:0{self.index_width}d}"
            for axis_index, axis in enumerate(any_profile.axis_labels):
                axis_label = cartesian_axes.get_axis_label(axis)
                stem = f"{self.field_name}-axis={axis_label}-{index_tag}-amr_level={self.amr_level}"
                file_path = extracted_dir / f"{stem}.json"
                if is_scalar:
                    comp_profile = comp_profiles_lookup[comp_labels[0]][position_index]
                    profile_models.ScalarProfile(
                        field_name=self.field_name,
                        step_time=step_time,
                        step_index=step_index,
                        profile_axis=axis_label,
                        position=comp_profile.get_domain(axis_index=axis_index),
                        field_value=comp_profile.get_values(axis_index=axis_index),
                        amr_level=self.amr_level,
                    ).save_to_file(file_path)
                else:
                    components = {
                        comp_profiles_lookup[comp_label][position_index].comp_name:
                        profile_models.ComponentArrays(
                            position=comp_profiles_lookup[comp_label][position_index].get_domain(
                                axis_index=axis_index,
                            ),
                            field_value=comp_profiles_lookup[comp_label][position_index].get_values(
                                axis_index=axis_index,
                            ),
                        )
                        for comp_label in comp_labels
                    }
                    profile_models.VectorProfile(
                        field_name=self.field_name,
                        step_time=step_time,
                        step_index=step_index,
                        profile_axis=axis_label,
                        components=components,
                        amr_level=self.amr_level,
                    ).save_to_file(file_path)

    @staticmethod
    def _style_axs(
        *,
        axs_grid: manage_plots.PlotAxesGrid,
        comp_labels: list[str],
        axis_labels: list[cartesian_axes.AxisLike_3D],
    ) -> None:
        num_rows = len(comp_labels)
        for row_index, comp_label in enumerate(comp_labels):
            is_bottom_row = row_index == num_rows - 1
            for col_index, axis_label in enumerate(axis_labels):
                ax = axs_grid[row_index][col_index]
                is_left_col = col_index == 0
                if is_left_col:
                    ax.set_ylabel(comp_label)
                else:
                    ax.tick_params(labelleft=False)
                if is_bottom_row:
                    axis_label_str = cartesian_axes.get_axis_label(axis_label)
                    ax.set_xlabel(axis_label_str if "$" in axis_label_str else f"${axis_label_str}$")
                else:
                    ax.tick_params(labelbottom=False)

    @staticmethod
    def _plot_comp_profile(
        *,
        axs_row: manage_plots.PlotAxesGrid,
        comp_profile: CompProfile,
        color: annotate_axis.ColorType,
    ) -> None:
        for axis_index in range(comp_profile.num_axes):
            ax = axs_row[axis_index]
            x = comp_profile.get_domain(axis_index=axis_index)
            y = comp_profile.get_values(axis_index=axis_index)
            ax.plot(
                x,
                y,
                lw=2.0,
                color=color,
            )

    def _plot_series_row(
        self,
        *,
        axs_row: manage_plots.PlotAxesGrid,
        comp_profiles: list[CompProfile],
    ) -> None:
        palette = add_color.make_palette(
            config=add_color.SequentialConfig(
                palette_name=field_registry.SEQUENTIAL_CMAP,
                palette_range=(0.25, 1.0),
            ),
            value_range=(
                0,
                max(
                    0,
                    len(comp_profiles) - 1,
                ),
            ),
        )
        for time_index, comp_profile in enumerate(comp_profiles):
            color = palette.mpl_cmap(
                palette.mpl_norm(
                    time_index,
                ),
            )
            RenderCompProfiles._plot_comp_profile(
                axs_row=axs_row,
                comp_profile=comp_profile,
                color=color,
            )
        add_color.add_colorbar(
            ax=axs_row[-1],
            palette=palette,
            label=r"snapshot index",
        )

    def run(
        self,
    ) -> None:
        ## compute midplane profiles for each snapshot and component
        compute_comp_profiles = ComputeCompProfiles(
            snapshot_dirs=self.snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
            field_name=self.field_name,
            field_loader=self.field_loader,
            comps_to_plot=self.comps_to_plot,
            axes_to_slice=self.axes_to_slice,
            amr_level=self.amr_level,
        )
        comp_profiles_lookup = compute_comp_profiles.run()
        if not comp_profiles_lookup:
            return
        comp_labels = list(
            comp_profiles_lookup.keys(),
        )
        axis_labels = comp_profiles_lookup[comp_labels[0]][0].axis_labels
        num_rows = len(comp_labels)
        num_cols = len(axis_labels)
        ## figure layout: one row per field component, one col per slice axis
        fig, axs_grid = manage_plots.create_figure_grid(
            num_rows=num_rows,
            num_cols=num_cols,
            x_spacing=0.05,
            y_spacing=0.15 if num_rows > 1 else 0.05,
        )
        ## plot each component row; use a sequential color series if there are multiple snapshots
        for row_index, comp_label in enumerate(comp_labels):
            comp_profiles = comp_profiles_lookup[comp_label]
            if len(comp_profiles) == 1:
                RenderCompProfiles._plot_comp_profile(
                    axs_row=axs_grid[row_index],
                    comp_profile=comp_profiles[0],
                    color="black",
                )
            else:
                self._plot_series_row(
                    axs_row=axs_grid[row_index],
                    comp_profiles=comp_profiles,
                )
        ## optionally write extracted profile data to JSON
        if self.extract_data:
            self._save_comp_profiles(
                comp_profiles_lookup=comp_profiles_lookup,
                extracted_dir=self.extracted_dir,
            )
        ## label axes and save; include snapshot index in filename if there is only one snapshot
        RenderCompProfiles._style_axs(
            axs_grid=axs_grid,
            comp_labels=comp_labels,
            axis_labels=axis_labels,
        )
        num_snapshots = len(comp_profiles_lookup[comp_labels[0]])
        if num_snapshots == 1:
            step_index = int(
                find_snapshots.get_step_index_string(
                    snapshot_dir=self.snapshot_dirs[0],
                    snapshot_tag=self.snapshot_tag,
                ),
            )
            padded_index = f"{step_index:0{self.index_width}d}"
            fig_path = self.figures_dir / f"{self.field_name}-profile-index={padded_index}.png"
        else:
            fig_path = self.figures_dir / f"{self.field_name}-profiles.png"
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
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...] | list[cartesian_axes.AxisLike_3D] | None,
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...] | list[cartesian_axes.AxisLike_3D] | None,
        extract_data: bool,
        extracted_dir: Path | None = None,
        figures_dir: Path | None = None,
        amr_level: int = 0,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        field_registry.validate_fields(field_names=fields_to_plot)
        if comps_to_plot is None:
            comps_to_plot = cartesian_axes.DEFAULT_3D_AXES_ORDER
        elif not set(comps_to_plot).issubset(set(cartesian_axes.DEFAULT_3D_AXES_ORDER)):
            raise ValueError("Provide one or more components (via -c) from: x_0, x_1, x_2")
        if axes_to_slice is None:
            axes_to_slice = cartesian_axes.DEFAULT_3D_AXES_ORDER
        elif not set(axes_to_slice).issubset(set(cartesian_axes.DEFAULT_3D_AXES_ORDER)):
            raise ValueError("Provide one or more axes (via -a) from: x_0, x_1, x_2")
        self.input_dir = Path(input_dir)
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = validate_types.as_tuple(param=fields_to_plot)
        self.comps_to_plot = validate_types.as_tuple(param=comps_to_plot)
        self.axes_to_slice = validate_types.as_tuple(param=axes_to_slice)
        self.extract_data = extract_data
        self.extracted_dir = Path(extracted_dir) if extracted_dir is not None else None
        self.figures_dir = Path(figures_dir) if figures_dir is not None else None
        self.amr_level = amr_level

    def run(
        self,
    ) -> None:
        ## find all snapshot dirs under input_dir whose names match snapshot_tag, sorted by index
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
        index_width = find_snapshots.get_max_index_width(
            snapshot_dirs=snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
        )
        ## compute and render profiles for each requested field
        for field_name in self.fields_to_plot:
            field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
            render_comp_profiles = RenderCompProfiles(
                snapshot_dirs=snapshot_dirs,
                snapshot_tag=self.snapshot_tag,
                index_width=index_width,
                extracted_dir=extracted_dir,
                figures_dir=figures_dir,
                field_name=field_name,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                field_loader=field_meta.loader,
                cmap_name=field_meta.cmap,
                extract_data=self.extract_data,
                amr_level=self.amr_level,
            )
            render_comp_profiles.run()


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    user_args = argparse.ArgumentParser(
        description="Plot midplane profiles of Quokka field components.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_slicing=True,
                produces_data=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        comps_to_plot=user_args.comps,
        axes_to_slice=user_args.axes,
        extract_data=user_args.save_data,
        extracted_dir=user_args.extracted_dir,
        figures_dir=user_args.figures_dir,
        amr_level=user_args.amr_level,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
