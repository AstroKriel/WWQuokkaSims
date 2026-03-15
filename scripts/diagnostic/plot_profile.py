## { SCRIPT

##
## === DEPENDENCIES
##

import re
import csv
import numpy
import argparse

from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

from jormi.ww_types import check_types
from jormi.ww_plots import manage_plots, add_color
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import field_types, domain_types

from ww_quokka_sims.sim_io import find_datasets, load_dataset

import utils

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class CompProfile:
    sim_time: float
    comp_label: str
    axis_labels: list[cartesian_axes.AxisLike_3D]
    x_array_by_axis: list[numpy.ndarray]
    y_array_by_axis: list[numpy.ndarray]

    @property
    def num_axes(
        self,
    ) -> int:
        return len(self.axis_labels)

    def get(
        self,
        *,
        axis_index: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        return (
            self.x_array_by_axis[axis_index],
            self.y_array_by_axis[axis_index],
        )


##
## === OPERATOR CLASSES
##


class ComputeCompProfiles:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        field_name: str,
        field_loader: Callable,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...],
    ):
        self.dataset_dirs = dataset_dirs
        self.field_name = field_name
        self.field_loader = field_loader
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice

    @staticmethod
    def _compute_cell_centers(
        *,
        udomain_3d: domain_types.UniformDomain_3D,
        axis_to_slice: cartesian_axes.AxisLike_3D,
    ) -> numpy.ndarray:
        (x_min, _), (y_min, _), (z_min, _) = udomain_3d.domain_bounds
        num_cells_x, num_cells_y, num_cells_z = udomain_3d.resolution
        cell_width_x, cell_width_y, cell_width_z = udomain_3d.cell_widths
        ax_idx = cartesian_axes.get_axis_index(axis_to_slice)
        if ax_idx == 0: return x_min + (numpy.arange(num_cells_x) + 0.5) * cell_width_x
        if ax_idx == 1: return y_min + (numpy.arange(num_cells_y) + 0.5) * cell_width_y
        if ax_idx == 2: return z_min + (numpy.arange(num_cells_z) + 0.5) * cell_width_z
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
        if ax_idx == 0: return data_3d[:, slice_index_y, slice_index_z]
        if ax_idx == 1: return data_3d[slice_index_x, :, slice_index_z]
        if ax_idx == 2: return data_3d[slice_index_x, slice_index_y, :]
        raise ValueError("axis must be one of: x_0, x_1, x_2")

    def _compute_scalar_profiles(
        self,
        *,
        field: field_types.ScalarField_3D,
        udomain_3d: domain_types.UniformDomain_3D,
    ) -> list[CompProfile]:
        field_types.ensure_3d_sfield(field)
        sim_time = utils.get_sim_time(field=field)
        axis_labels = list(self.axes_to_slice)
        x_array_by_axis: list[numpy.ndarray] = []
        y_array_by_axis: list[numpy.ndarray] = []
        for axis_to_slice in axis_labels:
            x_positions = ComputeCompProfiles._compute_cell_centers(
                udomain_3d=udomain_3d,
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
                sim_time=sim_time,
                axis_labels=axis_labels,
                comp_label=field_types.get_label(field),
                x_array_by_axis=x_array_by_axis,
                y_array_by_axis=y_array_by_axis,
            ),
        ]

    def _compute_vector_profiles(
        self,
        *,
        field: field_types.VectorField_3D,
        udomain_3d: domain_types.UniformDomain_3D,
    ) -> list[CompProfile]:
        if len(self.comps_to_plot) == 0:
            raise ValueError(
                f"Vector field `{self.field_name}` requires at least one component to plot; none provided.",
            )
        field_types.ensure_3d_vfield(field)
        sim_time = utils.get_sim_time(field=field)
        comp_names = sorted(self.comps_to_plot)
        axis_labels = list(self.axes_to_slice)
        comp_profiles: list[CompProfile] = []
        for comp_name in comp_names:
            comp_label = field_types.get_vcomp_label(field, comp_name)
            x_array_by_axis: list[numpy.ndarray] = []
            y_array_by_axis: list[numpy.ndarray] = []
            for axis_to_slice in axis_labels:
                x_positions = ComputeCompProfiles._compute_cell_centers(
                    udomain_3d=udomain_3d,
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
                    sim_time=sim_time,
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
        for dataset_dir in self.dataset_dirs:
            with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
                udomain_3d = ds.load_3d_uniform_domain()
                field = self.field_loader(ds)  # ScalarField or VectorField
            if isinstance(field, field_types.ScalarField_3D):
                comp_profiles = self._compute_scalar_profiles(
                    field=field,
                    udomain_3d=udomain_3d,
                )
            elif isinstance(field, field_types.VectorField_3D):
                comp_profiles = self._compute_vector_profiles(
                    field=field,
                    udomain_3d=udomain_3d,
                )
            else:
                raise ValueError(f"{self.field_name} is an unrecognised field type.")
            for comp_profile in comp_profiles:
                comp_label = comp_profile.comp_label
                if comp_label not in comp_profiles_lookup:
                    comp_profiles_lookup[comp_label] = []
                comp_profiles_lookup[comp_label].append(comp_profile)
        for comp_label in comp_profiles_lookup:
            comp_profiles_lookup[comp_label].sort(key=lambda item: item.sim_time)
        return comp_profiles_lookup


class RenderCompProfiles:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        dataset_tag: str,
        field_name: str,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...],
        field_loader: Callable,
        cmap_name: str,
        fig_dir: Path,
        save_profiles: bool,
    ):
        self.dataset_dirs = dataset_dirs
        self.dataset_tag = dataset_tag
        self.fig_dir = Path(fig_dir)
        self.field_name = field_name
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice
        self.field_loader = field_loader
        self.cmap_name = cmap_name
        self.save_profiles = save_profiles

    @staticmethod
    def _safe_slug(
        text: str,
    ) -> str:
        text = text.strip()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Za-z0-9_\-\.]+", "", text)
        return text if text else "profile"

    def _save_comp_profiles(
        self,
        *,
        comp_profiles_lookup: dict[str, list[CompProfile]],
        out_dir: Path,
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for comp_label, comp_profiles in comp_profiles_lookup.items():
            comp_slug = self._safe_slug(comp_label)
            for comp_profile in comp_profiles:
                t_str = f"{comp_profile.sim_time:.3f}"
                for axis_index, axis_label in enumerate(comp_profile.axis_labels):
                    domain, values = comp_profile.get(axis_index=axis_index)
                    file_name = f"{self.field_name}_comp={comp_slug}_along={axis_label}_t={t_str}.csv"
                    file_path = out_dir / file_name
                    with file_path.open("w", newline="") as fp:
                        writer = csv.writer(fp)
                        writer.writerow(["domain", "values"])
                        for position, value in zip(domain, values, strict=False):
                            writer.writerow([float(position), float(value)])

    @staticmethod
    def _style_axs(
        *,
        axs_grid,
        comp_labels: list[str],
        axis_labels: list[cartesian_axes.AxisLike_3D],
    ) -> None:
        for row_index, comp_label in enumerate(comp_labels):
            for col_index, axis_label in enumerate(axis_labels):
                ax = axs_grid[row_index][col_index]
                if col_index == 0:
                    ax.set_ylabel(comp_label)
                ax.set_xlabel(utils.as_latex_label(str(axis_label)))

    @staticmethod
    def _plot_comp_profile(
        *,
        axs_row,
        comp_profile: CompProfile,
        color,
    ) -> None:
        for axis_index in range(comp_profile.num_axes):
            ax = axs_row[axis_index]
            x, y = comp_profile.get(axis_index=axis_index)
            ax.plot(x, y, lw=2.0, color=color)

    def _plot_series_row(
        self,
        *,
        axs_row,
        comp_profiles: list[CompProfile],
    ) -> None:
        palette = add_color.make_palette(
            config=add_color.SequentialConfig(
                palette_name=self.cmap_name,
                palette_range=(0.25, 1.0),
            ),
            value_range=(0, max(0,
                                len(comp_profiles) - 1)),
        )
        for time_index, comp_profile in enumerate(comp_profiles):
            color = palette.mpl_cmap(palette.mpl_norm(time_index))
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
        compute_comp_profiles = ComputeCompProfiles(
            dataset_dirs=self.dataset_dirs,
            field_name=self.field_name,
            field_loader=self.field_loader,
            comps_to_plot=self.comps_to_plot,
            axes_to_slice=self.axes_to_slice,
        )
        comp_profiles_lookup = compute_comp_profiles.run()
        if not comp_profiles_lookup:
            return
        comp_labels = list(comp_profiles_lookup.keys())
        axis_labels = comp_profiles_lookup[comp_labels[0]][0].axis_labels
        num_rows = len(comp_labels)
        num_cols = len(axis_labels)
        fig, axs_grid = utils.create_figure(
            num_rows=num_rows,
            num_cols=num_cols,
        )
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
        if self.save_profiles:
            self._save_comp_profiles(
                comp_profiles_lookup=comp_profiles_lookup,
                out_dir=self.fig_dir,
            )
        RenderCompProfiles._style_axs(
            axs_grid=axs_grid,
            comp_labels=comp_labels,
            axis_labels=axis_labels,
        )
        num_snapshots = len(comp_profiles_lookup[comp_labels[0]])
        if num_snapshots == 1:
            snapshot_index = find_datasets.get_dataset_index_string(self.dataset_dirs[0], self.dataset_tag)
            fig_path = self.fig_dir / f"{self.field_name}_profile_{snapshot_index}.png"
        else:
            fig_path = self.fig_dir / f"{self.field_name}_profiles.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        dataset_tag: str,
        fields_to_plot: list[str],
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...] | list[cartesian_axes.AxisLike_3D] | None,
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...] | list[cartesian_axes.AxisLike_3D] | None,
        save_profiles: bool,
    ):
        check_types.ensure_nonempty_string(
            param=dataset_tag,
            param_name="dataset_tag",
        )
        utils.validate_fields(fields_to_plot)
        if comps_to_plot is None:
            comps_to_plot = cartesian_axes.DEFAULT_3D_AXES_ORDER
        elif not set(comps_to_plot).issubset(set(cartesian_axes.DEFAULT_3D_AXES_ORDER)):
            raise ValueError("Provide one or more components (via -c) from: x_0, x_1, x_2")
        if axes_to_slice is None:
            axes_to_slice = cartesian_axes.DEFAULT_3D_AXES_ORDER
        elif not set(axes_to_slice).issubset(set(cartesian_axes.DEFAULT_3D_AXES_ORDER)):
            raise ValueError("Provide one or more axes (via -a) from: x_0, x_1, x_2")
        self.input_dir = Path(input_dir)
        self.dataset_tag = dataset_tag
        self.fields_to_plot = check_types.as_tuple(param=fields_to_plot)
        self.comps_to_plot = check_types.as_tuple(param=comps_to_plot)
        self.axes_to_slice = check_types.as_tuple(param=axes_to_slice)
        self.save_profiles = save_profiles

    def run(
        self,
    ) -> None:
        dataset_dirs = find_datasets.resolve_dataset_dirs(
            input_dir=self.input_dir,
            dataset_tag=self.dataset_tag,
        )
        if not dataset_dirs:
            return
        fig_dir = dataset_dirs[0].parent
        for field_name in self.fields_to_plot:
            field_meta = utils.QUOKKA_FIELD_LOOKUP[field_name]
            render_comp_profiles = RenderCompProfiles(
                dataset_dirs=dataset_dirs,
                dataset_tag=self.dataset_tag,
                fig_dir=fig_dir,
                field_name=field_name,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                field_loader=field_meta["loader"],
                cmap_name=field_meta["cmap"],
                save_profiles=self.save_profiles,
            )
            render_comp_profiles.run()


##
## === PROGRAM MAIN
##


def main():
    parser = argparse.ArgumentParser(
        description="Plot midplane profiles of Quokka field components.",
        parents=[utils.base_parser()],
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        default=False,
        help="Save profiles as CSVs (default: False).",
    )
    user_args = parser.parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.dir,
        dataset_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        comps_to_plot=user_args.comps,
        axes_to_slice=user_args.axes,
        save_profiles=user_args.save,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
