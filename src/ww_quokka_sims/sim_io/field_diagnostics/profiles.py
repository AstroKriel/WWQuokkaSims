## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import pathlib
import re
import typing

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import domain_models, field_models
from jormi.ww_io import json_io, manage_io, manage_log
from jormi.ww_plots import add_color, annotate_panel, manage_figure
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims.sim_io.field_diagnostics import field_palettes
from ww_quokka_sims.sim_io.snapshots import field_registry, find_snapshots, load_snapshot

##
## === HELPERS
##


def _ensure_field_name(
    field_name: object,
) -> None:
    validate_types.ensure_nonempty_string(
        param=field_name,  # pyright: ignore[reportArgumentType]
        param_name="<field_name>",
    )
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", str(field_name)):
        raise ValueError(
            f"`<field_name>` must be a valid identifier, got: {field_name!r}",
        )


def _ensure_profile_axis(
    profile_axis: object,
) -> None:
    valid = cartesian_axes.VALID_3D_AXIS_LABELS
    if profile_axis not in valid:
        raise ValueError(
            f"`<profile_axis>` must be one of {valid}, got: {profile_axis!r}",
        )


def _ensure_profile_arrays(
    position: object,
    field_value: object,
) -> None:
    validate_types.ensure_ndarray_ndim(
        param=position,
        ndim=1,
        param_name="<position>",
    )
    validate_types.ensure_ndarray_ndim(
        param=field_value,
        ndim=1,
        param_name="<field_value>",
    )
    if len(position) == 0:  # pyright: ignore[reportArgumentType]
        raise ValueError("`<position>` must be non-empty.")
    if len(position) != len(field_value):  # pyright: ignore[reportArgumentType]
        raise ValueError(
            f"`<position>` and `<field_value>` must have the same length, "
            f"got {len(position)} and {len(field_value)}.",  # pyright: ignore[reportArgumentType]
        )


##
## === COMPONENT ARRAYS
##


@dataclasses.dataclass(frozen=True)
class ComponentArrays:
    position: NDArray[numpy.floating]
    field_value: NDArray[numpy.floating]
    label: str

    def __post_init__(
        self,
    ) -> None:
        _ensure_profile_arrays(
            position=self.position,
            field_value=self.field_value,
        )
        validate_types.ensure_nonempty_string(
            param=self.label,
            param_name="<label>",
        )


##
## === SCALAR PROFILE
##


@dataclasses.dataclass(frozen=True)
class ScalarProfile:
    field_name: str
    field_label: str
    step_time: float
    step_index: int
    profile_axis: str
    position: NDArray[numpy.floating]
    field_value: NDArray[numpy.floating]
    amr_level: int = 0

    def __post_init__(
        self,
    ) -> None:
        _ensure_field_name(self.field_name)
        validate_types.ensure_nonempty_string(
            param=self.field_label,
            param_name="<field_label>",
        )
        validate_types.ensure_finite_float(
            param=self.step_time,
            param_name="<step_time>",
            allow_none=False,
        )
        validate_types.ensure_finite_int(
            param=self.step_index,
            param_name="<step_index>",
            allow_none=False,
        )
        _ensure_profile_axis(self.profile_axis)
        _ensure_profile_arrays(
            position=self.position,
            field_value=self.field_value,
        )
        validate_types.ensure_finite_int(
            param=self.amr_level,
            param_name="<amr_level>",
            allow_none=False,
            require_positive=True,
            allow_zero=True,
        )

    def save_to_file(
        self,
        file_path: pathlib.Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "field_name": self.field_name,
                "field_label": self.field_label,
                "step_time": self.step_time,
                "step_index": self.step_index,
                "profile_axis": self.profile_axis,
                "position": self.position,
                "field_value": self.field_value,
                "amr_level": self.amr_level,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: pathlib.Path,
    ) -> "ScalarProfile":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={
                "field_name",
                "field_label",
                "step_time",
                "step_index",
                "profile_axis",
                "position",
                "field_value",
                "amr_level",
            },
            param_name="<ScalarProfile JSON>",
        )
        return cls(
            field_name=data["field_name"],
            field_label=data["field_label"],
            step_time=float(data["step_time"]),
            step_index=int(data["step_index"]),
            profile_axis=data["profile_axis"],
            position=numpy.asarray(data["position"]),
            field_value=numpy.asarray(data["field_value"]),
            amr_level=int(data["amr_level"]),
        )


##
## === VECTOR PROFILE
##


@dataclasses.dataclass(frozen=True)
class VectorProfile:
    field_name: str
    step_time: float
    step_index: int
    profile_axis: str
    components: dict[str, ComponentArrays]
    amr_level: int = 0

    def __post_init__(
        self,
    ) -> None:
        _ensure_field_name(self.field_name)
        validate_types.ensure_finite_float(
            param=self.step_time,
            param_name="<step_time>",
            allow_none=False,
        )
        validate_types.ensure_finite_int(
            param=self.step_index,
            param_name="<step_index>",
            allow_none=False,
        )
        _ensure_profile_axis(self.profile_axis)
        if not self.components:
            raise ValueError("`<components>` must be non-empty.")
        valid = cartesian_axes.VALID_3D_AXIS_LABELS
        for key in self.components:
            if key not in valid:
                raise ValueError(
                    f"`<components>` key must be one of {valid}, got: {key!r}",
                )
        validate_types.ensure_finite_int(
            param=self.amr_level,
            param_name="<amr_level>",
            allow_none=False,
            require_positive=True,
            allow_zero=True,
        )

    def save_to_file(
        self,
        file_path: pathlib.Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "field_name": self.field_name,
                "step_time": self.step_time,
                "step_index": self.step_index,
                "profile_axis": self.profile_axis,
                "field_comps": {
                    comp_axis: {
                        "position": comp.position,
                        "field_value": comp.field_value,
                        "label": comp.label,
                    }
                    for comp_axis, comp in self.components.items()
                },
                "amr_level": self.amr_level,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: pathlib.Path,
    ) -> "VectorProfile":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={
                "field_name",
                "step_time",
                "step_index",
                "profile_axis",
                "field_comps",
                "amr_level",
            },
            param_name="<VectorProfile JSON>",
        )
        components = {
            comp_axis:
            ComponentArrays(
                position=numpy.asarray(comp_data["position"]),
                field_value=numpy.asarray(comp_data["field_value"]),
                label=comp_data["label"],
            )
            for comp_axis, comp_data in data["field_comps"].items()
        }
        return cls(
            field_name=data["field_name"],
            step_time=float(data["step_time"]),
            step_index=int(data["step_index"]),
            profile_axis=data["profile_axis"],
            components=components,
            amr_level=int(data["amr_level"]),
        )


##
## === COMP PROFILE
##


@dataclasses.dataclass(frozen=True)
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


@typing.final
class ComputeCompProfiles:

    def __init__(
        self,
        *,
        registered_field: field_registry.RegisteredField,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...],
        amr_level: int = 0,
    ):
        self.registered_field = registered_field
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
        raise ValueError(f"axis must be one of the three cartesian axes, got {axis_to_slice!r}")

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
        raise ValueError(f"axis must be one of the three cartesian axes, got {axis_to_slice!r}")

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
            x_positions = self._compute_cell_centers(
                uniform_domain_3d=uniform_domain_3d,
                axis_to_slice=axis_to_slice,
            )
            field_profile = self._extract_1d_midplane_profile(
                data_3d=field.fdata.farray,
                axis_to_slice=axis_to_slice,
            )
            x_array_by_axis.append(x_positions)
            y_array_by_axis.append(field_profile)
        return [
            CompProfile(
                step_time=step_time,
                step_index=step_index,
                comp_name=self.registered_field.name,
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
                f"Vector field `{self.registered_field.name}` requires at least one component to plot; none provided.",
            )
        field_models.ensure_3d_vfield(field)
        step_time = field.sim_time
        assert step_time is not None
        comp_names = sorted(self.comps_to_plot)
        axis_labels = list(self.axes_to_slice)
        comp_profiles: list[CompProfile] = []
        for comp_name in comp_names:
            comp_label = field_models.get_vcomp_label(vfield_3d=field, comp_axis=comp_name)
            x_array_by_axis: list[numpy.ndarray] = []
            y_array_by_axis: list[numpy.ndarray] = []
            for axis_to_slice in axis_labels:
                x_positions = self._compute_cell_centers(
                    uniform_domain_3d=uniform_domain_3d,
                    axis_to_slice=axis_to_slice,
                )
                comp_index = cartesian_axes.get_axis_index(comp_name)
                comp_data_3d = field.fdata.farray[comp_index]
                comp_profile = self._extract_1d_midplane_profile(
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

    def compute_snapshot(
        self,
        *,
        snapshot_dir: pathlib.Path,
        snapshot_tag: str,
    ) -> list[CompProfile]:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=snapshot_tag,
            ),
        )
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as quokka_snapshot:
            uniform_domain_3d = quokka_snapshot.load_3d_uniform_domain(amr_level=self.amr_level)
            field = self.registered_field.load(
                quokka_snapshot=quokka_snapshot,
                amr_level=self.amr_level,
            )  # ScalarField or VectorField
        if isinstance(field, field_models.ScalarField_3D):
            return self._compute_scalar_profiles(
                field=field,
                uniform_domain_3d=uniform_domain_3d,
                step_index=step_index,
            )
        if isinstance(field, field_models.VectorField_3D):
            return self._compute_vector_profiles(
                field=field,
                uniform_domain_3d=uniform_domain_3d,
                step_index=step_index,
            )
        raise ValueError(f"{self.registered_field.name} is an unrecognised field type.")


##
## === FIGURE RENDERING
##


@typing.final
class GenerateCompProfiles:

    def __init__(
        self,
        *,
        snapshot_dirs: list[pathlib.Path],
        snapshot_tag: str,
        index_width: int,
        registered_field: field_registry.RegisteredField,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...],
        data_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        save_data: bool,
        save_figure: bool,
        overwrite: bool = False,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.index_width = index_width
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.registered_field = registered_field
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice
        self.save_data = save_data
        self.save_figure = save_figure
        self.overwrite = overwrite
        self.amr_level = amr_level

    def _data_file_path(
        self,
        *,
        axis_label: str,
        padded_index: str,
        data_dir: pathlib.Path,
    ) -> pathlib.Path:
        return data_dir / f"{self.registered_field.name}-axis={axis_label}-index={padded_index}-amr_level={self.amr_level}.json"

    def _snapshot_figure_file_path(
        self,
        *,
        figures_dir: pathlib.Path,
        padded_index: str,
    ) -> pathlib.Path:
        return figures_dir / f"{self.registered_field.name}-profile-index={padded_index}.png"

    def _save_snapshot_data(
        self,
        *,
        comp_profiles: list[CompProfile],
        data_dir: pathlib.Path,
        padded_index: str,
    ) -> None:
        manage_io.create_directory(
            directory=data_dir,
            verbose=False,
        )
        is_scalar = comp_profiles[0].comp_name == self.registered_field.name
        step_time = comp_profiles[0].step_time
        step_index = comp_profiles[0].step_index
        for axis_index, axis in enumerate(comp_profiles[0].axis_labels):
            axis_label = cartesian_axes.get_axis_label(axis)
            file_path = self._data_file_path(
                axis_label=axis_label,
                padded_index=padded_index,
                data_dir=data_dir,
            )
            if is_scalar:
                comp_profile = comp_profiles[0]
                ScalarProfile(
                    field_name=self.registered_field.name,
                    field_label=comp_profile.comp_label,
                    step_time=step_time,
                    step_index=step_index,
                    profile_axis=axis_label,
                    position=comp_profile.get_domain(axis_index=axis_index),
                    field_value=comp_profile.get_values(axis_index=axis_index),
                    amr_level=self.amr_level,
                ).save_to_file(file_path)
            else:
                components = {
                    comp_profile.comp_name:
                    ComponentArrays(
                        position=comp_profile.get_domain(axis_index=axis_index),
                        field_value=comp_profile.get_values(axis_index=axis_index),
                        label=comp_profile.comp_label,
                    )
                    for comp_profile in comp_profiles
                }
                VectorProfile(
                    field_name=self.registered_field.name,
                    step_time=step_time,
                    step_index=step_index,
                    profile_axis=axis_label,
                    components=components,
                    amr_level=self.amr_level,
                ).save_to_file(file_path)

    def _load_snapshot_data(
        self,
        *,
        data_paths: list[pathlib.Path],
    ) -> tuple[list[CompProfile], float] | None:
        if not all(path.exists() for path in data_paths):
            return None
        first_raw = json_io.read_json_file_into_dict(
            file_path=data_paths[0],
            verbose=False,
        )
        step_time = 0.0
        step_index = 0
        if "field_comps" not in first_raw:
            x_array_by_axis: list[numpy.ndarray] = []
            y_array_by_axis: list[numpy.ndarray] = []
            comp_label = ""
            for path in data_paths:
                scalar_profile = ScalarProfile.load_from_file(path)
                x_array_by_axis.append(scalar_profile.position)
                y_array_by_axis.append(scalar_profile.field_value)
                comp_label = scalar_profile.field_label
                step_time = scalar_profile.step_time
                step_index = scalar_profile.step_index
            comp_profiles = [
                CompProfile(
                    step_time=step_time,
                    step_index=step_index,
                    comp_name=self.registered_field.name,
                    axis_labels=list(self.axes_to_slice),
                    comp_label=comp_label,
                    x_array_by_axis=x_array_by_axis,
                    y_array_by_axis=y_array_by_axis,
                ),
            ]
            return comp_profiles, step_time
        vector_profiles = [VectorProfile.load_from_file(path) for path in data_paths]
        comp_keys = sorted(vector_profiles[0].components.keys())
        per_comp_x: dict[str, list[numpy.ndarray]] = {key: [] for key in comp_keys}
        per_comp_y: dict[str, list[numpy.ndarray]] = {key: [] for key in comp_keys}
        per_comp_label: dict[str, str] = {}
        for vector_profile in vector_profiles:
            step_time = vector_profile.step_time
            step_index = vector_profile.step_index
            for key in comp_keys:
                comp_arrays = vector_profile.components[key]
                per_comp_x[key].append(comp_arrays.position)
                per_comp_y[key].append(comp_arrays.field_value)
                per_comp_label[key] = comp_arrays.label
        comp_profiles = [
            CompProfile(
                step_time=step_time,
                step_index=step_index,
                comp_name=key,
                axis_labels=list(self.axes_to_slice),
                comp_label=per_comp_label[key],
                x_array_by_axis=per_comp_x[key],
                y_array_by_axis=per_comp_y[key],
            ) for key in comp_keys
        ]
        return comp_profiles, step_time

    @staticmethod
    def _style_axs(
        *,
        axs_grid: manage_figure.PanelGrid,
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
        axs_row: manage_figure.PanelGrid,
        comp_profile: CompProfile,
        color: annotate_panel.ColorType,
    ) -> None:
        for axis_index in range(comp_profile.num_axes):
            ax = axs_row[axis_index]
            x = comp_profile.get_domain(axis_index=axis_index)
            y = comp_profile.get_values(axis_index=axis_index)
            ax.plot(
                x,
                y,
                color=color,
            )

    def _plot_series_row(
        self,
        *,
        axs_row: manage_figure.PanelGrid,
        comp_profiles: list[CompProfile],
    ) -> None:
        palette = add_color.make_palette(
            config=add_color.SequentialConfig(
                palette_name=field_palettes.SEQUENTIAL_PALETTE_NAME,
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
            self._plot_comp_profile(
                axs_row=axs_row,
                comp_profile=comp_profile,
                color=color,
            )
        add_color.add_colorbar(
            panels=axs_row[-1],
            palette=palette,
            label=r"snapshot index",
        )

    def _save_snapshot_figure(
        self,
        *,
        comp_profiles: list[CompProfile],
        figure_path: pathlib.Path,
    ) -> None:
        axis_labels = comp_profiles[0].axis_labels
        comp_labels = [comp_profile.comp_label for comp_profile in comp_profiles]
        fig, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=len(comp_profiles),
            num_panel_cols=len(axis_labels),
        )
        for row_index, comp_profile in enumerate(comp_profiles):
            self._plot_comp_profile(
                axs_row=axs_grid[row_index],
                comp_profile=comp_profile,
                color="black",
            )
        self._style_axs(
            axs_grid=axs_grid,
            comp_labels=comp_labels,
            axis_labels=axis_labels,
        )
        manage_figure.save_figure(
            figure=fig,
            figure_path=figure_path,
            verbose=False,
        )

    def _save_summary_figure(
        self,
        *,
        comp_profiles_lookup: dict[str, list[CompProfile]],
        figures_dir: pathlib.Path,
    ) -> None:
        """Combined overlay across every saved snapshot; always rebuilt fresh from whatever is on
        disk (not from anything held in memory across the potentially-long per-snapshot loop above).
        """
        comp_labels = list(comp_profiles_lookup.keys())
        axis_labels = comp_profiles_lookup[comp_labels[0]][0].axis_labels
        fig, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=len(comp_labels),
            num_panel_cols=len(axis_labels),
        )
        for row_index, comp_label in enumerate(comp_labels):
            comp_profiles = comp_profiles_lookup[comp_label]
            if len(comp_profiles) == 1:
                self._plot_comp_profile(
                    axs_row=axs_grid[row_index],
                    comp_profile=comp_profiles[0],
                    color="black",
                )
            else:
                self._plot_series_row(
                    axs_row=axs_grid[row_index],
                    comp_profiles=comp_profiles,
                )
        self._style_axs(
            axs_grid=axs_grid,
            comp_labels=comp_labels,
            axis_labels=axis_labels,
        )
        fig_path = figures_dir / f"{self.registered_field.name}-profiles-summary.png"
        manage_figure.save_figure(
            figure=fig,
            figure_path=fig_path,
            verbose=True,
        )

    def _process_snapshot(
        self,
        *,
        compute_comp_profiles: ComputeCompProfiles,
        snapshot_dir: pathlib.Path,
        data_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        index_width: int,
    ) -> None:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            ),
        )
        padded_index = f"{step_index:0{index_width}d}"
        data_paths = [
            self._data_file_path(
                axis_label=cartesian_axes.get_axis_label(axis),
                padded_index=padded_index,
                data_dir=data_dir,
            ) for axis in self.axes_to_slice
        ]
        figure_path = self._snapshot_figure_file_path(
            figures_dir=figures_dir,
            padded_index=padded_index,
        )
        data_exists = all(path.exists() for path in data_paths)
        data_needed = self.save_data and (self.overwrite or not data_exists)
        figure_needed = self.save_figure and (self.overwrite or not figure_path.exists())

        if not data_needed and not figure_needed:
            return

        if figure_needed and not data_needed and data_exists:
            loaded = self._load_snapshot_data(data_paths=data_paths)
            if loaded is not None:
                ## cheap path: reconstruct the figure from already-saved data, skip the raw snapshot
                manage_log.log_hint(
                    text=(
                        f"`{self.registered_field.name}` at snapshot {step_index}: "
                        f"building figure from saved data, skipping the raw snapshot."
                    ),
                )
                comp_profiles, _step_time = loaded
                self._save_snapshot_figure(
                    comp_profiles=comp_profiles,
                    figure_path=figure_path,
                )
                return

        comp_profiles = compute_comp_profiles.compute_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot_tag=self.snapshot_tag,
        )
        if data_needed:
            self._save_snapshot_data(
                comp_profiles=comp_profiles,
                data_dir=data_dir,
                padded_index=padded_index,
            )
        if figure_needed:
            self._save_snapshot_figure(
                comp_profiles=comp_profiles,
                figure_path=figure_path,
            )

    def _load_all_saved_comp_profiles(
        self,
        *,
        data_dir: pathlib.Path,
    ) -> dict[str, list[CompProfile]]:
        first_axis_label = cartesian_axes.get_axis_label(self.axes_to_slice[0])
        pattern = f"{self.registered_field.name}-axis={first_axis_label}-index=*-amr_level={self.amr_level}.json"
        comp_profiles_lookup: dict[str, list[CompProfile]] = {}
        for first_axis_path in sorted(data_dir.glob(pattern)):
            raw = json_io.read_json_file_into_dict(
                file_path=first_axis_path,
                verbose=False,
            )
            padded_index = f"{int(raw['step_index']):0{self.index_width}d}"
            data_paths = [
                self._data_file_path(
                    axis_label=cartesian_axes.get_axis_label(axis),
                    padded_index=padded_index,
                    data_dir=data_dir,
                ) for axis in self.axes_to_slice
            ]
            loaded = self._load_snapshot_data(data_paths=data_paths)
            if loaded is None:
                continue
            comp_profiles, _step_time = loaded
            for comp_profile in comp_profiles:
                comp_profiles_lookup.setdefault(comp_profile.comp_label, []).append(comp_profile)
        for comp_label in comp_profiles_lookup:
            comp_profiles_lookup[comp_label].sort(key=lambda item: item.step_time)
        return comp_profiles_lookup

    def run(
        self,
    ) -> None:
        if self.save_data or self.save_figure:
            compute_comp_profiles = ComputeCompProfiles(
                registered_field=self.registered_field,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                amr_level=self.amr_level,
            )
            for snapshot_dir in self.snapshot_dirs:
                self._process_snapshot(
                    compute_comp_profiles=compute_comp_profiles,
                    snapshot_dir=snapshot_dir,
                    data_dir=self.data_dir,
                    figures_dir=self.figures_dir,
                    index_width=self.index_width,
                )
        if not self.save_figure:
            return
        ## the summary is only buildable from saved data; if none was ever saved for this field
        ## (eg. --save-figure was used without --save-data, ever), there's nothing to aggregate
        comp_profiles_lookup = self._load_all_saved_comp_profiles(data_dir=self.data_dir)
        if not comp_profiles_lookup:
            manage_log.log_hint(
                text=f"Skipping summary figure for `{self.registered_field.name}`: no saved data found in {self.data_dir}.",
            )
            return
        self._save_summary_figure(
            comp_profiles_lookup=comp_profiles_lookup,
            figures_dir=self.figures_dir,
        )


## } MODULE
