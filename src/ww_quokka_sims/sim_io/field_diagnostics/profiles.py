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
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import add_color, annotate_panel, latex_labels, manage_figure
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
## === SCALAR PROFILE
##


@dataclasses.dataclass(frozen=True)
class ScalarFieldProfile:
    field_name: str
    field_latex_label: latex_labels.LatexLabel
    sim_time: float
    step_index: find_snapshots.StepIndex
    profile_axis: str
    position: NDArray[numpy.floating]
    field_value: NDArray[numpy.floating]
    amr_level: int = 0

    def __post_init__(
        self,
    ) -> None:
        _ensure_field_name(self.field_name)
        validate_types.ensure_finite_float(
            param=self.sim_time,
            param_name="<sim_time>",
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
                "field_label": self.field_latex_label.content,
                "sim_time": self.sim_time,
                "step_index": self.step_index.value,
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
    ) -> "ScalarFieldProfile":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={
                "field_name",
                "field_label",
                "sim_time",
                "step_index",
                "profile_axis",
                "position",
                "field_value",
                "amr_level",
            },
            param_name="<ScalarFieldProfile JSON>",
        )
        return cls(
            field_name=data["field_name"],
            field_latex_label=latex_labels.LatexLabel(content=data["field_label"]),
            sim_time=float(data["sim_time"]),
            step_index=find_snapshots.StepIndex.from_value(int(data["step_index"])),
            profile_axis=data["profile_axis"],
            position=numpy.asarray(data["position"]),
            field_value=numpy.asarray(data["field_value"]),
            amr_level=int(data["amr_level"]),
        )


##
## === VECTOR COMPONENT
##


@dataclasses.dataclass(frozen=True)
class VectorComponent:
    field_value: NDArray[numpy.floating]
    latex_label: latex_labels.LatexLabel

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_ndarray_ndim(
            param=self.field_value,
            ndim=1,
            param_name="<field_value>",
        )


##
## === VECTOR PROFILE
##


@dataclasses.dataclass(frozen=True)
class VectorFieldProfile:
    field_name: str
    sim_time: float
    step_index: find_snapshots.StepIndex
    profile_axis: str
    position: NDArray[numpy.floating]
    components: dict[cartesian_axes.CartesianAxis_3D, VectorComponent]
    amr_level: int = 0

    def __post_init__(
        self,
    ) -> None:
        _ensure_field_name(self.field_name)
        validate_types.ensure_finite_float(
            param=self.sim_time,
            param_name="<sim_time>",
            allow_none=False,
        )
        _ensure_profile_axis(self.profile_axis)
        if not self.components:
            raise ValueError("`<components>` must be non-empty.")
        for component in self.components.values():
            _ensure_profile_arrays(
                position=self.position,
                field_value=component.field_value,
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
                "sim_time": self.sim_time,
                "step_index": self.step_index.value,
                "profile_axis": self.profile_axis,
                "position": self.position,
                "field_comps": {
                    comp_axis: {
                        "field_value": component.field_value,
                        "label": component.latex_label.content,
                    }
                    for comp_axis, component in self.components.items()
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
    ) -> "VectorFieldProfile":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={
                "field_name",
                "sim_time",
                "step_index",
                "profile_axis",
                "position",
                "field_comps",
                "amr_level",
            },
            param_name="<VectorFieldProfile JSON>",
        )
        components = {
            cartesian_axes.as_axis(comp_axis):
            VectorComponent(
                field_value=numpy.asarray(comp_data["field_value"]),
                latex_label=latex_labels.LatexLabel(content=comp_data["label"]),
            )
            for comp_axis, comp_data in data["field_comps"].items()
        }
        return cls(
            field_name=data["field_name"],
            sim_time=float(data["sim_time"]),
            step_index=find_snapshots.StepIndex.from_value(int(data["step_index"])),
            profile_axis=data["profile_axis"],
            position=numpy.asarray(data["position"]),
            components=components,
            amr_level=int(data["amr_level"]),
        )


##
## === COMP PROFILE
##


@dataclasses.dataclass(frozen=True)
class CompProfile:
    sim_time: float
    step_index: find_snapshots.StepIndex
    comp_name: str
    comp_latex_label: latex_labels.LatexLabel
    axis_labels: list[cartesian_axes.AxisLike_3D]
    domain_array_by_axis: list[numpy.ndarray]
    values_array_by_axis: list[numpy.ndarray]

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
        return self.domain_array_by_axis[axis_index]

    def get_values(
        self,
        *,
        axis_index: int,
    ) -> numpy.ndarray:
        return self.values_array_by_axis[axis_index]


##
## === FIELD PROCESSING
##


@typing.final
class ComputeCompProfiles:

    def __init__(
        self,
        *,
        snapshot_dirs: list[pathlib.Path],
        snapshot_tag: str,
        registered_field: field_registry.RegisteredField,
        index_width: int,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        axes_to_slice: tuple[cartesian_axes.AxisLike_3D, ...],
        save_data: bool,
        data_dir: pathlib.Path,
        overwrite: bool = False,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.registered_field = registered_field
        self.index_width = index_width
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice
        self.save_data = save_data
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.amr_level = amr_level

    def _get_data_path(
        self,
        *,
        axis_label: str,
        padded_step_index_string: str,
    ) -> pathlib.Path:
        return self.data_dir / f"{self.registered_field.name}-axis={axis_label}-index={padded_step_index_string}-amr_level={self.amr_level}.json"

    def _save_snapshot_data(
        self,
        *,
        comp_profiles: list[CompProfile],
        padded_step_index_string: str,
    ) -> None:
        is_scalar = comp_profiles[0].comp_name == self.registered_field.name
        sim_time = comp_profiles[0].sim_time
        step_index = comp_profiles[0].step_index
        for axis_index, axis_label in enumerate(comp_profiles[0].axis_labels):
            axis_label_str = cartesian_axes.get_axis_label(axis_label)
            file_path = self._get_data_path(
                axis_label=axis_label_str,
                padded_step_index_string=padded_step_index_string,
            )
            if is_scalar:
                comp_profile = comp_profiles[0]
                ScalarFieldProfile(
                    field_name=self.registered_field.name,
                    field_latex_label=comp_profile.comp_latex_label,
                    sim_time=sim_time,
                    step_index=step_index,
                    profile_axis=axis_label_str,
                    position=comp_profile.get_domain(axis_index=axis_index),
                    field_value=comp_profile.get_values(axis_index=axis_index),
                    amr_level=self.amr_level,
                ).save_to_file(file_path)
            else:
                position = comp_profiles[0].get_domain(axis_index=axis_index)
                components = {
                    cartesian_axes.as_axis(comp_profile.comp_name):
                    VectorComponent(
                        field_value=comp_profile.get_values(axis_index=axis_index),
                        latex_label=comp_profile.comp_latex_label,
                    )
                    for comp_profile in comp_profiles
                }
                VectorFieldProfile(
                    field_name=self.registered_field.name,
                    sim_time=sim_time,
                    step_index=step_index,
                    profile_axis=axis_label_str,
                    position=position,
                    components=components,
                    amr_level=self.amr_level,
                ).save_to_file(file_path)

    def _load_snapshot_data(
        self,
        *,
        data_paths: list[pathlib.Path],
    ) -> tuple[list[CompProfile], float]:
        """Load profiles already saved to `data_paths`; caller must confirm they all exist first."""
        first_raw = json_io.read_json_file_into_dict(
            file_path=data_paths[0],
            verbose=False,
        )
        sim_time = 0.0
        step_index = find_snapshots.StepIndex.from_value(0)
        if "field_comps" not in first_raw:
            domain_array_by_axis: list[numpy.ndarray] = []
            values_array_by_axis: list[numpy.ndarray] = []
            comp_latex_label: latex_labels.LatexLabel | None = None
            for data_path in data_paths:
                scalar_field_profile = ScalarFieldProfile.load_from_file(data_path)
                domain_array_by_axis.append(scalar_field_profile.position)
                values_array_by_axis.append(scalar_field_profile.field_value)
                comp_latex_label = scalar_field_profile.field_latex_label
                sim_time = scalar_field_profile.sim_time
                step_index = scalar_field_profile.step_index
            assert comp_latex_label is not None
            comp_profiles = [
                CompProfile(
                    sim_time=sim_time,
                    step_index=step_index,
                    comp_name=self.registered_field.name,
                    comp_latex_label=comp_latex_label,
                    axis_labels=list(self.axes_to_slice),
                    domain_array_by_axis=domain_array_by_axis,
                    values_array_by_axis=values_array_by_axis,
                ),
            ]
            return comp_profiles, sim_time
        else:
            vector_profiles = [VectorFieldProfile.load_from_file(data_path) for data_path in data_paths]
            comp_keys = sorted(vector_profiles[0].components.keys())
            per_comp_domain: dict[cartesian_axes.CartesianAxis_3D, list[numpy.ndarray]] = {key: [] for key in comp_keys}
            per_comp_values: dict[cartesian_axes.CartesianAxis_3D, list[numpy.ndarray]] = {key: [] for key in comp_keys}
            per_comp_latex_label: dict[cartesian_axes.CartesianAxis_3D, latex_labels.LatexLabel] = {}
            for vector_field_profile in vector_profiles:
                sim_time = vector_field_profile.sim_time
                step_index = vector_field_profile.step_index
                for key in comp_keys:
                    component = vector_field_profile.components[key]
                    per_comp_domain[key].append(vector_field_profile.position)
                    per_comp_values[key].append(component.field_value)
                    per_comp_latex_label[key] = component.latex_label
            comp_profiles = [
                CompProfile(
                    sim_time=sim_time,
                    step_index=step_index,
                    comp_name=key,
                    comp_latex_label=per_comp_latex_label[key],
                    axis_labels=list(self.axes_to_slice),
                    domain_array_by_axis=per_comp_domain[key],
                    values_array_by_axis=per_comp_values[key],
                ) for key in comp_keys
            ]
            return comp_profiles, sim_time

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
        elif ax_idx == 1:
            return y_min + (numpy.arange(num_cells_y) + 0.5) * cell_width_y
        elif ax_idx == 2:
            return z_min + (numpy.arange(num_cells_z) + 0.5) * cell_width_z
        else:
            raise ValueError(f"axis must be one of the three cartesian axes, got {axis_to_slice!r}")

    @staticmethod
    def _extract_1d_midplane_profile(
        *,
        sarray_3d: numpy.ndarray,
        axis_to_slice: cartesian_axes.AxisLike_3D,
    ) -> numpy.ndarray:
        num_cells_x, num_cells_y, num_cells_z = sarray_3d.shape
        slice_index_x = num_cells_x // 2
        slice_index_y = num_cells_y // 2
        slice_index_z = num_cells_z // 2
        ax_idx = cartesian_axes.get_axis_index(axis_to_slice)
        if ax_idx == 0:
            return sarray_3d[:, slice_index_y, slice_index_z]
        elif ax_idx == 1:
            return sarray_3d[slice_index_x, :, slice_index_z]
        elif ax_idx == 2:
            return sarray_3d[slice_index_x, slice_index_y, :]
        else:
            raise ValueError(f"axis must be one of the three cartesian axes, got {axis_to_slice!r}")

    def _compute_scalar_profiles(
        self,
        *,
        sfield_3d: field_models.ScalarField_3D,
        uniform_domain_3d: domain_models.UniformDomain_3D,
        step_index: find_snapshots.StepIndex,
    ) -> list[CompProfile]:
        field_models.ensure_3d_sfield(sfield_3d)
        sim_time = sfield_3d.sim_time
        assert sim_time is not None
        axis_labels = list(self.axes_to_slice)
        domain_array_by_axis: list[numpy.ndarray] = []
        values_array_by_axis: list[numpy.ndarray] = []
        for axis_to_slice in axis_labels:
            x_positions = self._compute_cell_centers(
                uniform_domain_3d=uniform_domain_3d,
                axis_to_slice=axis_to_slice,
            )
            field_profile = self._extract_1d_midplane_profile(
                sarray_3d=sfield_3d.fdata.farray,
                axis_to_slice=axis_to_slice,
            )
            domain_array_by_axis.append(x_positions)
            values_array_by_axis.append(field_profile)
        return [
            CompProfile(
                sim_time=sim_time,
                step_index=step_index,
                comp_name=self.registered_field.name,
                comp_latex_label=field_models.get_label(sfield_3d),
                axis_labels=axis_labels,
                domain_array_by_axis=domain_array_by_axis,
                values_array_by_axis=values_array_by_axis,
            ),
        ]

    def _compute_vector_profiles(
        self,
        *,
        vfield_3d: field_models.VectorField_3D,
        uniform_domain_3d: domain_models.UniformDomain_3D,
        step_index: find_snapshots.StepIndex,
    ) -> list[CompProfile]:
        if len(self.comps_to_plot) == 0:
            raise ValueError(
                f"Vector field `{self.registered_field.name}` requires at least one component to plot; none provided.",
            )
        field_models.ensure_3d_vfield(vfield_3d)
        sim_time = vfield_3d.sim_time
        assert sim_time is not None
        comp_names = sorted(self.comps_to_plot)
        axis_labels = list(self.axes_to_slice)
        comp_profiles: list[CompProfile] = []
        for comp_name in comp_names:
            comp_latex_label = field_models.get_vcomp_label(
                vfield_3d=vfield_3d,
                comp_axis=comp_name,
            )
            domain_array_by_axis: list[numpy.ndarray] = []
            values_array_by_axis: list[numpy.ndarray] = []
            for axis_to_slice in axis_labels:
                x_positions = self._compute_cell_centers(
                    uniform_domain_3d=uniform_domain_3d,
                    axis_to_slice=axis_to_slice,
                )
                comp_index = cartesian_axes.get_axis_index(comp_name)
                comp_sarray_3d = vfield_3d.fdata.farray[comp_index]
                comp_sarray_1d = self._extract_1d_midplane_profile(
                    sarray_3d=comp_sarray_3d,
                    axis_to_slice=axis_to_slice,
                )
                domain_array_by_axis.append(x_positions)
                values_array_by_axis.append(comp_sarray_1d)
            comp_profile = CompProfile(
                sim_time=sim_time,
                step_index=step_index,
                comp_name=cartesian_axes.get_axis_label(comp_name),
                comp_latex_label=comp_latex_label,
                axis_labels=axis_labels,
                domain_array_by_axis=domain_array_by_axis,
                values_array_by_axis=values_array_by_axis,
            )
            comp_profiles.append(comp_profile)
        return comp_profiles

    def _compute_snapshot(
        self,
        *,
        snapshot_dir: pathlib.Path,
        step_index: find_snapshots.StepIndex,
    ) -> list[CompProfile]:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as quokka_snapshot:
            uniform_domain_3d = quokka_snapshot.load_3d_uniform_domain(amr_level=self.amr_level)
            field_3d = self.registered_field.load(
                quokka_snapshot=quokka_snapshot,
                amr_level=self.amr_level,
            )
        if isinstance(field_3d, field_models.ScalarField_3D):
            return self._compute_scalar_profiles(
                sfield_3d=field_3d,
                uniform_domain_3d=uniform_domain_3d,
                step_index=step_index,
            )
        elif isinstance(field_3d, field_models.VectorField_3D):
            return self._compute_vector_profiles(
                vfield_3d=field_3d,
                uniform_domain_3d=uniform_domain_3d,
                step_index=step_index,
            )
        else:
            raise ValueError(f"{self.registered_field.name} is an unrecognised field type.")

    def run(
        self,
    ) -> list[list[CompProfile]]:
        all_comp_profiles: list[list[CompProfile]] = []
        for snapshot_dir in self.snapshot_dirs:
            step_index = find_snapshots.get_step_index(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            )
            padded_step_index_string = step_index.get_padded_string(index_width=self.index_width)
            data_paths = [
                self._get_data_path(
                    axis_label=cartesian_axes.get_axis_label(axis_to_slice),
                    padded_step_index_string=padded_step_index_string,
                ) for axis_to_slice in self.axes_to_slice
            ]
            data_is_complete = all(data_path.exists() for data_path in data_paths)
            if (not self.overwrite) and data_is_complete:
                comp_profiles, _sim_time = self._load_snapshot_data(data_paths=data_paths)
            else:
                comp_profiles = self._compute_snapshot(
                    snapshot_dir=snapshot_dir,
                    step_index=step_index,
                )
                if self.save_data:
                    manage_io.create_directory(
                        directory=self.data_dir,
                        verbose=False,
                    )
                    self._save_snapshot_data(
                        comp_profiles=comp_profiles,
                        padded_step_index_string=padded_step_index_string,
                    )
            all_comp_profiles.append(comp_profiles)
        all_comp_profiles.sort(key=lambda _comp_profiles: _comp_profiles[0].sim_time)
        return all_comp_profiles


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

    def _get_figure_path(
        self,
        *,
        figures_dir: pathlib.Path,
        padded_step_index_string: str,
    ) -> pathlib.Path:
        return figures_dir / f"{self.registered_field.name}-profile-index={padded_step_index_string}.png"

    @staticmethod
    def _style_axs(
        *,
        axs_grid: manage_figure.PanelGrid,
        comp_latex_labels: list[latex_labels.LatexLabel],
        axis_labels: list[cartesian_axes.AxisLike_3D],
    ) -> None:
        num_rows = len(comp_latex_labels)
        for row_index, comp_latex_label in enumerate(comp_latex_labels):
            is_bottom_row = row_index == num_rows - 1
            for col_index, axis_label in enumerate(axis_labels):
                ax = axs_grid[row_index][col_index]
                is_left_col = col_index == 0
                if is_left_col:
                    ax.set_ylabel(comp_latex_label.label)
                if is_bottom_row:
                    ax.set_xlabel(cartesian_axes.get_axis_latex_label(axis_label).label)
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
            x_values = comp_profile.get_domain(axis_index=axis_index)
            y_values = comp_profile.get_values(axis_index=axis_index)
            ax.plot(
                x_values,
                y_values,
                color=color,
            )

    def _plot_series_row(
        self,
        *,
        axs_row: manage_figure.PanelGrid,
        comp_profiles: list[CompProfile],
    ) -> None:
        last_series_index = max(0, len(comp_profiles) - 1)
        palette = add_color.make_palette(
            config=add_color.SequentialPaletteConfig(
                palette_name=field_palettes.SEQUENTIAL_PALETTE_NAME,
                palette_range=(0.25, 1.0),
            ),
            value_range=(0, last_series_index),
        )
        for time_index, comp_profile in enumerate(comp_profiles):
            color = palette.get_color(time_index)
            self._plot_comp_profile(
                axs_row=axs_row,
                comp_profile=comp_profile,
                color=color,
            )
        add_color.add_colorbar(
            panels=axs_row[-1],
            palette=palette,
            label=r"snapshot index",
            colorbar_gap_pt=15.0,
            label_gap_pt=10.0,
        )

    def _save_snapshot_figure(
        self,
        *,
        comp_profiles: list[CompProfile],
        figure_path: pathlib.Path,
    ) -> None:
        axis_labels = comp_profiles[0].axis_labels
        comp_latex_labels = [comp_profile.comp_latex_label for comp_profile in comp_profiles]
        figure, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=len(comp_profiles),
            num_panel_cols=len(axis_labels),
            panel_col_gap_pt=30.0,
        )
        for row_index, comp_profile in enumerate(comp_profiles):
            self._plot_comp_profile(
                axs_row=axs_grid[row_index],
                comp_profile=comp_profile,
                color="black",
            )
        self._style_axs(
            axs_grid=axs_grid,
            comp_latex_labels=comp_latex_labels,
            axis_labels=axis_labels,
        )
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
            verbose=False,
        )

    def _save_summary_figure(
        self,
        *,
        comp_profiles_lookup: dict[str, list[CompProfile]],
        figures_dir: pathlib.Path,
    ) -> None:
        """Combined overlay across every snapshot processed this run; always rebuilt fresh."""
        comp_names = list(comp_profiles_lookup.keys())
        axis_labels = comp_profiles_lookup[comp_names[0]][0].axis_labels
        comp_latex_labels = [comp_profiles_lookup[comp_name][0].comp_latex_label for comp_name in comp_names]
        figure, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=len(comp_names),
            num_panel_cols=len(axis_labels),
            panel_col_gap_pt=30.0,
        )
        for row_index, comp_name in enumerate(comp_names):
            comp_profiles = comp_profiles_lookup[comp_name]
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
            comp_latex_labels=comp_latex_labels,
            axis_labels=axis_labels,
        )
        figure_path = figures_dir / f"{self.registered_field.name}-profiles-summary.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
            verbose=True,
        )

    @staticmethod
    def _group_comp_profiles_by_name(
        all_comp_profiles: list[list[CompProfile]],
    ) -> dict[str, list[CompProfile]]:
        """Regroup per-snapshot profile lists into per-component series, sorted by sim_time."""
        comp_profiles_lookup: dict[str, list[CompProfile]] = {}
        for comp_profiles in all_comp_profiles:
            for comp_profile in comp_profiles:
                if comp_profile.comp_name not in comp_profiles_lookup:
                    comp_profiles_lookup[comp_profile.comp_name] = []
                comp_profiles_lookup[comp_profile.comp_name].append(comp_profile)
        for comp_profiles in comp_profiles_lookup.values():
            comp_profiles.sort(key=lambda _comp_profile: _comp_profile.sim_time)
        return comp_profiles_lookup

    def run(
        self,
    ) -> None:
        compute_comp_profiles_pipeline = ComputeCompProfiles(
            snapshot_dirs=self.snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
            registered_field=self.registered_field,
            index_width=self.index_width,
            comps_to_plot=self.comps_to_plot,
            axes_to_slice=self.axes_to_slice,
            save_data=self.save_data,
            data_dir=self.data_dir,
            overwrite=self.overwrite,
            amr_level=self.amr_level,
        )
        all_comp_profiles = compute_comp_profiles_pipeline.run()
        if all_comp_profiles and self.save_figure:
            for comp_profiles in all_comp_profiles:
                padded_step_index_string = comp_profiles[0].step_index.get_padded_string(index_width=self.index_width)
                figure_path = self._get_figure_path(
                    figures_dir=self.figures_dir,
                    padded_step_index_string=padded_step_index_string,
                )
                if self.overwrite or not figure_path.exists():
                    self._save_snapshot_figure(
                        comp_profiles=comp_profiles,
                        figure_path=figure_path,
                    )
            comp_profiles_lookup = self._group_comp_profiles_by_name(all_comp_profiles)
            self._save_summary_figure(
                comp_profiles_lookup=comp_profiles_lookup,
                figures_dir=self.figures_dir,
            )


## } MODULE
