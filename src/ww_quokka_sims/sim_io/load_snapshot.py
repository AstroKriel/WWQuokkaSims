## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from types import TracebackType
from typing import Any

## third-party
import numpy

from yt.loaders import load as yt_load
from yt.utilities.logger import ytLogger as yt_logger

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
)
from jormi import ww_lists
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_types

## local
from ._snapshot import (
    _DeriveEnergyFields,
    _DeriveMagneticFields,
    _DeriveMHDFields,
    _DeriveVelocityFields,
    FieldKey as FieldKey,  # explicit re-export so pyright treats it as public API
    LRUCache,
    YT_SFIELD_KEYS,
    YT_VFIELD_KEYS,
)

##
## === SNAPSHOT OPERATOR CLASS
##


class QuokkaSnapshot(
        _DeriveVelocityFields,
        _DeriveEnergyFields,
        _DeriveMagneticFields,
        _DeriveMHDFields,
):
    """Interface for loading Quokka snapshots with yt."""

    snapshot_dir: Path
    verbose: bool
    _yt_dataset: Any | None
    _in_context: bool
    _sim_time: float | None
    _covering_grid_cache: dict[int, Any]
    _uniform_domain_3d_cache: dict[int, domain_models.UniformDomain_3D]
    _field_cache: LRUCache

    ##
    ## --- SNAPSHOT LIFECYCLE
    ##

    def __init__(
        self,
        *,
        snapshot_dir: str | Path,
        verbose: bool = True,
    ):
        """Initialise a snapshot handle without opening the underlying yt dataset."""
        validate_types.ensure_bool(
            param=verbose,
            param_name="verbose",
        )
        self.snapshot_dir = Path(snapshot_dir)
        self.verbose = verbose
        self._yt_dataset = None
        self._in_context = False
        self._sim_time = None
        self._covering_grid_cache = {}
        self._uniform_domain_3d_cache = {}
        ## the following fields are cached: rho_sfield_3d, mom_vfield_3d, v_vfield_3d, and b_vfield_3d
        self._field_cache = LRUCache(max_size=4)

    def __enter__(
        self,
    ):
        """Enter the context; open the yt dataset if needed; validate simulation time."""
        self._in_context = True
        self._open_if_needed()
        _ = self.sim_time  # force implicit validation
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Exit the context; close the yt dataset."""
        self._in_context = False
        self._close()

    def _open_if_needed(
        self,
    ) -> None:
        """Open the yt dataset if not already open; cache the simulation time."""
        if self._yt_dataset is None:
            if not self.verbose:
                ## reduce yt verbosity: only print warnings, errors and critical messages
                yt_logger.setLevel("WARNING")
            yt_dataset = yt_load(str(self.snapshot_dir))
            self._sim_time = float(yt_dataset.current_time)
            self._yt_dataset = yt_dataset

    def _close_if_needed(
        self,
    ) -> None:
        """Close the yt dataset unless currently inside a context manager."""
        if not self._in_context:
            self._close()

    def _close(
        self,
    ) -> None:
        """Close the yt dataset; clear cached grid objects; keep simulation time cached."""
        if self._yt_dataset is not None:
            self._yt_dataset.close()
            self._yt_dataset = None
            self._covering_grid_cache = {}
            self._uniform_domain_3d_cache = {}
            self._field_cache.clear_cache()

    @property
    def is_open(
        self,
    ) -> bool:
        """`True` iff the yt dataset is currently open."""
        return self._yt_dataset is not None

    def close(
        self,
    ) -> None:
        """Close the yt dataset; exit any active context."""
        self._in_context = False
        self._close()

    ##
    ## --- PROBE SNAPSHOT
    ##

    @property
    def sim_time(
        self,
    ) -> float:
        """Simulation time in code units."""
        if self._sim_time is None:
            self._open_if_needed()
        sim_time = self._sim_time
        if (sim_time is None) or not numpy.isfinite(sim_time):
            msg = f"invalid simulation time in {self.snapshot_dir}: {sim_time!r}."
            manage_log.log_error(text=msg)
            raise RuntimeError(msg)
        return float(sim_time)

    def _validate_amr_level(
        self,
        amr_level: int,
    ) -> None:
        """Raise if `amr_level` exceeds the finest level actually present in this snapshot."""
        assert self._yt_dataset is not None
        max_level = int(self._yt_dataset.index.max_level)
        if not (0 <= amr_level <= max_level):
            msg = (
                f"requested AMR level {amr_level} but {self.snapshot_dir} only has "
                f"levels 0-{max_level} (max_level={max_level})."
            )
            manage_log.log_error(text=msg)
            raise ValueError(msg)

    def _get_covering_grid(
        self,
        *,
        amr_level: int = 0,
    ) -> Any:
        """
        Return a covering grid spanning the whole domain at `amr_level`'s resolution.

        For `amr_level > 0`, this is the composite of the finest data available up to and
        including `amr_level` (coarser regions are filled by interpolating from the highest
        level that does cover them); `amr_level=0` (the default) always reads the base level.
        """
        self._open_if_needed()
        assert self._yt_dataset is not None
        self._validate_amr_level(amr_level)
        if amr_level not in self._covering_grid_cache:
            refinement_ratio = int(self._yt_dataset.refine_by)
            num_cells = self._yt_dataset.domain_dimensions * (refinement_ratio**amr_level)
            self._covering_grid_cache[amr_level] = self._yt_dataset.covering_grid(
                level=amr_level,
                left_edge=self._yt_dataset.domain_left_edge,
                dims=num_cells,
            )
        return self._covering_grid_cache[amr_level]

    def _get_available_field_keys(
        self,
    ) -> list[FieldKey]:
        """Return all (field-group, field-name) yt keys available in the snapshot."""
        self._open_if_needed()
        assert self._yt_dataset is not None
        field_keys = sorted(set(self._yt_dataset.field_list))
        self._close_if_needed()
        return field_keys

    def list_available_field_keys(
        self,
    ) -> list[FieldKey]:
        """List all available yt field keys in this snapshot."""
        field_keys = self._get_available_field_keys()
        manage_log.log_items(
            title="Available Fields",
            items=field_keys,
            message=f"Stored under: {self.snapshot_dir}",
            message_position="bottom",
            show_time=False,
        )
        return field_keys

    def is_field_key_available(
        self,
        *,
        field_key: FieldKey,
    ) -> bool:
        """Return `True` iff `field_key` exists in the snapshot."""
        available_keys = set(self._get_available_field_keys())
        return field_key in available_keys

    ##
    ## --- RESOLVE FIELD
    ##

    def _resolve_sfield_key(
        self,
        field_name: str,
    ) -> FieldKey:
        """Resolve the yt key associated with a named scalar field."""
        if field_name not in YT_SFIELD_KEYS:
            valid_string = ww_lists.as_quoted_string(list(YT_SFIELD_KEYS.keys()))
            msg = f"unknown scalar field `{field_name}`; valid options: {valid_string}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return YT_SFIELD_KEYS[field_name]["key"]

    def _get_sfield_key(
        self,
        field_name: str,
    ) -> FieldKey:
        """Resolve and validate the yt key associated with a scalar field."""
        field_key = self._resolve_sfield_key(field_name)
        if not self.is_field_key_available(field_key=field_key):
            msg = f"scalar field `{field_name}` ({field_key[0]}:{field_key[1]}) not found; searched in {self.snapshot_dir}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return field_key

    def _resolve_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        """Return the component yt keys associated with a named vector field."""
        if field_name not in YT_VFIELD_KEYS:
            valid_string = ww_lists.as_quoted_string(list(YT_VFIELD_KEYS.keys()))
            msg = f"unknown vector field `{field_name}`; valid options: {valid_string}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return YT_VFIELD_KEYS[field_name]["keys"]

    def _get_missing_vfield_keys(
        self,
        field_name: str,
    ) -> list[FieldKey]:
        """Return missing component yt keys for `field_name`."""
        vfield_key_lookup = self._resolve_vfield_key_lookup(field_name)
        available_keys = set(self._get_available_field_keys())
        return [comp_key for comp_key in vfield_key_lookup.values() if comp_key not in available_keys]

    def _get_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        """Resolve and validate component keys associated with a named vector field."""
        missing_keys = self._get_missing_vfield_keys(field_name)
        if missing_keys:
            missing_string = ww_lists.as_quoted_string(
                [f"{yt_group}:{yt_field}" for yt_group, yt_field in missing_keys],
            )
            msg = f"vector field `{field_name}` is incomplete in {self.snapshot_dir}; missing components: {missing_string}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return self._resolve_vfield_key_lookup(field_name)

    def _is_vfield_keys_available(
        self,
        field_name: str,
    ) -> bool:
        """Return `True` iff all components associated with a named vector field exist."""
        return len(self._get_missing_vfield_keys(field_name)) == 0

    def _load_3d_sarray(
        self,
        field_key: FieldKey,
        *,
        amr_level: int = 0,
    ) -> numpy.ndarray:
        """Load a scalar field from the covering grid as a 3D `ndarray`."""
        self._open_if_needed()
        assert self._yt_dataset is not None
        covering_grid = self._get_covering_grid(amr_level=amr_level)
        if field_key not in self._yt_dataset.field_list:
            self._close_if_needed()
            raise KeyError(f"field {field_key} not found; searched in {self.snapshot_dir}.")
        sarray_3d = numpy.asarray(covering_grid[field_key], dtype=numpy.float64)
        if sarray_3d.ndim != 3:
            self._close_if_needed()
            raise ValueError(f"expected a 3D array for {field_key}; got shape {sarray_3d.shape}.")
        self._close_if_needed()
        return numpy.ascontiguousarray(sarray_3d)

    ##
    ## --- CORE FIELD LOADERS
    ##

    def _extract_3d_sarray(
        self,
        sfield_3d: field_models.ScalarField_3D,
        *,
        param_name: str,
    ) -> numpy.ndarray:
        return field_models.extract_3d_sarray(
            sfield_3d=sfield_3d,
            param_name=param_name,
        )

    def _extract_3d_varray(
        self,
        vfield_3d: field_models.VectorField_3D,
        *,
        param_name: str,
    ) -> numpy.ndarray:
        return field_models.extract_3d_varray(
            vfield_3d=vfield_3d,
            param_name=param_name,
        )

    def load_3d_sfield(
        self,
        *,
        field_key: FieldKey,
        field_name: str,
        latex_label: str,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Wrap a scalar array as `ScalarField_3D` with a `field_name`, `latex_label`, and `sim_time`."""
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        validate_types.ensure_nonempty_string(
            param=latex_label,
            param_name="latex_label",
        )
        sarray_3d = self._load_3d_sarray(field_key, amr_level=amr_level)
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray_3d,
            uniform_domain_3d=uniform_domain_3d,
            field_name=field_name,
            latex_label=latex_label,
            sim_time=self.sim_time,
        )

    def load_3d_vfield(
        self,
        *,
        vfield_key_lookup: dict[cartesian_axes.CartesianAxis_3D, FieldKey],
        field_name: str,
        latex_label: str,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        """Load and stack 3 components into a `VectorField_3D` with a `field_name`, `latex_label`, and `sim_time`."""
        if set(vfield_key_lookup) != set(cartesian_axes.DEFAULT_3D_AXES_ORDER):
            received_axes = [axis.value for axis in sorted(vfield_key_lookup.keys(), key=lambda a: a.value)]
            expected_axes = [axis.value for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER]
            msg = f"`vfield_key_lookup` must contain all 3 components {expected_axes}; got {received_axes}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        validate_types.ensure_nonempty_string(
            param=latex_label,
            param_name="latex_label",
        )
        self._open_if_needed()
        assert self._yt_dataset is not None
        covering_grid = self._get_covering_grid(amr_level=amr_level)
        grouped_sarrays: dict[cartesian_axes.CartesianAxis_3D, numpy.ndarray] = {}
        for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER:
            comp_key = vfield_key_lookup[comp_axis]
            if comp_key not in self._yt_dataset.field_list:
                self._close_if_needed()
                raise KeyError(f"field {comp_key} not found; searched in {self.snapshot_dir}.")
            comp_sarray = numpy.asarray(covering_grid[comp_key], dtype=numpy.float64)
            if comp_sarray.ndim != 3:
                self._close_if_needed()
                raise ValueError(f"expected a 3D array for {comp_key}; got shape {comp_sarray.shape}.")
            grouped_sarrays[comp_axis] = comp_sarray
        self._close_if_needed()
        sim_time = self.sim_time
        varray_3d = numpy.stack(
            [grouped_sarrays[comp_axis] for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER],
            axis=0,
        )
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=varray_3d,
            uniform_domain_3d=uniform_domain_3d,
            sim_time=sim_time,
            field_name=field_name,
            latex_label=latex_label,
        )

    ##
    ## --- DOMAIN
    ##

    def load_3d_uniform_domain(
        self,
        *,
        force_periodicity: bool = True,
        amr_level: int = 0,
    ) -> domain_models.UniformDomain_3D:
        """
        Return uniform domain metadata: bounds, resolution, and periodicity; result is cached per `amr_level`.

        `resolution` is the base-level `domain_dimensions` scaled by `refinement_ratio**amr_level`, matching the
        resolution `_get_covering_grid(amr_level=...)` actually returns, so the two stay consistent.
        `force_periodicity` only takes effect on the first call for a given `amr_level`; yt cannot read
        periodicity reliably.
        """
        validate_types.ensure_bool(
            param=force_periodicity,
            param_name="force_periodicity",
        )
        self._open_if_needed()
        assert self._yt_dataset is not None
        self._validate_amr_level(amr_level)
        if amr_level in self._uniform_domain_3d_cache:
            self._close_if_needed()
            return self._uniform_domain_3d_cache[amr_level]
        x_min, y_min, z_min = (float(value) for value in self._yt_dataset.domain_left_edge)
        x_max, y_max, z_max = (float(value) for value in self._yt_dataset.domain_right_edge)
        refinement_ratio = int(self._yt_dataset.refine_by)
        num_cells_x, num_cells_y, num_cells_z = (
            int(num_cells) * (refinement_ratio**amr_level) for num_cells in self._yt_dataset.domain_dimensions
        )
        is_periodic_x, is_periodic_y, is_periodic_z = (
            (bool(is_periodic) or force_periodicity) for is_periodic in self._yt_dataset.periodicity
        )
        self._close_if_needed()
        uniform_domain_3d = domain_models.UniformDomain_3D(
            periodicity=(is_periodic_x, is_periodic_y, is_periodic_z),
            resolution=(num_cells_x, num_cells_y, num_cells_z),
            domain_bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        )
        self._uniform_domain_3d_cache[amr_level] = uniform_domain_3d
        return uniform_domain_3d

    ##
    ## --- BASIC FIELDS
    ##

    def load_3d_density_sfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Load gas density: `rho`."""
        cache_key = f"density:{amr_level}"
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.ScalarField_3D):
            return cached_field
        rho_key = self._get_sfield_key("density")
        rho_sfield_3d = self.load_3d_sfield(
            field_key=rho_key,
            field_name="density",
            latex_label=r"\rho",
            amr_level=amr_level,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=rho_sfield_3d,
        )
        return rho_sfield_3d

    def load_3d_momentum_vfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        """Load momentum field: `vec(m) = rho vec(v)`."""
        cache_key = f"momentum:{amr_level}"
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.VectorField_3D):
            return cached_field
        mom_key_lookup = self._get_vfield_key_lookup("momentum")
        mom_vfield_3d = self.load_3d_vfield(
            vfield_key_lookup=mom_key_lookup,
            field_name="momentum",
            latex_label=r"\rho \,\vec{v}",
            amr_level=amr_level,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=mom_vfield_3d,
        )
        return mom_vfield_3d

    def load_3d_magnetic_vfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        """Load magnetic field: `vec(b)`."""
        cache_key = f"magnetic:{amr_level}"
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.VectorField_3D):
            return cached_field
        b_key_lookup = self._get_vfield_key_lookup("magnetic")
        b_vfield_3d = self.load_3d_vfield(
            vfield_key_lookup=b_key_lookup,
            field_name="magnetic",
            latex_label=r"\vec{b}",
            amr_level=amr_level,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=b_vfield_3d,
        )
        return b_vfield_3d

    def load_3d_total_energy_sfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Load total energy: `e_tot = e_int + e_kin + e_mag` (code units)."""
        cache_key = f"total_energy:{amr_level}"
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.ScalarField_3D):
            return cached_field
        E_tot_key = self._get_sfield_key("total_energy")
        E_tot_sfield_3d = self.load_3d_sfield(
            field_key=E_tot_key,
            field_name="total_energy",
            latex_label=r"E_\mathrm{tot}",
            amr_level=amr_level,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=E_tot_sfield_3d,
        )
        return E_tot_sfield_3d

    def load_3d_magnetic_divergence_sfield(
        self,
    ) -> field_models.ScalarField_3D:
        """
        Load magnetic field divergence: div(b).

        Quokka's native value, computed on its div-preserving staggered mesh, is used when available.
        Otherwise, a fallback estimate using a different stencil is calculated. The native value
        requires `derived_vars = "magnetic_divergence"` in the param TOML file.
        """
        cached_field = self._field_cache.get_cached_field("magnetic_divergence")
        if isinstance(cached_field, field_models.ScalarField_3D):
            return cached_field
        div_b_key = self._resolve_sfield_key("magnetic_divergence")
        if self.is_field_key_available(field_key=div_b_key):
            div_b_sfield_3d = self.load_3d_sfield(
                field_key=div_b_key,
                field_name="magnetic_divergence",
                latex_label=r"\nabla\cdot\vec{b}",
            )
        else:
            manage_log.log_warning(
                text=(
                    f"native `magnetic_divergence` field was not found in {self.snapshot_dir}; falling back "
                    "to an estimated field instead. Set derived_vars = \"magnetic_divergence\" in the "
                    "param TOML file to get the more accurate, solver-native value instead."
                ),
            )
            div_b_sfield_3d = self.compute_div_b_sfield()
        self._field_cache.cache_field(
            cache_key="magnetic_divergence",
            field_data=div_b_sfield_3d,
        )
        return div_b_sfield_3d


## } MODULE
