## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

from yt.loaders import load as yt_load
from yt.utilities.logger import ytLogger as yt_logger

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    _farray_operators,
    compute_fields,
    decompose_fields,
    domain_types,
    field_operators,
    field_types,
)
from jormi.ww_io import manage_log
from jormi.ww_types import check_types

##
## === DATA STRUCTURES
##

FieldKey = tuple[str, str]

## boxlib uses "x-", "y-", "z-" prefixes for vector component field names
_BOXLIB_XYZ_LABELS: dict[cartesian_axes.CartesianAxis_3D, str] = {
    cartesian_axes.CartesianAxis_3D.X0: "x",
    cartesian_axes.CartesianAxis_3D.X1: "y",
    cartesian_axes.CartesianAxis_3D.X2: "z",
}


@dataclass(frozen=True)
class HelmholtzKineticEnergy:
    Ekin_div_sfield_3d: field_types.ScalarField_3D
    Ekin_sol_sfield_3d: field_types.ScalarField_3D
    Ekin_bulk_sfield_3d: field_types.ScalarField_3D

    def __post_init__(self) -> None:
        field_types.ensure_3d_sfield(
            sfield_3d=self.Ekin_div_sfield_3d,
            param_name="<Ekin_div_sfield_3d>",
        )
        field_types.ensure_3d_sfield(
            sfield_3d=self.Ekin_sol_sfield_3d,
            param_name="<Ekin_sol_sfield_3d>",
        )
        field_types.ensure_3d_sfield(
            sfield_3d=self.Ekin_bulk_sfield_3d,
            param_name="<Ekin_bulk_sfield_3d>",
        )


##
## === YT FIELD MAPPINGS
##


def create_boxlib_vkeys(
    field_name: str,
) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
    """
    Create a mapping from CartesianAxis_3D -> yt field key for a given base field name.

    For example, for field_name="GasMomentum", this yields:
    {
      X0: ("boxlib", "x-GasMomentum"),
      X1: ("boxlib", "y-GasMomentum"),
      X2: ("boxlib", "z-GasMomentum"),
    }
    """
    return {
        axis: ("boxlib", f"{_BOXLIB_XYZ_LABELS[axis]}-{field_name}")
        for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER
    }


YT_VFIELD_KEYS: dict[str, dict] = {
    "momentum": {
        "keys": create_boxlib_vkeys("GasMomentum"),
        "description": "Momentum density components: vec(m) = rho * vec(v)",
    },
    "magnetic": {
        "keys": create_boxlib_vkeys("BField"),
        "description": "Magnetic field components (code units)",
    },
}

YT_SFIELD_KEYS: dict[str, dict] = {
    "density": {
        "key": ("boxlib", "gasDensity"),
        "description": "Gas density field",
    },
    "total_energy": {
        "key": ("boxlib", "gasEnergy"),
        "description": "Total energy density: E_tot = E_int + E_kin + E_mag",
    },
}

##
## === CACHE OPERATOR CLASS
##


class LRUCache:
    """Simple least-recently-used (LRU) cache for storing field objects."""

    def __init__(
        self,
        max_size: int = 3,
    ) -> None:
        check_types.ensure_finite_int(
            param=max_size,
            param_name="max_size",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        self._cache_lookup = OrderedDict()
        self._max_size = int(max_size)

    def get_cached_field(
        self,
        field_name: str,
    ):
        """Return cached value for `field_name`, or None if not found."""
        cached_field = self._cache_lookup.get(field_name)
        if cached_field is not None:
            self._cache_lookup.move_to_end(field_name)
        return cached_field

    def cache_field(
        self,
        field_name: str,
        field_data: field_types.ScalarField_3D | field_types.VectorField_3D,
    ) -> None:
        """Store `field_data` under `field_name` and evict the least-recently-used item if the cache is full."""
        self._cache_lookup[field_name] = field_data
        self._cache_lookup.move_to_end(field_name)
        while len(self._cache_lookup) > self._max_size:
            self._cache_lookup.popitem(last=False)

    def clear_cache(
        self,
    ) -> None:
        """Clear all cached fields."""
        self._cache_lookup.clear()


##
## === DATASET OPERATOR CLASS
##


class QuokkaDataset:
    """Interface for loading Quokka datasets with yt."""

    ##
    ## --- DATASET LIFECYCLE
    ##

    def __init__(
        self,
        dataset_dir: str | Path,
        verbose: bool = True,
    ):
        """Initialise a dataset handle without opening the underlying yt dataset."""
        check_types.ensure_bool(
            param=verbose,
            param_name="verbose",
        )
        self.dataset_dir = Path(dataset_dir)
        self.verbose = verbose
        self.dataset = None
        self._in_context = False
        self._sim_time = None
        self._covering_grid = None
        self._udomain_3d = None
        ## we will cache the following fields: rho_sfield_3d, mom_vfield_3d, v_vfield_3d, and b_vfield_3d
        self._field_cache = LRUCache(max_size=4)

    def __enter__(
        self,
    ):
        """Enter a context, open the dataset if needed, and validate the simulation time."""
        self._in_context = True
        self._open_dataset_if_needed()
        _ = self.sim_time  # force implicit validation
        return self

    def __exit__(
        self,
        _exc_type,
        _exc_value,
        _traceback,
    ):
        """Exit a context and close the dataset handle."""
        self._in_context = False
        self._close_dataset()

    def _open_dataset_if_needed(
        self,
    ) -> None:
        """Open the yt dataset if it is not already open and cache the simulation time."""
        if self.dataset is None:
            if not self.verbose:
                ## reduce yt verbosity: only print warnings, errors and critical messages
                yt_logger.setLevel("WARNING")
            self.dataset = yt_load(str(self.dataset_dir))
            self._sim_time = float(self.dataset.current_time)

    def _close_dataset_if_needed(
        self,
    ) -> None:
        """Close the yt dataset unless currently inside a context manager."""
        if not self._in_context:
            self._close_dataset()

    def _close_dataset(
        self,
    ) -> None:
        """Close the yt dataset and clear cached grid objects. Keep the simulation time cached."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
            self._covering_grid = None
            self._udomain_3d = None
            self._field_cache.clear_cache()

    @property
    def is_open(
        self,
    ) -> bool:
        """`True` iff the yt dataset handle is currently open."""
        return self.dataset is not None

    def close(
        self,
    ) -> None:
        self._in_context = False
        self._close_dataset()

##
## --- PROBE DATASET
##

    @property
    def sim_time(
        self,
    ) -> float:
        """Simulation time in code units."""
        if self._sim_time is None:
            self._open_dataset_if_needed()
        sim_time = self._sim_time
        if (sim_time is None) or not numpy.isfinite(sim_time):
            msg = f"Invalid simulation time in dataset: {self.dataset_dir} (sim_time = {sim_time!r})"
            manage_log.log_error(msg)
            raise RuntimeError(msg)
        return float(sim_time)

    def _get_covering_grid(
        self,
    ):
        """Return the coarsest (level 0) covering grid spanning the domain."""
        self._open_dataset_if_needed()
        assert self.dataset is not None
        if self._covering_grid is None:
            self._covering_grid = self.dataset.covering_grid(
                level=0,
                left_edge=self.dataset.domain_left_edge,
                dims=self.dataset.domain_dimensions,
            )
        return self._covering_grid

    def _get_available_field_keys(
        self,
    ) -> list[FieldKey]:
        """Return all (field-group, field-name) yt keys available in the dataset."""
        self._open_dataset_if_needed()
        assert self.dataset is not None
        field_keys = sorted(set(self.dataset.field_list))
        self._close_dataset_if_needed()
        return field_keys

    def list_available_field_keys(
        self,
    ) -> list[FieldKey]:
        """List all available yt field keys in this dataset."""
        field_keys = self._get_available_field_keys()
        manage_log.log_items(
            title="Available Fields",
            items=field_keys,
            message=f"Stored under: {self.dataset_dir}",
            message_position="bottom",
            show_time=False,
        )
        return field_keys

    def is_field_key_available(
        self,
        field_key: FieldKey,
    ) -> bool:
        """Return `True` iff a particular yt field key exists in the dataset."""
        available_keys = set(self._get_available_field_keys())
        return field_key in available_keys

##
## --- RESOLVE FIELD
##

    def _resolve_sfield_key(
        self,
        field_name: str,
    ) -> FieldKey:
        """Resolve the yt key for a named scalar field."""
        if field_name not in YT_SFIELD_KEYS:
            valid_string = ", ".join(YT_SFIELD_KEYS.keys())
            msg = f"Unknown scalar field `{field_name}`. Valid options: {valid_string}"
            manage_log.log_error(msg)
            raise KeyError(msg)
        return YT_SFIELD_KEYS[field_name]["key"]

    def _get_sfield_key(
        self,
        field_name: str,
    ) -> FieldKey:
        """Resolve and validate the yt key for a scalar field."""
        field_key = self._resolve_sfield_key(field_name)
        if not self.is_field_key_available(field_key):
            msg = f"Scalar field `{field_name}` ({field_key[0]}:{field_key[1]}) is not available in: {self.dataset_dir}."
            manage_log.log_error(msg)
            raise KeyError(msg)
        return field_key

    def _resolve_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        """Return the component yt keys for a named vector field."""
        if field_name not in YT_VFIELD_KEYS:
            valid_string = ", ".join(YT_VFIELD_KEYS.keys())
            msg = f"Unknown vector field `{field_name}`. Valid options: {valid_string}"
            manage_log.log_error(msg)
            raise KeyError(msg)
        return YT_VFIELD_KEYS[field_name]["keys"]

    def _get_missing_vfield_keys(
        self,
        field_name: str,
    ) -> list[FieldKey]:
        """Return the list of missing component keys for a named vector field."""
        vfield_key_lookup = self._resolve_vfield_key_lookup(field_name)
        available_keys = set(self._get_available_field_keys())
        return [comp_key for comp_key in vfield_key_lookup.values() if comp_key not in available_keys]

    def _get_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        """Resolve and validate component keys for a named vector field."""
        missing_keys = self._get_missing_vfield_keys(field_name)
        if missing_keys:
            missing_string = ", ".join(f"{yt_group}:{yt_field}" for yt_group, yt_field in missing_keys)
            msg = (
                f"Vector field `{field_name}` is incomplete in {self.dataset_dir}. "
                f"Missing components: {missing_string}"
            )
            manage_log.log_error(msg)
            raise KeyError(msg)
        return self._resolve_vfield_key_lookup(field_name)

    def _is_vfield_keys_available(
        self,
        field_name: str,
    ) -> bool:
        """Return `True` iff all components for the named vector field exist."""
        return len(self._get_missing_vfield_keys(field_name)) == 0

    def _load_3d_sarray(
        self,
        field_key: FieldKey,
    ) -> numpy.ndarray:
        """Load a scalar field from the covering grid as a 3D `ndarray`."""
        self._open_dataset_if_needed()
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        if field_key not in self.dataset.field_list:
            self._close_dataset_if_needed()
            raise KeyError(f"Field {field_key} not found in {self.dataset_dir}")
        sarray_3d = numpy.asarray(covering_grid[field_key], dtype=numpy.float64)
        if sarray_3d.ndim != 3:
            self._close_dataset_if_needed()
            raise ValueError(f"Expected a 3D field for {field_key}; received: {sarray_3d.shape}")
        self._close_dataset_if_needed()
        return numpy.ascontiguousarray(sarray_3d)

##
## --- CORE FIELD LOADERS
##

    def _extract_3d_sarray(
        self,
        sfield_3d: field_types.ScalarField_3D,
        *,
        param_name: str,
    ) -> numpy.ndarray:
        return field_types.extract_3d_sarray(
            sfield_3d=sfield_3d,
            param_name=param_name,
        )

    def _extract_3d_varray(
        self,
        vfield_3d: field_types.VectorField_3D,
        *,
        param_name: str,
    ) -> numpy.ndarray:
        return field_types.extract_3d_varray(
            vfield_3d=vfield_3d,
            param_name=param_name,
        )

    def load_3d_sfield(
        self,
        field_key: FieldKey,
        field_label: str,
    ) -> field_types.ScalarField_3D:
        """Wrap a scalar array as `ScalarField` with a `field_label` and `sim_time`."""
        check_types.ensure_nonempty_string(
            param=field_label,
            param_name="field_label",
        )
        sarray_3d = self._load_3d_sarray(field_key)
        udomain_3d = self.load_3d_uniform_domain()
        return field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray_3d,
            udomain_3d=udomain_3d,
            field_label=field_label,
            sim_time=self.sim_time,
        )

    def load_3d_vfield(
        self,
        vfield_key_lookup: dict[cartesian_axes.CartesianAxis_3D, FieldKey],
        field_label: str,
    ) -> field_types.VectorField_3D:
        """Load and stack 3 components into a `VectorField_3D` with a `field_label` and `sim_time`."""
        if set(vfield_key_lookup) != set(cartesian_axes.DEFAULT_3D_AXES_ORDER):
            received_axes = [axis.value for axis in sorted(vfield_key_lookup.keys(), key=lambda a: a.value)]
            expected_axes = [axis.value for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER]
            msg = (
                "`vfield_key_lookup` must contain all 3 components "
                f"{expected_axes}; only received: {received_axes}"
            )
            manage_log.log_error(msg)
            raise KeyError(msg)
        check_types.ensure_nonempty_string(
            param=field_label,
            param_name="field_label",
        )
        self._open_dataset_if_needed()
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        grouped_sarrays: dict[cartesian_axes.CartesianAxis_3D, numpy.ndarray] = {}
        for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER:
            comp_key = vfield_key_lookup[comp_axis]
            if comp_key not in self.dataset.field_list:
                self._close_dataset_if_needed()
                raise KeyError(f"Field {comp_key} was not found in {self.dataset_dir}")
            comp_sarray = numpy.asarray(covering_grid[comp_key], dtype=numpy.float64)
            if comp_sarray.ndim != 3:
                self._close_dataset_if_needed()
                raise ValueError(f"Expected a 3D field for {comp_key}; received: {comp_sarray.shape}")
            grouped_sarrays[comp_axis] = comp_sarray
        self._close_dataset_if_needed()
        sim_time = self.sim_time
        varray_3d = numpy.stack(
            [grouped_sarrays[comp_axis] for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER],
            axis=0,
        )
        udomain_3d = self.load_3d_uniform_domain()
        return field_types.VectorField_3D.from_3d_varray(
            varray_3d=varray_3d,
            udomain_3d=udomain_3d,
            sim_time=sim_time,
            field_label=field_label,
        )

##
## --- DOMAIN
##

    def load_3d_uniform_domain(
        self,
        force_periodicity: bool = True,
    ) -> domain_types.UniformDomain_3D:
        """
        Return uniform domain metadata (bounds, resolution, periodicity).

        Note: force_periodicity only affects the first call; subsequent calls returns the cached domain.
        This is required because yt cannot read this property reliably yet
        """
        check_types.ensure_bool(
            param=force_periodicity,
            param_name="force_periodicity",
        )
        if self._udomain_3d is not None:
            return self._udomain_3d
        self._open_dataset_if_needed()
        assert self.dataset is not None
        x_min, y_min, z_min = (float(value) for value in self.dataset.domain_left_edge)
        x_max, y_max, z_max = (float(value) for value in self.dataset.domain_right_edge)
        num_cells_x, num_cells_y, num_cells_z = (
            int(num_cells) for num_cells in self.dataset.domain_dimensions
        )
        is_periodic_x, is_periodic_y, is_periodic_z = (
            (bool(is_periodic) or force_periodicity) for is_periodic in self.dataset.periodicity
        )
        self._close_dataset_if_needed()
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(is_periodic_x, is_periodic_y, is_periodic_z),
            resolution=(num_cells_x, num_cells_y, num_cells_z),
            domain_bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        )
        self._udomain_3d = udomain_3d
        return udomain_3d

##
## --- BASIC FIELDS
##

    def load_3d_density_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Load gas density: `rho`."""
        cached_field = self._field_cache.get_cached_field("density")
        if cached_field is not None:
            return cached_field
        rho_key = self._get_sfield_key("density")
        rho_sfield_3d = self.load_3d_sfield(
            field_key=rho_key,
            field_label=r"\rho",
        )
        self._field_cache.cache_field(
            field_name="density",
            field_data=rho_sfield_3d,
        )
        return rho_sfield_3d

    def load_3d_momentum_vfield(
        self,
    ) -> field_types.VectorField_3D:
        """Load momentum field: `vec(m) = rho vec(v)`."""
        cached_field = self._field_cache.get_cached_field("momentum")
        if cached_field is not None:
            return cached_field
        mom_key_lookup = self._get_vfield_key_lookup("momentum")
        mom_vfield_3d = self.load_3d_vfield(
            vfield_key_lookup=mom_key_lookup,
            field_label=r"\rho \,\vec{v}",
        )
        self._field_cache.cache_field(
            field_name="momentum",
            field_data=mom_vfield_3d,
        )
        return mom_vfield_3d

    def load_3d_velocity_vfield(
        self,
    ) -> field_types.VectorField_3D:
        """Load velocity field: `vec(v) = vec(m) / rho`."""
        cached_field = self._field_cache.get_cached_field("velocity")
        if cached_field is not None:
            return cached_field
        rho_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_density_sfield(),
            param_name="<rho_sfield_3d>",
        )
        mom_varray_3d = self._extract_3d_varray(
            vfield_3d=self.load_3d_momentum_vfield(),
            param_name="<mom_vfield_3d>",
        )
        rho_has_zeros = compute_array_stats.check_no_zero_values(
            array=rho_sarray_3d,
            param_name="<rho_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            v_varray = mom_varray_3d / rho_sarray_3d[numpy.newaxis, ...]
        if not rho_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=v_varray,
                param_name="<v_vfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=v_varray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        udomain_3d = self.load_3d_uniform_domain()
        v_vfield_3d = field_types.VectorField_3D.from_3d_varray(
            varray_3d=v_varray,
            udomain_3d=udomain_3d,
            field_label=r"\vec{v}",
            sim_time=self.sim_time,
        )
        self._field_cache.cache_field(
            field_name="velocity",
            field_data=v_vfield_3d,
        )
        return v_vfield_3d

    def load_3d_velocity_magnitude_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute velocity magnitude: `|vec(v)|`."""
        v_vfield_3d = self.load_3d_velocity_vfield()
        return field_operators.compute_vfield_magnitude(
            vfield_3d=v_vfield_3d,
            field_label=r"|\vec{v}|",
        )

    def load_3d_magnetic_vfield(
        self,
    ) -> field_types.VectorField_3D:
        """Load magnetic field: `vec(b)`."""
        cached_field = self._field_cache.get_cached_field("magnetic")
        if cached_field is not None:
            return cached_field
        b_key_lookup = self._get_vfield_key_lookup("magnetic")
        b_vfield_3d = self.load_3d_vfield(
            vfield_key_lookup=b_key_lookup,
            field_label=r"\vec{b}",
        )
        self._field_cache.cache_field(
            field_name="magnetic",
            field_data=b_vfield_3d,
        )
        return b_vfield_3d

    def load_3d_total_energy_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Load total energy: `E_tot = E_int + E_kin + E_mag` (code units)."""
        Etot_key = self._get_sfield_key("total_energy")
        return self.load_3d_sfield(
            field_key=Etot_key,
            field_label=r"E_\mathrm{tot}",
        )

    def load_3d_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute kinetic energy density: `E_kin = 0.5 * rho * |v|^2`."""
        rho_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_density_sfield(),
            param_name="<rho_sfield_3d>",
        )
        mom_varray_3d = self._extract_3d_varray(
            vfield_3d=self.load_3d_momentum_vfield(),
            param_name="<mom_vfield_3d>",
        )
        rho_has_zeros = compute_array_stats.check_no_zero_values(
            array=rho_sarray_3d,
            param_name="<rho_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            Ekin_sarray_3d = 0.5 * _farray_operators.sum_of_varray_comps_squared(
                mom_varray_3d,
            ) / rho_sarray_3d
        if not rho_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=Ekin_sarray_3d,
                param_name="<Ekin_sfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=Ekin_sarray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        udomain_3d = self.load_3d_uniform_domain()
        return field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=Ekin_sarray_3d,
            udomain_3d=udomain_3d,
            field_label=r"E_\mathrm{kin}",
            sim_time=self.sim_time,
        )

    def load_3d_magnetic_energy_sfield(
        self,
        energy_prefactor: float = 0.5,
        field_label=r"E_\mathrm{mag}",
    ) -> field_types.ScalarField_3D:
        """Compute magnetic energy density: `E_mag = alpha * |b|^2` with `alpha=0.5` by default."""
        check_types.ensure_finite_float(
            param=energy_prefactor,
            param_name="energy_prefactor",
            allow_none=False,
        )
        b_vfield_3d = self.load_3d_magnetic_vfield()
        return compute_fields.compute_magnetic_energy_density_sfield(
            vfield_3d_b=b_vfield_3d,
            energy_prefactor=energy_prefactor,
            field_label=field_label,
        )

    def load_3d_internal_energy_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute internal energy: `E_int = E_tot - E_kin - E_mag` (`E_mag = 0` if `vec(b)` is not available)."""
        Etot_sarray = self._extract_3d_sarray(
            sfield_3d=self.load_3d_total_energy_sfield(),
            param_name="<Etot_sfield_3d>",
        )
        Ekin_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_kinetic_energy_sfield(),
            param_name="<Ekin_sfield_3d>",
        )
        Eint_sarray = Etot_sarray - Ekin_sarray_3d
        if self._is_vfield_keys_available("magnetic"):
            Emag_sarray = self._extract_3d_sarray(
                sfield_3d=self.load_3d_magnetic_energy_sfield(),
                param_name="<Emag_sfield_3d>",
            )
            Eint_sarray -= Emag_sarray
        compute_array_stats.check_no_nonfinite_values(
            array=Eint_sarray,
            param_name="<Eint_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=Eint_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=Eint_sarray,
            udomain_3d=self.load_3d_uniform_domain(),
            field_label=r"E_\mathrm{int}",
            sim_time=self.sim_time,
        )

    def load_3d_pressure_sfield(
        self,
        gamma: float = 5.0 / 3.0,
    ) -> field_types.ScalarField_3D:
        """Compute thermal pressure: `p = (gamma - 1) * E_int`."""
        check_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        Eint_sarray = self._extract_3d_sarray(
            sfield_3d=self.load_3d_internal_energy_sfield(),
            param_name="<Eint_sfield_3d>",
        )
        p_sarray = (gamma - 1.0) * Eint_sarray
        return field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=p_sarray,
            udomain_3d=self.load_3d_uniform_domain(),
            field_label=r"p",
            sim_time=self.sim_time,
        )

##
## --- VELOCITY FIELDS
##

    def load_3d_divu_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField_3D:
        """Compute divergence of velocity: `nabla cdot vec(v)` using a `grad_order` accurate stencil."""
        v_vfield_3d = self.load_3d_velocity_vfield()
        return field_operators.compute_vfield_divergence(
            vfield_3d=v_vfield_3d,
            field_label=r"\nabla\cdot\vec{v}",
            grad_order=grad_order,
        )

    def load_3d_vorticity_vfield(
        self,
        grad_order: int = 2,
    ) -> field_types.VectorField_3D:
        """Compute vorticity vector: `curl(vec(v))` using a `grad_order` accurate stencil."""
        v_vfield_3d = self.load_3d_velocity_vfield()
        return field_operators.compute_vfield_curl(
            vfield_3d=v_vfield_3d,
            grad_order=grad_order,
            field_label=r"\nabla\times\vec{v}",
        )

    def load_3d_vorticity_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField_3D:
        """Compute vorticity magnitude: `|curl(vec(v))|`."""
        omega_vfield_3d = self.load_3d_vorticity_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=omega_vfield_3d,
            field_label=r"|\nabla\times\vec{v}|",
        )

    def load_3d_kinetic_helicity_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField_3D:
        """Compute kinetic helicity density: `curl(vec(v)) dot vec(v)`."""
        omega_vfield_3d = self.load_3d_vorticity_vfield(grad_order=grad_order)
        v_vfield_3d = self.load_3d_velocity_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_3d_a=omega_vfield_3d,
            vfield_3d_b=v_vfield_3d,
            field_label=r"(\nabla\times\vec{v})\cdot\vec{v}",
        )

    def load_3d_helmholtz_kinetic_energy(
        self,
    ) -> HelmholtzKineticEnergy:
        """Compute Helmholtz-decomposed kinetic energies from `vec(v) = vec(v)_div + vec(v)_sol + vec(v)_bulk`."""
        udomain_3d = self.load_3d_uniform_domain()
        v_vfield_3d = self.load_3d_velocity_vfield()
        rho_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_density_sfield(),
            param_name="<rho_sfield_3d>",
        )
        helmholtz_vfields = decompose_fields.compute_helmholtz_decomposed_fields(vfield_3d_q=v_vfield_3d)
        v_div_varray = self._extract_3d_varray(
            vfield_3d=helmholtz_vfields.vfield_3d_div,
            param_name="<vfield_3d_div>",
        )
        v_sol_varray = self._extract_3d_varray(
            vfield_3d=helmholtz_vfields.vfield_3d_sol,
            param_name="<vfield_3d_sol>",
        )
        v_bulk_varray = self._extract_3d_varray(
            vfield_3d=helmholtz_vfields.vfield_3d_bulk,
            param_name="<vfield_3d_bulk>",
        )
        Ekin_div_sarray = 0.5 * rho_sarray_3d * _farray_operators.sum_of_varray_comps_squared(v_div_varray)
        Ekin_sol_sarray = 0.5 * rho_sarray_3d * _farray_operators.sum_of_varray_comps_squared(v_sol_varray)
        Ekin_bulk_sarray = 0.5 * rho_sarray_3d * _farray_operators.sum_of_varray_comps_squared(v_bulk_varray)
        compute_array_stats.check_no_nonfinite_values(
            array=Ekin_div_sarray,
            param_name="<Ekin_div_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=Ekin_div_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        compute_array_stats.check_no_nonfinite_values(
            array=Ekin_sol_sarray,
            param_name="<Ekin_sol_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=Ekin_sol_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        compute_array_stats.check_no_nonfinite_values(
            array=Ekin_bulk_sarray,
            param_name="<Ekin_bulk_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=Ekin_bulk_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        Ekin_div_sfield_3d = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=Ekin_div_sarray,
            udomain_3d=udomain_3d,
            field_label=r"E_{\mathrm{kin}, \parallel}",
            sim_time=self.sim_time,
        )
        Ekin_sol_sfield_3d = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=Ekin_sol_sarray,
            udomain_3d=udomain_3d,
            field_label=r"E_{\mathrm{kin}, \perp}",
            sim_time=self.sim_time,
        )
        Ekin_bulk_sfield_3d = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=Ekin_bulk_sarray,
            udomain_3d=udomain_3d,
            field_label=r"E_{\mathrm{kin}, \mathrm{bulk}}",
            sim_time=self.sim_time,
        )
        return HelmholtzKineticEnergy(
            Ekin_div_sfield_3d=Ekin_div_sfield_3d,
            Ekin_sol_sfield_3d=Ekin_sol_sfield_3d,
            Ekin_bulk_sfield_3d=Ekin_bulk_sfield_3d,
        )

    def load_3d_div_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute kinetic energy in irrotational (curl-free) velocity modes: `E_kin,div = 0.5 rho (v_div)^2`."""
        helmholtz_Ekin = self.load_3d_helmholtz_kinetic_energy()
        return helmholtz_Ekin.Ekin_div_sfield_3d

    def load_3d_sol_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute kinetic energy in solenoidal (divergence-free) velocity modes: `E_kin,sol = 0.5 rho (v_sol)^2`."""
        helmholtz_Ekin = self.load_3d_helmholtz_kinetic_energy()
        return helmholtz_Ekin.Ekin_sol_sfield_3d

    def load_3d_bulk_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute kinetic energy in bulk velocity: `E_kin,bulk = 0.5 rho (v_bulk)^2`."""
        helmholtz_Ekin = self.load_3d_helmholtz_kinetic_energy()
        return helmholtz_Ekin.Ekin_bulk_sfield_3d

##
## --- MAGNETIC FIELDS
##

    def load_3d_plasma_beta_sfield(
        self,
        gamma: float = 5.0 / 3.0,
    ) -> field_types.ScalarField_3D:
        """Compute plasma beta: `beta = 2 p / |vec(b)|^2`."""
        check_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        p_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_pressure_sfield(gamma=gamma),
            param_name="<p_sfield_3d>",
        )
        b_varray_3d = self._extract_3d_varray(
            vfield_3d=self.load_3d_magnetic_vfield(),
            param_name="<b_vfield_3d>",
        )
        b_sq_sarray_3d = _farray_operators.sum_of_varray_comps_squared(b_varray_3d)
        b_sq_has_zeros = compute_array_stats.check_no_zero_values(
            array=b_sq_sarray_3d,
            param_name="<|b|^2>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            beta_sarray_3d = 2.0 * p_sarray_3d / b_sq_sarray_3d
        if not b_sq_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=beta_sarray_3d,
                param_name="<beta_sfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=beta_sarray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=beta_sarray_3d,
            udomain_3d=self.load_3d_uniform_domain(),
            field_label=r"\beta",
            sim_time=self.sim_time,
        )

    def load_3d_alfven_speed_vfield(
        self,
    ) -> field_types.VectorField_3D:
        """Compute Alfven speed: `vec(v_A) = vec(b) / sqrt(rho)`."""
        b_varray_3d = self._extract_3d_varray(
            vfield_3d=self.load_3d_magnetic_vfield(),
            param_name="<b_vfield_3d>",
        )
        rho_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_density_sfield(),
            param_name="<rho_sfield_3d>",
        )
        rho_has_zeros = compute_array_stats.check_no_zero_values(
            array=rho_sarray_3d,
            param_name="<rho_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            va_varray_3d = b_varray_3d / numpy.sqrt(rho_sarray_3d)[numpy.newaxis, ...]
        if not rho_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=va_varray_3d,
                param_name="<va_vfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=va_varray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_types.VectorField_3D.from_3d_varray(
            varray_3d=va_varray_3d,
            udomain_3d=self.load_3d_uniform_domain(),
            field_label=r"\vec{v}_A",
            sim_time=self.sim_time,
        )

    def load_3d_alfven_speed_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute Alfven speed magnitude: `|vec(v_A)|`."""
        va_vfield_3d = self.load_3d_alfven_speed_vfield()
        return field_operators.compute_vfield_magnitude(
            vfield_3d=va_vfield_3d,
            field_label=r"|\vec{v}_A|",
        )

    def load_3d_divb_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField_3D:
        """Compute magnetic divergence: `div[vec(b)]`."""
        b_vfield_3d = self.load_3d_magnetic_vfield()
        return field_operators.compute_vfield_divergence(
            vfield_3d=b_vfield_3d,
            field_label=r"\nabla\cdot\vec{b}",
            grad_order=grad_order,
        )

    def load_3d_current_density_vfield(
        self,
        grad_order: int = 2,
    ) -> field_types.VectorField_3D:
        """Compute current density: `curl[vec(b)]`."""
        b_vfield_3d = self.load_3d_magnetic_vfield()
        return field_operators.compute_vfield_curl(
            vfield_3d=b_vfield_3d,
            field_label=r"\nabla\times\vec{b}",
            grad_order=grad_order,
        )

    def load_3d_current_density_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField_3D:
        """Compute current magnitude: `|curl[vec(b)]|`."""
        j_vfield_3d = self.load_3d_current_density_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=j_vfield_3d,
            field_label=r"|\nabla\times\vec{b}|",
        )

    def load_3d_current_helicity_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField_3D:
        """Compute current helicity density: `curl[vec(b)] cdot vec(b)`."""
        j_vfield_3d = self.load_3d_current_density_vfield(grad_order=grad_order)
        b_vfield_3d = self.load_3d_magnetic_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_3d_a=j_vfield_3d,
            vfield_3d_b=b_vfield_3d,
            field_label=r"(\nabla\times\vec{b})\cdot\vec{b}",
        )

##
## --- MHD COMPOSITE FIELDS
##

    def load_3d_cross_helicity_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Compute cross helicity density: `vec(v) cdot vec(b)`."""
        v_vfield_3d = self.load_3d_velocity_vfield()
        b_vfield_3d = self.load_3d_magnetic_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_3d_a=v_vfield_3d,
            vfield_3d_b=b_vfield_3d,
            field_label=r"\vec{v}\cdot\vec{b}",
        )

    def load_3d_lorentz_force_vfield(
        self,
        grad_order: int = 2,
    ) -> field_types.VectorField_3D:
        """Compute Lorentz force: `curl[vec(b)] x vec(b)`."""
        j_vfield_3d = self.load_3d_current_density_vfield(grad_order=grad_order)
        b_vfield_3d = self.load_3d_magnetic_vfield()
        return field_operators.compute_vfield_cross_product(
            vfield_3d_a=j_vfield_3d,
            vfield_3d_b=b_vfield_3d,
            field_label=r"(\nabla\times\vec{b})\times\vec{b}",
        )

    def load_3d_lorentz_force_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField_3D:
        """Compute Lorentz force magnitude: `| curl[vec(b)] x vec(b) |`."""
        lf_vfield_3d = self.load_3d_lorentz_force_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=lf_vfield_3d,
            field_label=r"|(\nabla\times\vec{b})\times\vec{b}|",
        )

    def load_3d_energy_ratio_sfield(
        self,
        energy_prefactor: float = 0.5,
    ) -> field_types.ScalarField_3D:
        """Compute magnetic-to-kinetic energy ratio: `E_mag / E_kin`."""
        Emag_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_magnetic_energy_sfield(energy_prefactor=energy_prefactor),
            param_name="<Emag_sfield_3d>",
        )
        Ekin_sarray_3d = self._extract_3d_sarray(
            sfield_3d=self.load_3d_kinetic_energy_sfield(),
            param_name="<Ekin_sfield_3d>",
        )
        Ekin_has_zeros = compute_array_stats.check_no_zero_values(
            array=Ekin_sarray_3d,
            param_name="<Ekin_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            Eratio_sarray_3d = Emag_sarray_3d / Ekin_sarray_3d
        if not Ekin_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=Eratio_sarray_3d,
                param_name="<Eratio_sfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=Eratio_sarray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=Eratio_sarray_3d,
            udomain_3d=self.load_3d_uniform_domain(),
            field_label=r"E_\mathrm{mag} / E_\mathrm{kin}",
            sim_time=self.sim_time,
        )

    def load_3d_poynting_flux_vfield(
        self,
    ) -> field_types.VectorField_3D:
        """Compute Poynting-flux-like vector: `vec(b) x [vec(v) x vec(b)]`."""
        v_vfield_3d = self.load_3d_velocity_vfield()
        b_vfield_3d = self.load_3d_magnetic_vfield()
        vxb_vfield_3d = field_operators.compute_vfield_cross_product(
            vfield_3d_a=v_vfield_3d,
            vfield_3d_b=b_vfield_3d,
            field_label=r"\vec{v}\times\vec{b}",
        )
        return field_operators.compute_vfield_cross_product(
            vfield_3d_a=b_vfield_3d,
            vfield_3d_b=vxb_vfield_3d,
            field_label=r"\vec{b}\times(\vec{v}\times\vec{b})",
        )


## } MODULE
