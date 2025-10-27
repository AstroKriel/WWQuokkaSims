## { MODULE

##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from yt.loaders import load as yt_load
from dataclasses import dataclass
from collections import OrderedDict
from yt.utilities.logger import ytLogger as yt_logger
from jormi.utils import type_utils
from jormi.ww_io import log_manager
from jormi.ww_fields import farray_operators, field_types, field_operators, decompose_fields

##
## === HELPER FUNCTIONS
##

def create_boxlib_vkeys(
    field_name: str
) -> dict[field_types.CompAxis, tuple[str, str]]:
    return {
        comp_axis: ("boxlib", f"{comp_axis}-{field_name}")
        for comp_axis in field_types.DEFAULT_COMP_AXES_ORDER
    }


##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class HelmholtzKineticEnergy:
    Ekin_div_sfield: field_types.ScalarField
    Ekin_sol_sfield: field_types.ScalarField
    Ekin_bulk_sfield: field_types.ScalarField

    def __post_init__(
        self,
    ) -> None:
        field_types.ensure_sfield(self.Ekin_div_sfield)
        field_types.ensure_sfield(self.Ekin_sol_sfield)
        field_types.ensure_sfield(self.Ekin_bulk_sfield)


class LRUCache:
    """Simple least-recently-used (LRU) cache for storing field objects."""

    def __init__(
        self,
        max_size: int = 3,
    ) -> None:
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
        field_data: field_types.ScalarField | field_types.VectorField,
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
## === YT FIELD MAPPINGS
##

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
## === OPERATOR CLASS
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
        self.dataset_dir = Path(dataset_dir)
        self.verbose = bool(verbose)
        self.dataset = None
        self._in_context = False
        self._sim_time = None
        self._covering_grid = None
        self._uniform_domain = None
        ## cache: density, momentum, velocity, and magnetic fields
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
            self._uniform_domain = None
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
            log_manager.log_error(msg)
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
    ) -> list[tuple[str, str]]:
        """Return all (field-group, field-name) yt keys available in the dataset."""
        self._open_dataset_if_needed()
        assert self.dataset is not None
        field_keys = sorted(set(self.dataset.field_list))
        self._close_dataset_if_needed()
        return field_keys

    def list_available_field_keys(
        self,
    ) -> list[tuple[str, str]]:
        """List all available yt field keys in this dataset."""
        field_keys = self._get_available_field_keys()
        log_manager.log_items(
            title="Available Fields",
            items=field_keys,
            message=f"Stored under: {self.dataset_dir}",
            message_position="bottom",
            show_time=False,
        )
        return field_keys

    def is_field_key_available(
        self,
        field_key: tuple[str, str],
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
    ) -> tuple[str, str]:
        """Resolve the yt key for a named scalar field."""
        if field_name not in YT_SFIELD_KEYS:
            valid_str = ", ".join(YT_SFIELD_KEYS.keys())
            msg = f"Unknown scalar field '{field_name}'. Valid options: {valid_str}"
            log_manager.log_error(msg)
            raise KeyError(msg)
        return YT_SFIELD_KEYS[field_name]["key"]

    def _get_sfield_key(
        self,
        field_name: str,
    ) -> tuple[str, str]:
        """Resolve and validate the yt key for a scalar field."""
        field_key = self._resolve_sfield_key(field_name)
        if not self.is_field_key_available(field_key):
            msg = f"Scalar field '{field_name}' ({field_key[0]}:{field_key[1]}) is not available in: {self.dataset_dir}."
            log_manager.log_error(msg)
            raise KeyError(msg)
        return field_key

    def _resolve_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[field_types.CompAxis, tuple[str, str]]:
        """Return the component yt keys for a named vector field."""
        if field_name not in YT_VFIELD_KEYS:
            valid_str = ", ".join(YT_VFIELD_KEYS.keys())
            msg = f"Unknown vector field '{field_name}'. Valid options: {valid_str}"
            log_manager.log_error(msg)
            raise KeyError(msg)
        return YT_VFIELD_KEYS[field_name]["keys"]

    def _get_missing_vfield_keys(
        self,
        field_name: str,
    ) -> list[tuple[str, str]]:
        """Return the list of missing component keys for a named vector field."""
        vfield_key_lookup = self._resolve_vfield_key_lookup(field_name)
        available_keys = set(self._get_available_field_keys())
        return [comp_key for comp_key in vfield_key_lookup.values() if comp_key not in available_keys]

    def _get_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[field_types.CompAxis, tuple[str, str]]:
        """Resolve and validate component keys for a named vector field."""
        missing_keys = self._get_missing_vfield_keys(field_name)
        if missing_keys:
            missing_str = ", ".join([f"{yt_group}:{yt_field}" for yt_group, yt_field in missing_keys])
            msg = f"Vector field '{field_name}' is incomplete in {self.dataset_dir}. Missing components: {missing_str}"
            log_manager.log_error(msg)
            raise KeyError(msg)
        return self._resolve_vfield_key_lookup(field_name)

    def _is_vfield_keys_available(
        self,
        field_name: str,
    ) -> bool:
        """Return `True` iff all components for the named vector field exist."""
        return len(self._get_missing_vfield_keys(field_name)) == 0

    def _load_sfield_data(
        self,
        field_key: tuple[str, str],
    ) -> numpy.ndarray:
        """Load a scalar field from the covering grid as a 3D `ndarray`."""
        self._open_dataset_if_needed()
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        if field_key not in self.dataset.field_list:
            self._close_dataset_if_needed()
            raise KeyError(f"Field {field_key} not found in {self.dataset_dir}")
        field_sarray = numpy.asarray(covering_grid[field_key], dtype=numpy.float64)
        if field_sarray.ndim != 3:
            self._close_dataset_if_needed()
            raise ValueError(f"Expected a 3D field for {field_key}; received: {field_sarray.shape}")
        self._close_dataset_if_needed()
        return numpy.ascontiguousarray(field_sarray)

##
## --- CORE FIELD LOADERS
##

    def load_sfield(
        self,
        field_key: tuple[str, str],
        field_label: str,
    ) -> field_types.ScalarField:
        """Wrap a scalar array as `ScalarField` with a `field_label` and `sim_time`."""
        type_utils.ensure_nonempty_str(
            var_obj=field_label,
            var_name="field_label",
        )
        field_sarray = self._load_sfield_data(field_key)
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=field_sarray,
            field_label=field_label,
        )

    def load_vfield(
        self,
        vfield_key_lookup: dict[field_types.CompAxis, tuple[str, str]],
        field_label: str,
    ) -> field_types.VectorField:
        """Load and stack 3 components into a `VectorField` with a `field_label` and `sim_time`."""
        if set(vfield_key_lookup) != set(field_types.DEFAULT_COMP_AXES_ORDER):
            msg = f"`vfield_key_lookup` must contain all 3 components: x, y, z; only received: {sorted(vfield_key_lookup.keys())}"
            log_manager.log_error(msg)
            raise KeyError(msg)
        type_utils.ensure_nonempty_str(
            var_obj=field_label,
            var_name="field_label",
        )
        self._open_dataset_if_needed()
        sim_time = self.sim_time
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        grouped_data_sarrays: dict[field_types.CompAxis, numpy.ndarray] = {}
        for comp_axis in field_types.DEFAULT_COMP_AXES_ORDER:
            comp_key = vfield_key_lookup[comp_axis]
            if comp_key not in self.dataset.field_list:
                self._close_dataset_if_needed()
                raise KeyError(f"Field {comp_key} was not found in {self.dataset_dir}")
            comp_sarray = numpy.asarray(covering_grid[comp_key], dtype=numpy.float64)
            if comp_sarray.ndim != 3:
                self._close_dataset_if_needed()
                raise ValueError(f"Expected a 3D field for {comp_key}; received: {comp_sarray.shape}")
            grouped_data_sarrays[comp_axis] = comp_sarray
        self._close_dataset_if_needed()
        data_varray = numpy.stack(
            [
                grouped_data_sarrays[comp_axis]
                for comp_axis in field_types.DEFAULT_COMP_AXES_ORDER
            ],
            axis=0,
        )
        return field_types.VectorField(
            sim_time=sim_time,
            data=data_varray,
            field_label=field_label,
        )

##
## --- DOMAIN
##

    def load_uniform_domain(
        self,
        force_periodicity: bool = True,
    ) -> field_types.UniformDomain:
        """Return uniform domain metadata (bounds, resolution, periodicity)."""
        ## Note: force_periodicity only affects the first call; subsequent calls returns the cached domain.
        ## This is required because yt cannot read this property reliably yet
        if self._uniform_domain is not None:
            return self._uniform_domain
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
        uniform_domain = field_types.UniformDomain(
            periodicity=(is_periodic_x, is_periodic_y, is_periodic_z),
            resolution=(num_cells_x, num_cells_y, num_cells_z),
            domain_bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        )
        self._uniform_domain = uniform_domain
        return uniform_domain

##
## --- BASIC FIELDS
##

    def load_density_sfield(
        self,
    ) -> field_types.ScalarField:
        """Load gas density: `rho`."""
        cached_field = self._field_cache.get_cached_field("density")
        if cached_field is not None:
            return cached_field
        rho_key = self._get_sfield_key("density")
        rho_sfield = self.load_sfield(
            field_key=rho_key,
            field_label=r"$\rho$",
        )
        self._field_cache.cache_field(
            field_name="density",
            field_data=rho_sfield,
        )
        return rho_sfield

    def load_momentum_vfield(
        self,
    ) -> field_types.VectorField:
        """Load momentum field: `vec(m) = rho vec(v)`."""
        cached_field = self._field_cache.get_cached_field("momentum")
        if cached_field is not None:
            return cached_field
        mom_key_lookup = self._get_vfield_key_lookup("momentum")
        mom_vfield = self.load_vfield(
            vfield_key_lookup=mom_key_lookup,
            field_label=r"$\rho \,\vec{v}$",
        )
        self._field_cache.cache_field(
            field_name="momentum",
            field_data=mom_vfield,
        )
        return mom_vfield

    def load_velocity_vfield(
        self,
    ) -> field_types.VectorField:
        """Load velocity field: `vec(v) = vec(m) / rho`."""
        cached_field = self._field_cache.get_cached_field("velocity")
        if cached_field is not None:
            return cached_field
        rho_sarray = self.load_density_sfield().data
        mom_varray = self.load_momentum_vfield().data
        with numpy.errstate(divide="ignore", invalid="ignore"):
            v_varray = mom_varray / rho_sarray[numpy.newaxis, ...]
        v_varray = numpy.nan_to_num(v_varray, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        v_vfield = field_types.VectorField(
            sim_time=self.sim_time,
            data=v_varray,
            field_label=r"$\vec{v}$",
        )
        self._field_cache.cache_field(
            field_name="velocity",
            field_data=v_vfield,
        )
        return v_vfield

    def load_magnetic_vfield(
        self,
    ) -> field_types.VectorField:
        """Load magnetic field: `vec(b)`."""
        cached_field = self._field_cache.get_cached_field("magnetic")
        if cached_field is not None:
            return cached_field
        b_key_lookup = self._get_vfield_key_lookup("magnetic")
        b_vfield = self.load_vfield(
            vfield_key_lookup=b_key_lookup,
            field_label=r"$\vec{b}$",
        )
        self._field_cache.cache_field(
            field_name="magnetic",
            field_data=b_vfield,
        )
        return b_vfield

    def load_total_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        """Load total energy: `E_tot = E_int + E_kin + E_mag` (code units)."""
        Etot_key = self._get_sfield_key("total_energy")
        return self.load_sfield(
            field_key=Etot_key,
            field_label=r"$E_\mathrm{tot}$",
        )

    def load_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        """Compute kinetic energy density: `E_kin = 0.5 * rho * |v|^2`."""
        mom_varray = self.load_momentum_vfield().data
        rho_sarray = self.load_density_sfield().data
        with numpy.errstate(divide="ignore", invalid="ignore"):
            Ekin_sarray = 0.5 * farray_operators.sum_of_squared_components(mom_varray) / rho_sarray
        Ekin_sarray = numpy.nan_to_num(Ekin_sarray, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=Ekin_sarray,
            field_label=r"$E_\mathrm{kin}$",
        )

    def load_magnetic_energy_sfield(
        self,
        energy_prefactor: float = 0.5,
        field_label=r"$E_\mathrm{mag}$",
    ) -> field_types.ScalarField:
        """Compute magnetic energy density: `E_mag = alpha * |b|^2` with `alpha=0.5` by default."""
        b_vfield = self.load_magnetic_vfield()
        return field_operators.compute_magnetic_energy_density(
            vfield=b_vfield,
            energy_prefactor=energy_prefactor,
            field_label=field_label,
        )

    def load_internal_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        """Compute internal energy: `E_int = E_tot - E_kin - E_mag` (`E_mag = 0` if `vec(b)` is not available)."""
        Etot_sarray = self.load_total_energy_sfield().data
        Ekin_sarray = self.load_kinetic_energy_sfield().data
        Eint_sarray = Etot_sarray - Ekin_sarray
        if self._is_vfield_keys_available("magnetic"):
            Emag_sarray = self.load_magnetic_energy_sfield().data
            Eint_sarray -= Emag_sarray
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=Eint_sarray,
            field_label=r"$E_\mathrm{int}$",
        )

    def load_pressure_sfield(
        self,
        gamma: float = 5.0 / 3.0,
    ) -> field_types.ScalarField:
        """Compute thermal pressure: `p = (gamma - 1) * E_int`."""
        Eint_sarray = self.load_internal_energy_sfield().data
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=(gamma - 1.0) * Eint_sarray,
            field_label=r"$p$",
        )

##
## --- VELOCITY FIELDS
##

    def load_divu_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField:
        """Compute divergence of velocity: `nabla cdot vec(v)` using a `grad_order` accurate stencil."""
        v_vfield = self.load_velocity_vfield()
        uniform_domain = self.load_uniform_domain()
        return field_operators.compute_vfield_divergence(
            vfield=v_vfield,
            uniform_domain=uniform_domain,
            field_label=r"$\nabla\cdot\vec{v}$",
            grad_order=grad_order,
        )

    def load_vorticity_vfield(
        self,
        grad_order: int = 2,
    ) -> field_types.VectorField:
        """Compute vorticity vector: `curl(vec(v))` using a `grad_order` accurate stencil."""
        v_vfield = self.load_velocity_vfield()
        uniform_domain = self.load_uniform_domain()
        return field_operators.compute_vfield_curl(
            vfield=v_vfield,
            uniform_domain=uniform_domain,
            grad_order=grad_order,
            field_label=r"$\nabla\times\vec{v}$",
        )

    def load_vorticity_magnitude_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField:
        """Compute vorticity magnitude: `|curl(vec(v))|`."""
        omega_vfield = self.load_vorticity_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield=omega_vfield,
            field_label=r"$|\nabla\times\vec{v}|$",
        )

    def load_kinetic_helicity_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField:
        """Compute kinetic helicity density: `curl(vec(v)) dot vec(v)`."""
        omega_vfield = self.load_vorticity_vfield(grad_order=grad_order)
        v_vfield = self.load_velocity_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_a=omega_vfield,
            vfield_b=v_vfield,
            field_label=r"$(\nabla\times\vec{v})\cdot\vec{v}$",
        )

    def load_helmholtz_kinetic_energy(
        self,
    ) -> HelmholtzKineticEnergy:
        """Compute Helmholtz-decomposed kinetic energies from `vec(v) = vec(v)_div + vec(v)_sol + vec(v)_bulk`."""
        uniform_domain = self.load_uniform_domain()
        rho_sarray = self.load_density_sfield().data
        v_vfield = self.load_velocity_vfield()
        helmholtz_vfields = decompose_fields.compute_helmholtz_decomposition(
            vfield=v_vfield,
            uniform_domain=uniform_domain,
        )
        v_div_varray = helmholtz_vfields.div_vfield.data
        v_sol_varray = helmholtz_vfields.sol_vfield.data
        v_bulk_varray = helmholtz_vfields.bulk_vfield.data
        Ekin_div_sarray = 0.5 * rho_sarray * farray_operators.sum_of_squared_components(v_div_varray)
        Ekin_sol_sarray = 0.5 * rho_sarray * farray_operators.sum_of_squared_components(v_sol_varray)
        Ekin_bulk_sarray = 0.5 * rho_sarray * farray_operators.sum_of_squared_components(v_bulk_varray)
        numpy.nan_to_num(Ekin_div_sarray, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        numpy.nan_to_num(Ekin_sol_sarray, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        numpy.nan_to_num(Ekin_bulk_sarray, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        Ekin_div_sfield = field_types.ScalarField(
            sim_time=self.sim_time,
            data=Ekin_div_sarray,
            field_label=r"$E_{\mathrm{kin}, \parallel}$",
        )
        Ekin_sol_sfield = field_types.ScalarField(
            sim_time=self.sim_time,
            data=Ekin_sol_sarray,
            field_label=r"$E_{\mathrm{kin}, \perp}$",
        )
        Ekin_bulk_sfield = field_types.ScalarField(
            sim_time=self.sim_time,
            data=Ekin_bulk_sarray,
            field_label=r"$E_{\mathrm{kin}, \mathrm{bulk}}$",
        )
        return HelmholtzKineticEnergy(
            Ekin_div_sfield=Ekin_div_sfield,
            Ekin_sol_sfield=Ekin_sol_sfield,
            Ekin_bulk_sfield=Ekin_bulk_sfield,
        )

    def load_div_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        """Compute kinetic energy in irrotational (curl-free) velocity modes: `E_kin,div = 0.5 rho (v_div)^2`."""
        helmholtz_Ekin = self.load_helmholtz_kinetic_energy()
        return helmholtz_Ekin.Ekin_div_sfield

    def load_sol_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        """Compute kinetic energy in solenoidal (divergence-free) velocity modes: `E_kin,sol = 0.5 rho (v_sol)^2`."""
        helmholtz_Ekin = self.load_helmholtz_kinetic_energy()
        return helmholtz_Ekin.Ekin_sol_sfield

    def load_bulk_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        """Compute kinetic energy in bulk velocity: `E_kin,bulk = 0.5 rho (v_bulk)^2`."""
        helmholtz_Ekin = self.load_helmholtz_kinetic_energy()
        return helmholtz_Ekin.Ekin_bulk_sfield

##
## --- MAGNETIC FIELDS
##

    def load_plasma_beta_sfield(
        self,
        gamma: float = 5.0 / 3.0,
    ) -> field_types.ScalarField:
        """Compute plasma beta: `beta = 2 p / |vec(b)|^2`."""
        p_sarray = self.load_pressure_sfield(gamma=gamma).data
        b_varray = self.load_magnetic_vfield().data
        b_sq_sarray = farray_operators.sum_of_squared_components(b_varray)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            beta_sarray = 2.0 * p_sarray / b_sq_sarray
        beta_sarray = numpy.nan_to_num(beta_sarray, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=beta_sarray,
            field_label=r"$\beta$",
        )

    def load_alfven_speed_vfield(
        self,
    ) -> field_types.VectorField:
        """Compute Alfven speed: `vec(v_A) = vec(b) / sqrt(rho)`."""
        b_varray = self.load_magnetic_vfield().data
        rho_sarray = self.load_density_sfield().data
        with numpy.errstate(divide="ignore", invalid="ignore"):
            va_varray = b_varray / numpy.sqrt(rho_sarray)[numpy.newaxis, ...]
        va_varray = numpy.nan_to_num(va_varray, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return field_types.VectorField(
            sim_time=self.sim_time,
            data=va_varray,
            field_label=r"$\vec{v}_A$",
        )

    def load_alfven_speed_magnitude_sfield(
        self,
    ) -> field_types.ScalarField:
        """Compute Alfven speed magnitude: `|vec(v_A)|`."""
        va_vfield = self.load_alfven_speed_vfield()
        return field_operators.compute_vfield_magnitude(
            vfield=va_vfield,
            field_label=r"$|\vec{v}_A|$",
        )

    def load_divb_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField:
        """Compute magnetic divergence: `div[vec(b)]`."""
        b_vfield = self.load_magnetic_vfield()
        uniform_domain = self.load_uniform_domain()
        return field_operators.compute_vfield_divergence(
            vfield=b_vfield,
            uniform_domain=uniform_domain,
            field_label=r"$\nabla\cdot\vec{b}$",
            grad_order=grad_order,
        )

    def load_current_density_vfield(
        self,
        grad_order: int = 2,
    ) -> field_types.VectorField:
        """Compute current density: `curl[vec(b)]`."""
        b_vfield = self.load_magnetic_vfield()
        uniform_domain = self.load_uniform_domain()
        return field_operators.compute_vfield_curl(
            vfield=b_vfield,
            uniform_domain=uniform_domain,
            field_label=r"$\nabla\times\vec{b}$",
            grad_order=grad_order,
        )

    def load_current_density_magnitude_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField:
        """Compute current magnitude: `|curl[vec(b)]|`."""
        j_vfield = self.load_current_density_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield=j_vfield,
            field_label=r"$|\nabla\times\vec{b}|$",
        )

    def load_current_helicity_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField:
        """Compute current helicity density: `curl[vec(b)] cdot vec(b)`."""
        j_vfield = self.load_current_density_vfield(grad_order=grad_order)
        b_vfield = self.load_magnetic_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_a=j_vfield,
            vfield_b=b_vfield,
            field_label=r"$(\nabla\times\vec{b})\cdot\vec{b}$",
        )

##
## --- MHD COMPOSITE FIELDS
##

    def load_cross_helicity_sfield(
        self,
    ) -> field_types.ScalarField:
        """Compute cross helicity density: `vec(v) cdot vec(b)`."""
        v_vfield = self.load_velocity_vfield()
        b_vfield = self.load_magnetic_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_a=v_vfield,
            vfield_b=b_vfield,
            field_label=r"$\vec{v}\cdot\vec{b}$",
        )

    def load_lorentz_force_vfield(
        self,
        grad_order: int = 2,
    ) -> field_types.VectorField:
        """Compute Lorentz force: `curl[vec(b)] x vec(b)`."""
        j_vfield = self.load_current_density_vfield(grad_order=grad_order)
        b_vfield = self.load_magnetic_vfield()
        return field_operators.compute_vfield_cross_product(
            vfield_a=j_vfield,
            vfield_b=b_vfield,
            field_label=r"$(\nabla\times\vec{b})\times\vec{b}$",
        )

    def load_lorentz_force_magnitude_sfield(
        self,
        grad_order: int = 2,
    ) -> field_types.ScalarField:
        """Compute Lorentz force magnitude: `| curl[vec(b)] x vec(b) |`."""
        lf_vfield = self.load_lorentz_force_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield=lf_vfield,
            field_label=r"$|(\nabla\times\vec{b})\times\vec{b}|$",
        )

    def load_poynting_flux_vfield(
        self,
    ) -> field_types.VectorField:
        """Compute Poynting-flux-like vector: `vec(b) x [vec(v) x vec(b)]`."""
        v_vfield = self.load_velocity_vfield()
        b_vfield = self.load_magnetic_vfield()
        vxb_vfield = field_operators.compute_vfield_cross_product(
            vfield_a=v_vfield,
            vfield_b=b_vfield,
            field_label=r"$\vec{v}\times\vec{b}$",
        )
        return field_operators.compute_vfield_cross_product(
            vfield_a=b_vfield,
            vfield_b=vxb_vfield,
            field_label=r"$\vec{b}\times(\vec{v}\times\vec{b})$",
        )


## } MODULE
