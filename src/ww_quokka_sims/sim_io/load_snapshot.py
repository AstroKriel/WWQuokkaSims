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
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_types

## local
from ._energy_fields import _EnergyFieldsMixin
from ._magnetic_fields import _MagneticFieldsMixin
from ._mhd_fields import _MHDFieldsMixin
from ._read_fields import (
    FieldKey,
    LRUCache,
    YT_SFIELD_KEYS,
    YT_VFIELD_KEYS,
)
from ._velocity_fields import _VelocityFieldsMixin

##
## === SNAPSHOT OPERATOR CLASS
##


class QuokkaSnapshot(
    _VelocityFieldsMixin,
    _EnergyFieldsMixin,
    _MagneticFieldsMixin,
    _MHDFieldsMixin,
):
    """Interface for loading Quokka snapshots with yt."""

    dataset_dir: Path
    verbose: bool
    dataset: Any | None
    _in_context: bool
    _sim_time: float | None
    _covering_grid: Any | None
    _udomain_3d: domain_models.UniformDomain_3D | None
    _field_cache: LRUCache

    ##
    ## --- SNAPSHOT LIFECYCLE
    ##

    def __init__(
        self,
        dataset_dir: str | Path,
        verbose: bool = True,
    ):
        """Initialise a snapshot handle without opening the underlying yt dataset."""
        validate_types.ensure_bool(
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
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
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
            ds = yt_load(str(self.dataset_dir))
            self._sim_time = float(ds.current_time)
            self.dataset = ds

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
## --- PROBE SNAPSHOT
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
            manage_log.log_error(text=msg)
            raise RuntimeError(msg)
        return float(sim_time)

    def _get_covering_grid(
        self,
    ) -> Any:
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
        field_keys = sorted(
            set(
                self.dataset.field_list,
            ),
        )
        self._close_dataset_if_needed()
        return field_keys

    def list_available_field_keys(
        self,
    ) -> list[FieldKey]:
        """List all available yt field keys in this snapshot."""
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
        available_keys = set(
            self._get_available_field_keys(),
        )
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
            valid_string = ", ".join(
                YT_SFIELD_KEYS.keys(),
            )
            msg = f"Unknown scalar field `{field_name}`. Valid options: {valid_string}"
            manage_log.log_error(text=msg)
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
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return field_key

    def _resolve_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        """Return the component yt keys for a named vector field."""
        if field_name not in YT_VFIELD_KEYS:
            valid_string = ", ".join(
                YT_VFIELD_KEYS.keys(),
            )
            msg = f"Unknown vector field `{field_name}`. Valid options: {valid_string}"
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return YT_VFIELD_KEYS[field_name]["keys"]

    def _get_missing_vfield_keys(
        self,
        field_name: str,
    ) -> list[FieldKey]:
        """Return the list of missing component keys for a named vector field."""
        vfield_key_lookup = self._resolve_vfield_key_lookup(field_name)
        available_keys = set(
            self._get_available_field_keys(),
        )
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
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return self._resolve_vfield_key_lookup(field_name)

    def _is_vfield_keys_available(
        self,
        field_name: str,
    ) -> bool:
        """Return `True` iff all components for the named vector field exist."""
        return len(
            self._get_missing_vfield_keys(
                field_name,
            ),
        ) == 0

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
        field_key: FieldKey,
        field_label: str,
    ) -> field_models.ScalarField_3D:
        """Wrap a scalar array as `ScalarField` with a `field_label` and `sim_time`."""
        validate_types.ensure_nonempty_string(
            param=field_label,
            param_name="field_label",
        )
        sarray_3d = self._load_3d_sarray(field_key)
        udomain_3d = self.load_uniform_domain()
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray_3d,
            udomain_3d=udomain_3d,
            field_label=field_label,
            sim_time=self.sim_time,
        )

    def load_3d_vfield(
        self,
        vfield_key_lookup: dict[cartesian_axes.CartesianAxis_3D, FieldKey],
        field_label: str,
    ) -> field_models.VectorField_3D:
        """Load and stack 3 components into a `VectorField_3D` with a `field_label` and `sim_time`."""
        if set(vfield_key_lookup) != set(cartesian_axes.DEFAULT_3D_AXES_ORDER):
            received_axes = [axis.value for axis in sorted(vfield_key_lookup.keys(), key=lambda a: a.value)]
            expected_axes = [axis.value for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER]
            msg = (
                "`vfield_key_lookup` must contain all 3 components "
                f"{expected_axes}; only received: {received_axes}"
            )
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        validate_types.ensure_nonempty_string(
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
        udomain_3d = self.load_uniform_domain()
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=varray_3d,
            udomain_3d=udomain_3d,
            sim_time=sim_time,
            field_label=field_label,
        )

##
## --- DOMAIN
##

    def load_uniform_domain(
        self,
        force_periodicity: bool = True,
    ) -> domain_models.UniformDomain_3D:
        """
        Return uniform domain metadata (bounds, resolution, periodicity).

        Note: force_periodicity only affects the first call; subsequent calls returns the cached domain.
        This is required because yt cannot read this property reliably yet
        """
        validate_types.ensure_bool(
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
        udomain_3d = domain_models.UniformDomain_3D(
            periodicity=(is_periodic_x, is_periodic_y, is_periodic_z),
            resolution=(num_cells_x, num_cells_y, num_cells_z),
            domain_bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        )
        self._udomain_3d = udomain_3d
        return udomain_3d

##
## --- BASIC FIELDS
##

    def load_density_sfield(
        self,
    ) -> field_models.ScalarField_3D:
        """Load gas density: `rho`."""
        cached_field = self._field_cache.get_cached_field("density")
        if isinstance(cached_field, field_models.ScalarField_3D):
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

    def load_momentum_vfield(
        self,
    ) -> field_models.VectorField_3D:
        """Load momentum field: `vec(m) = rho vec(v)`."""
        cached_field = self._field_cache.get_cached_field("momentum")
        if isinstance(cached_field, field_models.VectorField_3D):
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

    def load_magnetic_vfield(
        self,
    ) -> field_models.VectorField_3D:
        """Load magnetic field: `vec(b)`."""
        cached_field = self._field_cache.get_cached_field("magnetic")
        if isinstance(cached_field, field_models.VectorField_3D):
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

    def load_total_energy_sfield(
        self,
    ) -> field_models.ScalarField_3D:
        """Load total energy: `E_tot = E_int + E_kin + E_mag` (code units)."""
        Etot_key = self._get_sfield_key("total_energy")
        return self.load_3d_sfield(
            field_key=Etot_key,
            field_label=r"E_\mathrm{tot}",
        )


## } MODULE
