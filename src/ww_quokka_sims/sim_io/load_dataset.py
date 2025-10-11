## { MODULE

##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from yt.loaders import load as yt_load
from yt.utilities.logger import ytLogger as yt_logger
from jormi.ww_io import log_manager
from jormi.ww_fields import field_types, field_operators

##
## === OPERATOR CLASS
##


class QuokkaDataset:
    """
    Interface for loading Quokka datasets with yt.
    """

    YT_VFIELD_KEYS: dict[str, dict] = {
        "momentum": {
            "keys": {
                comp_str: ("boxlib", f"{comp_str}-GasMomentum")
                for comp_str in ("x", "y", "z")
            },
            "description": "Momentum density components: rho v",
        },
        "magnetic": {
            "keys": {
                comp_str: ("boxlib", f"{comp_str}-BField")
                for comp_str in ("x", "y", "z")
            },
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
            "description": "Total energy density (internal + kinetic + magnetic)",
        },
    }

    def __init__(
        self,
        dataset_dir: str | Path,
        verbose: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.verbose = bool(verbose)
        self.dataset = None
        self._sim_time: float | None = None  # remains cached even after the dataset is closed
        self.covering_grid = None
        self._in_context = False

    def __enter__(
        self,
    ):
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
        self._in_context = False
        self._close_dataset()

    def _open_dataset_if_needed(
        self,
    ) -> None:
        if self.dataset is None:
            if not self.verbose:
                ## reduce yt verbosity: only print warnings, errors and critical messages
                yt_logger.setLevel("WARNING")
            self.dataset = yt_load(str(self.dataset_dir))
            self._sim_time = float(self.dataset.current_time)

    def _close_dataset_if_needed(
        self,
    ) -> None:
        if not self._in_context:
            self._close_dataset()

    def _close_dataset(
        self,
    ) -> None:
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
            self.covering_grid = None
            ## NOTE: keep self._sim_time cached

    @property
    def sim_time(
        self,
    ) -> float:
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
        self._open_dataset_if_needed()
        assert self.dataset is not None
        if self.covering_grid is None:
            self.covering_grid = self.dataset.covering_grid(
                level=0,
                left_edge=self.dataset.domain_left_edge,
                dims=self.dataset.domain_dimensions,
            )
        return self.covering_grid

    def _get_available_field_keys(
        self,
    ) -> list[tuple[str, str]]:
        self._open_dataset_if_needed()
        assert self.dataset is not None
        field_keys = sorted(set(self.dataset.field_list))
        self._close_dataset_if_needed()
        return field_keys

    def list_available_field_keys(
        self,
    ) -> list[tuple[str, str]]:
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
        """True iff a particular yt field key exists in the dataset."""
        available_keys = set(self._get_available_field_keys())
        return field_key in available_keys

    def _resolve_sfield_key(
        self,
        field_name: str,
    ) -> tuple[str, str]:
        """Return the yt key for a named scalar field."""
        if field_name not in self.YT_SFIELD_KEYS:
            valid_str = ", ".join(self.YT_SFIELD_KEYS.keys())
            msg = f"Unknown scalar field '{field_name}'. Valid options: {valid_str}"
            log_manager.log_error(msg)
            raise KeyError(msg)
        return self.YT_SFIELD_KEYS[field_name]["key"]

    def _get_sfield_key(
        self,
        field_name: str,
    ) -> tuple[str, str]:
        field_key = self._resolve_sfield_key(field_name)
        if not self.is_field_key_available(field_key):
            msg = f"Scalar field '{field_name}' ({field_key[0]}:{field_key[1]}) is not available in: {self.dataset_dir}."
            log_manager.log_error(msg)
            raise KeyError(msg)
        return field_key

    def _resolve_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[str, tuple[str, str]]:
        """Return the component yt keys for a named vector field."""
        if field_name not in self.YT_VFIELD_KEYS:
            valid_str = ", ".join(self.YT_VFIELD_KEYS.keys())
            msg = f"Unknown vector field '{field_name}'. Valid options: {valid_str}"
            log_manager.log_error(msg)
            raise KeyError(msg)
        return self.YT_VFIELD_KEYS[field_name]["keys"]

    def _get_missing_vfield_keys(
        self,
        field_name: str,
    ) -> list[tuple[str, str]]:
        """Return a list of missing yt keys for the named vector field (no logging)."""
        vfield_key_lookup = self._resolve_vfield_key_lookup(field_name)
        available_keys = set(self._get_available_field_keys())
        return [comp_key for comp_key in vfield_key_lookup.values() if comp_key not in available_keys]

    def _get_vfield_key_lookup(
        self,
        field_name: str,
    ) -> dict[str, tuple[str, str]]:
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
        """True iff *all components* for a named vector field exist."""
        return len(self._get_missing_vfield_keys(field_name)) == 0

    def _load_sfield_data(
        self,
        field_key: tuple[str, str],
    ) -> numpy.ndarray:
        self._open_dataset_if_needed()
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        if field_key not in self.dataset.field_list:
            self._close_dataset_if_needed()
            raise KeyError(f"Field {field_key} not found in {self.dataset_dir}")
        field_data = numpy.asarray(covering_grid[field_key], dtype=field_types.DEFAULT_FLOAT_TYPE)
        if field_data.ndim != 3:
            self._close_dataset_if_needed()
            raise ValueError(f"Expected a 3D field for {field_key}; received: {field_data.shape}")
        self._close_dataset_if_needed()
        return numpy.ascontiguousarray(field_data)

    def load_sfield(
        self,
        field_key: tuple[str, str],
        field_label: str,
    ) -> field_types.ScalarField:
        field_data = self._load_sfield_data(field_key)
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=field_data,
            label=field_label,
        )

    def load_vfield(
        self,
        vfield_key_lookup: dict[str, tuple[str, str]],
        comp_labels: tuple[str, str, str],
    ) -> field_types.VectorField:
        req_comp_axes = ("x", "y", "z")
        if set(vfield_key_lookup) != set(req_comp_axes):
            msg = f"vfield_key_lookup must contain all 3 components: x, y, z; received: {sorted(vfield_key_lookup.keys())}"
            log_manager.log_error(msg)
            raise KeyError(msg)
        if len(comp_labels) != 3:
            msg = f"comp_labels must be length-3; received: {comp_labels!r}"
            log_manager.log_error(msg)
            raise ValueError(msg)
        self._open_dataset_if_needed()
        sim_time = self.sim_time
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        data_arrays: dict[str, numpy.ndarray] = {}
        for comp_axis in req_comp_axes:
            comp_key = vfield_key_lookup[comp_axis]
            if comp_key not in self.dataset.field_list:
                self._close_dataset_if_needed()
                raise KeyError(f"Field {comp_key} not found in {self.dataset_dir}")
            comp_data = numpy.asarray(covering_grid[comp_key], dtype=field_types.DEFAULT_FLOAT_TYPE)
            if comp_data.ndim != 3:
                self._close_dataset_if_needed()
                raise ValueError(f"Expected a 3D field for {comp_key}; received: {comp_data.shape}")
            data_arrays[comp_axis] = comp_data
        self._close_dataset_if_needed()
        stacked_data_arrays = numpy.stack(
            [
                data_arrays["x"],
                data_arrays["y"],
                data_arrays["z"],
            ],
            axis=0,
        )
        return field_types.VectorField(
            sim_time=sim_time,
            data=stacked_data_arrays,
            labels=comp_labels,
        )

    def load_domain(
        self,
    ) -> field_types.UniformDomain:
        self._open_dataset_if_needed()
        assert self.dataset is not None
        x_min, y_min, z_min = (float(value) for value in self.dataset.domain_left_edge)
        x_max, y_max, z_max = (float(value) for value in self.dataset.domain_right_edge)
        n_cells_x, n_cells_y, n_cells_z = (int(num_cells) for num_cells in self.dataset.domain_dimensions)
        is_periodic_x, is_periodic_y, is_periodic_z = (
            bool(is_periodic) for is_periodic in self.dataset.periodicity
        )
        domain = field_types.UniformDomain(
            periodicity=(is_periodic_x, is_periodic_y, is_periodic_z),
            resolution=(n_cells_x, n_cells_y, n_cells_z),
            domain_bounds=((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        )
        self._close_dataset_if_needed()
        return domain

    def load_density_sfield(
        self,
    ) -> field_types.ScalarField:
        rho_key = self._get_sfield_key("density")
        return self.load_sfield(
            field_key=rho_key,
            field_label=r"$\rho$"
        )

    def load_velocity_vfield(
        self,
    ) -> field_types.VectorField:
        mom_data = self.load_momentum_vfield().data
        rho_data = self.load_density_sfield().data
        with numpy.errstate(divide="ignore", invalid="ignore"):
            vel_data = mom_data / rho_data[numpy.newaxis, ...]
        vel_data = numpy.nan_to_num(vel_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return field_types.VectorField(
            sim_time=self.sim_time,
            data=vel_data,
            labels=(r"$v_x$", r"$v_y$", r"$v_z$"),
        )

    def load_momentum_vfield(
        self,
    ) -> field_types.VectorField:
        mom_key_lookup = self._get_vfield_key_lookup("momentum")
        return self.load_vfield(
            vfield_key_lookup=mom_key_lookup,
            comp_labels=(r"$m_x$", r"$m_y$", r"$m_z$"),
        )

    def load_kinetic_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        mom_data = self.load_momentum_vfield().data
        rho_data = self.load_density_sfield().data
        with numpy.errstate(divide="ignore", invalid="ignore"):
            Ekin_data = 0.5 * numpy.sum(mom_data * mom_data, axis=0) / rho_data
        Ekin_data = numpy.nan_to_num(Ekin_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=Ekin_data,
            label=r"$E_\mathrm{kin}$",
        )

    def load_magnetic_vfield(
        self,
    ) -> field_types.VectorField:
        b_key_lookup = self._get_vfield_key_lookup("magnetic")
        return self.load_vfield(
            vfield_key_lookup=b_key_lookup,
            comp_labels=(r"$b_x$", r"$b_y$", r"$b_z$"),
        )

    def load_magnetic_energy_sfield(
        self,
        energy_prefactor: float = 0.5,
    ) -> field_types.ScalarField:
        b_vfield = self.load_magnetic_vfield()
        return field_operators.compute_magnetic_energy_density(
            vfield=b_vfield,
            energy_prefactor=energy_prefactor,
            label=r"$E_\mathrm{mag}$",
        )

    def load_div_b_sfield(
        self,
    ) -> field_types.ScalarField:
        b_vfield = self.load_magnetic_vfield()
        domain = self.load_domain()
        divb_sfield = field_operators.compute_vfield_divergence(
            vfield=b_vfield,
            domain=domain,
        )
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=divb_sfield.data,
            label=r"$\nabla \cdot \vec{b}$",
        )

    def load_total_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        Etot_key = self._get_sfield_key("total_energy")
        return self.load_sfield(
            field_key=Etot_key,
            field_label=r"$E_\mathrm{tot}$"
        )

    def load_internal_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        Etot_data = self.load_total_energy_sfield().data
        Ekin_data = self.load_kinetic_energy_sfield().data
        Eint_data = Etot_data - Ekin_data
        if self._is_vfield_keys_available("magnetic"):
            Emag_data = self.load_magnetic_energy_sfield().data
            Eint_data -= Emag_data
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=Eint_data,
            label=r"$E_\mathrm{int}$",
        )

    def load_pressure_sfield(
        self,
        gamma: float = 5.0 / 3.0,
    ) -> field_types.ScalarField:
        Eint_data = self.load_internal_energy_sfield().data
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=(gamma - 1.0) * Eint_data,
            label=r"$p$",
        )

    @property
    def is_open(
        self,
    ) -> bool:
        ## true iff the yt dataset handle is currently open
        return self.dataset is not None

    def close(
        self,
    ) -> None:
        self._in_context = False
        self._close_dataset()


## } MODULE
