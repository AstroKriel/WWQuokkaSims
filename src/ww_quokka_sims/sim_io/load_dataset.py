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

    def __init__(
        self,
        dataset_dir: str | Path,
        verbose: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.verbose = bool(verbose)
        self.dataset = None
        self._sim_time: float | None = None  # remains cached even after dataset is closed
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

    def list_available_fields(
        self,
    ) -> list[tuple[str, str]]:
        self._open_dataset_if_needed()
        assert self.dataset is not None
        fields = sorted(set(self.dataset.field_list))
        self._close_dataset_if_needed()
        log_manager.log_items(
            title="Available Fields",
            items=fields,
            message=f"Stored under: {self.dataset_dir}",
            message_position="bottom",
            show_time=False,
        )
        return fields

    def _load_sfield(
        self,
        field_key: tuple[str, str],
    ) -> numpy.ndarray:
        self._open_dataset_if_needed()
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        if field_key not in self.dataset.field_list:
            self._close_dataset_if_needed()
            raise KeyError(f"Field {field_key} not found in {self.dataset_dir}")
        data_array = numpy.asarray(covering_grid[field_key], dtype=numpy.float64)
        if data_array.ndim != 3:
            self._close_dataset_if_needed()
            raise ValueError(f"Expected a 3D field for {field_key}, got {data_array.shape}")
        self._close_dataset_if_needed()
        return numpy.ascontiguousarray(data_array)

    def _load_vfield(
        self,
        component_keys: dict[str, tuple[str, str]],
        labels: tuple[str, str, str],
    ) -> field_types.VectorField:
        self._open_dataset_if_needed()
        sim_time = self.sim_time
        assert self.dataset is not None
        covering_grid = self._get_covering_grid()
        data_arrays: dict[str, numpy.ndarray] = {}
        for axis in ("x", "y", "z"):
            field_key = component_keys[axis]
            if field_key not in self.dataset.field_list:
                self._close_dataset_if_needed()
                raise KeyError(f"Field {field_key} not found in {self.dataset_dir}")
            data_array = numpy.asarray(covering_grid[field_key], dtype=numpy.float64)
            if data_array.ndim != 3:
                self._close_dataset_if_needed()
                raise ValueError(f"Expected a 3D field for {field_key}, got {data_array.shape}")
            data_arrays[axis] = data_array
        self._close_dataset_if_needed()
        vfield_arrays = numpy.stack(
            [
                data_arrays["x"],
                data_arrays["y"],
                data_arrays["z"],
            ],
            axis=0,
        )
        return field_types.VectorField(
            sim_time=sim_time,
            data=vfield_arrays,
            labels=labels,
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
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=self._load_sfield(("boxlib", "gasDensity")),
            label="rho",
        )

    def load_total_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=self._load_sfield(("boxlib", "gasEnergy")),
            label="E_tot",
        )

    def load_internal_energy_sfield(
        self,
    ) -> field_types.ScalarField:
        return field_types.ScalarField(
            sim_time=self.sim_time,
            data=self._load_sfield(("boxlib", "gasInternalEnergy")),
            label="E_int",
        )

    def load_momentum_vfield(
        self,
    ) -> field_types.VectorField:
        return self._load_vfield(
            component_keys={
                "x": ("boxlib", "x-GasMomentum"),
                "y": ("boxlib", "y-GasMomentum"),
                "z": ("boxlib", "z-GasMomentum"),
            },
            labels=("M_x", "M_y", "M_z"),
        )

    def load_magnetic_vfield(
        self,
    ) -> field_types.VectorField:
        return self._load_vfield(
            component_keys={
                "x": ("boxlib", "x-BField"),
                "y": ("boxlib", "y-BField"),
                "z": ("boxlib", "z-BField"),
            },
            labels=("B_x", "B_y", "B_z"),
        )

    def load_magnetic_energy(self, coeff: float = 0.5) -> field_types.ScalarField:
        return field_operators.compute_magnetic_energy(
            vfield_b=self.load_magnetic_vfield(),
            coeff=coeff,
            label="E_mag",
        )

    def load_velocity_vfield(
        self,
    ) -> field_types.VectorField:
        vfield_mom = self.load_momentum_vfield()
        sfield_rho = self.load_density_sfield()
        with numpy.errstate(divide="ignore", invalid="ignore"):
            vfield_vel = vfield_mom.data / sfield_rho.data[numpy.newaxis, ...]
        return field_types.VectorField(
            sim_time=self.sim_time,
            data=vfield_vel,
            labels=("V_x", "V_y", "V_z"),
        )

    @property
    def is_open(self) -> bool:
        ## true iff the yt dataset handle is currently open
        return self.dataset is not None

    def close(self) -> None:
        self._in_context = False
        self._close_dataset()


## } MODULE
