## { MODULE

##
## === DEPENDENCIES
##

## local
from .._sim_params import param_groups
from .._sim_params import save_params
from .._sim_params import scheme_lookup

##
## === CONSTANTS
##

_BOUNDARY_CONDITIONS = ("periodic", "periodic", "periodic")
_SNAPSHOT_PREFIX = "snapshots/plt"
_INTEGRATOR_ORDER = 2
_RUN_SIM = True
_RUN_CONVERGENCE = False

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction_order_key: str,
    num_modes_x: int,
    num_modes_y: int,
    num_modes_z: int,
    angle_between_k_b0: float,
    stop_time: float,
    max_time_steps: int,
    error_tol: float = 0.002,
    domain_lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_cells: tuple[int, int, int] = (128, 128, 128),
    blocking_factor: int | tuple[int, int, int] = 128,
    max_grid_size: int | tuple[int, int, int] = 128,
    cfl: float = 0.3,
    snapshot_index_interval: int = 25,
    checkpoint_index_interval: int = -1,
) -> save_params.SimParams:
    """
    Build the full parameter set for one fixed-resolution correctness run.

    `stop_time`/`max_time_steps` have no default: the right `stop_time` (one wave period) depends
    on `num_modes_x/y/z`/`angle_between_k_b0`, so a fixed default would silently be wrong.
    """
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction_order_key)
    return save_params.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            boundary_conditions=_BOUNDARY_CONDITIONS,
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=num_cells,
            blocking_factor=blocking_factor,
            max_grid_size=max_grid_size,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_prefix=_SNAPSHOT_PREFIX,
            snapshot_index_interval=snapshot_index_interval,
            checkpoint_index_interval=checkpoint_index_interval,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=cfl,
            stop_time=stop_time,
            max_time_steps=max_time_steps,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=_INTEGRATOR_ORDER,
            reconstruction_order=reconstruction_order,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key),
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key),
            reconstruction_order=reconstruction_order,
        ),
        setup_params=param_groups.SetupParams(
            param_values={
                "run_sim": _RUN_SIM,
                "run_convergence": _RUN_CONVERGENCE,
                "angle_between_k_b0": angle_between_k_b0,
                "num_modes_x": num_modes_x,
                "num_modes_y": num_modes_y,
                "num_modes_z": num_modes_z,
                "error_tol": error_tol,
            },
        ),
    )


## } MODULE
