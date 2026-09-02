## { MODULE

##
## === DEPENDENCIES
##

## local
from .. import param_groups
from .. import scheme_lookup
from .. import write_params

##
## === CONSTANTS
##

_BOUNDARY_CONDITIONS = ("periodic", "periodic", "periodic")
_SNAPSHOT_PREFIX = "snapshots/plt"
_INTEGRATOR_ORDER = 2
_USE_DUAL_ENERGY = 0
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
    error_tol: float = 0.003,
    domain_lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_cells: tuple[int, int, int] = (256, 8, 8),
    blocking_factor: int | tuple[int, int, int] = (256, 8, 8),
    max_grid_size: int | tuple[int, int, int] = 256,
    cfl: float = 0.3,
    stop_time: float = 1.0,
    max_time_steps: int = 20_000,
    snapshot_index_interval: int = 100,
    checkpoint_index_interval: int = -1,
) -> write_params.SimParams:
    """Build the full parameter set for one fixed-resolution correctness run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction_order_key)
    return write_params.SimParams(
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
            use_dual_energy=_USE_DUAL_ENERGY,
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
                "error_tol": error_tol,
            },
        ),
    )


## } MODULE
