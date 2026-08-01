## { MODULE

##
## === DEPENDENCIES
##

## local
from .. import param_groups
from .. import write_param_groups

## import scheme_lookup directly from its module file, not via the package __init__, to avoid a static import cycle
import ww_quokka_sims.sim_io.sim_params.sim_types.scheme_lookup as scheme_lookup

##
## === CONSTANTS
##

PROBLEM_KEY = "AlfvenWaveCircular-Correctness"

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction: str,
    error_tol: float = 0.003,
    domain_lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_cells: tuple[int, int, int] = (256, 8, 8),
    blocking_factor: int | tuple[int, int, int] = (256, 8, 8),
    max_grid_size: int | tuple[int, int, int] = 256,
    cfl: float = 0.3,
    ## wave direction/mode count are hardcoded, giving wavelength=1/alfven_speed=1, so one full
    ## period is exactly 1.0 -- safe to default here (nothing exposed changes it)
    stop_time: float = 1.0,
    max_time_steps: int = 20_000,
    snapshot_index_interval: int = 100,
    checkpoint_index_interval: int = -1,
) -> write_param_groups.SimParams:
    """Build the full parameter set for one `AlfvenWaveCircular` fixed-resolution correctness run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction).value
    return write_param_groups.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            boundary_conditions=("periodic", "periodic", "periodic"),
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=num_cells,
            blocking_factor=blocking_factor,
            max_grid_size=max_grid_size,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_prefix="snapshots/plt",
            snapshot_index_interval=snapshot_index_interval,
            checkpoint_index_interval=checkpoint_index_interval,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=cfl,
            stop_time=stop_time,
            max_time_steps=max_time_steps,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=2,
            reconstruction_order=reconstruction_order,
            use_dual_energy=0,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
        ),
        setup_params=param_groups.SetupParams(
            param_values={
                "run_sim": True,
                "run_convergence": False,
                "error_tol": error_tol,
            },
        ),
    )


## } MODULE
