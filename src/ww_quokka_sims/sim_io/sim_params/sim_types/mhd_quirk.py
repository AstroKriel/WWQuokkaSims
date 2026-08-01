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

PROBLEM_KEY = "MHDQuirk"

_DOMAIN_LO = (0.0, 0.0, 0.0)
_DOMAIN_HI = (1.0, 1.0, 1.0)
_BOUNDARY_CONDITIONS = ("ext_dir", "periodic", "periodic")
_CFL = 0.4
_STOP_TIME = 0.4
_MAX_TIME_STEPS = 2000

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction_order_key: str,
    num_cells: tuple[int, int, int] = (128, 128, 8),
    blocking_factor: int | tuple[int, int, int] = (16, 16, 8),
    max_grid_size: int | tuple[int, int, int] = 32,
    snapshot_prefix: str = "snapshots/plt",
    snapshot_index_interval: int | None = 25,
    snapshot_time_interval: float | None = None,
    checkpoint_index_interval: int | None = -1,
    checkpoint_time_interval: float | None = None,
    checkpoint_prefix: str | None = None,
) -> write_param_groups.SimParams:
    """Build the full parameter set for one `MHDQuirk` run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction_order_key).value
    return write_param_groups.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=_DOMAIN_LO,
            domain_hi=_DOMAIN_HI,
            boundary_conditions=_BOUNDARY_CONDITIONS,
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=num_cells,
            blocking_factor=blocking_factor,
            max_grid_size=max_grid_size,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_prefix=snapshot_prefix,
            snapshot_index_interval=snapshot_index_interval,
            snapshot_time_interval=snapshot_time_interval,
            checkpoint_index_interval=checkpoint_index_interval,
            checkpoint_time_interval=checkpoint_time_interval,
            checkpoint_prefix=checkpoint_prefix,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=_CFL,
            stop_time=_STOP_TIME,
            max_time_steps=_MAX_TIME_STEPS,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=2,
            reconstruction_order=reconstruction_order,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
        ),
    )


## } MODULE
