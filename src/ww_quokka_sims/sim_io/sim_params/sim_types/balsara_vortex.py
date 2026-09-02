## { MODULE

##
## === DEPENDENCIES
##

## local
from .. import param_groups
from .. import write_params

## import scheme_lookup directly from its module file, not via the package __init__, to avoid a static import cycle
import ww_quokka_sims.sim_io.sim_params.sim_types.scheme_lookup as scheme_lookup

##
## === CONSTANTS
##

PROBLEM_KEY = "MHDBalsaraVortex"

_BOUNDARY_CONDITIONS = ("periodic", "periodic", "periodic")

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction_order_key: str,
    domain_lo: tuple[float, float, float],
    domain_hi: tuple[float, float, float],
    num_cells: tuple[int, int, int],
    blocking_factor: int | tuple[int, int, int],
    max_grid_size: int | tuple[int, int, int],
    max_time_steps: int,
    vortex_mach: float = 0.01,
    vortex_b_magn: float = 0.01,
    advection: int = 1,
    num_orbits: int = 3,
    cfl: float = 0.3,
    snapshot_prefix: str = "snapshots/plt",
    snapshot_index_interval: int | None = 5000,
    snapshot_time_interval: float | None = None,
    checkpoint_index_interval: int | None = -1,
    checkpoint_time_interval: float | None = None,
    checkpoint_prefix: str | None = None,
) -> write_params.SimParams:
    """Build the full parameter set for one `MHDBalsaraVortex` run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction_order_key).value
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
            snapshot_prefix=snapshot_prefix,
            snapshot_index_interval=snapshot_index_interval,
            snapshot_time_interval=snapshot_time_interval,
            checkpoint_index_interval=checkpoint_index_interval,
            checkpoint_time_interval=checkpoint_time_interval,
            checkpoint_prefix=checkpoint_prefix,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=cfl,
            max_time_steps=max_time_steps,
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
        setup_params=param_groups.SetupParams(
            param_values={
                "vortex_Mach": vortex_mach,
                "vortex_b_magn": vortex_b_magn,
                "advection": advection,
                "num_orbits": num_orbits,
            },
        ),
    )


## } MODULE
