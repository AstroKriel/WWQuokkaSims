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

PROBLEM_KEY = "CurrentSheet"

_DOMAIN_LO = (-0.5, -0.5, -0.5)
_DOMAIN_HI = (0.5, 0.5, 0.5)
_BOUNDARY_CONDITIONS = ("periodic", "periodic", "periodic")

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction_order_key: str,
    num_cells: tuple[int, int, int] = (1024, 1024, 8),
    blocking_factor: int | tuple[int, int, int] = (16, 16, 8),
    max_grid_size: int | tuple[int, int, int] = 128,
    max_time_steps: int = 1_000_000,
    cfl: float = 0.2,
    stop_time: float = 10.0,
    use_reflux: int = 0,
    use_subcycle: int = 0,
    resistivity: float | None = None,
    snapshot_prefix: str = "snapshots/plt",
    snapshot_index_interval: int | None = None,
    snapshot_time_interval: float | None = 0.5,
    checkpoint_index_interval: int | None = None,
    checkpoint_time_interval: float | None = 1.0,
    checkpoint_prefix: str | None = "chk",
) -> write_params.SimParams:
    """Build the full parameter set for one `CurrentSheet` run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction_order_key).value
    return write_params.SimParams(
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
            cfl=cfl,
            use_reflux=use_reflux,
            use_subcycle=use_subcycle,
            stop_time=stop_time,
            max_time_steps=max_time_steps,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=2,
            reconstruction_order=reconstruction_order,
            ## Quokka requires use_dual_energy=0 whenever resistivity is nonzero
            use_dual_energy=0 if resistivity is not None else None,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
            resistivity=resistivity,
        ),
    )


## } MODULE
