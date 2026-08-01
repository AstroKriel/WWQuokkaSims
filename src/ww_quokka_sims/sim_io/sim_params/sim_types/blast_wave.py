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

PROBLEM_KEY = "MHDBlast"

## the blast is centered at the origin with a hardcoded radius, independent of the domain;
## domain must stay symmetric around the origin for the blast to remain centered
_DOMAIN_LO = (-0.5, -0.5, -0.5)
_DOMAIN_HI = (0.5, 0.5, 0.5)

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction: str,
    num_cells: tuple[int, int, int],
    blocking_factor: int | tuple[int, int, int],
    max_grid_size: int | tuple[int, int, int],
    max_time_steps: int,
    cfl: float = 0.3,
    stop_time: float = 0.05,
    use_reflux: int = 0,
    use_subcycle: int = 0,
    snapshot_prefix: str = "snapshots/plt",
    snapshot_index_interval: int | None = None,
    snapshot_time_interval: float | None = None,
    checkpoint_index_interval: int | None = None,
    checkpoint_time_interval: float | None = None,
    checkpoint_prefix: str | None = None,
    derived_vars: tuple[str, ...] | None = ("magnetic_divergence", ),
) -> write_param_groups.SimParams:
    """Build the full parameter set for one `MHDBlast` run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction).value
    return write_param_groups.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=_DOMAIN_LO,
            domain_hi=_DOMAIN_HI,
            boundary_conditions=("periodic", "periodic", "periodic"),
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
            derived_vars=derived_vars,
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
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
        ),
    )


## } MODULE
