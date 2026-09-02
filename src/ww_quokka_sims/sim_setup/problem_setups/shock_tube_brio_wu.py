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

_DOMAIN_LO = (0.0, 0.0, 0.0)
_DOMAIN_HI = (1.0, 1.0, 1.0)
_BOUNDARY_CONDITIONS = ("ext_dir", "periodic", "periodic")
_INTEGRATOR_ORDER = 2

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction_order_key: str,
    num_cells: tuple[int, int, int],
    blocking_factor: int | tuple[int, int, int],
    max_grid_size: int | tuple[int, int, int],
    max_time_steps: int,
    cfl: float = 0.4,
    stop_time: float = 0.1,
    use_reflux: int = 0,
    use_subcycle: int = 0,
    use_dual_energy: int | None = 0,
    snapshot_prefix: str = "snapshots/plt",
    snapshot_index_interval: int | None = None,
    snapshot_time_interval: float | None = None,
    checkpoint_index_interval: int | None = None,
    checkpoint_time_interval: float | None = None,
    checkpoint_prefix: str | None = None,
) -> save_params.SimParams:
    """Build the full parameter set for one run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction_order_key)
    return save_params.SimParams(
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
            integrator_order=_INTEGRATOR_ORDER,
            reconstruction_order=reconstruction_order,
            use_dual_energy=use_dual_energy,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key),
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key),
            reconstruction_order=reconstruction_order,
        ),
    )


## } MODULE
