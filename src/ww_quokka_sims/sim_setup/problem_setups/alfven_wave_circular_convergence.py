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
_IS_BOUNDARY_PERIODIC = (1, 1, 1)
_NUM_CELLS = (64, 8, 8)
_BLOCKING_FACTOR = 8
_MAX_GRID_SIZE = 128
_AMR_VERBOSITY = param_groups.AmrVerbosity.SILENT
_SNAPSHOT_INDEX_INTERVAL = -1
_CFL = 0.3
_USE_REFLUX = 0
_USE_SUBCYCLE = 0
_INTEGRATOR_ORDER = 2
_USE_DUAL_ENERGY = 0
_MACHINE_PRECISION_TARGET = 0
_NX_MAX = 2048

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction_order_key: str,
) -> save_params.SimParams:
    """Build the full parameter set for one Richardson-convergence-sweep combination."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction_order_key)
    return save_params.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=_DOMAIN_LO,
            domain_hi=_DOMAIN_HI,
            is_boundary_periodic=_IS_BOUNDARY_PERIODIC,
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=_NUM_CELLS,
            blocking_factor=_BLOCKING_FACTOR,
            max_grid_size=_MAX_GRID_SIZE,
        ),
        verbosity_params=param_groups.VerbosityParams(level=_AMR_VERBOSITY),
        output_file_params=param_groups.OutputFileParams(
            snapshot_index_interval=_SNAPSHOT_INDEX_INTERVAL,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=_CFL,
            use_reflux=_USE_REFLUX,
            use_subcycle=_USE_SUBCYCLE,
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
                "machine_precision_target": _MACHINE_PRECISION_TARGET,
                "nx_max": _NX_MAX,
            },
        ),
    )


## } MODULE
