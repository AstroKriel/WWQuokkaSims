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

PROBLEM_KEY = "AlfvenWaveCircular-Convergence"

## `run_convergence` overrides domain/resolution/stop_time/max_timesteps internally; values
## below match the reference file only. No `num_modes`/`angle_between_k_b0`: hardcoded here.
_DOMAIN_LO = (0.0, 0.0, 0.0)
_DOMAIN_HI = (1.0, 1.0, 1.0)
_NUM_CELLS = (64, 8, 8)
_NX_MAX = 2048

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction: str,
) -> write_param_groups.SimParams:
    """Build the full parameter set for one `AlfvenWaveCircular` Richardson-convergence-sweep combination."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction).value
    return write_param_groups.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=_DOMAIN_LO,
            domain_hi=_DOMAIN_HI,
            is_boundary_periodic=(1, 1, 1),
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=_NUM_CELLS,
            blocking_factor=8,
            max_grid_size=128,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_index_interval=-1,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=0.3,
            use_reflux=0,
            use_subcycle=0,
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
            group_title="wave setup",
            param_values={
                "machine_precision_target": 0,
                "nx_max": _NX_MAX,
            },
        ),
        amr_verbosity=0,
    )


## } MODULE
