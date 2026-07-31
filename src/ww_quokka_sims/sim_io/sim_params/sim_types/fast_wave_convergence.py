## { MODULE

##
## === DEPENDENCIES
##

## local
from .. import param_groups
from .. import write_param_groups

## fully-dotted, not `from . import scheme_lookup`: that form triggers `reportImportCycles`,
## since `sim_types/__init__.py` imports this module before `scheme_lookup` resolves
import ww_quokka_sims.sim_io.sim_params.sim_types.scheme_lookup as scheme_lookup

##
## === CONSTANTS
##

PROBLEM_NAME = "FastWaveConvergence"

## `problem_main()` defaults to `setup.run_convergence=true` (Richardson sweep), which overrides
## `amr.n_cell`/`geometry.*`/`stop_time`/`max_timesteps` per sweep iteration, ignoring the toml
## (see threads/dead-toml-params/); domain/resolution below are fixed to match the reference file.
##
## `cfl`/`use_reflux`/`use_subcycle`/`use_tracers` are genuinely respected but kept fixed too: the
## sweep's pass/fail check (`expected_rate=2.0`/`tolerance=0.3`, hardcoded, no toml key) can fail
## from temporal error alone if `cfl` is too large, unrelated to reconstruction/EMF-scheme accuracy.
_BASE_NUM_CELLS = (128, 8, 8)
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
    """Build the full parameter set for one `FastWaveConvergence` scheme combo (e.g. `q26-b25-ppm_ep`)."""
    ## `.value` unwraps the Enum to the plain int/str `sim_params.param_groups` dataclasses declare;
    ## an Enum member's `str()`/f-string form is `"ClassName.MEMBER"`, not its underlying value
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction).value
    return write_param_groups.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=(0.0, 0.0, 0.0),
            domain_hi=(1.0, 1.0, 1.0),
            is_boundary_periodic=(1, 1, 1),
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=_BASE_NUM_CELLS,
            blocking_factor=(16, 8, 8),
            max_grid_size=128,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_index_interval=-1,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=0.3,
            use_reflux=0,
            use_subcycle=0,
            use_tracers=1,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=2,
            reconstruction_order=reconstruction_order,
            use_dual_energy=0,
        ),
        ## no `resistivity`: not physically meaningful for this ideal-MHD wave test
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
        ),
        setup_params=param_groups.SetupParams(
            group_title="wave setup",
            param_values={
                "num_modes_x": 1,
                "num_modes_y": 0,
                "num_modes_z": 0,
                "angle_between_k_b0": 90,
                "machine_precision_target": 0,
                "nx_max": _NX_MAX,
            },
        ),
    )


## } MODULE
