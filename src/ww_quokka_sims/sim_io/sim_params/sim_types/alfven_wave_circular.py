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

PROBLEM_NAME = "AlfvenWaveCircular"

## `problem_main()` defaults to `setup.run_convergence=true` (Richardson convergence sweep,
## `runWaveTest`), which overrides `amr.n_cell`/`geometry.*` and `sim.stopTime_`/`maxTimesteps_`
## internally for every sweep iteration, ignoring whatever the toml sets (see
## threads/dead-toml-params/); domain/resolution below are fixed to match what the real
## reference files render, not because they control anything under the default mode
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
    """Build the full parameter set for one `AlfvenWaveCircular` scheme combo (e.g. `b25-b25-pcm`)."""
    ## `.value` unwraps the Enum to the plain int/str `sim_params.param_groups` dataclasses declare;
    ## an Enum member's `str()`/f-string form is `"ClassName.MEMBER"`, not its underlying value
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
        ## `AlfvenWaveCircular` asserts `mhd.resistivity == 0` and aborts otherwise; no `resistivity`
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
