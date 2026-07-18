## { MODULE

##
## === DEPENDENCIES
##

## local
from ..models import (
    GeometryParams,
    HydroParams,
    MHDParams,
    OutputParams,
    ResolutionParams,
    SetupParams,
    TimeIntegrationParams,
)
from ..write import SimParamsBundle
## importing names directly from the sibling module (not `from . import scheme_lookup`) avoids
## a `sim_types/__init__.py` import cycle, since that would require the package's `__init__`
## to finish executing before this module (which it imports) can run
from .scheme_lookup import (
    emf_averaging_scheme_by_key,
    emf_compute_scheme_by_key,
    interpolation_by_key,
)

##
## === CONSTANTS
##

PROBLEM_TYPE = "FastWaveConvergence"

## fixed for every combo of this problem type: only the EMF/reconstruction scheme varies
## across a sweep (see `build_combo`); everything below is boilerplate the Quokka problem
## generator itself reads, unrelated to which scheme is under test
_BASE_N_CELL = (128, 8, 8)
_NX_MAX = 2048

##
## === PROGRAM MAIN
##


def build_combo(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    interpolation: str,
) -> SimParamsBundle:
    """Build the full parameter set for one `FastWaveConvergence` scheme combo (e.g. `q26-b25-ppm_ep`)."""
    ## `.value` unwraps the Enum to the plain int/str `sim_params.models` dataclasses declare;
    ## an Enum member's `str()`/f-string form is `"ClassName.MEMBER"`, not its underlying value
    reconstruction_order = interpolation_by_key(interpolation).value
    return SimParamsBundle(
        geometry=GeometryParams(
            prob_lo=(0.0, 0.0, 0.0),
            prob_hi=(1.0, 1.0, 1.0),
            is_periodic=(1, 1, 1),
        ),
        resolution=ResolutionParams(
            n_cell=_BASE_N_CELL,
            blocking_factor=(16, 8, 8),
            max_grid_size=128,
        ),
        output=OutputParams(
            plotfile_interval=-1,
        ),
        time_integration=TimeIntegrationParams(
            cfl=0.3,
            do_reflux=0,
            do_subcycle=0,
            do_tracers=1,
        ),
        hydro=HydroParams(
            rk_integrator_order=2,
            reconstruction_order=reconstruction_order,
            use_dual_energy=0,
        ),
        mhd=MHDParams(
            emf_compute_scheme=emf_compute_scheme_by_key(compute_scheme_key).value,
            emf_averaging_scheme=emf_averaging_scheme_by_key(averaging_scheme_key).value,
            emf_reconstruction_order=reconstruction_order,
        ),
        setup=SetupParams(
            group_title="wave setup",
            values={
                "num_modes_x": 1,
                "num_modes_y": 0,
                "num_modes_z": 0,
                "angle_between_k_b0": 90,
                "machine_precision_target": 0,
                "nx_max": _NX_MAX,
            },
        ),
        problem_type=PROBLEM_TYPE,
    )


## } MODULE
