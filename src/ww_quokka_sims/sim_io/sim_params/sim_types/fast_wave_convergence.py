## { MODULE

##
## === DEPENDENCIES
##

## local
from ..param_groups import (
    GeometryParams,
    HydroParams,
    MHDParams,
    OutputFileParams,
    ResolutionParams,
    SetupParams,
    TimeIntegrationParams,
)
from ..write_param_groups import SimParams
## direct names, not `from . import scheme_lookup`: avoids a `sim_types/__init__.py` import cycle
from .scheme_lookup import (
    emf_averaging_scheme_by_key,
    emf_compute_scheme_by_key,
    interpolation_by_key,
)

##
## === CONSTANTS
##

PROBLEM_NAME = "FastWaveConvergence"

_BASE_NUM_CELLS = (128, 8, 8)
_NX_MAX = 2048

##
## === PROGRAM MAIN
##


def build_combo(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    interpolation: str,
) -> SimParams:
    """Build the full parameter set for one `FastWaveConvergence` scheme combo (e.g. `q26-b25-ppm_ep`)."""
    ## `.value` unwraps the Enum to the plain int/str `sim_params.param_groups` dataclasses declare;
    ## an Enum member's `str()`/f-string form is `"ClassName.MEMBER"`, not its underlying value
    interpolation_order = interpolation_by_key(interpolation).value
    return SimParams(
        geometry_params=GeometryParams(
            domain_lo=(0.0, 0.0, 0.0),
            domain_hi=(1.0, 1.0, 1.0),
            is_boundary_periodic=(1, 1, 1),
        ),
        resolution_params=ResolutionParams(
            num_cells=_BASE_NUM_CELLS,
            blocking_factor=(16, 8, 8),
            max_grid_size=128,
        ),
        output_file_params=OutputFileParams(
            snapshot_index_interval=-1,
        ),
        time_integration_params=TimeIntegrationParams(
            cfl=0.3,
            use_reflux=0,
            use_subcycle=0,
            use_tracers=1,
        ),
        hydro_params=HydroParams(
            integrator_order=2,
            interpolation_order=interpolation_order,
            use_dual_energy=0,
        ),
        ## no `resistivity`: not physically meaningful for this ideal-MHD wave test
        mhd_params=MHDParams(
            emf_compute_scheme=emf_compute_scheme_by_key(compute_scheme_key).value,
            emf_averaging_scheme=emf_averaging_scheme_by_key(averaging_scheme_key).value,
            interpolation_order=interpolation_order,
        ),
        setup_params=SetupParams(
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
