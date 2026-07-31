from collections.abc import Callable

from .. import write_param_groups
from . import alfven_wave_circular as alfven_wave_circular
from . import alfven_wave_linear as alfven_wave_linear
from . import balsara_vortex as balsara_vortex
from . import blast_wave as blast_wave
from . import brio_wu_shock_tube as brio_wu_shock_tube
from . import current_sheet as current_sheet
from . import fast_wave_convergence as fast_wave_convergence
from . import field_loop as field_loop
from . import mhd_quirk as mhd_quirk
from . import orszag_tang as orszag_tang
from . import ryu_jones_2a_shock_tube as ryu_jones_2a_shock_tube
from . import scheme_lookup as scheme_lookup

## add one entry per new file added under `sim_types/`
_SIM_PARAMS_BUILDER_LOOKUP = {
    alfven_wave_circular.PROBLEM_NAME: alfven_wave_circular.build_sim_params,
    alfven_wave_linear.PROBLEM_NAME: alfven_wave_linear.build_sim_params,
    balsara_vortex.PROBLEM_NAME: balsara_vortex.build_sim_params,
    blast_wave.PROBLEM_NAME: blast_wave.build_sim_params,
    brio_wu_shock_tube.PROBLEM_NAME: brio_wu_shock_tube.build_sim_params,
    current_sheet.PROBLEM_NAME: current_sheet.build_sim_params,
    fast_wave_convergence.PROBLEM_NAME: fast_wave_convergence.build_sim_params,
    field_loop.PROBLEM_NAME: field_loop.build_sim_params,
    mhd_quirk.PROBLEM_NAME: mhd_quirk.build_sim_params,
    orszag_tang.PROBLEM_NAME: orszag_tang.build_sim_params,
    ryu_jones_2a_shock_tube.PROBLEM_NAME: ryu_jones_2a_shock_tube.build_sim_params,
}


def resolve_sim_params_builder(
    problem_name: str,
) -> Callable[..., write_param_groups.SimParams]:
    """Resolve `problem_name` to its profile's `build_sim_params` function."""
    if problem_name not in _SIM_PARAMS_BUILDER_LOOKUP:
        raise ValueError(
            f"unknown problem_name {problem_name!r}; expected one of "
            f"{sorted(_SIM_PARAMS_BUILDER_LOOKUP)}.",
        )
    return _SIM_PARAMS_BUILDER_LOOKUP[problem_name]
