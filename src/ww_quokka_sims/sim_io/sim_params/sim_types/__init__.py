from collections.abc import Callable

from .. import write_param_groups
from . import alfven_wave_circular_convergence as alfven_wave_circular_convergence
from . import alfven_wave_circular_correctness as alfven_wave_circular_correctness
from . import alfven_wave_linear_convergence as alfven_wave_linear_convergence
from . import alfven_wave_linear_correctness as alfven_wave_linear_correctness
from . import balsara_vortex as balsara_vortex
from . import blast_wave as blast_wave
from . import brio_wu_shock_tube as brio_wu_shock_tube
from . import current_sheet as current_sheet
from . import fast_wave_convergence as fast_wave_convergence
from . import fast_wave_correctness as fast_wave_correctness
from . import field_loop as field_loop
from . import mhd_quirk as mhd_quirk
from . import orszag_tang as orszag_tang
from . import ryu_jones_2a_shock_tube as ryu_jones_2a_shock_tube
from . import scheme_lookup as scheme_lookup
from . import slow_wave_convergence as slow_wave_convergence
from . import slow_wave_correctness as slow_wave_correctness

## add one entry per new file added under `sim_types/`; `PROBLEM_KEY` is `[problem_name]` or,
## when a problem has multiple modes on one Quokka binary, `[problem_name]-[mode]`.
_SIM_PARAMS_BUILDER_LOOKUP = {
    alfven_wave_circular_convergence.PROBLEM_KEY: alfven_wave_circular_convergence.build_sim_params,
    alfven_wave_circular_correctness.PROBLEM_KEY: alfven_wave_circular_correctness.build_sim_params,
    alfven_wave_linear_convergence.PROBLEM_KEY: alfven_wave_linear_convergence.build_sim_params,
    alfven_wave_linear_correctness.PROBLEM_KEY: alfven_wave_linear_correctness.build_sim_params,
    balsara_vortex.PROBLEM_KEY: balsara_vortex.build_sim_params,
    blast_wave.PROBLEM_KEY: blast_wave.build_sim_params,
    brio_wu_shock_tube.PROBLEM_KEY: brio_wu_shock_tube.build_sim_params,
    current_sheet.PROBLEM_KEY: current_sheet.build_sim_params,
    fast_wave_convergence.PROBLEM_KEY: fast_wave_convergence.build_sim_params,
    fast_wave_correctness.PROBLEM_KEY: fast_wave_correctness.build_sim_params,
    field_loop.PROBLEM_KEY: field_loop.build_sim_params,
    mhd_quirk.PROBLEM_KEY: mhd_quirk.build_sim_params,
    orszag_tang.PROBLEM_KEY: orszag_tang.build_sim_params,
    ryu_jones_2a_shock_tube.PROBLEM_KEY: ryu_jones_2a_shock_tube.build_sim_params,
    slow_wave_convergence.PROBLEM_KEY: slow_wave_convergence.build_sim_params,
    slow_wave_correctness.PROBLEM_KEY: slow_wave_correctness.build_sim_params,
}


def resolve_sim_params_builder(
    problem_key: str,
) -> Callable[..., write_param_groups.SimParams]:
    """Resolve `problem_key` to its profile's `build_sim_params` function."""
    if problem_key not in _SIM_PARAMS_BUILDER_LOOKUP:
        raise ValueError(
            f"unknown problem_key {problem_key!r}; expected one of "
            f"{sorted(_SIM_PARAMS_BUILDER_LOOKUP)}.",
        )
    return _SIM_PARAMS_BUILDER_LOOKUP[problem_key]
