from collections.abc import Callable
from enum import Enum
from typing import cast

from jormi.ww_validation import validate_enums

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


class ProblemKey(str, Enum):
    ALFVEN_WAVE_CIRCULAR_CONVERGENCE = alfven_wave_circular_convergence.PROBLEM_KEY
    ALFVEN_WAVE_CIRCULAR_CORRECTNESS = alfven_wave_circular_correctness.PROBLEM_KEY
    ALFVEN_WAVE_LINEAR_CONVERGENCE = alfven_wave_linear_convergence.PROBLEM_KEY
    ALFVEN_WAVE_LINEAR_CORRECTNESS = alfven_wave_linear_correctness.PROBLEM_KEY
    MHD_BALSARA_VORTEX = balsara_vortex.PROBLEM_KEY
    MHD_BLAST = blast_wave.PROBLEM_KEY
    BRIO_WU_SHOCK_TUBE = brio_wu_shock_tube.PROBLEM_KEY
    CURRENT_SHEET = current_sheet.PROBLEM_KEY
    FAST_WAVE_CONVERGENCE = fast_wave_convergence.PROBLEM_KEY
    FAST_WAVE_CORRECTNESS = fast_wave_correctness.PROBLEM_KEY
    FIELD_LOOP = field_loop.PROBLEM_KEY
    MHD_QUIRK = mhd_quirk.PROBLEM_KEY
    ORSZAG_TANG = orszag_tang.PROBLEM_KEY
    RYU_JONES_2A_SHOCK_TUBE = ryu_jones_2a_shock_tube.PROBLEM_KEY
    SLOW_WAVE_CONVERGENCE = slow_wave_convergence.PROBLEM_KEY
    SLOW_WAVE_CORRECTNESS = slow_wave_correctness.PROBLEM_KEY


## add one entry per new file added under `sim_types/`; `PROBLEM_KEY` is `[problem_name]` or,
## when a problem has multiple modes on one Quokka binary, `[problem_name]-[mode]`.
_SIM_PARAMS_BUILDER_LOOKUP = {
    ProblemKey.ALFVEN_WAVE_CIRCULAR_CONVERGENCE: alfven_wave_circular_convergence.build_sim_params,
    ProblemKey.ALFVEN_WAVE_CIRCULAR_CORRECTNESS: alfven_wave_circular_correctness.build_sim_params,
    ProblemKey.ALFVEN_WAVE_LINEAR_CONVERGENCE: alfven_wave_linear_convergence.build_sim_params,
    ProblemKey.ALFVEN_WAVE_LINEAR_CORRECTNESS: alfven_wave_linear_correctness.build_sim_params,
    ProblemKey.MHD_BALSARA_VORTEX: balsara_vortex.build_sim_params,
    ProblemKey.MHD_BLAST: blast_wave.build_sim_params,
    ProblemKey.BRIO_WU_SHOCK_TUBE: brio_wu_shock_tube.build_sim_params,
    ProblemKey.CURRENT_SHEET: current_sheet.build_sim_params,
    ProblemKey.FAST_WAVE_CONVERGENCE: fast_wave_convergence.build_sim_params,
    ProblemKey.FAST_WAVE_CORRECTNESS: fast_wave_correctness.build_sim_params,
    ProblemKey.FIELD_LOOP: field_loop.build_sim_params,
    ProblemKey.MHD_QUIRK: mhd_quirk.build_sim_params,
    ProblemKey.ORSZAG_TANG: orszag_tang.build_sim_params,
    ProblemKey.RYU_JONES_2A_SHOCK_TUBE: ryu_jones_2a_shock_tube.build_sim_params,
    ProblemKey.SLOW_WAVE_CONVERGENCE: slow_wave_convergence.build_sim_params,
    ProblemKey.SLOW_WAVE_CORRECTNESS: slow_wave_correctness.build_sim_params,
}


def resolve_sim_params_builder(
    problem_key: ProblemKey | str,
) -> Callable[..., write_param_groups.SimParams]:
    """Resolve `problem_key` (a `ProblemKey` member or its string value) to its profile's `build_sim_params`."""
    resolved_key = cast(
        ProblemKey, validate_enums.resolve_member(
            member=problem_key,
            valid_enums=ProblemKey,
        )
    )
    return _SIM_PARAMS_BUILDER_LOOKUP[resolved_key]
