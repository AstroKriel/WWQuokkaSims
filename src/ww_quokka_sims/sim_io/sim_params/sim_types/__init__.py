from collections.abc import Callable
from enum import Enum

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


class ProblemKey(Enum):
    """Each member's value is its profile module; add one member per new file added under `sim_types/`."""

    ALFVEN_WAVE_CIRCULAR_CONVERGENCE = alfven_wave_circular_convergence
    ALFVEN_WAVE_CIRCULAR_CORRECTNESS = alfven_wave_circular_correctness
    ALFVEN_WAVE_LINEAR_CONVERGENCE = alfven_wave_linear_convergence
    ALFVEN_WAVE_LINEAR_CORRECTNESS = alfven_wave_linear_correctness
    MHD_BALSARA_VORTEX = balsara_vortex
    MHD_BLAST = blast_wave
    BRIO_WU_SHOCK_TUBE = brio_wu_shock_tube
    CURRENT_SHEET = current_sheet
    FAST_WAVE_CONVERGENCE = fast_wave_convergence
    FAST_WAVE_CORRECTNESS = fast_wave_correctness
    FIELD_LOOP = field_loop
    MHD_QUIRK = mhd_quirk
    ORSZAG_TANG = orszag_tang
    RYU_JONES_2A_SHOCK_TUBE = ryu_jones_2a_shock_tube
    SLOW_WAVE_CONVERGENCE = slow_wave_convergence
    SLOW_WAVE_CORRECTNESS = slow_wave_correctness

    @property
    def problem_key(
        self,
    ) -> str:
        return self.value.PROBLEM_KEY

    @property
    def build_sim_params(
        self,
    ) -> Callable[..., write_param_groups.SimParams]:
        return self.value.build_sim_params


def resolve_sim_params_builder(
    problem_key: ProblemKey | str,
) -> Callable[..., write_param_groups.SimParams]:
    """Resolve `problem_key` (a `ProblemKey` member or its string value) to its profile's `build_sim_params`."""
    if isinstance(problem_key, ProblemKey):
        return problem_key.build_sim_params
    for member in ProblemKey:
        if member.problem_key == problem_key:
            return member.build_sim_params
    raise ValueError(
        f"unknown problem_key {problem_key!r}; expected one of "
        f"{sorted(member.problem_key for member in ProblemKey)}.",
    )
