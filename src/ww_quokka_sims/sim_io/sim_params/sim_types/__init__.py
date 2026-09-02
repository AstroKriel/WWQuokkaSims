from collections.abc import Callable
from enum import Enum

from .. import _save_params
from . import alfven_wave_circular_convergence as alfven_wave_circular_convergence
from . import alfven_wave_circular_correctness as alfven_wave_circular_correctness
from . import alfven_wave_linear_convergence as alfven_wave_linear_convergence
from . import alfven_wave_linear_correctness as alfven_wave_linear_correctness
from . import blast_wave as blast_wave
from . import carbuncle as carbuncle
from . import current_sheet as current_sheet
from . import fast_wave_convergence as fast_wave_convergence
from . import fast_wave_correctness as fast_wave_correctness
from . import magnetic_loop as magnetic_loop
from . import shock_tube_brio_wu as shock_tube_brio_wu
from . import shock_tube_ryu_jones_2a as shock_tube_ryu_jones_2a
from . import slow_wave_convergence as slow_wave_convergence
from . import slow_wave_correctness as slow_wave_correctness
from . import vortex_balsara as vortex_balsara
from . import vortex_orszag_tang as vortex_orszag_tang


class ProblemSetup(Enum):
    """Each member's value is the module defining that specific problem setup; add one member per new file added under `sim_types/`."""

    ALFVEN_WAVE_CIRCULAR_CONVERGENCE = alfven_wave_circular_convergence
    ALFVEN_WAVE_CIRCULAR_CORRECTNESS = alfven_wave_circular_correctness
    ALFVEN_WAVE_LINEAR_CONVERGENCE = alfven_wave_linear_convergence
    ALFVEN_WAVE_LINEAR_CORRECTNESS = alfven_wave_linear_correctness
    BLAST_WAVE = blast_wave
    CARBUNCLE = carbuncle
    CURRENT_SHEET = current_sheet
    FAST_WAVE_CONVERGENCE = fast_wave_convergence
    FAST_WAVE_CORRECTNESS = fast_wave_correctness
    MAGNETIC_LOOP = magnetic_loop
    SHOCK_TUBE_BRIO_WU = shock_tube_brio_wu
    SHOCK_TUBE_RYU_JONES_2A = shock_tube_ryu_jones_2a
    SLOW_WAVE_CONVERGENCE = slow_wave_convergence
    SLOW_WAVE_CORRECTNESS = slow_wave_correctness
    VORTEX_BALSARA = vortex_balsara
    VORTEX_ORSZAG_TANG = vortex_orszag_tang

    @property
    def build_sim_params(
        self,
    ) -> Callable[..., _save_params.SimParams]:
        return self.value.build_sim_params


def resolve_sim_params_builder(
    problem_setup: ProblemSetup,
) -> Callable[..., _save_params.SimParams]:
    """Resolve `problem_setup` to its profile's `build_sim_params`."""
    return problem_setup.build_sim_params
