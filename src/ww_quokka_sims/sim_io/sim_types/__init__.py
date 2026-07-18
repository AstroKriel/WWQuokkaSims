from . import fast_wave_convergence as fast_wave_convergence
from . import scheme_lookup as scheme_lookup

## registry: Quokka problem-generator class name -> its profile's `build_combo` function.
## Add one entry per new file added under `sim_types/`.
PROFILES_BY_PROBLEM_TYPE = {
    fast_wave_convergence.PROBLEM_TYPE: fast_wave_convergence.build_combo,
}
