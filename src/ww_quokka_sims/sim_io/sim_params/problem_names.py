## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from enum import Enum

##
## === KNOWN PROBLEM NAMES
## Canonical Quokka C++ problem-generator class names. Defined once here so `write.py`'s
## guardrails and each `sim_types/` profile module reference the same member, rather than
## independently retyping a matching string that could silently drift out of sync. Lives
## alongside `models.py` (not inside `sim_types/`) since `write.py` cannot import anything
## under `sim_types/` without cycling back through `sim_types/__init__.py` -> a profile
## module -> `write.py`.
##


class ProblemName(str, Enum):
    FAST_WAVE_CONVERGENCE = "FastWaveConvergence"


## } MODULE
