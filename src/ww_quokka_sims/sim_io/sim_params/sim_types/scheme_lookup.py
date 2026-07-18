## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from enum import Enum
from typing import cast

## personal
from jormi.ww_validation import validate_enums

##
## === SCHEME ENUMS
## Shorthand scheme letters, as used in dataset directory/combo names (e.g. `q26-b25-ppm_ep`),
## mapped to the values Quokka's MHD implementation actually expects. Shared across every
## problem-type profile in `sim_types/`, since these mappings are universal, not per-problem.
##
## Enum members over plain dicts: an invalid letter (e.g. a typo) is caught by
## `resolve_member` with a clear error listing valid names, rather than a bare `KeyError`.
##


class InterpolationScheme(int, Enum):
    """`hydro.reconstruction_order` / `mhd.emf_reconstruction_order` by interpolation letter."""

    PCM = 1
    PLM = 2
    PPM = 3
    PPM_EP = 5


class EMFComputeScheme(str, Enum):
    """`mhd.emf_compute_scheme` by compute-scheme letter."""

    Q26 = "Quokka2026"
    FS17 = "FelkerStone2017"
    B25 = "Balsara2025"


class EMFAveragingScheme(str, Enum):
    """`mhd.emf_averaging_scheme` by averaging-scheme letter."""

    B25 = "Balsara2025"
    LD04 = "LondrilloDelZanna2004"


##
## === LETTER RESOLUTION
##


def interpolation_by_letter(
    letter: str,
) -> InterpolationScheme:
    return cast(
        InterpolationScheme,
        validate_enums.resolve_member(member=letter, valid_enums=InterpolationScheme),
    )


def emf_compute_scheme_by_letter(
    letter: str,
) -> EMFComputeScheme:
    return cast(
        EMFComputeScheme,
        validate_enums.resolve_member(member=letter, valid_enums=EMFComputeScheme),
    )


def emf_averaging_scheme_by_letter(
    letter: str,
) -> EMFAveragingScheme:
    return cast(
        EMFAveragingScheme,
        validate_enums.resolve_member(member=letter, valid_enums=EMFAveragingScheme),
    )


## } MODULE
