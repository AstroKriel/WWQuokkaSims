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
##


class InterpolationScheme(int, Enum):
    """`hydro.reconstruction_order` / `mhd.emf_reconstruction_order` by interpolation key."""

    PCM = 1
    PLM = 2
    PPM = 3
    PPM_EP = 5


class EMFComputeScheme(str, Enum):
    """`mhd.emf_compute_scheme` by compute-scheme key."""

    Q26 = "Quokka2026"
    FS17 = "FelkerStone2017"
    B25 = "Balsara2025"


class EMFAveragingScheme(str, Enum):
    """`mhd.emf_averaging_scheme` by averaging-scheme key."""

    B25 = "Balsara2025"
    LD04 = "LondrilloDelZanna2004"


##
## === KEY RESOLUTION
##


def resolve_interpolation_scheme(
    key: str,
) -> InterpolationScheme:
    """Resolve `key` (a name, value, or shorthand combo-name key) to an `InterpolationScheme` member."""
    return cast(
        InterpolationScheme,
        validate_enums.resolve_member(member=key, valid_enums=InterpolationScheme),
    )


def resolve_emf_compute_scheme(
    key: str,
) -> EMFComputeScheme:
    """Resolve `key` (a name, value, or shorthand combo-name key) to an `EMFComputeScheme` member."""
    return cast(
        EMFComputeScheme,
        validate_enums.resolve_member(member=key, valid_enums=EMFComputeScheme),
    )


def resolve_emf_averaging_scheme(
    key: str,
) -> EMFAveragingScheme:
    """Resolve `key` (a name, value, or shorthand combo-name key) to an `EMFAveragingScheme` member."""
    return cast(
        EMFAveragingScheme,
        validate_enums.resolve_member(member=key, valid_enums=EMFAveragingScheme),
    )


## } MODULE
