## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import enum
import typing

## personal
from jormi.ww_validation import validate_enums

##
## === RECONSTRUCTION SCHEME
##


class ReconstructionScheme(enum.IntEnum):
    """`hydro.reconstruction_order` / `mhd.emf_reconstruction_order` by reconstruction key."""

    PCM = 1
    PLM = 2
    PPM = 3
    PPM_EP = 5


def resolve_reconstruction_scheme(
    *,
    key: str,
) -> ReconstructionScheme:
    """Resolve `key` (a name, value, or shorthand combination-name key) to a `ReconstructionScheme` member."""
    return typing.cast(
        ReconstructionScheme,
        validate_enums.resolve_member(
            member=key,
            valid_enums=ReconstructionScheme,
        ),
    )


##
## === EMF COMPUTE SCHEME
##


class EMFComputeScheme(enum.StrEnum):
    """`mhd.emf_compute_scheme` by compute-scheme key."""

    Q26 = "Quokka2026"
    FS17 = "FelkerStone2017"
    B25 = "Balsara2025"


def resolve_emf_compute_scheme(
    *,
    key: str,
) -> EMFComputeScheme:
    """Resolve `key` (a name, value, or shorthand combination-name key) to an `EMFComputeScheme` member."""
    return typing.cast(
        EMFComputeScheme,
        validate_enums.resolve_member(
            member=key,
            valid_enums=EMFComputeScheme,
        ),
    )


##
## === EMF AVERAGING SCHEME
##


class EMFAveragingScheme(enum.StrEnum):
    """`mhd.emf_averaging_scheme` by averaging-scheme key."""

    B25 = "Balsara2025"
    LD04 = "LondrilloDelZanna2004"


def resolve_emf_averaging_scheme(
    *,
    key: str,
) -> EMFAveragingScheme:
    """Resolve `key` (a name, value, or shorthand combination-name key) to an `EMFAveragingScheme` member."""
    return typing.cast(
        EMFAveragingScheme,
        validate_enums.resolve_member(
            member=key,
            valid_enums=EMFAveragingScheme,
        ),
    )


## } MODULE
