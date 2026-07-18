## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass, field
from typing import Any

## personal
from jormi.ww_validation import validate_types

##
## === HELPERS
##


def _ensure_axis_triple(
    param: object,
    *,
    param_name: str,
) -> None:
    validate_types.ensure_tuple_of_numbers(
        param=param,
        param_name=param_name,
        seq_length=3,
    )


def _ensure_scalar_or_axis_triple(
    param: object,
    *,
    param_name: str,
) -> None:
    if isinstance(param, tuple):
        _ensure_axis_triple(param, param_name=param_name)
        return
    validate_types.ensure_finite_int(
        param=param,
        param_name=param_name,
        require_positive=True,
        allow_zero=False,
    )


##
## === GEOMETRY
##


@dataclass(frozen=True)
class GeometryParams:
    """
    Domain bounds and boundary conditions.

    Exactly one of `is_periodic` or `bc` must be set: `is_periodic` for a fully
    periodic domain, `bc` (rendered as `quokka.bc`) when any boundary is `ext_dir`.
    """

    prob_lo: tuple[float, float, float]
    prob_hi: tuple[float, float, float]
    is_periodic: tuple[int, int, int] | None = None
    bc: tuple[str, str, str] | None = None

    def __post_init__(
        self,
    ) -> None:
        _ensure_axis_triple(self.prob_lo, param_name="<prob_lo>")
        _ensure_axis_triple(self.prob_hi, param_name="<prob_hi>")
        if (self.is_periodic is None) == (self.bc is None):
            raise ValueError(
                "exactly one of `<is_periodic>` or `<bc>` must be set, "
                f"got is_periodic={self.is_periodic!r}, bc={self.bc!r}.",
            )
        if self.is_periodic is not None:
            validate_types.ensure_tuple_of_ints(
                param=self.is_periodic,
                param_name="<is_periodic>",
                seq_length=3,
            )
        if self.bc is not None:
            validate_types.ensure_tuple_of_strings(
                param=self.bc,
                param_name="<bc>",
                seq_length=3,
            )


##
## === RESOLUTION
##


@dataclass(frozen=True)
class ResolutionParams:
    """
    Grid resolution and domain decomposition.

    `blocking_factor` and `max_grid_size` accept either a single value (applied to
    all three axes) or a per-axis `(x, y, z)` tuple; the builder always renders
    these explicitly per axis (`amr.blocking_factor_x/_y/_z`), never as a bare
    fallback key, since AMReX interprets a bare `amr.blocking_factor = ...` array
    as one value per AMR *level*, not per axis.
    """

    n_cell: tuple[int, int, int]
    blocking_factor: int | tuple[int, int, int]
    max_grid_size: int | tuple[int, int, int]
    max_level: int = 0
    n_error_buf: int | None = None

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_tuple_of_ints(
            param=self.n_cell,
            param_name="<n_cell>",
            seq_length=3,
        )
        for value in self.n_cell:
            if value < 8:
                raise ValueError(
                    f"`<n_cell>` entries must be >= 8 (AMReX minimum under periodic BCs), got: {self.n_cell}.",
                )
        _ensure_scalar_or_axis_triple(self.blocking_factor, param_name="<blocking_factor>")
        _ensure_scalar_or_axis_triple(self.max_grid_size, param_name="<max_grid_size>")
        validate_types.ensure_finite_int(
            param=self.max_level,
            param_name="<max_level>",
            require_positive=True,
            allow_zero=True,
        )
        if (self.n_error_buf is not None) and (self.max_level == 0):
            raise ValueError(
                "`<n_error_buf>` only applies when `<max_level>` > 0 (AMR refinement enabled).",
            )


##
## === OUTPUT
##


@dataclass(frozen=True)
class OutputParams:
    """Checkpoint and plotfile cadence. Exactly one of `plotfile_interval` or `plottime_interval` must be set."""

    plotfile_prefix: str | None = None
    plotfile_interval: int | None = None
    plottime_interval: float | None = None
    checkpoint_interval: int | None = None
    checkpoint_prefix: str | None = None

    def __post_init__(
        self,
    ) -> None:
        if (self.plotfile_interval is None) == (self.plottime_interval is None):
            raise ValueError(
                "exactly one of `<plotfile_interval>` or `<plottime_interval>` must be set, got "
                f"plotfile_interval={self.plotfile_interval!r}, plottime_interval={self.plottime_interval!r}.",
            )
        if self.plotfile_prefix is not None:
            validate_types.ensure_nonempty_string(
                param=self.plotfile_prefix,
                param_name="<plotfile_prefix>",
            )
        if self.checkpoint_interval is not None:
            validate_types.ensure_finite_int(
                param=self.checkpoint_interval,
                param_name="<checkpoint_interval>",
            )


##
## === TIME INTEGRATION
##


@dataclass(frozen=True)
class TimeIntegrationParams:
    """CFL and stop-condition parameters."""

    cfl: float
    do_reflux: int | None = None
    do_subcycle: int | None = None
    do_tracers: int | None = None
    stop_time: float | None = None
    max_timesteps: int | None = None

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_in_bounds(
            param=self.cfl,
            param_name="<cfl>",
            min_value=0.0,
            max_value=1.0,
        )


##
## === HYDRO
##


@dataclass(frozen=True)
class HydroParams:
    """Reconstruction scheme: pcm=1, plm=2, ppm=3, ppm_ep=5."""

    rk_integrator_order: int
    reconstruction_order: int
    use_dual_energy: int | None = None

    def __post_init__(
        self,
    ) -> None:
        if self.reconstruction_order not in (1, 2, 3, 5):
            raise ValueError(
                f"`<reconstruction_order>` must be one of (1, 2, 3, 5), got: {self.reconstruction_order}.",
            )


##
## === MHD
##

VALID_EMF_COMPUTE_SCHEMES = ("Quokka2026", "FelkerStone2017", "Balsara2025")
VALID_EMF_AVERAGING_SCHEMES = ("Balsara2025", "LondrilloDelZanna2004")


@dataclass(frozen=True)
class MHDParams:
    """EMF reconstruction and averaging scheme; `resistivity` is only set for resistive-correctness runs."""

    emf_compute_scheme: str
    emf_averaging_scheme: str
    emf_reconstruction_order: int
    resistivity: float | None = None

    def __post_init__(
        self,
    ) -> None:
        if self.emf_compute_scheme not in VALID_EMF_COMPUTE_SCHEMES:
            raise ValueError(
                f"`<emf_compute_scheme>` must be one of {VALID_EMF_COMPUTE_SCHEMES}, got: {self.emf_compute_scheme!r}.",
            )
        if self.emf_averaging_scheme not in VALID_EMF_AVERAGING_SCHEMES:
            raise ValueError(
                f"`<emf_averaging_scheme>` must be one of {VALID_EMF_AVERAGING_SCHEMES}, got: {self.emf_averaging_scheme!r}.",
            )
        if self.emf_reconstruction_order not in (1, 2, 3, 5):
            raise ValueError(
                f"`<emf_reconstruction_order>` must be one of (1, 2, 3, 5), got: {self.emf_reconstruction_order}.",
            )


##
## === SETUP (problem-specific, always rendered last)
##


@dataclass(frozen=True)
class SetupParams:
    """
    Problem-specific parameters, rendered last under a caller-chosen section title
    (e.g. `"wave setup"`, `"problem setup"`), matching the two titles seen in the corpus.
    Keys are rendered under a caller-chosen prefix (default `"setup"`); at least one
    known problem (field-loop) uses its own prefix instead of `"setup"`.
    """

    values: dict[str, Any] = field(default_factory=dict)
    title: str = "problem setup"
    key_prefix: str = "setup"

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_dict(
            param=self.values,
            param_name="<values>",
        )
        if not self.values:
            raise ValueError("`<values>` must be non-empty; omit `setup` entirely instead of passing an empty one.")
        validate_types.ensure_nonempty_string(
            param=self.title,
            param_name="<title>",
        )
        validate_types.ensure_nonempty_string(
            param=self.key_prefix,
            param_name="<key_prefix>",
        )


## } MODULE
