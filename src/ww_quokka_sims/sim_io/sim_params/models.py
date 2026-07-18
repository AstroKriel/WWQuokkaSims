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

    Fields
    ---
    - `prob_lo`:
        Lower domain corner, `(x, y, z)`.

    - `prob_hi`:
        Upper domain corner, `(x, y, z)`.

    - `is_periodic`:
        Per-axis periodicity flags; mutually exclusive with `bc`.

    - `bc`:
        Per-axis boundary condition names (e.g. `"ext_dir"`, `"periodic"`); mutually exclusive with `is_periodic`.
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

    Fields
    ---
    - `n_cell`:
        Base-level cell count per axis; each entry must be >= 8.

    - `blocking_factor`:
        Minimum grid tile size; scalar (applied to all axes) or per-axis.

    - `max_grid_size`:
        Maximum grid tile size; scalar (applied to all axes) or per-axis.

    - `max_level`:
        Number of AMR refinement levels above the base grid; `0` disables AMR.

    - `n_error_buf`:
        Refinement buffer cell count; only valid when `max_level` > 0.
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
                    f"`<n_cell>` entries must be >= 8 (AMReX minimum under periodic BCs), got {self.n_cell}.",
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
    """
    Checkpoint and plotfile cadence. Exactly one of `plotfile_interval` or `plottime_interval` must be set.

    Fields
    ---
    - `plotfile_prefix`:
        Directory/filename prefix for plotfiles; omit to use Quokka's default.

    - `plotfile_interval`:
        Emit a plotfile every N steps; mutually exclusive with `plottime_interval`.

    - `plottime_interval`:
        Emit a plotfile every `plottime_interval` of simulation time; mutually exclusive with `plotfile_interval`.

    - `checkpoint_interval`:
        Emit a checkpoint every N steps; omit to disable checkpointing.

    - `checkpoint_prefix`:
        Directory/filename prefix for checkpoints; omit to use Quokka's default.
    """

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
    """
    CFL and stop-condition parameters.

    Fields
    ---
    - `cfl`:
        Courant number, in `[0.0, 1.0]`.

    - `do_reflux`:
        `1` to reflux at coarse-fine boundaries, `0` to disable; omit for Quokka's default.

    - `do_subcycle`:
        `1` to subcycle in time across AMR levels, `0` to disable; omit for Quokka's default.

    - `do_tracers`:
        `1` to advect tracer particles, `0` to disable; omit for Quokka's default.

    - `stop_time`:
        Simulation time at which to stop; omit to run until `max_timesteps`.

    - `max_timesteps`:
        Maximum number of timesteps to run; omit to run until `stop_time`.
    """

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
    """
    Reconstruction scheme: pcm=1, plm=2, ppm=3, ppm_ep=5.

    Fields
    ---
    - `rk_integrator_order`:
        Runge-Kutta time-integrator order.

    - `reconstruction_order`:
        Spatial reconstruction order; must be one of `(1, 2, 3, 5)`.

    - `use_dual_energy`:
        `1` to enable the dual-energy formalism, `0` to disable; omit for Quokka's default.
    """

    rk_integrator_order: int
    reconstruction_order: int
    use_dual_energy: int | None = None

    def __post_init__(
        self,
    ) -> None:
        if self.reconstruction_order not in (1, 2, 3, 5):
            raise ValueError(
                f"`<reconstruction_order>` must be one of (1, 2, 3, 5); got {self.reconstruction_order}.",
            )


##
## === MHD
##

VALID_EMF_COMPUTE_SCHEMES = ("Quokka2026", "FelkerStone2017", "Balsara2025")
VALID_EMF_AVERAGING_SCHEMES = ("Balsara2025", "LondrilloDelZanna2004")


@dataclass(frozen=True)
class MHDParams:
    """
    EMF reconstruction and averaging scheme; `resistivity` is only set for resistive-correctness runs.

    Fields
    ---
    - `emf_compute_scheme`:
        EMF compute scheme name; must be one of `VALID_EMF_COMPUTE_SCHEMES`.

    - `emf_averaging_scheme`:
        EMF averaging scheme name; must be one of `VALID_EMF_AVERAGING_SCHEMES`.

    - `emf_reconstruction_order`:
        Spatial reconstruction order for the EMF; must be one of `(1, 2, 3, 5)`.

    - `resistivity`:
        Physical resistivity; omit for ideal MHD. See `write_sim_params_toml`'s guardrails for when this is disallowed.
    """

    emf_compute_scheme: str
    emf_averaging_scheme: str
    emf_reconstruction_order: int
    resistivity: float | None = None

    def __post_init__(
        self,
    ) -> None:
        if self.emf_compute_scheme not in VALID_EMF_COMPUTE_SCHEMES:
            raise ValueError(
                f"`<emf_compute_scheme>` must be one of {VALID_EMF_COMPUTE_SCHEMES}; got {self.emf_compute_scheme!r}.",
            )
        if self.emf_averaging_scheme not in VALID_EMF_AVERAGING_SCHEMES:
            raise ValueError(
                f"`<emf_averaging_scheme>` must be one of {VALID_EMF_AVERAGING_SCHEMES}; got {self.emf_averaging_scheme!r}.",
            )
        if self.emf_reconstruction_order not in (1, 2, 3, 5):
            raise ValueError(
                f"`<emf_reconstruction_order>` must be one of (1, 2, 3, 5); got {self.emf_reconstruction_order}.",
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

    Fields
    ---
    - `values`:
        Problem-specific key/value pairs; must be non-empty.

    - `title`:
        Section header text this block is rendered under.

    - `key_prefix`:
        Prefix each key in `values` is rendered under, e.g. `"setup"` -> `setup.<key>`.
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
