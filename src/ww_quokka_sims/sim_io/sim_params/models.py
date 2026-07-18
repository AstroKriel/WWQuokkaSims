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

    Exactly one of `is_boundary_periodic` or `boundary_conditions` must be set:
    `is_boundary_periodic` for a fully periodic domain, `boundary_conditions`
    (rendered as `quokka.bc`) when any boundary is `ext_dir`.

    Fields
    ---
    - `domain_lo`:
        Lower domain corner, `(x, y, z)`.

    - `domain_hi`:
        Upper domain corner, `(x, y, z)`.

    - `is_boundary_periodic`:
        Per-axis periodicity flags; mutually exclusive with `boundary_conditions`.

    - `boundary_conditions`:
        Per-axis boundary condition names (e.g. `"ext_dir"`, `"periodic"`); mutually exclusive with
        `is_boundary_periodic`.
    """

    domain_lo: tuple[float, float, float]
    domain_hi: tuple[float, float, float]
    is_boundary_periodic: tuple[int, int, int] | None = None
    boundary_conditions: tuple[str, str, str] | None = None

    def __post_init__(
        self,
    ) -> None:
        _ensure_axis_triple(self.domain_lo, param_name="<domain_lo>")
        _ensure_axis_triple(self.domain_hi, param_name="<domain_hi>")
        if (self.is_boundary_periodic is None) == (self.boundary_conditions is None):
            raise ValueError(
                "exactly one of `<is_boundary_periodic>` or `<boundary_conditions>` must be set, "
                f"got is_boundary_periodic={self.is_boundary_periodic!r}, "
                f"boundary_conditions={self.boundary_conditions!r}.",
            )
        if self.is_boundary_periodic is not None:
            validate_types.ensure_tuple_of_ints(
                param=self.is_boundary_periodic,
                param_name="<is_boundary_periodic>",
                seq_length=3,
            )
        if self.boundary_conditions is not None:
            validate_types.ensure_tuple_of_strings(
                param=self.boundary_conditions,
                param_name="<boundary_conditions>",
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
    - `num_cells`:
        Base-level cell count per axis; each entry must be >= 8.

    - `blocking_factor`:
        Minimum grid tile size; scalar (applied to all axes) or per-axis.

    - `max_grid_size`:
        Maximum grid tile size; scalar (applied to all axes) or per-axis.

    - `max_amr_levels`:
        Number of AMR refinement levels above the base grid; `0` disables AMR.

    - `num_refinement_buffer_cells`:
        Cells to pad around AMR-tagged regions before refining; only valid when `max_amr_levels` > 0.
    """

    num_cells: tuple[int, int, int]
    blocking_factor: int | tuple[int, int, int]
    max_grid_size: int | tuple[int, int, int]
    max_amr_levels: int = 0
    num_refinement_buffer_cells: int | None = None

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_tuple_of_ints(
            param=self.num_cells,
            param_name="<num_cells>",
            seq_length=3,
        )
        for value in self.num_cells:
            if value < 8:
                raise ValueError(
                    f"`<num_cells>` entries must be >= 8 (AMReX minimum under periodic BCs), got {self.num_cells}.",
                )
        _ensure_scalar_or_axis_triple(self.blocking_factor, param_name="<blocking_factor>")
        _ensure_scalar_or_axis_triple(self.max_grid_size, param_name="<max_grid_size>")
        validate_types.ensure_finite_int(
            param=self.max_amr_levels,
            param_name="<max_amr_levels>",
            require_positive=True,
            allow_zero=True,
        )
        if (self.num_refinement_buffer_cells is not None) and (self.max_amr_levels == 0):
            raise ValueError(
                "`<num_refinement_buffer_cells>` only applies when `<max_amr_levels>` > 0 (AMR refinement enabled).",
            )


##
## === OUTPUT
##


@dataclass(frozen=True)
class OutputParams:
    """
    Checkpoint and plotfile cadence. Exactly one of `snapshot_index_interval` or `snapshot_time_interval`
    must be set.

    Fields
    ---
    - `snapshot_prefix`:
        Directory/filename prefix for plotfiles; omit to use Quokka's default.

    - `snapshot_index_interval`:
        Emit a plotfile every N steps; mutually exclusive with `snapshot_time_interval`.

    - `snapshot_time_interval`:
        Emit a plotfile every `snapshot_time_interval` of simulation time; mutually exclusive with
        `snapshot_index_interval`.

    - `checkpoint_index_interval`:
        Emit a checkpoint every N steps; omit to disable checkpointing.

    - `checkpoint_prefix`:
        Directory/filename prefix for checkpoints; omit to use Quokka's default.
    """

    snapshot_prefix: str | None = None
    snapshot_index_interval: int | None = None
    snapshot_time_interval: float | None = None
    checkpoint_index_interval: int | None = None
    checkpoint_prefix: str | None = None

    def __post_init__(
        self,
    ) -> None:
        if (self.snapshot_index_interval is None) == (self.snapshot_time_interval is None):
            raise ValueError(
                "exactly one of `<snapshot_index_interval>` or `<snapshot_time_interval>` must be set, got "
                f"snapshot_index_interval={self.snapshot_index_interval!r}, "
                f"snapshot_time_interval={self.snapshot_time_interval!r}.",
            )
        if self.snapshot_prefix is not None:
            validate_types.ensure_nonempty_string(
                param=self.snapshot_prefix,
                param_name="<snapshot_prefix>",
            )
        if self.checkpoint_index_interval is not None:
            validate_types.ensure_finite_int(
                param=self.checkpoint_index_interval,
                param_name="<checkpoint_index_interval>",
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

    - `use_reflux`:
        `1` to reflux at coarse-fine boundaries, `0` to disable; omit for Quokka's default.

    - `use_subcycle`:
        `1` to subcycle in time across AMR levels, `0` to disable; omit for Quokka's default.

    - `use_tracers`:
        `1` to advect tracer particles, `0` to disable; omit for Quokka's default.

    - `stop_time`:
        Simulation time at which to stop; omit to run until `max_time_steps`.

    - `max_time_steps`:
        Maximum number of timesteps to run; omit to run until `stop_time`.
    """

    cfl: float
    use_reflux: int | None = None
    use_subcycle: int | None = None
    use_tracers: int | None = None
    stop_time: float | None = None
    max_time_steps: int | None = None

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

## pcm=1, plm=2, ppm=3, ppm_ep=5; shared with `MHDParams.interpolation_order` below,
## since both select from the same set of interpolation schemes
VALID_INTERPOLATION_ORDERS = (1, 2, 3, 5)


@dataclass(frozen=True)
class HydroParams:
    """
    Reconstruction scheme: pcm=1, plm=2, ppm=3, ppm_ep=5.

    Fields
    ---
    - `integrator_order`:
        Runge-Kutta time-integrator order.

    - `interpolation_order`:
        Spatial reconstruction order; must be one of `VALID_INTERPOLATION_ORDERS`.

    - `use_dual_energy`:
        `1` to enable the dual-energy formalism, `0` to disable; omit for Quokka's default.
    """

    integrator_order: int
    interpolation_order: int
    use_dual_energy: int | None = None

    def __post_init__(
        self,
    ) -> None:
        if self.interpolation_order not in VALID_INTERPOLATION_ORDERS:
            raise ValueError(
                f"`<interpolation_order>` must be one of {VALID_INTERPOLATION_ORDERS}; "
                f"got {self.interpolation_order}.",
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

    - `interpolation_order`:
        Spatial reconstruction order for the EMF; must be one of `VALID_INTERPOLATION_ORDERS`.

    - `resistivity`:
        Physical resistivity; omit for ideal MHD. See `write_sim_params_toml`'s guardrails for when this is disallowed.
    """

    emf_compute_scheme: str
    emf_averaging_scheme: str
    interpolation_order: int
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
        if self.interpolation_order not in VALID_INTERPOLATION_ORDERS:
            raise ValueError(
                f"`<interpolation_order>` must be one of {VALID_INTERPOLATION_ORDERS}; "
                f"got {self.interpolation_order}.",
            )


##
## === SETUP (problem-specific, always rendered last)
##


@dataclass(frozen=True)
class SetupParams:
    """
    Problem-specific parameters, rendered last under a caller-chosen param group title
    (e.g. `"wave setup"`, `"problem setup"`), matching the two titles seen in the corpus.
    Keys are rendered under a caller-chosen prefix (default `"setup"`); at least one
    known problem (field-loop) uses its own prefix instead of `"setup"`.

    Fields
    ---
    - `param_values`:
        Problem-specific key/value pairs; must be non-empty.

    - `group_title`:
        Param group header text this block is rendered under.

    - `key_prefix`:
        Prefix each key in `param_values` is rendered under, e.g. `"setup"` -> `setup.<key>`.
    """

    param_values: dict[str, Any] = field(default_factory=dict)
    group_title: str = "problem setup"
    key_prefix: str = "setup"

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_dict(
            param=self.param_values,
            param_name="<param_values>",
        )
        if not self.param_values:
            raise ValueError(
                "`<param_values>` must be non-empty; omit `setup` entirely instead of passing an empty one.",
            )
        validate_types.ensure_nonempty_string(
            param=self.group_title,
            param_name="<group_title>",
        )
        validate_types.ensure_nonempty_string(
            param=self.key_prefix,
            param_name="<key_prefix>",
        )


## } MODULE
