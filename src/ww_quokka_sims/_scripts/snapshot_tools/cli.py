## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import argparse

from dataclasses import dataclass
from pathlib import Path

## personal
from jormi import ww_lists
from jormi.ww_fields import cartesian_axes
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._scripts.snapshot_tools import field_registry
from ww_quokka_sims.sim_io.snapshots import find_snapshots

##
## === CONSTANTS
##

AXIS_LABELS_TEXT = ww_lists.as_string(elems=list(cartesian_axes.VALID_3D_AXIS_LABELS))

##
## === ARG GROUPS
##


@dataclass(
    frozen=True,
    kw_only=True,
)
class SnapshotArgs:
    """Bound from `--input-dir`/`--tag`; present on every script that scans a directory of snapshots."""

    input_dir: Path
    snapshot_tag: str

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_nonempty_string(
            param=self.snapshot_tag,
            param_name="snapshot_tag",
        )

    @classmethod
    def from_user_args(
        cls,
        user_args: argparse.Namespace,
    ) -> "SnapshotArgs":
        return cls(
            input_dir=user_args.input_dir,
            snapshot_tag=user_args.tag,
        )


@dataclass(
    frozen=True,
    kw_only=True,
)
class DataOutputArgs:
    """Bound from `--data-dir`/`--overwrite`; added by `base_parser(allow_write=True)`."""

    overwrite: bool = False
    data_dir: Path | None = None

    @classmethod
    def from_user_args(
        cls,
        user_args: argparse.Namespace,
    ) -> "DataOutputArgs":
        return cls(
            overwrite=user_args.overwrite,
            data_dir=user_args.data_dir,
        )


@dataclass(
    frozen=True,
    kw_only=True,
)
class DiagnosticOutputArgs(DataOutputArgs):
    """`DataOutputArgs` plus a save-figure/save-data choice; added by `base_parser(allow_figures=True)`."""

    save_data: bool
    save_figure: bool
    figures_dir: Path | None = None

    def __post_init__(
        self,
    ) -> None:
        _ensure_save_flag_selected(
            save_figure=self.save_figure,
            save_data=self.save_data,
        )

    @classmethod
    def from_user_args(
        cls,
        user_args: argparse.Namespace,
    ) -> "DiagnosticOutputArgs":
        return cls(
            overwrite=user_args.overwrite,
            data_dir=user_args.data_dir,
            save_data=user_args.save_data,
            save_figure=user_args.save_figure,
            figures_dir=user_args.figures_dir,
        )


@dataclass(
    frozen=True,
    kw_only=True,
)
class FieldArgs:
    """Bound from `--fields`/`--amr-level`; the raw, pre-lookup field request.

    Each field name is resolved separately, downstream, into its own script-local
    `ResolvedFieldArgs` (via `field_registry.REGISTERED_FIELD_LOOKUP`).
    """

    fields: tuple[str, ...] | list[str] | None
    amr_level: int = 0

    @classmethod
    def from_user_args(
        cls,
        user_args: argparse.Namespace,
    ) -> "FieldArgs":
        return cls(
            fields=user_args.fields,
            amr_level=user_args.amr_level,
        )


@dataclass(
    frozen=True,
    kw_only=True,
)
class FieldCompArgs(FieldArgs):
    """`FieldArgs` plus `--comps`; the field itself may still be scalar or vector,
    `comps` only applies when it turns out to be a vector field.
    """

    comps: tuple[str, ...] | list[str] | None = None

    def __post_init__(
        self,
    ) -> None:
        if (self.comps is not None) and not set(self.comps).issubset(set(cartesian_axes.VALID_3D_AXIS_LABELS)):
            raise ValueError(f"Provide one or more components (via -c) from: {AXIS_LABELS_TEXT}")

    @classmethod
    def from_user_args(
        cls,
        user_args: argparse.Namespace,
    ) -> "FieldCompArgs":
        return cls(
            fields=user_args.fields,
            amr_level=user_args.amr_level,
            comps=user_args.comps,
        )


@dataclass(
    frozen=True,
    kw_only=True,
)
class FieldCompAxesArgs(FieldCompArgs):
    """`FieldCompArgs` plus `--axes`, for scripts that slice along specific axes."""

    axes: tuple[str, ...] | list[str] | None = None

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        if (self.axes is not None) and not set(self.axes).issubset(set(cartesian_axes.VALID_3D_AXIS_LABELS)):
            raise ValueError(f"Provide one or more axes (via -a) from: {AXIS_LABELS_TEXT}")

    @classmethod
    def from_user_args(
        cls,
        user_args: argparse.Namespace,
    ) -> "FieldCompAxesArgs":
        return cls(
            fields=user_args.fields,
            amr_level=user_args.amr_level,
            comps=user_args.comps,
            axes=user_args.axes,
        )


##
## === PARSER
##


def base_parser(
    *,
    num_dirs: int = 1,
    allow_fields: bool = True,
    allow_vfields: bool = False,
    allow_slicing: bool = False,
    allow_write: bool = False,
    allow_figures: bool = False,
    allow_parallel: bool = False,
) -> argparse.ArgumentParser:
    """
    Shared argument parser for diagnostic scripts.

    Returns a base parser intended to be used as a parent via `parents=[base_parser()]`.
    The child parser inherits all arguments defined here, which can then be accessed
    as usual on the parsed namespace (e.g. `args.fields`, `args.tag`, `args.dir`).

    Parameters
    ---
    - `num_dirs`:
        Number of input directory arguments to add.
        Default: `num_dirs = 1` adds a single optional `--input-dir`.
        `num_dirs = N > 1` adds `--input-dir-1`, `--input-dir-2`, ... `--input-dir-N` (all required).

    - `allow_fields`:
        `True` adds `--fields` argument; default: `True`.
        Set to `False` for scripts that operate on all fields.

    - `allow_vfields`:
        `True` adds `--comps` argument for selecting vector field components; default: `False`.

    - `allow_slicing`:
        `True` adds `--axes` argument for selecting slice axes; default: `False`.

    - `allow_write`:
        `True` adds `--data-dir` and `--overwrite`; default: `False`. Set to `False` for scripts that
        write nothing to disk. Scripts resume by default (skip snapshots whose output already exists);
        `--overwrite` disables that.

    - `allow_figures`:
        `True` adds `--save-figure`/`--figures-dir` and `--save-data`, on top of what `allow_write` adds;
        requires `allow_write=True`. Default: `False`. `DiagnosticOutputArgs.__post_init__` enforces at
        least one of `--save-figure`/`--save-data` being selected.

    - `allow_parallel`:
        `True` adds `--num-workers` for scripts that dispatch work across snapshots via
        `jormi.ww_fns.parallel_dispatch`; default: `False`. `None` (the flag's default) means all
        available cores; `1` runs serially.

    Example
    ---
    parser = argparse.ArgumentParser(parents=[cli.base_parser(...)], description="...")

    args = parser.parse_args()
    """
    if allow_figures and not allow_write:
        raise ValueError("`allow_figures=True` requires `allow_write=True`.")
    field_list = ww_lists.as_string(
        elems=sorted(
            field_registry.REGISTERED_FIELD_LOOKUP.keys(),
        ),
    )
    parser = argparse.ArgumentParser(add_help=False)
    ## directory arguments (shape depends on num_dirs)
    if num_dirs == 1:
        parser.add_argument(
            "--input-dir",
            type=lambda path: Path(path).expanduser().resolve(),
            default=None,
            help=(
                "Path to a directory containing snapshot dirs (matched by --tag), or to a single snapshot dir."
            ),
        )
        if allow_write:
            parser.add_argument(
                "--data-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                default=None,
                help="Output directory for data written to disk; defaults to the parent directory of the snapshot.",
            )
        if allow_figures:
            parser.add_argument(
                "--figures-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                default=None,
                help="Output directory for figures (used with --save-figure); defaults to --data-dir.",
            )
    else:
        for dir_index in range(1, num_dirs + 1):
            parser.add_argument(
                f"--input-dir-{dir_index}",
                type=lambda path: Path(path).expanduser().resolve(),
                required=True,
                help=f"Input directory {dir_index} of {num_dirs}.",
            )
        if allow_write:
            parser.add_argument(
                "--data-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                required=True,
                help="Output directory for data written to disk.",
            )
        if allow_figures:
            parser.add_argument(
                "--figures-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                default=None,
                help="Output directory for figures (used with --save-figure); defaults to --data-dir.",
            )
    ## always-present arguments
    parser.add_argument(
        "--tag",
        default="plt",
        help="Snapshot prefix tag; default: `plt`.",
    )
    parser.add_argument(
        "--amr-level",
        type=int,
        default=0,
        help=(
            "AMR level to read (composite of the finest data available up to and including this level); "
            "default: 0 (base level). Errors if the snapshot does not have this many levels."
        ),
    )
    if allow_fields:
        parser.add_argument(
            "--fields",
            nargs="+",
            default=None,
            help=f"Fields to plot; options: {field_list}",
        )
    ## optional vector field component argument
    if allow_vfields:
        parser.add_argument(
            "--comps",
            nargs="+",
            default=None,
            help=f"Vector field components to show; options: {AXIS_LABELS_TEXT}",
        )
    ## optional slice axis argument
    if allow_slicing:
        parser.add_argument(
            "--axes",
            nargs="+",
            default=None,
            help=f"Axes to slice along; options: {AXIS_LABELS_TEXT}",
        )
    ## optional save-figure/save-data flags
    if allow_figures:
        parser.add_argument(
            "--save-figure",
            action="store_true",
            default=False,
            help="Save the figure to disk; default: False.",
        )
        parser.add_argument(
            "--save-data",
            action="store_true",
            default=False,
            help="Save the extracted data to disk; default: False.",
        )
    if allow_write:
        parser.add_argument(
            "--overwrite",
            action="store_true",
            default=False,
            help=(
                "Redo snapshots whose output already exists, instead of skipping them; "
                "default: False (resume where a prior run left off)."
            ),
        )
    ## optional worker-count flag
    if allow_parallel:
        parser.add_argument(
            "--num-workers",
            type=int,
            default=None,
            help="Number of worker processes; default: all available cores. Set to 1 to run serially.",
        )
    return parser


##
## === OUTPUT DIRECTORIES
##


def resolve_output_dir(
    *,
    output_dir: Path | None,
    default_dir: Path,
) -> Path:
    """Resolve `output_dir` to `default_dir` if unset, creating it if needed."""
    resolved_dir = output_dir if output_dir is not None else default_dir
    resolved_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return resolved_dir


def _ensure_save_flag_selected(
    *,
    save_figure: bool,
    save_data: bool,
) -> None:
    """Ensure at least one of `--save-figure`/`--save-data` was passed."""
    if not (save_figure or save_data):
        raise ValueError("must pass `--save-figure` and/or `--save-data`; neither was given.")


##
## === SNAPSHOT INPUTS
##


@dataclass(frozen=True)
class ResolvedInputs:
    """`figures_dir`/`index_width` are `None` unless the caller opted into resolving them."""

    snapshot_dirs: list[Path]
    data_dir: Path
    figures_dir: Path | None = None
    index_width: int | None = None


def resolve_inputs(
    *,
    snapshot_args: SnapshotArgs,
    output_args: DataOutputArgs,
    allow_index_width: bool = True,
    max_elems: int | None = None,
) -> ResolvedInputs | None:
    """Resolve snapshot dirs and output locations shared by every diagnostic pipeline.

    `figures_dir` is resolved only when `output_args` is a `DiagnosticOutputArgs` (it has no
    meaning for a plain `DataOutputArgs`, which never produces figures).
    """
    snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
        input_dir=snapshot_args.input_dir,
        snapshot_tag=snapshot_args.snapshot_tag,
        max_elems=max_elems,
    )
    if not snapshot_dirs:
        return None
    data_dir = resolve_output_dir(
        output_dir=output_args.data_dir,
        default_dir=snapshot_dirs[0].parent,
    )
    figures_dir = None
    if isinstance(output_args, DiagnosticOutputArgs):
        figures_dir = resolve_output_dir(
            output_dir=output_args.figures_dir,
            default_dir=data_dir,
        )
    index_width = None
    if allow_index_width:
        index_width = find_snapshots.get_max_index_width(
            snapshot_dirs=snapshot_dirs,
            snapshot_tag=snapshot_args.snapshot_tag,
        )
    return ResolvedInputs(
        snapshot_dirs=snapshot_dirs,
        data_dir=data_dir,
        figures_dir=figures_dir,
        index_width=index_width,
    )


##
## === AXES
##


def parse_axes(
    *,
    axes: tuple[str, ...] | list[str] | None,
) -> tuple[cartesian_axes.CartesianAxis_3D, ...]:
    """Resolve `comps`/`axes` to a canonical tuple; `None` defaults to all three axes."""
    if axes is None:
        return tuple(cartesian_axes.DEFAULT_3D_AXES_ORDER)
    parsed_axes: list[cartesian_axes.CartesianAxis_3D] = []
    for axis_name in axes:
        try:
            parsed_axes.append(
                cartesian_axes.as_axis(
                    axis=axis_name,
                ),
            )
        except (TypeError, ValueError):
            raise ValueError(f"Provide one or more axes from: {AXIS_LABELS_TEXT}")
    return tuple(parsed_axes)


## } MODULE
