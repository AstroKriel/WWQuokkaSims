## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import argparse

from pathlib import Path

## personal
from jormi import ww_lists
from jormi.ww_fields import cartesian_axes

## local
from ww_quokka_sims._script_tools import field_registry

##
## === PARSER
##


def base_parser(
    *,
    num_dirs: int = 1,
    allow_fields: bool = True,
    allow_vfields: bool = False,
    allow_slicing: bool = False,
    allow_output: bool = False,
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

    - `allow_output`:
        `True` adds `--save-figure`/`--figures-dir` and `--save-data`/`--data-dir`, two independent
        pairs; default: `False`. Set to `False` for scripts that write no data or figures to disk.
        Pair with `ensure_save_flag_selected` to require at least one of the two flags.

    - `allow_parallel`:
        `True` adds `--num-workers` for scripts that dispatch work across snapshots via
        `jormi.ww_fns.parallel_dispatch`; default: `False`. `None` (the flag's default) means all
        available cores; `1` runs serially.

    Example
    ---
    parser = argparse.ArgumentParser(parents=[cli.base_parser(...)], description="...")

    args = parser.parse_args()
    """
    field_list = ww_lists.as_string(
        elems=sorted(
            field_registry.QUOKKA_FIELD_LOOKUP.keys(),
        ),
    )
    axis_list = ww_lists.as_string(
        elems=list(
            cartesian_axes.VALID_3D_AXIS_LABELS,
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
        if allow_output:
            parser.add_argument(
                "--data-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                default=None,
                help=(
                    "Output directory for extracted data (used with --save-data); defaults to the parent "
                    "directory of the snapshot."
                ),
            )
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
        if allow_output:
            parser.add_argument(
                "--data-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                required=True,
                help="Output directory for extracted data (used with --save-data).",
            )
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
            help=f"Vector field components to show; options: {axis_list}",
        )
    ## optional slice axis argument
    if allow_slicing:
        parser.add_argument(
            "--axes",
            nargs="+",
            default=None,
            help=f"Axes to slice along; options: {axis_list}",
        )
    ## optional save-figure/save-data flags
    if allow_output:
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


def ensure_save_flag_selected(
    *,
    save_figure: bool,
    save_data: bool,
) -> None:
    """Ensure at least one of `--save-figure`/`--save-data` was passed."""
    if not (save_figure or save_data):
        raise ValueError("must pass `--save-figure` and/or `--save-data`; neither was given.")


## } MODULE
