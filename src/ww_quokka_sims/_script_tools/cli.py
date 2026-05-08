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
    num_dirs: int = 1,
    allow_fields: bool = True,
    allow_vfields: bool = False,
    allow_slicing: bool = False,
    produces_data: bool = False,
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

    - `produces_data`:
        `True` adds arguments `--out-dir` and `--save-data`; default: `False`.
        Set to `False` for scripts that write no data or figures to disk.

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
            help="Path to a Quokka simulation or snapshot directory.",
        )
        if produces_data:
            parser.add_argument(
                "--out-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                default=None,
                help=
                "Output directory for figures and extracted data; defaults to the snapshot's parent directory.",
            )
    else:
        for dir_index in range(1, num_dirs + 1):
            parser.add_argument(
                f"--input-dir-{dir_index}",
                type=lambda path: Path(path).expanduser().resolve(),
                required=True,
                help=f"Input directory {dir_index} of {num_dirs}.",
            )
        if produces_data:
            parser.add_argument(
                "--out-dir",
                type=lambda path: Path(path).expanduser().resolve(),
                required=True,
                help="Output directory for figures and extracted data.",
            )
    ## always-present arguments
    parser.add_argument(
        "--tag",
        default="plt",
        help="Snapshot prefix tag; default: `plt`.",
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
    ## optional save-data flag
    if produces_data:
        parser.add_argument(
            "--save-data",
            action="store_true",
            default=False,
            help="Save plotted data to disk; default: False.",
        )
    return parser


## } MODULE
