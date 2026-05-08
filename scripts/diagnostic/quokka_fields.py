## { MODULE

##
## === DEPENDENCIES
##

import argparse

from pathlib import Path
from collections.abc import Callable

from jormi import ww_lists
from jormi.ww_fields import cartesian_axes

from ww_quokka_sims.sim_io import load_snapshot

##
## === DEFAULT COLORMAPS
##

SEQUENTIAL_CMAP = "cmr.lavender"
DIVERGING_CMAP = "cmr.iceburn"

##
## === FIELD REGISTRY
##


def _field_entry(
    loader: Callable,
    cmap: str,
) -> dict:
    return {
        "loader": loader,
        "cmap": cmap,
    }


QUOKKA_FIELD_LOOKUP = {
    "rho":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_density_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "vel":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "vel_magn":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_magnitude_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "mag":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_magnetic_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_tot":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_total_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_int":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_internal_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_kin":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_kin_div":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_kin_sol":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_sol_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_kin_bulk":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_bulk_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_mag":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_magnetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "E_ratio":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_energy_ratio_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "pressure":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_pressure_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "divb":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_b_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "cur":
    _field_entry(
        loader=load_snapshot.QuokkaSnapshot.compute_current_density_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
}

##
## === VALIDATION
##


def validate_fields(
    field_names: list[str] | tuple[str, ...] | None,
) -> None:
    valid_field_names = set(
        QUOKKA_FIELD_LOOKUP.keys(),
    )
    if not field_names or not set(field_names).issubset(valid_field_names):
        raise ValueError(f"Provide fields via -f from: {sorted(valid_field_names)}")


##
## === PARSER
##


def base_parser(
    num_dirs: int = 1,
    allow_vfields: bool = True,
    allow_slicing: bool = True,
    allow_fields: bool = True,
    produces_data: bool = True,
) -> argparse.ArgumentParser:
    """
    Shared argument parser for diagnostic scripts.

    Returns a base parser intended to be used as a parent via `parents=[base_parser()]`.
    The child parser inherits all arguments defined here, which can then be accessed
    as usual on the parsed namespace (e.g. `args.fields`, `args.tag`, `args.dir`).

    Parameters
    ---
    - `num_dirs`:
        Number of input directory arguments to add. If 1, adds a single optional `--input-dir`.
        If >1, adds `--input-dir-1`, `--input-dir-2`, ... (all required).

    - `allow_vfields`:
        If True, adds `--comps` for selecting vector field components.

    - `allow_slicing`:
        If True, adds `--axes` for selecting slice axes. Typically paired with `allow_vfields`.

    - `allow_fields`:
        If True, adds `--fields`. Set to False for scripts that operate on all fields (e.g. compare_datasets).

    - `produces_data`:
        If True, adds `--out-dir` and `--save-data`. Default: True.
        Set to False for scripts that write no data or figures to disk (e.g. inspection or comparison scripts).

    Example
    ---
    parser = argparse.ArgumentParser(
        parents=[quokka_fields.base_parser(
            num_dirs=1,
            allow_vfields=True,
            allow_slicing=True,
            produces_data=True,
        )],
        description="...",
    )
    args = parser.parse_args() # args.input_dir, args.tag, args.fields, args.comps, args.axes, args.save_data
    """
    field_list = ww_lists.as_string(
        elems=sorted(
            QUOKKA_FIELD_LOOKUP.keys(),
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
                help="Output directory for figures and extracted data. Defaults to the snapshot parent directory.",
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
        help="Snapshot tag (e.g. `plt_` -> plt_0000000, plt_0000100). Default: `plt`.",
    )
    if allow_fields:
        parser.add_argument(
            "--fields",
            nargs="+",
            default=None,
            help=f"Fields to plot. Options: {field_list}",
        )
    ## optional vector field component argument
    if allow_vfields:
        parser.add_argument(
            "--comps",
            nargs="+",
            default=None,
            help=f"Vector field components to show. Options: {axis_list}",
        )
    ## optional slice axis argument
    if allow_slicing:
        parser.add_argument(
            "--axes",
            nargs="+",
            default=None,
            help=f"Axes to slice along. Options: {axis_list}",
        )
    ## optional extract argument (skip for non-plot scripts)
    if produces_data:
        parser.add_argument(
            "--save-data",
            action="store_true",
            default=False,
            help="Save plotted data to disk (default: False).",
        )
    return parser


## } MODULE
