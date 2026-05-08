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
    parser = argparse.ArgumentParser(parents=[quokka_fields.base_parser(...)], description="...")

    args = parser.parse_args()
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
                help="Output directory for figures and extracted data; defaults to the snapshot's parent directory.",
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
        help="Snapshot prefex tag; default: `plt`.",
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
    ## optional extract argument (skip for non-plot scripts)
    if produces_data:
        parser.add_argument(
            "--save-data",
            action="store_true",
            default=False,
            help="Save plotted data to disk; default: False.",
        )
    return parser


## } MODULE
