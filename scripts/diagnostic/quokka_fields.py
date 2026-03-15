## { MODULE

##
## === DEPENDENCIES
##

import argparse

from pathlib import Path
from collections.abc import Callable

from jormi import ww_lists
from jormi.ww_fields import cartesian_axes

from ww_quokka_sims.sim_io import load_dataset

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
        loader=load_dataset.QuokkaDataset.load_3d_density_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "vel":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_velocity_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "vel_magn":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_velocity_magnitude_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "mag":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_magnetic_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Etot":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_total_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Eint":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_internal_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Ekin":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Ekin_div":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_div_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Ekin_sol":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_sol_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Ekin_bulk":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_bulk_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Emag":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_magnetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "Eratio":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_energy_ratio_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "pressure":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_pressure_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "divb":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_divb_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "cur":
    _field_entry(
        loader=load_dataset.QuokkaDataset.load_3d_current_density_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
}

##
## === VALIDATION
##


def validate_fields(
    field_names: list[str] | tuple[str, ...] | None,
) -> None:
    valid_field_names = set(QUOKKA_FIELD_LOOKUP.keys())
    if not field_names or not set(field_names).issubset(valid_field_names):
        raise ValueError(f"Provide fields via -f from: {sorted(valid_field_names)}")


##
## === PARSER
##


def base_parser(
    num_dirs: int = 1,
    allow_vfields: bool = True,
) -> argparse.ArgumentParser:
    """
    Shared parser arguments for diagnostic scripts.

    Parameters
    ---
    - `num_dirs`:
        Number of input directory arguments to add.

    - `allow_vfields`:
        If True, adds --comps/-c and --axes/-a arguments for vector field components and slice axes.

    Use as a parent:
        parser = argparse.ArgumentParser(parents=[quokka_fields.base_parser()], description="...")
    """
    field_list = ww_lists.as_string(elems=sorted(QUOKKA_FIELD_LOOKUP.keys()))
    axis_list = ww_lists.as_string(elems=list(cartesian_axes.VALID_3D_AXIS_LABELS))
    parser = argparse.ArgumentParser(add_help=False)
    ## directory arguments (shape depends on num_dirs)
    if num_dirs == 1:
        parser.add_argument(
            "--dir",
            "-d",
            type=lambda path: Path(path).expanduser().resolve(),
            default=None,
            help="Path to a Quokka simulation or dataset directory.",
        )
    else:
        for dir_index in range(1, num_dirs + 1):
            parser.add_argument(
                f"--dir-{dir_index}",
                f"-d{dir_index}",
                type=lambda path: Path(path).expanduser().resolve(),
                required=True,
                help=f"Input directory {dir_index} of {num_dirs}.",
            )
        parser.add_argument(
            "--out",
            "-o",
            type=lambda path: Path(path).expanduser().resolve(),
            required=True,
            help="Output directory for figures.",
        )
    ## always-present arguments
    parser.add_argument(
        "--tag",
        "-t",
        default="plt",
        help="Dataset tag (e.g. `plt` -> plt00010, plt00020). Default: `plt`.",
    )
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=None,
        help=f"Fields to plot. Options: {field_list}",
    )
    ## optional vector field arguments (skip for purely scalar scripts)
    if allow_vfields:
        parser.add_argument(
            "--comps",
            "-c",
            nargs="+",
            default=None,
            help=f"Vector field components to show. Options: {axis_list}",
        )
        parser.add_argument(
            "--axes",
            "-a",
            nargs="+",
            default=None,
            help=f"Axes to slice along. Options: {axis_list}",
        )
    return parser


## } MODULE
