## { MODULE

##
## === DEPENDENCIES
##

import numpy
import argparse

from pathlib import Path
from matplotlib.figure import Figure as mpl_Figure

from jormi.ww_types import check_types
from jormi import ww_lists
from jormi.ww_plots import manage_plots
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import field_types

##
## === QUOKKA FIELDS
##

QUOKKA_FIELD_LOOKUP = {
    "rho": {
        "loader": "load_3d_density_sfield",
        "cmap": "cmr.lavender",
    },
    "vel": {
        "loader": "load_3d_velocity_vfield",
        "cmap": "cmr.lavender",
    },
    "vel_magn": {
        "loader": "load_3d_velocity_magnitude_sfield",
        "cmap": "cmr.lavender",
    },
    "mag": {
        "loader": "load_3d_magnetic_vfield",
        "cmap": "cmr.lavender",
    },
    "Etot": {
        "loader": "load_3d_total_energy_sfield",
        "cmap": "cmr.lavender",
    },
    "Eint": {
        "loader": "load_3d_internal_energy_sfield",
        "cmap": "cmr.lavender",
    },
    "Ekin": {
        "loader": "load_3d_kinetic_energy_sfield",
        "cmap": "cmr.lavender",
    },
    "Ekin_div": {
        "loader": "load_3d_div_kinetic_energy_sfield",
        "cmap": "cmr.lavender",
    },
    "Ekin_sol": {
        "loader": "load_3d_sol_kinetic_energy_sfield",
        "cmap": "cmr.lavender",
    },
    "Ekin_bulk": {
        "loader": "load_3d_bulk_kinetic_energy_sfield",
        "cmap": "cmr.lavender",
    },
    "Emag": {
        "loader": "load_3d_magnetic_energy_sfield",
        "cmap": "cmr.lavender",
    },
    "Eratio": {
        "loader": "load_3d_energy_ratio_sfield",
        "cmap": "cmr.iceburn",
    },
    "pressure": {
        "loader": "load_3d_pressure_sfield",
        "cmap": "cmr.lavender",
    },
    "divb": {
        "loader": "load_3d_divb_sfield",
        "cmap": "cmr.fusion",
    },
    "cur": {
        "loader": "load_current_density_sfield",
        "cmap": "cmr.lavender",
    },
}

##
## === HELPER FUNCTIONS
##


def get_sim_time(
    field: field_types.ScalarField_3D | field_types.VectorField_3D,
) -> float:
    sim_time = field.sim_time
    check_types.ensure_finite_float(
        param=sim_time,
        param_name="sim_time",
        allow_none=False,
    )
    assert sim_time is not None
    return float(sim_time)


def as_latex_label(
    label: str,
) -> str:
    if "$" in label:
        return label
    return f"${label}$"


def validate_fields(
    fields_to_plot: list[str] | tuple[str, ...] | None,
) -> None:
    valid_fields = set(QUOKKA_FIELD_LOOKUP.keys())
    if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
        raise ValueError(f"Provide fields via -f from: {sorted(valid_fields)}")


def base_parser() -> argparse.ArgumentParser:
    """
    Shared parser arguments for diagnostic scripts.
    
    Use as a parent:
        parser = argparse.ArgumentParser(parents=[utils.base_parser()], description="...")
    """
    field_list = ww_lists.as_string(elems=sorted(QUOKKA_FIELD_LOOKUP.keys()))
    axis_list = ww_lists.as_string(elems=list(cartesian_axes.VALID_3D_AXIS_LABELS))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dir",
        "-d",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Path to a Quokka simulation or dataset directory.",
    )
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


def create_figure(
    num_rows: int = 1,
    num_cols: int = 1,
    add_cbar_space: bool = False,
) -> tuple[mpl_Figure, manage_plots.PlotAxesArray]:
    if (num_rows == 1) and (num_cols == 1):
        fig, ax = manage_plots.create_figure()
        if add_cbar_space:
            fig.subplots_adjust(right=0.82)
        axs_grid = numpy.asarray([[ax]], dtype=object)
        return fig, axs_grid
    fig, axs_grid = manage_plots.create_figure(
        num_rows=num_rows,
        num_cols=num_cols,
        y_spacing=0.25,
        x_spacing=0.75 if add_cbar_space else 0.25,
    )
    return fig, axs_grid


## } MODULE
