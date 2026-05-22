## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from collections.abc import Callable
from dataclasses import dataclass

## local
from ww_quokka_sims.sim_io import load_snapshot

##
## === DEFAULT COLORMAPS
##

SEQUENTIAL_CMAP = "cmr.lavender"
DIVERGING_CMAP = "cmr.iceburn"

##
## === FIELD REGISTRY
##


@dataclass(frozen=True)
class FieldEntry:
    loader: Callable
    cmap: str
    latex_label: str


QUOKKA_FIELD_LOOKUP = {
    "rho":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_density_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"\rho",
    ),
    "vel":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_vfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"\vec{v}",
    ),
    "vel_magn":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_magnitude_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"|\vec{v}|",
    ),
    "mag":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_magnetic_vfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"\vec{b}",
    ),
    "E_tot":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_total_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{tot}",
    ),
    "E_int":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_internal_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{int}",
    ),
    "E_kin":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{kin}",
    ),
    "E_kin_div":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_{\mathrm{kin}, \parallel}",
    ),
    "E_kin_sol":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_sol_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_{\mathrm{kin}, \perp}",
    ),
    "E_kin_bulk":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_bulk_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_{\mathrm{kin}, \mathrm{bulk}}",
    ),
    "E_mag":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_magnetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{mag}",
    ),
    "E_ratio":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_energy_ratio_sfield,
        cmap=DIVERGING_CMAP,
        latex_label=r"E_\mathrm{mag} / E_\mathrm{kin}",
    ),
    "pressure":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_pressure_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"p",
    ),
    "div_b":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_b_sfield,
        cmap=DIVERGING_CMAP,
        latex_label=r"\nabla\cdot\vec{b}",
    ),
    "cur":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_current_density_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"|\nabla\times\vec{b}|",
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
        raise ValueError(f"Provide fields via --fields from: {sorted(valid_field_names)}")


## } MODULE
