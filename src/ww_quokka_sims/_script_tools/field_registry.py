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
    "density":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_density_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"\rho",
    ),
    "velocity":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_vfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"\vec{v}",
    ),
    "velocity_magnitude":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_magnitude_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"|\vec{v}|",
    ),
    "magnetic":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_magnetic_vfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"\vec{b}",
    ),
    "total_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_total_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{tot}",
    ),
    "internal_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_internal_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{int}",
    ),
    "kinetic_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{kin}",
    ),
    "kinetic_energy_compressive":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_{\mathrm{kin}, \parallel}",
    ),
    "kinetic_energy_solenoidal":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_sol_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_{\mathrm{kin}, \perp}",
    ),
    "kinetic_energy_bulk":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_bulk_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_{\mathrm{kin}, \mathrm{bulk}}",
    ),
    "magnetic_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_magnetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"E_\mathrm{mag}",
    ),
    "energy_ratio":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_energy_ratio_sfield,
        cmap=DIVERGING_CMAP,
        latex_label=r"E_\mathrm{mag} / E_\mathrm{kin}",
    ),
    "plasma_beta":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_plasma_beta_sfield,
        cmap=DIVERGING_CMAP,
        latex_label=r"\beta",
    ),
    "pressure":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_pressure_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"p",
    ),
    "magnetic_divergence":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_b_sfield,
        cmap=DIVERGING_CMAP,
        latex_label=r"\nabla\cdot\vec{b}",
    ),
    "current_density_magnitude":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_current_density_sfield,
        cmap=SEQUENTIAL_CMAP,
        latex_label=r"|\nabla\times\vec{b}|",
    ),
    "current_density":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_current_density_vfield,
        cmap=DIVERGING_CMAP,
        latex_label=r"\nabla\times\vec{b}",
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
