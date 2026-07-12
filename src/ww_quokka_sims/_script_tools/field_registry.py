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


QUOKKA_FIELD_LOOKUP = {
    "density":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_density_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "velocity":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "velocity_magnitude":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_magnitude_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "magnetic":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_magnetic_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "total_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_total_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "internal_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_internal_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "kinetic_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "kinetic_energy_compressive":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "kinetic_energy_solenoidal":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_sol_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "kinetic_energy_bulk":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_bulk_kinetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "magnetic_energy":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_magnetic_energy_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "energy_ratio":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_energy_ratio_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "plasma_beta":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_plasma_beta_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "pressure":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_pressure_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "velocity_divergence":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_div_v_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "vorticity":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_vorticity_vfield,
        cmap=DIVERGING_CMAP,
    ),
    "vorticity_magnitude":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_vorticity_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "magnetic_divergence":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.load_3d_magnetic_divergence_sfield,
        cmap=DIVERGING_CMAP,
    ),
    "current_density_magnitude":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_current_density_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "current_density":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_current_density_vfield,
        cmap=DIVERGING_CMAP,
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
