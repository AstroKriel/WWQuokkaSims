## { MODULE

##
## === DEPENDENCIES
##

from collections.abc import Callable

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
        raise ValueError(f"Provide fields via --fields from: {sorted(valid_field_names)}")


## } MODULE
