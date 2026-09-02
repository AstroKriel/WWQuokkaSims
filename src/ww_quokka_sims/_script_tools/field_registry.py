## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass

## personal
from jormi.ww_fields.fields_3d import field_models

## local
from ww_quokka_sims.sim_io.snapshots import load_snapshot

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
    "velocity_gradient":
    FieldEntry(
        loader=load_snapshot.QuokkaSnapshot.compute_velocity_gradient_r2tfield,
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


def get_field_type(
    field_name: str,
) -> type[field_models.AnyField_3D]:
    """Return the concrete field type `field_name` resolves to, read off its loader's return-type
    annotation. Doesn't load any data or call the loader."""
    loader = QUOKKA_FIELD_LOOKUP[field_name].loader
    return_type = inspect.signature(loader).return_annotation
    if not isinstance(return_type, type):
        raise TypeError(f"loader for `{field_name}` has no return-type annotation.")
    return return_type


def validate_fields(
    field_names: list[str] | tuple[str, ...] | None,
    *,
    allowed_types: tuple[type, ...] | None = None,
) -> None:
    """Ensure every name in `field_names` is registered, and (if `allowed_types` is given) resolves
    to one of those types."""
    valid_field_names = set(
        QUOKKA_FIELD_LOOKUP.keys(),
    )
    if not field_names or not set(field_names).issubset(valid_field_names):
        raise ValueError(f"Provide fields via --fields from: {sorted(valid_field_names)}.")
    if allowed_types is None:
        return
    for field_name in field_names:
        field_type = get_field_type(field_name)
        if not issubclass(field_type, allowed_types):
            allowed_names = sorted(allowed_type.__name__ for allowed_type in allowed_types)
            raise ValueError(
                f"`{field_name}` resolves to {field_type.__name__}, which is not supported here;"
                f" supported types: {allowed_names}.",
            )


## } MODULE
