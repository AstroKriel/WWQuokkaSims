## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import inspect

from collections import abc as collections_abc

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_io import manage_log

## local
from ww_quokka_sims.sim_io.snapshots import load_snapshot

##
## === FIELD REGISTRY
##


@dataclasses.dataclass(frozen=True)
class ExpectedProperties:
    """Declared facts about a field's values, used to derive presentation choices instead of
    guessing them (e.g., a cmap centred at `pivot_value`, or whether `log10` is safe to apply
    directly). Leaves room for more properties as new needs come up."""

    pivot_value: float | None
    is_strictly_positive: bool


@dataclasses.dataclass(frozen=True)
class RegisteredField:
    name: str
    loader_fn: collections_abc.Callable
    expected_properties: ExpectedProperties

    def load(
        self,
        quokka_snapshot: load_snapshot.QuokkaSnapshot,
        *,
        amr_level: int = 0,
    ) -> field_models.AnyField_3D:
        """Load this field from `quokka_snapshot`, warning if it breaks its own declared properties."""
        field = self.loader_fn(quokka_snapshot, amr_level=amr_level)
        if self.expected_properties.is_strictly_positive:
            sarray_3d = field_models.extract_3d_sarray(
                sfield_3d=field,
                param_name=f"<{self.name}_sfield_3d>",
            )
            if not numpy.all(sarray_3d >= 0):
                manage_log.log_warning(
                    text=f"`{self.name}` is declared strictly positive but loaded values include negatives.",
                )
        return field


REGISTERED_FIELD_LOOKUP = {
    registered_field.name: registered_field
    for registered_field in (
        RegisteredField(
            name="density",
            loader_fn=load_snapshot.QuokkaSnapshot.load_3d_density_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="velocity",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_velocity_vfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="velocity_magnitude",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_velocity_magnitude_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="magnetic",
            loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_vfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="total_energy",
            loader_fn=load_snapshot.QuokkaSnapshot.load_3d_total_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="internal_energy",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_internal_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="kinetic_energy",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="kinetic_energy_compressive",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_div_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="kinetic_energy_solenoidal",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_sol_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="kinetic_energy_bulk",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_bulk_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="magnetic_energy",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_magnetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="energy_ratio",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_energy_ratio_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=1.0,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="plasma_beta",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_plasma_beta_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=1.0,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="pressure",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_pressure_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="velocity_divergence",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_div_v_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="velocity_gradient",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_velocity_gradient_r2tfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="vorticity",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_vorticity_vfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="vorticity_magnitude",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_vorticity_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="magnetic_divergence",
            loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_divergence_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="current_density_magnitude",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_current_density_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="current_density",
            loader_fn=load_snapshot.QuokkaSnapshot.compute_current_density_vfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
    )
}

##
## === VALIDATION
##


def get_field_type(
    field_name: str,
) -> type[field_models.AnyField_3D]:
    """Return the concrete field type `field_name` resolves to, read off its loader's return-type
    annotation. Doesn't load any data or call the loader."""
    loader_fn = REGISTERED_FIELD_LOOKUP[field_name].loader_fn
    return_type = inspect.signature(loader_fn).return_annotation
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
        REGISTERED_FIELD_LOOKUP.keys(),
    )
    if not field_names or not set(field_names).issubset(valid_field_names):
        raise ValueError(f"`field_names` must be a non-empty subset of: {sorted(valid_field_names)}.")
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
