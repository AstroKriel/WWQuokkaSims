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
from jormi.ww_fields import cartesian_axes
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
    field_loader_fn: collections_abc.Callable
    expected_properties: ExpectedProperties
    amr_leaves_loader_fn: collections_abc.Callable | None = None
    native_slice_loader_fn: collections_abc.Callable | None = None

    def load(
        self,
        *,
        quokka_snapshot: load_snapshot.QuokkaSnapshot,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.AnyField_3D:
        """
        Load this field from `quokka_snapshot`, warning if it breaks its own declared properties.

        `use_chunked_reader` is only forwarded when `True`: not every registered field's loader
        accepts it yet, and most callers never request it, so leaving it off by default keeps
        every currently-working field loading exactly as it did before this parameter existed.
        """
        loader_kwargs: dict[str, int | bool] = {"amr_level": amr_level}
        if use_chunked_reader:
            loader_kwargs["use_chunked_reader"] = use_chunked_reader
        field = self.field_loader_fn(quokka_snapshot, **loader_kwargs)
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

    def load_amr_leaves(
        self,
        *,
        quokka_snapshot: load_snapshot.QuokkaSnapshot,
    ) -> load_snapshot.AMRLeaves | dict[cartesian_axes.CartesianAxis_3D, load_snapshot.AMRLeaves]:
        """
        Load this field at every leaf cell across the full AMR hierarchy, each at its own
        native resolution (never resampled onto one uniform grid).

        Only supported for fields registered with an `amr_leaves_loader_fn`; see
        `validate_fields_support_amr_leaves` to check a batch of field names upfront.
        """
        if self.amr_leaves_loader_fn is None:
            raise ValueError(f"`{self.name}` does not support AMR-leaf loading.")
        return self.amr_leaves_loader_fn(quokka_snapshot)

    def load_native_slice(
        self,
        *,
        quokka_snapshot: load_snapshot.QuokkaSnapshot,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray] | dict[cartesian_axes.CartesianAxis_3D, tuple[numpy.ndarray,
                                                                                           numpy.ndarray]]:
        """
        Load this field, and its per-pixel native `dx`, on a genuine AMR-native slice (each
        pixel read from whichever box actually covers it, at the finest level present).

        Only supported for fields registered with a `native_slice_loader_fn`; see
        `validate_fields_support_native_slice` to check a batch of field names upfront.
        """
        if self.native_slice_loader_fn is None:
            raise ValueError(f"`{self.name}` does not support native-slice loading.")
        return self.native_slice_loader_fn(
            quokka_snapshot,
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )


REGISTERED_FIELD_LOOKUP = {
    registered_field.name: registered_field
    for registered_field in (
        RegisteredField(
            name="density",
            field_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_density_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_density_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_density_native_slice,
        ),
        RegisteredField(
            name="velocity",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_velocity_vfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_velocity_amr_leaves_by_axis,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_velocity_native_slice_by_axis,
        ),
        RegisteredField(
            name="velocity_magnitude",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_velocity_magnitude_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_velocity_magnitude_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_velocity_magnitude_native_slice,
        ),
        RegisteredField(
            name="magnetic",
            field_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_vfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_amr_leaves_by_axis,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_native_slice_by_axis,
        ),
        RegisteredField(
            name="total_energy",
            field_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_total_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_total_energy_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_total_energy_native_slice,
        ),
        RegisteredField(
            name="internal_energy",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_internal_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_internal_energy_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_internal_energy_native_slice,
        ),
        RegisteredField(
            name="kinetic_energy",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_kinetic_energy_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_kinetic_energy_native_slice,
        ),
        RegisteredField(
            name="kinetic_energy_compressive",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_div_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="kinetic_energy_solenoidal",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_sol_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="kinetic_energy_bulk",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_bulk_kinetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="magnetic_energy",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_magnetic_energy_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_energy_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_energy_native_slice,
        ),
        RegisteredField(
            name="energy_ratio",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_energy_ratio_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=1.0,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_energy_ratio_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_energy_ratio_native_slice,
        ),
        RegisteredField(
            name="plasma_beta",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_plasma_beta_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=1.0,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_plasma_beta_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_plasma_beta_native_slice,
        ),
        RegisteredField(
            name="pressure",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_pressure_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_pressure_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_pressure_native_slice,
        ),
        RegisteredField(
            name="velocity_divergence",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_div_v_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="velocity_gradient",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_velocity_gradient_r2tfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="vorticity",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_vorticity_vfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        ),
        RegisteredField(
            name="vorticity_magnitude",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_vorticity_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="magnetic_divergence",
            field_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_divergence_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
            amr_leaves_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_divergence_amr_leaves,
            native_slice_loader_fn=load_snapshot.QuokkaSnapshot.load_3d_magnetic_divergence_native_slice,
        ),
        RegisteredField(
            name="current_density_magnitude",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_current_density_sfield,
            expected_properties=ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        ),
        RegisteredField(
            name="current_density",
            field_loader_fn=load_snapshot.QuokkaSnapshot.compute_current_density_vfield,
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
    *,
    field_name: str,
) -> type[field_models.AnyField_3D]:
    """Return the concrete field type `field_name` resolves to, read off its loader's return-type
    annotation. Doesn't load any data or call the loader."""
    field_loader_fn = REGISTERED_FIELD_LOOKUP[field_name].field_loader_fn
    return_type = inspect.signature(field_loader_fn).return_annotation
    if not isinstance(return_type, type):
        raise TypeError(f"loader for `{field_name}` has no return-type annotation.")
    return return_type


def validate_fields(
    *,
    field_names: list[str] | tuple[str, ...] | None,
    allowed_types: tuple[type, ...] | None = None,
) -> None:
    """Ensure every name in `field_names` is registered, and (if `allowed_types` is given) resolves
    to one of those types."""
    valid_field_names = set(
        REGISTERED_FIELD_LOOKUP.keys(),
    )
    if not field_names or not set(field_names).issubset(valid_field_names):
        raise ValueError(f"`field_names` must be a non-empty subset of: {sorted(valid_field_names)}.")
    if allowed_types is not None:
        for field_name in field_names:
            field_type = get_field_type(field_name=field_name)
            if not issubclass(field_type, allowed_types):
                allowed_names = sorted(allowed_type.__name__ for allowed_type in allowed_types)
                raise ValueError(
                    f"`{field_name}` resolves to {field_type.__name__}, which is not supported here;"
                    f" supported types: {allowed_names}.",
                )


def validate_fields_support_amr_leaves(
    *,
    field_names: list[str] | tuple[str, ...],
) -> None:
    """Ensure every name in `field_names` has a registered `amr_leaves_loader_fn`; lists every
    unsupported field at once (not just the first one hit), so a multi-field request fails
    upfront and completely instead of partway through processing. Assumes `field_names` are
    already known-registered (see `validate_fields`)."""
    unsupported_field_names = sorted(
        field_name for field_name in field_names
        if REGISTERED_FIELD_LOOKUP[field_name].amr_leaves_loader_fn is None
    )
    if unsupported_field_names:
        raise ValueError(f"the following fields do not support AMR-leaf loading: {unsupported_field_names}.")


def validate_fields_support_native_slice(
    *,
    field_names: list[str] | tuple[str, ...],
) -> None:
    """Ensure every name in `field_names` has a registered `native_slice_loader_fn`; lists
    every unsupported field at once (not just the first one hit), so a multi-field request
    fails upfront and completely instead of partway through processing. Assumes `field_names`
    are already known-registered (see `validate_fields`)."""
    unsupported_field_names = sorted(
        field_name for field_name in field_names
        if REGISTERED_FIELD_LOOKUP[field_name].native_slice_loader_fn is None
    )
    if unsupported_field_names:
        raise ValueError(
            f"the following fields do not support native-slice loading: {unsupported_field_names}."
        )


def field_supports_chunked_reader(
    *,
    field_name: str,
) -> bool:
    """Return `True` iff `field_name`'s loader accepts `use_chunked_reader`. Doesn't load any
    data or call the loader."""
    field_loader_fn = REGISTERED_FIELD_LOOKUP[field_name].field_loader_fn
    return "use_chunked_reader" in inspect.signature(field_loader_fn).parameters


def validate_fields_support_chunked_reader(
    *,
    field_names: list[str] | tuple[str, ...],
) -> None:
    """Ensure every name in `field_names` resolves to a loader accepting `use_chunked_reader`;
    lists every unsupported field at once (not just the first one hit), so a multi-field
    request fails upfront and completely instead of partway through processing with a raw
    `TypeError`. Assumes `field_names` are already known-registered (see `validate_fields`)."""
    unsupported_field_names = sorted(
        field_name for field_name in field_names if not field_supports_chunked_reader(field_name=field_name)
    )
    if unsupported_field_names:
        raise ValueError(
            f"the following fields do not support use_chunked_reader: {unsupported_field_names}.",
        )


## } MODULE
