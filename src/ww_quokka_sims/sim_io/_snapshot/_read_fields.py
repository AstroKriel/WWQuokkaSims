## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, TypeAlias

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    field_models,
)
from jormi.ww_validation import validate_types

##
## === DATA STRUCTURES
##

FieldKey: TypeAlias = tuple[str, str]

## boxlib uses "x-", "y-", "z-" prefixes for vector component field names
_BOXLIB_XYZ_LABELS: dict[cartesian_axes.CartesianAxis_3D, str] = {
    cartesian_axes.CartesianAxis_3D.X0: "x",
    cartesian_axes.CartesianAxis_3D.X1: "y",
    cartesian_axes.CartesianAxis_3D.X2: "z",
}


@dataclass(frozen=True)
class HelmholtzKineticEnergy:
    """Helmholtz-decomposed kinetic energy fields: divergent, solenoidal, and bulk."""

    E_kin_div_sfield_3d: field_models.ScalarField_3D
    E_kin_sol_sfield_3d: field_models.ScalarField_3D
    E_kin_bulk_sfield_3d: field_models.ScalarField_3D

    def __post_init__(
        self,
    ) -> None:
        field_models.ensure_3d_sfield(
            sfield_3d=self.E_kin_div_sfield_3d,
            param_name="<E_kin_div_sfield_3d>",
        )
        field_models.ensure_3d_sfield(
            sfield_3d=self.E_kin_sol_sfield_3d,
            param_name="<E_kin_sol_sfield_3d>",
        )
        field_models.ensure_3d_sfield(
            sfield_3d=self.E_kin_bulk_sfield_3d,
            param_name="<E_kin_bulk_sfield_3d>",
        )


##
## === YT FIELD MAPPINGS
##


def create_boxlib_vkeys(
    field_name: str,
) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
    """Map `CartesianAxis_3D` to yt field keys using the pattern `("boxlib", "<axis>-<field_name>")` for each axis."""
    return {
        axis: ("boxlib", f"{_BOXLIB_XYZ_LABELS[axis]}-{field_name}")
        for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER
    }


YT_VFIELD_KEYS: dict[str, dict[str, Any]] = {
    "momentum": {
        "keys": create_boxlib_vkeys("GasMomentum"),
        "description": "Momentum density components: vec(m) = rho * vec(v)",
    },
    "magnetic": {
        "keys": create_boxlib_vkeys("BField"),
        "description": "Magnetic field components (code units)",
    },
}

YT_SFIELD_KEYS: dict[str, dict[str, Any]] = {
    "density": {
        "key": ("boxlib", "gasDensity"),
        "description": "Gas density field",
    },
    "total_energy": {
        "key": ("boxlib", "gasEnergy"),
        "description": "Total energy density: E_tot = E_int + E_kin + E_mag",
    },
}

##
## === CACHE OPERATOR CLASS
##


class LRUCache:
    """LRU cache for field objects, keyed by field name."""

    _cache_lookup: OrderedDict[str, field_models.ScalarField_3D | field_models.VectorField_3D]
    _max_size: int

    def __init__(
        self,
        max_size: int = 3,
    ) -> None:
        validate_types.ensure_finite_int(
            param=max_size,
            param_name="max_size",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        self._cache_lookup = OrderedDict()
        self._max_size = int(max_size)

    def get_cached_field(
        self,
        field_name: str,
    ):
        """Return cached value for `field_name`, or None if not found."""
        cached_field = self._cache_lookup.get(field_name)
        if cached_field is not None:
            self._cache_lookup.move_to_end(field_name)
        return cached_field

    def cache_field(
        self,
        field_name: str,
        field_data: field_models.ScalarField_3D | field_models.VectorField_3D,
    ) -> None:
        """Store `field_data` under `field_name`; evict the LRU entry if at capacity."""
        self._cache_lookup[field_name] = field_data
        self._cache_lookup.move_to_end(field_name)
        while len(self._cache_lookup) > self._max_size:
            self._cache_lookup.popitem(last=False)

    def clear_cache(
        self,
    ) -> None:
        """Clear all cached fields."""
        self._cache_lookup.clear()


## } MODULE
