## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import collections
import dataclasses
import typing

## personal
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_validation import validate_types

## local
from .._snapshot_readers import read_fields

##
## === DATA STRUCTURES
##


@dataclasses.dataclass(frozen=True)
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

YT_VFIELD_KEYS: dict[str, dict[str, typing.Any]] = {
    "momentum": {
        "keys": read_fields.create_boxlib_vkeys(field_name="GasMomentum"),
        "description": "Momentum density components: vec(m) = rho * vec(v)",
    },
    "magnetic": {
        "keys": read_fields.create_boxlib_vkeys(field_name="BField"),
        "description": "Magnetic field components (code units)",
    },
}

YT_SFIELD_KEYS: dict[str, dict[str, typing.Any]] = {
    "density": {
        "key": ("boxlib", "gasDensity"),
        "description": "Gas density field",
    },
    "total_energy": {
        "key": ("boxlib", "gasEnergy"),
        "description": "Total energy density: e_tot = e_int + e_kin + e_mag",
    },
    "magnetic_divergence": {
        "key": ("boxlib", "magnetic_divergence"),
        "description": "Magnetic divergence: div(b) computed on the code's native staggered mesh",
    },
}

##
## === CACHE OPERATOR CLASS
##


class LRUCache:
    """LRU cache for field objects, keyed by cache key."""

    _cache_lookup: collections.OrderedDict[str, field_models.ScalarField_3D | field_models.VectorField_3D]
    _max_size: int

    def __init__(
        self,
        *,
        max_size: int = 3,
    ) -> None:
        validate_types.ensure_finite_int(
            param=max_size,
            param_name="max_size",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        self._cache_lookup = collections.OrderedDict()
        self._max_size = int(max_size)

    def get_cached_field(
        self,
        *,
        cache_key: str,
    ):
        """Return cached value for `cache_key`, or None if not found."""
        cached_field = self._cache_lookup.get(cache_key)
        if cached_field is not None:
            self._cache_lookup.move_to_end(cache_key)
        return cached_field

    def cache_field(
        self,
        *,
        cache_key: str,
        field_data: field_models.ScalarField_3D | field_models.VectorField_3D,
    ) -> None:
        """Store `field_data` under `cache_key`; evict the LRU entry if at capacity."""
        self._cache_lookup[cache_key] = field_data
        self._cache_lookup.move_to_end(cache_key)
        while len(self._cache_lookup) > self._max_size:
            self._cache_lookup.popitem(last=False)

    def clear_cache(
        self,
    ) -> None:
        """Clear all cached fields."""
        self._cache_lookup.clear()


## } MODULE
