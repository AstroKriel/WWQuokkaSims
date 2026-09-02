## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_io import manage_log

## local
## direct-name import, not the usual module import: `_snapshot_fields/__init__.py`
## re-exports this file's own contents, so `from . import fields_protocol`/`read_fields`
## would need the package fully resolved while it is still mid-import -- a real circular
## dependency
from .fields_protocol import FieldsProtocol

##
## === LOAD CLASS
##


class _LoadStoredFields:
    """Fields read directly from the snapshot (not computed from other fields)."""

    ##
    ## --- STORED FIELDS
    ##

    def _field_cache_key(
        self: FieldsProtocol,
        field_name: str,
        *,
        amr_level: int,
        use_chunked_reader: bool = False,
    ) -> str:
        """Build the `_field_cache` key for `field_name` at `amr_level`."""
        return f"{field_name}:level-{amr_level}:reader-{'chunked' if use_chunked_reader else 'whole_domain'}"

    def load_3d_density_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Load gas density: `rho`. See `_load_3d_sarray` for `use_chunked_reader`."""
        cache_key = self._field_cache_key("density", amr_level=amr_level, use_chunked_reader=use_chunked_reader)
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.ScalarField_3D):
            return cached_field
        rho_key = self._get_sfield_key("density")
        rho_sfield_3d = self.load_3d_sfield(
            field_key=rho_key,
            field_name="density",
            latex_label=r"\rho",
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=rho_sfield_3d,
        )
        return rho_sfield_3d

    def load_3d_momentum_vfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        """Load momentum field: `vec(m) = rho vec(v)`. See `_load_3d_sarray` for `use_chunked_reader`."""
        cache_key = self._field_cache_key("momentum", amr_level=amr_level, use_chunked_reader=use_chunked_reader)
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.VectorField_3D):
            return cached_field
        mom_key_lookup = self._get_vfield_key_lookup("momentum")
        mom_vfield_3d = self.load_3d_vfield(
            vfield_key_lookup=mom_key_lookup,
            field_name="momentum",
            latex_label=r"\rho \,\vec{v}",
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=mom_vfield_3d,
        )
        return mom_vfield_3d

    def load_3d_magnetic_vfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        """Load magnetic field: `vec(b)`. See `_load_3d_sarray` for `use_chunked_reader`."""
        cache_key = self._field_cache_key("magnetic", amr_level=amr_level, use_chunked_reader=use_chunked_reader)
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.VectorField_3D):
            return cached_field
        b_key_lookup = self._get_vfield_key_lookup("magnetic")
        b_vfield_3d = self.load_3d_vfield(
            vfield_key_lookup=b_key_lookup,
            field_name="magnetic",
            latex_label=r"\vec{b}",
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=b_vfield_3d,
        )
        return b_vfield_3d

    def load_3d_total_energy_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Load total energy: `e_tot = e_int + e_kin + e_mag` (code units)."""
        cache_key = self._field_cache_key("total_energy", amr_level=amr_level)
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.ScalarField_3D):
            return cached_field
        E_tot_key = self._get_sfield_key("total_energy")
        E_tot_sfield_3d = self.load_3d_sfield(
            field_key=E_tot_key,
            field_name="total_energy",
            latex_label=r"E_\mathrm{tot}",
            amr_level=amr_level,
        )
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=E_tot_sfield_3d,
        )
        return E_tot_sfield_3d

    def load_3d_magnetic_divergence_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """
        Load magnetic field divergence: div(b).

        Quokka's native value, computed on its div-preserving staggered mesh, is used when available.
        Otherwise, a fallback estimate using a different stencil is calculated. The native value
        requires `derived_vars = "magnetic_divergence"` in the param TOML file.
        """
        cache_key = self._field_cache_key("magnetic_divergence", amr_level=amr_level)
        cached_field = self._field_cache.get_cached_field(cache_key)
        if isinstance(cached_field, field_models.ScalarField_3D):
            return cached_field
        div_b_key = self._resolve_sfield_key("magnetic_divergence")
        if self.is_field_key_available(field_key=div_b_key):
            div_b_sfield_3d = self.load_3d_sfield(
                field_key=div_b_key,
                field_name="magnetic_divergence",
                latex_label=r"\nabla\cdot\vec{b}",
                amr_level=amr_level,
            )
        else:
            manage_log.log_warning(
                text=(
                    f"native `magnetic_divergence` field was not found in {self.snapshot_dir}; falling back "
                    "to an estimated field instead. Set derived_vars = \"magnetic_divergence\" in the "
                    "param TOML file to get the more accurate, solver-native value instead."
                ),
            )
            div_b_sfield_3d = self.compute_div_b_sfield(amr_level=amr_level)
        self._field_cache.cache_field(
            cache_key=cache_key,
            field_data=div_b_sfield_3d,
        )
        return div_b_sfield_3d


## } MODULE
