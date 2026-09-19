## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_arrays.farrays_3d import farray_operators
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    decompose_fields,
    field_models,
    compute_fields,
)
from jormi.ww_validation import validate_types

## local
## direct-name import, not the usual module import: `_snapshot_fields/__init__.py`
## re-exports this file's own contents, so `from .. import fields_protocol` would need
## the package fully resolved while it is still mid-import -- a real circular dependency
from ..fields_protocol import FieldsProtocol
from ..read_fields import HelmholtzKineticEnergy
from ..._snapshot_readers import read_fields
from ..._snapshot_readers.native_resolution import native_slice, native_values

##
## === DERIVE CLASS
##


class _DeriveEnergyFields:
    """Energy fields derived from a snapshot."""

    ##
    ## --- ENERGY FIELDS
    ##

    def compute_kinetic_energy_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute kinetic energy density: `e_kin = 0.5 * rho * |v|^2`. See `_load_3d_sarray` for
        `use_chunked_reader`."""
        rho_sfield_3d = self.load_3d_density_sfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        rho_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=rho_sfield_3d,
            param_name="<rho_sfield_3d>",
        )
        mom_vfield_3d = self.load_3d_momentum_vfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        mom_varray_3d = field_models.extract_3d_varray(
            vfield_3d=mom_vfield_3d,
            param_name="<mom_vfield_3d>",
        )
        rho_has_zeros = compute_array_stats.check_no_zero_values(
            array=rho_sarray_3d,
            param_name="<rho_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            E_kin_sarray_3d = 0.5 * farray_operators.compute_sum_of_varray_comps_squared(
                mom_varray_3d,
            ) / rho_sarray_3d
        if not rho_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=E_kin_sarray_3d,
                param_name="<E_kin_sfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=E_kin_sarray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_kin_sarray_3d,
            uniform_domain_3d=uniform_domain_3d,
            field_name="kinetic_energy",
            latex_label=r"E_\mathrm{kin}",
            sim_time=self.sim_time,
        )

    def load_3d_kinetic_energy_amr_leaves(
        self: FieldsProtocol,
    ) -> read_fields.AMRLeaves:
        """Load kinetic energy density `e_kin = 0.5 rho |v|^2` at every leaf cell across the full
        AMR hierarchy."""
        momentum_key_lookup = self._get_vfield_key_lookup(field_name="momentum")
        density_key = self._get_sfield_key(field_name="density")
        field_keys = tuple(momentum_key_lookup[axis]
                           for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER) + (density_key, )

        def derive_kinetic_energy_fn(
            raw_box_farray: numpy.ndarray,
        ) -> numpy.ndarray:
            momentum_box_varray = raw_box_farray[:3]
            density_box_farray = raw_box_farray[3]
            momentum_sq_box_farray = farray_operators.compute_sum_of_varray_comps_squared(momentum_box_varray)
            return 0.5 * momentum_sq_box_farray / density_box_farray

        return self._load_amr_leaves_of_derived_field(
            field_keys=field_keys,
            derive_fn=derive_kinetic_energy_fn,
        )

    def load_3d_kinetic_energy_native_slice(
        self: FieldsProtocol,
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Load kinetic energy density `e_kin = 0.5 rho |v|^2`, and its per-pixel native `dx`,
        on a genuine AMR-native slice."""
        momentum_key_lookup = self._get_vfield_key_lookup(field_name="momentum")
        density_key = self._get_sfield_key(field_name="density")
        field_keys = tuple(momentum_key_lookup[axis]
                           for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER) + (density_key, )

        def derive_kinetic_energy_fn(
            raw_sarray_stack: numpy.ndarray,
        ) -> numpy.ndarray:
            momentum_varray_2d = raw_sarray_stack[:3]
            density_sarray_2d = raw_sarray_stack[3]
            ## `farray_operators.compute_sum_of_varray_comps_squared` requires a 3D-domain (4D
            ## total) varray; a native slice only has 2 spatial dims, so this is done directly
            momentum_sq_sarray_2d = numpy.sum(momentum_varray_2d**2, axis=0)
            return 0.5 * momentum_sq_sarray_2d / density_sarray_2d

        return self._load_native_slice_of_derived_field(
            field_keys=field_keys,
            derive_fn=derive_kinetic_energy_fn,
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )

    def compute_magnetic_energy_sfield(
        self: FieldsProtocol,
        *,
        energy_prefactor: float = 0.5,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute magnetic energy density: `e_mag = alpha * |b|^2` with `alpha=0.5` by default. See
        `_load_3d_sarray` for `use_chunked_reader`."""
        validate_types.ensure_finite_float(
            param=energy_prefactor,
            param_name="energy_prefactor",
            allow_none=False,
        )
        magnetic_vfield_3d = self.load_3d_magnetic_vfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        return compute_fields.compute_magnetic_energy_density_sfield(
            magnetic_vfield_3d=magnetic_vfield_3d,
            energy_prefactor=energy_prefactor,
            field_name="magnetic_energy",
            latex_label=r"E_\mathrm{mag}",
        )

    def load_3d_magnetic_energy_amr_leaves(
        self: FieldsProtocol,
        *,
        energy_prefactor: float = 0.5,
    ) -> read_fields.AMRLeaves:
        """Load magnetic energy density `e_mag = alpha |b|^2` at every leaf cell across the full
        AMR hierarchy."""
        validate_types.ensure_finite_float(
            param=energy_prefactor,
            param_name="energy_prefactor",
            allow_none=False,
        )
        b_key_lookup = self._get_vfield_key_lookup(field_name="magnetic")
        field_keys = tuple(b_key_lookup[axis] for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER)

        def derive_magnetic_energy_fn(
            raw_box_farray: numpy.ndarray,
        ) -> numpy.ndarray:
            b_sq_box_farray = farray_operators.compute_sum_of_varray_comps_squared(raw_box_farray)
            return energy_prefactor * b_sq_box_farray

        return self._load_amr_leaves_of_derived_field(
            field_keys=field_keys,
            derive_fn=derive_magnetic_energy_fn,
        )

    def load_3d_magnetic_energy_native_slice(
        self: FieldsProtocol,
        *,
        energy_prefactor: float = 0.5,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Load magnetic energy density `e_mag = alpha |b|^2`, and its per-pixel native `dx`,
        on a genuine AMR-native slice."""
        validate_types.ensure_finite_float(
            param=energy_prefactor,
            param_name="energy_prefactor",
            allow_none=False,
        )
        b_key_lookup = self._get_vfield_key_lookup(field_name="magnetic")
        field_keys = tuple(b_key_lookup[axis] for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER)

        def derive_magnetic_energy_fn(
            raw_sarray_stack: numpy.ndarray,
        ) -> numpy.ndarray:
            ## `farray_operators.compute_sum_of_varray_comps_squared` requires a 3D-domain (4D
            ## total) varray; a native slice only has 2 spatial dims, so this is done directly
            b_sq_sarray_2d = numpy.sum(raw_sarray_stack**2, axis=0)
            return energy_prefactor * b_sq_sarray_2d

        return self._load_native_slice_of_derived_field(
            field_keys=field_keys,
            derive_fn=derive_magnetic_energy_fn,
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )

    def compute_internal_energy_sfield(
        self: FieldsProtocol,
        *,
        magnetic_energy_sfield_3d: field_models.ScalarField_3D | None = None,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute internal energy: `e_int = e_tot - e_kin - e_mag`; `e_mag = 0` if the snapshot did not store `vec(b)`.

        `magnetic_energy_sfield_3d` lets a caller that already computed `e_mag` pass it in,
        instead of paying for `|vec(b)|^2` a second time. See `_load_3d_sarray` for `use_chunked_reader`.
        """
        total_energy_sfield_3d = self.load_3d_total_energy_sfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        E_tot_sarray = field_models.extract_3d_sarray(
            sfield_3d=total_energy_sfield_3d,
            param_name="<E_tot_sfield_3d>",
        )
        kinetic_energy_sfield_3d = self.compute_kinetic_energy_sfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        E_kin_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=kinetic_energy_sfield_3d,
            param_name="<E_kin_sfield_3d>",
        )
        E_int_sarray = E_tot_sarray - E_kin_sarray_3d
        if self._is_vfield_keys_available(field_name="magnetic"):
            if magnetic_energy_sfield_3d is None:
                resolved_magnetic_energy_sfield_3d = self.compute_magnetic_energy_sfield(
                    amr_level=amr_level,
                    use_chunked_reader=use_chunked_reader,
                )
            else:
                resolved_magnetic_energy_sfield_3d = magnetic_energy_sfield_3d
            E_mag_sarray = field_models.extract_3d_sarray(
                sfield_3d=resolved_magnetic_energy_sfield_3d,
                param_name="<E_mag_sfield_3d>",
            )
            E_int_sarray -= E_mag_sarray
        compute_array_stats.check_no_nonfinite_values(
            array=E_int_sarray,
            param_name="<E_int_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=E_int_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_int_sarray,
            uniform_domain_3d=uniform_domain_3d,
            field_name="internal_energy",
            latex_label=r"E_\mathrm{int}",
            sim_time=self.sim_time,
        )

    def load_3d_internal_energy_amr_leaves(
        self: FieldsProtocol,
        *,
        magnetic_energy_leaves: read_fields.AMRLeaves | None = None,
    ) -> read_fields.AMRLeaves:
        """
        Load internal energy `e_int = e_tot - e_kin - e_mag` at every leaf cell across the full
        AMR hierarchy; `e_mag = 0` if the snapshot did not store `vec(b)`.

        `magnetic_energy_leaves` lets a caller that already computed `e_mag` pass it in,
        instead of paying for `|vec(b)|^2` a second time.
        """
        total_energy_leaves = self.load_3d_total_energy_amr_leaves()
        kinetic_energy_leaves = self.load_3d_kinetic_energy_amr_leaves()
        native_values.ensure_consistent_leaf_ordering(
            reference_leaves=total_energy_leaves,
            other_leaves=kinetic_energy_leaves,
        )
        internal_energy_values = total_energy_leaves.values - kinetic_energy_leaves.values
        if self._is_vfield_keys_available(field_name="magnetic"):
            if magnetic_energy_leaves is None:
                resolved_magnetic_energy_leaves = self.load_3d_magnetic_energy_amr_leaves()
            else:
                resolved_magnetic_energy_leaves = magnetic_energy_leaves
            native_values.ensure_consistent_leaf_ordering(
                reference_leaves=total_energy_leaves,
                other_leaves=resolved_magnetic_energy_leaves,
            )
            internal_energy_values = internal_energy_values - resolved_magnetic_energy_leaves.values
        compute_array_stats.check_no_nonfinite_values(
            array=internal_energy_values,
            param_name="<internal_energy_leaves.values>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=internal_energy_values,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return read_fields.AMRLeaves(
            values=internal_energy_values,
            cell_width=total_energy_leaves.cell_width,
            positions=total_energy_leaves.positions,
        )

    def load_3d_internal_energy_native_slice(
        self: FieldsProtocol,
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
        magnetic_energy_native_slice: tuple[numpy.ndarray, numpy.ndarray] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Load internal energy `e_int = e_tot - e_kin - e_mag`, and its per-pixel native `dx`, on
        a genuine AMR-native slice; `e_mag = 0` if the snapshot did not store `vec(b)`.

        `magnetic_energy_native_slice` lets a caller that already computed `e_mag` pass it in,
        instead of paying for `|vec(b)|^2` a second time.
        """
        total_energy_sarray_2d, total_energy_cell_width_2d = self.load_3d_total_energy_native_slice(
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )
        kinetic_energy_sarray_2d, kinetic_energy_cell_width_2d = self.load_3d_kinetic_energy_native_slice(
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )
        native_slice.ensure_consistent_slice_geometry(
            reference_cell_width_2d=total_energy_cell_width_2d,
            other_cell_width_2d=kinetic_energy_cell_width_2d,
        )
        internal_energy_sarray_2d = total_energy_sarray_2d - kinetic_energy_sarray_2d
        if self._is_vfield_keys_available(field_name="magnetic"):
            if magnetic_energy_native_slice is None:
                resolved_magnetic_energy_sarray_2d, resolved_magnetic_energy_cell_width_2d = self.load_3d_magnetic_energy_native_slice(
                    axis_to_slice=axis_to_slice,
                    slice_coordinate=slice_coordinate,
                )
            else:
                resolved_magnetic_energy_sarray_2d, resolved_magnetic_energy_cell_width_2d = magnetic_energy_native_slice
            native_slice.ensure_consistent_slice_geometry(
                reference_cell_width_2d=total_energy_cell_width_2d,
                other_cell_width_2d=resolved_magnetic_energy_cell_width_2d,
            )
            internal_energy_sarray_2d = internal_energy_sarray_2d - resolved_magnetic_energy_sarray_2d
        compute_array_stats.check_no_nonfinite_values(
            array=internal_energy_sarray_2d,
            param_name="<internal_energy_sarray_2d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=internal_energy_sarray_2d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return internal_energy_sarray_2d, total_energy_cell_width_2d

    def compute_pressure_sfield(
        self: FieldsProtocol,
        *,
        gamma: float = 5.0 / 3.0,
        magnetic_energy_sfield_3d: field_models.ScalarField_3D | None = None,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """
        Compute thermal pressure: `p = (gamma - 1) * e_int`.

        `magnetic_energy_sfield_3d` lets a caller that already computed `e_mag` pass it in,
        instead of paying for `|vec(b)|^2` a second time. See `_load_3d_sarray` for `use_chunked_reader`.
        """
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        internal_energy_sfield_3d = self.compute_internal_energy_sfield(
            magnetic_energy_sfield_3d=magnetic_energy_sfield_3d,
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        E_int_sarray = field_models.extract_3d_sarray(
            sfield_3d=internal_energy_sfield_3d,
            param_name="<E_int_sfield_3d>",
        )
        p_sarray = (gamma - 1.0) * E_int_sarray
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=p_sarray,
            uniform_domain_3d=uniform_domain_3d,
            field_name="pressure",
            latex_label=r"p",
            sim_time=self.sim_time,
        )

    def load_3d_pressure_amr_leaves(
        self: FieldsProtocol,
        *,
        gamma: float = 5.0 / 3.0,
        magnetic_energy_leaves: read_fields.AMRLeaves | None = None,
    ) -> read_fields.AMRLeaves:
        """
        Load thermal pressure `p = (gamma - 1) * e_int` at every leaf cell across the full AMR
        hierarchy.

        `magnetic_energy_leaves` lets a caller that already computed `e_mag` pass it in,
        instead of paying for `|vec(b)|^2` a second time.
        """
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        internal_energy_leaves = self.load_3d_internal_energy_amr_leaves(
            magnetic_energy_leaves=magnetic_energy_leaves,
        )
        pressure_values = (gamma - 1.0) * internal_energy_leaves.values
        return read_fields.AMRLeaves(
            values=pressure_values,
            cell_width=internal_energy_leaves.cell_width,
            positions=internal_energy_leaves.positions,
        )

    def load_3d_pressure_native_slice(
        self: FieldsProtocol,
        *,
        gamma: float = 5.0 / 3.0,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
        magnetic_energy_native_slice: tuple[numpy.ndarray, numpy.ndarray] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Load thermal pressure `p = (gamma - 1) * e_int`, and its per-pixel native `dx`, on a
        genuine AMR-native slice.

        `magnetic_energy_native_slice` lets a caller that already computed `e_mag` pass it in,
        instead of paying for `|vec(b)|^2` a second time.
        """
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        internal_energy_sarray_2d, internal_energy_cell_width_2d = self.load_3d_internal_energy_native_slice(
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
            magnetic_energy_native_slice=magnetic_energy_native_slice,
        )
        pressure_sarray_2d = (gamma - 1.0) * internal_energy_sarray_2d
        return pressure_sarray_2d, internal_energy_cell_width_2d

    def compute_helmholtz_kinetic_energy(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> HelmholtzKineticEnergy:
        """
        Compute Helmholtz-decomposed kinetic energies; splits `vec(v)` into `vec(v)_div + vec(v)_sol + vec(v)_bulk`.

        `use_chunked_reader` only affects reading `vec(v)`/`rho` in (see `_load_3d_sarray`); the
        Helmholtz decomposition itself still needs the full domain array, the same as an FFT.
        """
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        v_vfield_3d = self.compute_velocity_vfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        rho_sfield_3d = self.load_3d_density_sfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        rho_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=rho_sfield_3d,
            param_name="<rho_sfield_3d>",
        )
        helmholtz_vfields = decompose_fields.compute_helmholtz_decomposed_fields(vfield_3d=v_vfield_3d)
        v_div_varray = field_models.extract_3d_varray(
            vfield_3d=helmholtz_vfields.div_vfield_3d,
            param_name="<div_vfield_3d>",
        )
        v_sol_varray = field_models.extract_3d_varray(
            vfield_3d=helmholtz_vfields.sol_vfield_3d,
            param_name="<sol_vfield_3d>",
        )
        v_bulk_varray = field_models.extract_3d_varray(
            vfield_3d=helmholtz_vfields.bulk_vfield_3d,
            param_name="<bulk_vfield_3d>",
        )
        E_kin_div_sarray = 0.5 * rho_sarray_3d * farray_operators.compute_sum_of_varray_comps_squared(
            v_div_varray,
        )
        E_kin_sol_sarray = 0.5 * rho_sarray_3d * farray_operators.compute_sum_of_varray_comps_squared(
            v_sol_varray,
        )
        E_kin_bulk_sarray = 0.5 * rho_sarray_3d * farray_operators.compute_sum_of_varray_comps_squared(
            v_bulk_varray,
        )
        compute_array_stats.check_no_nonfinite_values(
            array=E_kin_div_sarray,
            param_name="<E_kin_div_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=E_kin_div_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        compute_array_stats.check_no_nonfinite_values(
            array=E_kin_sol_sarray,
            param_name="<E_kin_sol_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=E_kin_sol_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        compute_array_stats.check_no_nonfinite_values(
            array=E_kin_bulk_sarray,
            param_name="<E_kin_bulk_sfield_3d>",
            raise_error=False,
        )
        compute_array_stats.make_nonfinites_zero(
            array=E_kin_bulk_sarray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        E_kin_div_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_kin_div_sarray,
            uniform_domain_3d=uniform_domain_3d,
            field_name="kinetic_energy_div",
            latex_label=r"E_{\mathrm{kin}, \parallel}",
            sim_time=self.sim_time,
        )
        E_kin_sol_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_kin_sol_sarray,
            uniform_domain_3d=uniform_domain_3d,
            field_name="kinetic_energy_sol",
            latex_label=r"E_{\mathrm{kin}, \perp}",
            sim_time=self.sim_time,
        )
        E_kin_bulk_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_kin_bulk_sarray,
            uniform_domain_3d=uniform_domain_3d,
            field_name="kinetic_energy_bulk",
            latex_label=r"E_{\mathrm{kin}, \mathrm{bulk}}",
            sim_time=self.sim_time,
        )
        return HelmholtzKineticEnergy(
            E_kin_div_sfield_3d=E_kin_div_sfield_3d,
            E_kin_sol_sfield_3d=E_kin_sol_sfield_3d,
            E_kin_bulk_sfield_3d=E_kin_bulk_sfield_3d,
        )

    def compute_div_kinetic_energy_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute irrotational kinetic energy density: `e_kin,div = 0.5 rho |v_div|^2`. See
        `compute_helmholtz_kinetic_energy` for `use_chunked_reader`."""
        helmholtz_e_kin = self.compute_helmholtz_kinetic_energy(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        return helmholtz_e_kin.E_kin_div_sfield_3d

    def compute_sol_kinetic_energy_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute solenoidal kinetic energy density: `e_kin,sol = 0.5 rho |v_sol|^2`. See
        `compute_helmholtz_kinetic_energy` for `use_chunked_reader`."""
        helmholtz_e_kin = self.compute_helmholtz_kinetic_energy(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        return helmholtz_e_kin.E_kin_sol_sfield_3d

    def compute_bulk_kinetic_energy_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute bulk kinetic energy density: `e_kin,bulk = 0.5 rho |v_bulk|^2`. See
        `compute_helmholtz_kinetic_energy` for `use_chunked_reader`."""
        helmholtz_e_kin = self.compute_helmholtz_kinetic_energy(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        return helmholtz_e_kin.E_kin_bulk_sfield_3d


## } MODULE
