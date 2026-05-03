## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_arrays.farrays_3d import farray_operators
from jormi.ww_fields.fields_3d import (
    decompose_fields,
    field_models,
    compute_fields,
)
from jormi.ww_validation import validate_types

## local
from ._read_fields import HelmholtzKineticEnergy
from ._fields_protocol import _FieldsProtocol

##
## === DERIVE CLASS
##


class _DeriveEnergyFields:
    """Energy fields derived from a snapshot."""

    ##
    ## --- ENERGY FIELDS
    ##

    def compute_kinetic_energy_sfield(
        self: _FieldsProtocol,
    ) -> field_models.ScalarField_3D:
        """Compute kinetic energy density: `E_kin = 0.5 * rho * |v|^2`."""
        rho_sfield_3d = self.load_3d_density_sfield()
        rho_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=rho_sfield_3d,
            param_name="<rho_sfield_3d>",
        )
        mom_vfield_3d = self.load_3d_momentum_vfield()
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
        udomain_3d = self.load_3d_uniform_domain()
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_kin_sarray_3d,
            udomain_3d=udomain_3d,
            field_label=r"E_\mathrm{kin}",
            sim_time=self.sim_time,
        )

    def compute_magnetic_energy_sfield(
        self: _FieldsProtocol,
        energy_prefactor: float = 0.5,
        field_label: str = r"E_\mathrm{mag}",
    ) -> field_models.ScalarField_3D:
        """Compute magnetic energy density: `E_mag = alpha * |b|^2` with `alpha=0.5` by default."""
        validate_types.ensure_finite_float(
            param=energy_prefactor,
            param_name="energy_prefactor",
            allow_none=False,
        )
        b_vfield_3d = self.load_3d_magnetic_vfield()
        return compute_fields.compute_magnetic_energy_density_sfield(
            vfield_3d_b=b_vfield_3d,
            energy_prefactor=energy_prefactor,
            field_label=field_label,
        )

    def compute_internal_energy_sfield(
        self: _FieldsProtocol,
    ) -> field_models.ScalarField_3D:
        """Compute internal energy: `E_int = E_tot - E_kin - E_mag`; `E_mag = 0` if the snapshot did not store `vec(b)`."""
        E_tot_sarray = field_models.extract_3d_sarray(
            sfield_3d=self.load_3d_total_energy_sfield(),
            param_name="<E_tot_sfield_3d>",
        )
        E_kin_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.compute_kinetic_energy_sfield(),
            param_name="<E_kin_sfield_3d>",
        )
        E_int_sarray = E_tot_sarray - E_kin_sarray_3d
        if self._is_vfield_keys_available("magnetic"):
            E_mag_sarray = field_models.extract_3d_sarray(
                sfield_3d=self.compute_magnetic_energy_sfield(),
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
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_int_sarray,
            udomain_3d=self.load_3d_uniform_domain(),
            field_label=r"E_\mathrm{int}",
            sim_time=self.sim_time,
        )

    def compute_pressure_sfield(
        self: _FieldsProtocol,
        gamma: float = 5.0 / 3.0,
    ) -> field_models.ScalarField_3D:
        """Compute thermal pressure: `p = (gamma - 1) * E_int`."""
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        E_int_sarray = field_models.extract_3d_sarray(
            sfield_3d=self.compute_internal_energy_sfield(),
            param_name="<E_int_sfield_3d>",
        )
        p_sarray = (gamma - 1.0) * E_int_sarray
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=p_sarray,
            udomain_3d=self.load_3d_uniform_domain(),
            field_label=r"p",
            sim_time=self.sim_time,
        )

    def compute_helmholtz_kinetic_energy(
        self: _FieldsProtocol,
    ) -> HelmholtzKineticEnergy:
        """Compute Helmholtz-decomposed kinetic energies; splits `vec(v)` into `vec(v)_div + vec(v)_sol + vec(v)_bulk`."""
        udomain_3d = self.load_3d_uniform_domain()
        v_vfield_3d = self.compute_velocity_vfield()
        rho_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.load_3d_density_sfield(),
            param_name="<rho_sfield_3d>",
        )
        helmholtz_vfields = decompose_fields.compute_helmholtz_decomposed_fields(vfield_3d=v_vfield_3d)
        v_div_varray = field_models.extract_3d_varray(
            vfield_3d=helmholtz_vfields.vfield_3d_div,
            param_name="<vfield_3d_div>",
        )
        v_sol_varray = field_models.extract_3d_varray(
            vfield_3d=helmholtz_vfields.vfield_3d_sol,
            param_name="<vfield_3d_sol>",
        )
        v_bulk_varray = field_models.extract_3d_varray(
            vfield_3d=helmholtz_vfields.vfield_3d_bulk,
            param_name="<vfield_3d_bulk>",
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
            udomain_3d=udomain_3d,
            field_label=r"E_{\mathrm{kin}, \parallel}",
            sim_time=self.sim_time,
        )
        E_kin_sol_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_kin_sol_sarray,
            udomain_3d=udomain_3d,
            field_label=r"E_{\mathrm{kin}, \perp}",
            sim_time=self.sim_time,
        )
        E_kin_bulk_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=E_kin_bulk_sarray,
            udomain_3d=udomain_3d,
            field_label=r"E_{\mathrm{kin}, \mathrm{bulk}}",
            sim_time=self.sim_time,
        )
        return HelmholtzKineticEnergy(
            E_kin_div_sfield_3d=E_kin_div_sfield_3d,
            E_kin_sol_sfield_3d=E_kin_sol_sfield_3d,
            E_kin_bulk_sfield_3d=E_kin_bulk_sfield_3d,
        )

    def compute_div_kinetic_energy_sfield(
        self: _FieldsProtocol,
    ) -> field_models.ScalarField_3D:
        """Compute irrotational kinetic energy density: `E_kin,div = 0.5 rho |v_div|^2`."""
        helmholtz_E_kin = self.compute_helmholtz_kinetic_energy()
        return helmholtz_E_kin.E_kin_div_sfield_3d

    def compute_sol_kinetic_energy_sfield(
        self: _FieldsProtocol,
    ) -> field_models.ScalarField_3D:
        """Compute solenoidal kinetic energy density: `E_kin,sol = 0.5 rho |v_sol|^2`."""
        helmholtz_E_kin = self.compute_helmholtz_kinetic_energy()
        return helmholtz_E_kin.E_kin_sol_sfield_3d

    def compute_bulk_kinetic_energy_sfield(
        self: _FieldsProtocol,
    ) -> field_models.ScalarField_3D:
        """Compute bulk kinetic energy density: `E_kin,bulk = 0.5 rho |v_bulk|^2`."""
        helmholtz_E_kin = self.compute_helmholtz_kinetic_energy()
        return helmholtz_E_kin.E_kin_bulk_sfield_3d


## } MODULE
