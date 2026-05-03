## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields.fields_3d import (
    field_models,
    field_operators,
)

## local
from ._fields_protocol import _FieldsProtocol

##
## === DERIVE CLASS
##


class _DeriveMHDFields:
    """MHD composite field computations."""

    ##
    ## --- MHD COMPOSITE FIELDS
    ##

    def compute_cross_helicity_sfield(
        self: _FieldsProtocol,
    ) -> field_models.ScalarField_3D:
        """Compute cross helicity density: `vec(v) cdot vec(b)`."""
        v_vfield_3d = self.compute_velocity_vfield()
        b_vfield_3d = self.load_magnetic_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_3d_a=v_vfield_3d,
            vfield_3d_b=b_vfield_3d,
            field_label=r"\vec{v}\cdot\vec{b}",
        )

    def compute_lorentz_force_vfield(
        self: _FieldsProtocol,
        grad_order: int = 2,
    ) -> field_models.VectorField_3D:
        """Compute Lorentz force: `curl[vec(b)] x vec(b)`."""
        j_vfield_3d = self.compute_current_density_vfield(grad_order=grad_order)
        b_vfield_3d = self.load_magnetic_vfield()
        return field_operators.compute_vfield_cross_product(
            vfield_3d_a=j_vfield_3d,
            vfield_3d_b=b_vfield_3d,
            field_label=r"(\nabla\times\vec{b})\times\vec{b}",
        )

    def compute_lorentz_force_sfield(
        self: _FieldsProtocol,
        grad_order: int = 2,
    ) -> field_models.ScalarField_3D:
        """Compute Lorentz force magnitude: `| curl[vec(b)] x vec(b) |`."""
        lf_vfield_3d = self.compute_lorentz_force_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=lf_vfield_3d,
            field_label=r"|(\nabla\times\vec{b})\times\vec{b}|",
        )

    def compute_energy_ratio_sfield(
        self: _FieldsProtocol,
        energy_prefactor: float = 0.5,
    ) -> field_models.ScalarField_3D:
        """Compute magnetic-to-kinetic energy ratio: `E_mag / E_kin`."""
        Emag_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.compute_magnetic_energy_sfield(energy_prefactor=energy_prefactor),
            param_name="<Emag_sfield_3d>",
        )
        Ekin_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.compute_kinetic_energy_sfield(),
            param_name="<Ekin_sfield_3d>",
        )
        Ekin_has_zeros = compute_array_stats.check_no_zero_values(
            array=Ekin_sarray_3d,
            param_name="<Ekin_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            Eratio_sarray_3d = Emag_sarray_3d / Ekin_sarray_3d
        if not Ekin_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=Eratio_sarray_3d,
                param_name="<Eratio_sfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=Eratio_sarray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=Eratio_sarray_3d,
            udomain_3d=self.load_uniform_domain(),
            field_label=r"E_\mathrm{mag} / E_\mathrm{kin}",
            sim_time=self.sim_time,
        )

    def compute_poynting_flux_vfield(
        self: _FieldsProtocol,
    ) -> field_models.VectorField_3D:
        """Compute Poynting-flux-like vector: `vec(b) x [vec(v) x vec(b)]`."""
        v_vfield_3d = self.compute_velocity_vfield()
        b_vfield_3d = self.load_magnetic_vfield()
        vxb_vfield_3d = field_operators.compute_vfield_cross_product(
            vfield_3d_a=v_vfield_3d,
            vfield_3d_b=b_vfield_3d,
            field_label=r"\vec{v}\times\vec{b}",
        )
        return field_operators.compute_vfield_cross_product(
            vfield_3d_a=b_vfield_3d,
            vfield_3d_b=vxb_vfield_3d,
            field_label=r"\vec{b}\times(\vec{v}\times\vec{b})",
        )


## } MODULE
