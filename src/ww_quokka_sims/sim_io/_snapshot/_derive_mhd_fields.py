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
from ._fields_protocol import FieldsProtocol

##
## === DERIVE CLASS
##


class _DeriveMHDFields:
    """MHD fields derived from a snapshot."""

    ##
    ## --- MHD COMPOSITE FIELDS
    ##

    def compute_cross_helicity_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute cross helicity density: `vec(v) cdot vec(b)`."""
        v_vfield_3d = self.compute_velocity_vfield(amr_level=amr_level)
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_dot_product(
            f_vfield_3d=v_vfield_3d,
            g_vfield_3d=b_vfield_3d,
            field_name="cross_helicity",
            latex_label=r"\vec{v}\cdot\vec{b}",
        )

    def compute_lorentz_force_vfield(
        self: FieldsProtocol,
        grad_order: int = 2,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        """Compute Lorentz force: `curl[vec(b)] x vec(b)`."""
        j_vfield_3d = self.compute_current_density_vfield(grad_order=grad_order, amr_level=amr_level)
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_cross_product(
            f_vfield_3d=j_vfield_3d,
            g_vfield_3d=b_vfield_3d,
            field_name="lorentz_force",
            latex_label=r"(\nabla\times\vec{b})\times\vec{b}",
        )

    def compute_lorentz_force_sfield(
        self: FieldsProtocol,
        grad_order: int = 2,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute Lorentz force magnitude: `| curl[vec(b)] x vec(b) |`."""
        lorentz_force_vfield_3d = self.compute_lorentz_force_vfield(grad_order=grad_order, amr_level=amr_level)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=lorentz_force_vfield_3d,
            field_name="lorentz_force_magnitude",
            latex_label=r"|(\nabla\times\vec{b})\times\vec{b}|",
        )

    def compute_energy_ratio_sfield(
        self: FieldsProtocol,
        energy_prefactor: float = 0.5,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute magnetic-to-kinetic energy ratio: `e_mag / e_kin`."""
        E_mag_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.compute_magnetic_energy_sfield(energy_prefactor=energy_prefactor, amr_level=amr_level),
            param_name="<E_mag_sfield_3d>",
        )
        E_kin_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.compute_kinetic_energy_sfield(amr_level=amr_level),
            param_name="<E_kin_sfield_3d>",
        )
        E_kin_has_zeros = compute_array_stats.check_no_zero_values(
            array=E_kin_sarray_3d,
            param_name="<E_kin_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            energy_ratio_sarray_3d = E_mag_sarray_3d / E_kin_sarray_3d
        if not E_kin_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=energy_ratio_sarray_3d,
                param_name="<E_ratio_sfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=energy_ratio_sarray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=energy_ratio_sarray_3d,
            uniform_domain_3d=self.load_3d_uniform_domain(amr_level=amr_level),
            field_name="energy_ratio",
            latex_label=r"E_\mathrm{mag} / E_\mathrm{kin}",
            sim_time=self.sim_time,
        )

    def compute_poynting_flux_vfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        """Compute Poynting-flux-like vector: `vec(b) x [vec(v) x vec(b)]`."""
        v_vfield_3d = self.compute_velocity_vfield(amr_level=amr_level)
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        vxb_vfield_3d = field_operators.compute_vfield_cross_product(
            f_vfield_3d=v_vfield_3d,
            g_vfield_3d=b_vfield_3d,
            field_name="velocity_cross_magnetic",
            latex_label=r"\vec{v}\times\vec{b}",
        )
        return field_operators.compute_vfield_cross_product(
            f_vfield_3d=b_vfield_3d,
            g_vfield_3d=vxb_vfield_3d,
            field_name="poynting_flux",
            latex_label=r"\vec{b}\times(\vec{v}\times\vec{b})",
        )


## } MODULE
