## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields.fields_3d import (
    field_models,
    field_operators,
)
from jormi.ww_validation import validate_types

## local
from ._fields_protocol import FieldsProtocol

##
## === DERIVE CLASS
##


class _DeriveMagneticFields:
    """Magnetic fields derived from a snapshot."""

    ##
    ## --- MAGNETIC FIELDS
    ##

    def compute_alfven_speed_vfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        """Compute Alfven speed: `vec(v_A) = vec(b) / sqrt(rho)`."""
        b_varray_3d = field_models.extract_3d_varray(
            vfield_3d=self.load_3d_magnetic_vfield(amr_level=amr_level),
            param_name="<b_vfield_3d>",
        )
        rho_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.load_3d_density_sfield(amr_level=amr_level),
            param_name="<rho_sfield_3d>",
        )
        rho_has_zeros = compute_array_stats.check_no_zero_values(
            array=rho_sarray_3d,
            param_name="<rho_sfield_3d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            v_A_varray_3d = b_varray_3d / numpy.sqrt(rho_sarray_3d)[numpy.newaxis, ...]
        if not rho_has_zeros:
            ## warns if nonfinites arise from a source other than zero rho
            compute_array_stats.check_no_nonfinite_values(
                array=v_A_varray_3d,
                param_name="<v_A_vfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=v_A_varray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=v_A_varray_3d,
            uniform_domain_3d=self.load_3d_uniform_domain(amr_level=amr_level),
            field_name="alfven_velocity",
            latex_label=r"\vec{v}_A",
            sim_time=self.sim_time,
        )

    def compute_alfven_speed_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute Alfven speed magnitude: `|vec(v_A)|`."""
        v_A_vfield_3d = self.compute_alfven_speed_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=v_A_vfield_3d,
            field_name="alfven_speed",
            latex_label=r"|\vec{v}_A|",
        )

    def compute_div_b_sfield(
        self: FieldsProtocol,
        grad_order: int = 2,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Approximate magnetic divergence: `div[vec(b)]` (fallback for `load_3d_magnetic_divergence_sfield`)."""
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_divergence(
            vfield_3d=b_vfield_3d,
            field_name="magnetic_divergence",
            latex_label=r"\nabla\cdot\vec{b}",
            grad_order=grad_order,
        )

    def compute_current_density_vfield(
        self: FieldsProtocol,
        grad_order: int = 2,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        """Compute current density: `curl[vec(b)]`."""
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_curl(
            vfield_3d=b_vfield_3d,
            field_name="current_density",
            latex_label=r"\nabla\times\vec{b}",
            grad_order=grad_order,
        )

    def compute_current_density_sfield(
        self: FieldsProtocol,
        grad_order: int = 2,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute current magnitude: `|curl[vec(b)]|`."""
        j_vfield_3d = self.compute_current_density_vfield(grad_order=grad_order, amr_level=amr_level)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=j_vfield_3d,
            field_name="current_density_magnitude",
            latex_label=r"|\nabla\times\vec{b}|",
        )

    def compute_current_helicity_sfield(
        self: FieldsProtocol,
        grad_order: int = 2,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute current helicity density: `curl[vec(b)] cdot vec(b)`."""
        j_vfield_3d = self.compute_current_density_vfield(grad_order=grad_order, amr_level=amr_level)
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_dot_product(
            f_vfield_3d=j_vfield_3d,
            g_vfield_3d=b_vfield_3d,
            field_name="current_helicity",
            latex_label=r"(\nabla\times\vec{b})\cdot\vec{b}",
        )

    def compute_plasma_beta_sfield(
        self: FieldsProtocol,
        gamma: float = 5.0 / 3.0,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute plasma beta: `beta = 2 p / |vec(b)|^2`."""
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        energy_prefactor = 0.5
        e_mag_sfield_3d = self.compute_magnetic_energy_sfield(energy_prefactor=energy_prefactor, amr_level=amr_level)
        p_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=self.compute_pressure_sfield(
                gamma=gamma,
                magnetic_energy_sfield_3d=e_mag_sfield_3d,
                amr_level=amr_level,
            ),
            param_name="<p_sfield_3d>",
        )
        e_mag_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=e_mag_sfield_3d,
            param_name="<E_mag_sfield_3d>",
        )
        b_sq_sarray_3d = e_mag_sarray_3d / energy_prefactor
        b_sq_has_zeros = compute_array_stats.check_no_zero_values(
            array=b_sq_sarray_3d,
            param_name="<|b|^2>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            beta_sarray_3d = 2.0 * p_sarray_3d / b_sq_sarray_3d
        if not b_sq_has_zeros:
            ## warns if nonfinites arise from a source other than zero |b|^2
            compute_array_stats.check_no_nonfinite_values(
                array=beta_sarray_3d,
                param_name="<beta_sfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=beta_sarray_3d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=beta_sarray_3d,
            uniform_domain_3d=self.load_3d_uniform_domain(amr_level=amr_level),
            field_name="plasma_beta",
            latex_label=r"\beta",
            sim_time=self.sim_time,
        )


## } MODULE
