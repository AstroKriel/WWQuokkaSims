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

## local
from ._snapshot_protocol import _QuokkaSnapshotProtocol

##
## === MIXIN CLASS
##


class _VelocityFieldMethods:
    """Mixin providing velocity-derived field computations."""

    ##
    ## --- VELOCITY FIELDS
    ##

    def compute_velocity_vfield(
        self: _QuokkaSnapshotProtocol,
    ) -> field_models.VectorField_3D:
        """Load velocity field: `vec(v) = vec(m) / rho`."""
        rho_sfield_3d = self.load_density_sfield()
        rho_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=rho_sfield_3d,
            param_name="<rho_sfield_3d>",
        )
        mom_vfield_3d = self.load_momentum_vfield()
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
            v_varray = mom_varray_3d / rho_sarray_3d[numpy.newaxis, ...]
        if not rho_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=v_varray,
                param_name="<v_vfield_3d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=v_varray,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        udomain_3d = self.load_uniform_domain()
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=v_varray,
            udomain_3d=udomain_3d,
            field_label=r"\vec{v}",
            sim_time=self.sim_time,
        )

    def compute_velocity_magnitude_sfield(
        self: _QuokkaSnapshotProtocol,
    ) -> field_models.ScalarField_3D:
        """Compute velocity magnitude: `|vec(v)|`."""
        v_vfield_3d = self.compute_velocity_vfield()
        return field_operators.compute_vfield_magnitude(
            vfield_3d=v_vfield_3d,
            field_label=r"|\vec{v}|",
        )

    def compute_divu_sfield(
        self: _QuokkaSnapshotProtocol,
        grad_order: int = 2,
    ) -> field_models.ScalarField_3D:
        """Compute divergence of velocity: `nabla cdot vec(v)` using a `grad_order` accurate stencil."""
        v_vfield_3d = self.compute_velocity_vfield()
        return field_operators.compute_vfield_divergence(
            vfield_3d=v_vfield_3d,
            field_label=r"\nabla\cdot\vec{v}",
            grad_order=grad_order,
        )

    def compute_vorticity_vfield(
        self: _QuokkaSnapshotProtocol,
        grad_order: int = 2,
    ) -> field_models.VectorField_3D:
        """Compute vorticity vector: `curl(vec(v))` using a `grad_order` accurate stencil."""
        v_vfield_3d = self.compute_velocity_vfield()
        return field_operators.compute_vfield_curl(
            vfield_3d=v_vfield_3d,
            grad_order=grad_order,
            field_label=r"\nabla\times\vec{v}",
        )

    def compute_vorticity_sfield(
        self: _QuokkaSnapshotProtocol,
        grad_order: int = 2,
    ) -> field_models.ScalarField_3D:
        """Compute vorticity magnitude: `|curl(vec(v))|`."""
        omega_vfield_3d = self.compute_vorticity_vfield(grad_order=grad_order)
        return field_operators.compute_vfield_magnitude(
            vfield_3d=omega_vfield_3d,
            field_label=r"|\nabla\times\vec{v}|",
        )

    def compute_kinetic_helicity_sfield(
        self: _QuokkaSnapshotProtocol,
        grad_order: int = 2,
    ) -> field_models.ScalarField_3D:
        """Compute kinetic helicity density: `curl(vec(v)) dot vec(v)`."""
        omega_vfield_3d = self.compute_vorticity_vfield(grad_order=grad_order)
        v_vfield_3d = self.compute_velocity_vfield()
        return field_operators.compute_vfield_dot_product(
            vfield_3d_a=omega_vfield_3d,
            vfield_3d_b=v_vfield_3d,
            field_label=r"(\nabla\times\vec{v})\cdot\vec{v}",
        )


## } MODULE
