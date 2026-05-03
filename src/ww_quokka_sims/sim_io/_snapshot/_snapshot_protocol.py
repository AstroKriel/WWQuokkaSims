## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Protocol

## personal
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
)

## local
from ._read_fields import HelmholtzKineticEnergy

##
## === PROTOCOL
##


class _QuokkaSnapshotProtocol(Protocol):
    """Typing-only protocol declaring the raw interface that mixins rely on."""

    @property
    def sim_time(
        self,
    ) -> float: ...

    def load_uniform_domain(
        self,
        force_periodicity: bool = True,
    ) -> domain_models.UniformDomain_3D: ...

    def load_density_sfield(
        self,
    ) -> field_models.ScalarField_3D: ...

    def load_momentum_vfield(
        self,
    ) -> field_models.VectorField_3D: ...

    def load_magnetic_vfield(
        self,
    ) -> field_models.VectorField_3D: ...

    def load_total_energy_sfield(
        self,
    ) -> field_models.ScalarField_3D: ...

    def _is_vfield_keys_available(
        self,
        field_name: str,
    ) -> bool: ...

    def compute_velocity_vfield(
        self,
    ) -> field_models.VectorField_3D: ...

    def compute_kinetic_energy_sfield(
        self,
    ) -> field_models.ScalarField_3D: ...

    def compute_magnetic_energy_sfield(
        self,
        energy_prefactor: float = 0.5,
    ) -> field_models.ScalarField_3D: ...

    def compute_internal_energy_sfield(
        self,
    ) -> field_models.ScalarField_3D: ...

    def compute_pressure_sfield(
        self,
        gamma: float,
    ) -> field_models.ScalarField_3D: ...

    def compute_vorticity_vfield(
        self,
        grad_order: int,
    ) -> field_models.VectorField_3D: ...

    def compute_current_density_vfield(
        self,
        grad_order: int,
    ) -> field_models.VectorField_3D: ...

    def compute_helmholtz_kinetic_energy(
        self,
    ) -> HelmholtzKineticEnergy: ...

    def compute_lorentz_force_vfield(
        self,
        grad_order: int,
    ) -> field_models.VectorField_3D: ...

    def compute_alfven_speed_vfield(
        self,
    ) -> field_models.VectorField_3D: ...


## } MODULE
