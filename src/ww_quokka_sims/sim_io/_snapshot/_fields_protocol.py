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


class _FieldsProtocol(Protocol):
    """
    Protocol declaring the interface that `_Derive*` classes interact via.

    Each `_Derive*` method annotates `self` as `_FieldsProtocol`; basedpyright resolves
    all `self.*` calls using the stubs (empty function definitions) provided here. All
    functions called via `self.*`, including calls within the same `_Derive*` class,
    must have a stub here.
    """

    ##
    ## --- QuokkaSnapshot
    ##

    @property
    def sim_time(
        self,
    ) -> float:
        ...

    def load_3d_uniform_domain(
        self,
        *,
        force_periodicity: bool = True,
    ) -> domain_models.UniformDomain_3D:
        ...

    def load_3d_density_sfield(
        self,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_momentum_vfield(
        self,
    ) -> field_models.VectorField_3D:
        ...

    def load_3d_magnetic_vfield(
        self,
    ) -> field_models.VectorField_3D:
        ...

    def load_3d_total_energy_sfield(
        self,
    ) -> field_models.ScalarField_3D:
        ...

    def _is_vfield_keys_available(
        self,
        field_name: str,
    ) -> bool:
        ...

    ##
    ## --- _DeriveVelocityFields
    ##

    def compute_velocity_vfield(
        self,
    ) -> field_models.VectorField_3D:
        ...

    def compute_vorticity_vfield(
        self,
        grad_order: int,
    ) -> field_models.VectorField_3D:
        ...

    ##
    ## --- _DeriveEnergyFields
    ##

    def compute_kinetic_energy_sfield(
        self,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_magnetic_energy_sfield(
        self,
        energy_prefactor: float = 0.5,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_internal_energy_sfield(
        self,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_pressure_sfield(
        self,
        gamma: float,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_helmholtz_kinetic_energy(
        self,
    ) -> HelmholtzKineticEnergy:
        ...

    ##
    ## --- _DeriveMagneticFields
    ##

    def compute_alfven_speed_vfield(
        self,
    ) -> field_models.VectorField_3D:
        ...

    def compute_current_density_vfield(
        self,
        grad_order: int,
    ) -> field_models.VectorField_3D:
        ...

    ##
    ## --- _DeriveMHDFields
    ##

    def compute_lorentz_force_vfield(
        self,
        grad_order: int,
    ) -> field_models.VectorField_3D:
        ...


## } MODULE
