## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from collections.abc import Callable, Iterator
from typing import Protocol

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
)

## local
## direct-name import, not the usual module import: `_snapshot_fields/__init__.py`
## re-exports this file's own contents, so `from . import read_fields` would need the
## package fully resolved while it is still mid-import -- a real circular dependency
from .read_fields import HelmholtzKineticEnergy

##
## === PROTOCOL
##


class FieldsProtocol(Protocol):
    """
    Protocol declaring the interface that `_Derive*` classes interact via.

    Each `_Derive*` method annotates `self` as `FieldsProtocol`; basedpyright resolves
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
        amr_level: int = 0,
    ) -> domain_models.UniformDomain_3D:
        ...

    def load_3d_density_sfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_momentum_vfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        ...

    def load_3d_magnetic_vfield(
        self,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...

    def load_3d_total_energy_sfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        ...

    def _is_vfield_keys_available(
        self,
        field_name: str,
    ) -> bool:
        ...

    def _load_expanded_vfield_boxes(
        self,
        *,
        field_name: str,
        num_extra_cells: int,
        amr_level: int = 0,
    ) -> Iterator[tuple[numpy.ndarray, tuple[slice, slice, slice]]]:
        ...

    def _compute_chunked_derived_vfield(
        self,
        *,
        field_name: str,
        grad_order: int,
        amr_level: int,
        local_compute_fn: Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.VectorField_3D:
        ...

    ##
    ## --- _DeriveVelocityFields
    ##

    def compute_velocity_vfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        ...

    def compute_velocity_gradient_r2tfield(
        self,
        grad_order: int,
        *,
        amr_level: int = 0,
    ) -> field_models.RankTwoTensorField_3D:
        ...

    def compute_vorticity_vfield(
        self,
        grad_order: int,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        ...

    ##
    ## --- _DeriveEnergyFields
    ##

    def compute_kinetic_energy_sfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_magnetic_energy_sfield(
        self,
        energy_prefactor: float = 0.5,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_internal_energy_sfield(
        self,
        magnetic_energy_sfield_3d: field_models.ScalarField_3D | None = None,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_pressure_sfield(
        self,
        gamma: float = 5.0 / 3.0,
        magnetic_energy_sfield_3d: field_models.ScalarField_3D | None = None,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_helmholtz_kinetic_energy(
        self,
        *,
        amr_level: int = 0,
    ) -> HelmholtzKineticEnergy:
        ...

    ##
    ## --- _DeriveMagneticFields
    ##

    def compute_alfven_speed_vfield(
        self,
        *,
        amr_level: int = 0,
    ) -> field_models.VectorField_3D:
        ...

    def compute_current_density_vfield(
        self,
        grad_order: int,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...

    ##
    ## --- _DeriveMHDFields
    ##

    def compute_lorentz_force_vfield(
        self,
        grad_order: int,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...


## } MODULE
