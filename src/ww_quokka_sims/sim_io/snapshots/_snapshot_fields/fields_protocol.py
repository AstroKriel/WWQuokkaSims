## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import pathlib
import typing

from collections import abc as collections_abc

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
)

## local
from .._snapshot_readers.read_fields import AMRLeaves, FieldKey
from .._snapshot_readers.uniform_resolution import expanded_boxes
from .read_fields import HelmholtzKineticEnergy, LRUCache

##
## === PROTOCOL
##


class FieldsProtocol(typing.Protocol):
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

    snapshot_dir: pathlib.Path
    _field_cache: LRUCache

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

    def _resolve_sfield_key(
        self,
        *,
        field_name: str,
    ) -> FieldKey:
        ...

    def _get_sfield_key(
        self,
        *,
        field_name: str,
    ) -> FieldKey:
        ...

    def _get_vfield_key_lookup(
        self,
        *,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        ...

    def is_field_key_available(
        self,
        *,
        field_key: FieldKey,
    ) -> bool:
        ...

    def load_3d_sfield(
        self,
        *,
        field_key: FieldKey,
        field_name: str,
        latex_label: str,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_vfield(
        self,
        *,
        vfield_key_lookup: dict[cartesian_axes.CartesianAxis_3D, FieldKey],
        field_name: str,
        latex_label: str,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...

    def load_amr_leaves(
        self,
        *,
        field_key: FieldKey,
    ) -> AMRLeaves:
        ...

    def _load_amr_leaves_of_derived_field(
        self,
        *,
        field_keys: tuple[FieldKey, ...],
        derive_fn: collections_abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> AMRLeaves:
        ...

    def load_native_slice_sarray(
        self,
        *,
        field_key: FieldKey,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ...

    def _load_native_slice_of_derived_field(
        self,
        *,
        field_keys: tuple[FieldKey, ...],
        derive_fn: collections_abc.Callable[[numpy.ndarray], numpy.ndarray],
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ...

    def _field_cache_key(
        self,
        *,
        field_name: str,
        amr_level: int,
        use_chunked_reader: bool = False,
    ) -> str:
        ...

    def load_3d_density_sfield(
        self,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_momentum_vfield(
        self,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
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
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_total_energy_amr_leaves(
        self,
    ) -> AMRLeaves:
        ...

    def load_3d_total_energy_native_slice(
        self,
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ...

    def _is_vfield_keys_available(
        self,
        *,
        field_name: str,
    ) -> bool:
        ...

    def _iterate_expanded_boxes_of_vfield(
        self,
        *,
        field_name: str,
        num_extra_cells: int,
        amr_level: int = 0,
    ) -> collections_abc.Iterator[expanded_boxes.ExpandedFArray]:
        ...

    def _derive_chunked_vfield_from_field_name(
        self,
        *,
        field_name: str,
        grad_order: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.VectorField_3D:
        ...

    def _iterate_expanded_boxes_of_velocity_vfield(
        self,
        *,
        num_extra_cells: int,
        amr_level: int = 0,
    ) -> collections_abc.Iterator[expanded_boxes.ExpandedFArray]:
        ...

    def _derive_chunked_vfield_from_source(
        self,
        *,
        expanded_box_source: collections_abc.Iterator[expanded_boxes.ExpandedFArray],
        num_extra_cells: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.VectorField_3D:
        ...

    def _derive_chunked_sfield_from_source(
        self,
        *,
        expanded_box_source: collections_abc.Iterator[expanded_boxes.ExpandedFArray],
        num_extra_cells: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.ScalarField_3D:
        ...

    def _derive_chunked_r2tfield_from_source(
        self,
        *,
        expanded_box_source: collections_abc.Iterator[expanded_boxes.ExpandedFArray],
        num_extra_cells: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.RankTwoTensorField_3D:
        ...

    ##
    ## --- _DeriveVelocityFields
    ##

    def compute_velocity_vfield(
        self,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...

    def compute_vorticity_vfield(
        self,
        *,
        grad_order: int,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...

    ##
    ## --- _DeriveEnergyFields
    ##

    def compute_kinetic_energy_sfield(
        self,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_kinetic_energy_amr_leaves(
        self,
    ) -> AMRLeaves:
        ...

    def load_3d_kinetic_energy_native_slice(
        self,
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ...

    def compute_magnetic_energy_sfield(
        self,
        *,
        energy_prefactor: float = 0.5,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_magnetic_energy_amr_leaves(
        self,
        *,
        energy_prefactor: float = 0.5,
    ) -> AMRLeaves:
        ...

    def load_3d_magnetic_energy_native_slice(
        self,
        *,
        energy_prefactor: float = 0.5,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ...

    def compute_internal_energy_sfield(
        self,
        *,
        magnetic_energy_sfield_3d: field_models.ScalarField_3D | None = None,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_internal_energy_amr_leaves(
        self,
        *,
        magnetic_energy_leaves: AMRLeaves | None = None,
    ) -> AMRLeaves:
        ...

    def load_3d_internal_energy_native_slice(
        self,
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
        magnetic_energy_native_slice: tuple[numpy.ndarray, numpy.ndarray] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ...

    def compute_pressure_sfield(
        self,
        *,
        gamma: float = 5.0 / 3.0,
        magnetic_energy_sfield_3d: field_models.ScalarField_3D | None = None,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def load_3d_pressure_amr_leaves(
        self,
        *,
        gamma: float = 5.0 / 3.0,
        magnetic_energy_leaves: AMRLeaves | None = None,
    ) -> AMRLeaves:
        ...

    def load_3d_pressure_native_slice(
        self,
        *,
        gamma: float = 5.0 / 3.0,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
        magnetic_energy_native_slice: tuple[numpy.ndarray, numpy.ndarray] | None = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        ...

    def compute_helmholtz_kinetic_energy(
        self,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
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

    def compute_div_b_sfield(
        self,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        ...

    def compute_current_density_vfield(
        self,
        *,
        grad_order: int,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...

    ##
    ## --- _DeriveMHDFields
    ##

    def compute_lorentz_force_vfield(
        self,
        *,
        grad_order: int,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        ...


## } MODULE
