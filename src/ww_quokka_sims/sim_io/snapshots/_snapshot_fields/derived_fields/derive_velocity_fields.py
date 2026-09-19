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
    field_models,
    field_operators,
)

## local
## direct-name import, not the usual module import: `_snapshot_fields/__init__.py`
## re-exports this file's own contents, so `from .. import fields_protocol` would need
## the package fully resolved while it is still mid-import -- a real circular dependency
from ..fields_protocol import FieldsProtocol
from ..._snapshot_readers import read_fields
from ..._snapshot_readers.uniform_resolution import expanded_boxes

##
## === DERIVE CLASS
##


class _DeriveVelocityFields:
    """Velocity fields derived from a snapshot."""

    ##
    ## --- VELOCITY FIELDS
    ##

    def compute_velocity_vfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        """Compute velocity field: `vec(v) = vec(m) / rho`. See `_load_3d_sarray` for `use_chunked_reader`."""
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
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=v_varray,
            uniform_domain_3d=uniform_domain_3d,
            field_name="velocity",
            latex_label=r"\vec{v}",
            sim_time=self.sim_time,
        )

    def compute_velocity_magnitude_sfield(
        self: FieldsProtocol,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute velocity magnitude: `|vec(v)|`. See `_load_3d_sarray` for `use_chunked_reader`."""
        v_vfield_3d = self.compute_velocity_vfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        return field_operators.compute_vfield_magnitude(
            vfield_3d=v_vfield_3d,
            field_name="velocity_magnitude",
            latex_label=r"|\vec{v}|",
        )

    def load_3d_velocity_amr_leaves_by_axis(
        self: FieldsProtocol,
    ) -> dict[cartesian_axes.CartesianAxis_3D, read_fields.AMRLeaves]:
        """Load `v_x`, `v_y`, `v_z` at every leaf cell across the full AMR hierarchy, keyed by axis."""
        momentum_key_lookup = self._get_vfield_key_lookup(field_name="momentum")
        density_key = self._get_sfield_key(field_name="density")

        def derive_velocity_axis_fn(
            raw_box_farray: numpy.ndarray,
        ) -> numpy.ndarray:
            momentum_axis_box_farray = raw_box_farray[0]
            density_box_farray = raw_box_farray[1]
            return momentum_axis_box_farray / density_box_farray

        return {
            axis:
            self._load_amr_leaves_of_derived_field(
                field_keys=(momentum_key_lookup[axis], density_key),
                derive_fn=derive_velocity_axis_fn,
            )
            for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER
        }

    def load_3d_velocity_magnitude_amr_leaves(
        self: FieldsProtocol,
    ) -> read_fields.AMRLeaves:
        """Load velocity magnitude `|vec(v)|` at every leaf cell across the full AMR hierarchy."""
        momentum_key_lookup = self._get_vfield_key_lookup(field_name="momentum")
        density_key = self._get_sfield_key(field_name="density")
        field_keys = tuple(momentum_key_lookup[axis]
                           for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER) + (density_key, )

        def derive_velocity_magnitude_fn(
            raw_box_farray: numpy.ndarray,
        ) -> numpy.ndarray:
            momentum_box_varray = raw_box_farray[:3]
            density_box_farray = raw_box_farray[3]
            velocity_box_varray = momentum_box_varray / density_box_farray[numpy.newaxis, ...]
            return farray_operators.compute_varray_magnitude(velocity_box_varray)

        return self._load_amr_leaves_of_derived_field(
            field_keys=field_keys,
            derive_fn=derive_velocity_magnitude_fn,
        )

    def load_3d_velocity_native_slice_by_axis(
        self: FieldsProtocol,
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> dict[cartesian_axes.CartesianAxis_3D, tuple[numpy.ndarray, numpy.ndarray]]:
        """Load `v_x`, `v_y`, `v_z`, each with its per-pixel native `dx`, on a genuine AMR-native slice."""
        momentum_key_lookup = self._get_vfield_key_lookup(field_name="momentum")
        density_key = self._get_sfield_key(field_name="density")

        def derive_velocity_axis_fn(
            raw_sarray_stack: numpy.ndarray,
        ) -> numpy.ndarray:
            momentum_axis_sarray_2d = raw_sarray_stack[0]
            density_sarray_2d = raw_sarray_stack[1]
            return momentum_axis_sarray_2d / density_sarray_2d

        return {
            axis:
            self._load_native_slice_of_derived_field(
                field_keys=(momentum_key_lookup[axis], density_key),
                derive_fn=derive_velocity_axis_fn,
                axis_to_slice=axis_to_slice,
                slice_coordinate=slice_coordinate,
            )
            for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER
        }

    def load_3d_velocity_magnitude_native_slice(
        self: FieldsProtocol,
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Load velocity magnitude `|vec(v)|`, and its per-pixel native `dx`, on a genuine
        AMR-native slice."""
        momentum_key_lookup = self._get_vfield_key_lookup(field_name="momentum")
        density_key = self._get_sfield_key(field_name="density")
        field_keys = tuple(momentum_key_lookup[axis]
                           for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER) + (density_key, )

        def derive_velocity_magnitude_fn(
            raw_sarray_stack: numpy.ndarray,
        ) -> numpy.ndarray:
            momentum_varray_2d = raw_sarray_stack[:3]
            density_sarray_2d = raw_sarray_stack[3]
            velocity_varray_2d = momentum_varray_2d / density_sarray_2d[numpy.newaxis, ...]
            ## `farray_operators.compute_varray_magnitude` requires a 3D-domain (4D total)
            ## varray; a native slice only has 2 spatial dims, so sum-of-squares is done
            ## directly instead
            return numpy.sqrt(
                numpy.sum(
                    velocity_varray_2d**2,
                    axis=0,
                ),
            )

        return self._load_native_slice_of_derived_field(
            field_keys=field_keys,
            derive_fn=derive_velocity_magnitude_fn,
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )

    def compute_div_v_sfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """
        Compute velocity divergence `nabla cdot vec(v)`; `grad_order` controls stencil accuracy.

        By default, reads velocity via a full-domain covering grid first, then
        differentiates it in one pass. `use_chunked_reader=True` instead computes it
        box-by-box: velocity is never a full-domain array, only the returned divergence
        is. Only supports amr_level=0.
        """
        if not use_chunked_reader:
            v_vfield_3d = self.compute_velocity_vfield(amr_level=amr_level)
            return field_operators.compute_vfield_divergence(
                vfield_3d=v_vfield_3d,
                field_name="div_velocity",
                latex_label=r"\nabla\cdot\vec{v}",
                grad_order=grad_order,
            )
        else:
            cell_widths_3d = self.load_3d_uniform_domain(amr_level=amr_level).cell_widths
            num_extra_cells = expanded_boxes.compute_num_extra_cells(grad_order=grad_order)

            def derive_fn(
                expanded_v_varray: numpy.ndarray,
                num_extra_cells: int,
            ) -> numpy.ndarray:
                local_div_sarray = farray_operators.compute_varray_divergence(
                    varray_3d=expanded_v_varray,
                    cell_widths_3d=cell_widths_3d,
                    grad_order=grad_order,
                )
                return expanded_boxes.trim_expanded_box(
                    expanded_farray=local_div_sarray,
                    num_extra_cells=num_extra_cells,
                )

            expanded_box_source = self._iterate_expanded_boxes_of_velocity_vfield(
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
            )
            return self._derive_chunked_sfield_from_source(
                expanded_box_source=expanded_box_source,
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
                derive_fn=derive_fn,
                output_field_name="div_velocity",
                output_latex_label=r"\nabla\cdot\vec{v}",
            )

    def compute_velocity_gradient_r2tfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.RankTwoTensorField_3D:
        """
        Compute velocity gradient tensor `nabla vec(v)`; `grad_order` controls stencil accuracy.

        By default, reads velocity via a full-domain covering grid first, then
        differentiates it in one pass. `use_chunked_reader=True` instead computes it
        box-by-box: velocity is never a full-domain array, only the returned gradient
        tensor is. Only supports amr_level=0.
        """
        if not use_chunked_reader:
            v_vfield_3d = self.compute_velocity_vfield(amr_level=amr_level)
            return field_operators.compute_vfield_gradient(
                vfield_3d=v_vfield_3d,
                grad_order=grad_order,
                field_name="velocity_gradient",
                latex_label=r"\nabla\vec{v}",
            )
        else:
            cell_widths_3d = self.load_3d_uniform_domain(amr_level=amr_level).cell_widths
            num_extra_cells = expanded_boxes.compute_num_extra_cells(grad_order=grad_order)

            def derive_fn(
                expanded_v_varray: numpy.ndarray,
                num_extra_cells: int,
            ) -> numpy.ndarray:
                local_grad_r2tarray = farray_operators.compute_varray_grad(
                    varray_3d=expanded_v_varray,
                    cell_widths_3d=cell_widths_3d,
                    grad_order=grad_order,
                )
                return expanded_boxes.trim_expanded_box(
                    expanded_farray=local_grad_r2tarray,
                    num_extra_cells=num_extra_cells,
                )

            expanded_box_source = self._iterate_expanded_boxes_of_velocity_vfield(
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
            )
            return self._derive_chunked_r2tfield_from_source(
                expanded_box_source=expanded_box_source,
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
                derive_fn=derive_fn,
                output_field_name="velocity_gradient",
                output_latex_label=r"\nabla\vec{v}",
            )

    def compute_vorticity_vfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        """
        Compute vorticity vector `curl(vec(v))`; `grad_order` controls stencil accuracy.

        By default, reads velocity via a full-domain covering grid first, then
        differentiates it in one pass. `use_chunked_reader=True` instead computes it
        box-by-box: velocity is never a full-domain array, only the returned vorticity
        is. Only supports amr_level=0.
        """
        if not use_chunked_reader:
            v_vfield_3d = self.compute_velocity_vfield(amr_level=amr_level)
            return field_operators.compute_vfield_curl(
                vfield_3d=v_vfield_3d,
                grad_order=grad_order,
                field_name="vorticity",
                latex_label=r"\nabla\times\vec{v}",
            )
        else:
            cell_widths_3d = self.load_3d_uniform_domain(amr_level=amr_level).cell_widths
            num_extra_cells = expanded_boxes.compute_num_extra_cells(grad_order=grad_order)

            def derive_fn(
                expanded_v_varray: numpy.ndarray,
                num_extra_cells: int,
            ) -> numpy.ndarray:
                local_curl_varray = farray_operators.compute_varray_curl(
                    varray_3d=expanded_v_varray,
                    cell_widths_3d=cell_widths_3d,
                    grad_order=grad_order,
                )
                return expanded_boxes.trim_expanded_box(
                    expanded_farray=local_curl_varray,
                    num_extra_cells=num_extra_cells,
                )

            expanded_box_source = self._iterate_expanded_boxes_of_velocity_vfield(
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
            )
            return self._derive_chunked_vfield_from_source(
                expanded_box_source=expanded_box_source,
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
                derive_fn=derive_fn,
                output_field_name="vorticity",
                output_latex_label=r"\nabla\times\vec{v}",
            )

    def compute_vorticity_sfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute vorticity magnitude: `|curl(vec(v))|`. See `compute_vorticity_vfield` for
        `use_chunked_reader`."""
        omega_vfield_3d = self.compute_vorticity_vfield(
            grad_order=grad_order,
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        return field_operators.compute_vfield_magnitude(
            vfield_3d=omega_vfield_3d,
            field_name="vorticity_magnitude",
            latex_label=r"|\nabla\times\vec{v}|",
        )

    def compute_kinetic_helicity_sfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute kinetic helicity density: `curl(vec(v)) dot vec(v)`."""
        omega_vfield_3d = self.compute_vorticity_vfield(
            grad_order=grad_order,
            amr_level=amr_level,
        )
        v_vfield_3d = self.compute_velocity_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_dot_product(
            f_vfield_3d=omega_vfield_3d,
            g_vfield_3d=v_vfield_3d,
            field_name="kinetic_helicity",
            latex_label=r"(\nabla\times\vec{v})\cdot\vec{v}",
        )


## } MODULE
