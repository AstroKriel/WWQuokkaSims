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
from jormi.ww_validation import validate_types

## local
## direct-name import, not the usual module import: `_snapshot_fields/__init__.py`
## re-exports this file's own contents, so `from .. import fields_protocol` would need
## the package fully resolved while it is still mid-import -- a real circular dependency
from ..fields_protocol import FieldsProtocol
from ..._snapshot_readers import read_fields
from ..._snapshot_readers.native_resolution import native_slice, native_values
from ..._snapshot_readers.uniform_resolution import expanded_boxes

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
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        b_varray_3d = field_models.extract_3d_varray(
            vfield_3d=b_vfield_3d,
            param_name="<b_vfield_3d>",
        )
        rho_sfield_3d = self.load_3d_density_sfield(amr_level=amr_level)
        rho_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=rho_sfield_3d,
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
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=v_A_varray_3d,
            uniform_domain_3d=uniform_domain_3d,
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
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """
        Approximate magnetic divergence: `div[vec(b)]` (fallback for `load_3d_magnetic_divergence_sfield`).

        `use_chunked_reader=True` computes it box-by-box: `vec(b)` is never a full-domain
        array, only the returned divergence is. Only supports amr_level=0.
        """
        if not use_chunked_reader:
            b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
            return field_operators.compute_vfield_divergence(
                vfield_3d=b_vfield_3d,
                field_name="magnetic_divergence",
                latex_label=r"\nabla\cdot\vec{b}",
                grad_order=grad_order,
            )
        else:
            cell_widths_3d = self.load_3d_uniform_domain(amr_level=amr_level).cell_widths
            num_extra_cells = expanded_boxes.compute_num_extra_cells(grad_order=grad_order)

            def derive_fn(
                expanded_b_varray: numpy.ndarray,
                num_extra_cells: int,
            ) -> numpy.ndarray:
                local_div_sarray = farray_operators.compute_varray_divergence(
                    varray_3d=expanded_b_varray,
                    cell_widths_3d=cell_widths_3d,
                    grad_order=grad_order,
                )
                return expanded_boxes.trim_expanded_box(
                    expanded_farray=local_div_sarray,
                    num_extra_cells=num_extra_cells,
                )

            expanded_box_source = self._iterate_expanded_boxes_of_vfield(
                field_name="magnetic",
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
            )
            return self._derive_chunked_sfield_from_source(
                expanded_box_source=expanded_box_source,
                num_extra_cells=num_extra_cells,
                amr_level=amr_level,
                derive_fn=derive_fn,
                output_field_name="magnetic_divergence",
                output_latex_label=r"\nabla\cdot\vec{b}",
            )

    def compute_current_density_vfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        """
        Compute current density: `curl[vec(b)]`.

        By default, reads the whole magnetic field via `covering_grid` first, then
        differentiates it in one pass. `use_chunked_reader=True` instead computes it
        box-by-box via `_derive_chunked_vfield_from_field_name`: `vec(b)` is never a
        full-domain array, only the returned current density is. Only supports
        amr_level=0.
        """
        if not use_chunked_reader:
            b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
            return field_operators.compute_vfield_curl(
                vfield_3d=b_vfield_3d,
                field_name="current_density",
                latex_label=r"\nabla\times\vec{b}",
                grad_order=grad_order,
            )
        else:
            cell_widths_3d = self.load_3d_uniform_domain(amr_level=amr_level).cell_widths

            def derive_fn(
                expanded_b_varray: numpy.ndarray,
                num_extra_cells: int,
            ) -> numpy.ndarray:
                local_curl_varray = farray_operators.compute_varray_curl(
                    varray_3d=expanded_b_varray,
                    cell_widths_3d=cell_widths_3d,
                    grad_order=grad_order,
                )
                return expanded_boxes.trim_expanded_box(
                    expanded_farray=local_curl_varray,
                    num_extra_cells=num_extra_cells,
                )

            return self._derive_chunked_vfield_from_field_name(
                field_name="magnetic",
                grad_order=grad_order,
                amr_level=amr_level,
                derive_fn=derive_fn,
                output_field_name="current_density",
                output_latex_label=r"\nabla\times\vec{b}",
            )

    def compute_current_density_sfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute current magnitude: `|curl[vec(b)]|`. See `compute_current_density_vfield` for
        `use_chunked_reader`."""
        j_vfield_3d = self.compute_current_density_vfield(
            grad_order=grad_order,
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        return field_operators.compute_vfield_magnitude(
            vfield_3d=j_vfield_3d,
            field_name="current_density_magnitude",
            latex_label=r"|\nabla\times\vec{b}|",
        )

    def compute_current_helicity_sfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute current helicity density: `curl[vec(b)] cdot vec(b)`."""
        j_vfield_3d = self.compute_current_density_vfield(
            grad_order=grad_order,
            amr_level=amr_level,
        )
        b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
        return field_operators.compute_vfield_dot_product(
            f_vfield_3d=j_vfield_3d,
            g_vfield_3d=b_vfield_3d,
            field_name="current_helicity",
            latex_label=r"(\nabla\times\vec{b})\cdot\vec{b}",
        )

    def compute_plasma_beta_sfield(
        self: FieldsProtocol,
        *,
        gamma: float = 5.0 / 3.0,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute plasma beta: `beta = 2 p / |vec(b)|^2`. See `_load_3d_sarray` for `use_chunked_reader`."""
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        energy_prefactor = 0.5
        e_mag_sfield_3d = self.compute_magnetic_energy_sfield(
            energy_prefactor=energy_prefactor,
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        pressure_sfield_3d = self.compute_pressure_sfield(
            gamma=gamma,
            magnetic_energy_sfield_3d=e_mag_sfield_3d,
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        p_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=pressure_sfield_3d,
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
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=beta_sarray_3d,
            uniform_domain_3d=uniform_domain_3d,
            field_name="plasma_beta",
            latex_label=r"\beta",
            sim_time=self.sim_time,
        )

    def load_3d_plasma_beta_amr_leaves(
        self: FieldsProtocol,
        *,
        gamma: float = 5.0 / 3.0,
    ) -> read_fields.AMRLeaves:
        """Load plasma beta `beta = 2 p / |vec(b)|^2` at every leaf cell across the full AMR
        hierarchy."""
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        energy_prefactor = 0.5
        magnetic_energy_leaves = self.load_3d_magnetic_energy_amr_leaves(energy_prefactor=energy_prefactor)
        pressure_leaves = self.load_3d_pressure_amr_leaves(
            gamma=gamma,
            magnetic_energy_leaves=magnetic_energy_leaves,
        )
        native_values.ensure_consistent_leaf_ordering(
            reference_leaves=pressure_leaves,
            other_leaves=magnetic_energy_leaves,
        )
        b_sq_values = magnetic_energy_leaves.values / energy_prefactor
        b_sq_has_zeros = compute_array_stats.check_no_zero_values(
            array=b_sq_values,
            param_name="<|b|^2>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            beta_values = 2.0 * pressure_leaves.values / b_sq_values
        if not b_sq_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=beta_values,
                param_name="<beta_leaves.values>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=beta_values,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return read_fields.AMRLeaves(
            values=beta_values,
            cell_width=pressure_leaves.cell_width,
            positions=pressure_leaves.positions,
        )

    def load_3d_plasma_beta_native_slice(
        self: FieldsProtocol,
        *,
        gamma: float = 5.0 / 3.0,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Load plasma beta `beta = 2 p / |vec(b)|^2`, and its per-pixel native `dx`, on a
        genuine AMR-native slice."""
        validate_types.ensure_finite_float(
            param=gamma,
            param_name="gamma",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        energy_prefactor = 0.5
        magnetic_energy_sarray_2d, magnetic_energy_cell_width_2d = self.load_3d_magnetic_energy_native_slice(
            energy_prefactor=energy_prefactor,
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )
        pressure_sarray_2d, pressure_cell_width_2d = self.load_3d_pressure_native_slice(
            gamma=gamma,
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
            magnetic_energy_native_slice=(magnetic_energy_sarray_2d, magnetic_energy_cell_width_2d),
        )
        native_slice.ensure_consistent_slice_geometry(
            reference_cell_width_2d=pressure_cell_width_2d,
            other_cell_width_2d=magnetic_energy_cell_width_2d,
        )
        b_sq_sarray_2d = magnetic_energy_sarray_2d / energy_prefactor
        b_sq_has_zeros = compute_array_stats.check_no_zero_values(
            array=b_sq_sarray_2d,
            param_name="<|b|^2>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            beta_sarray_2d = 2.0 * pressure_sarray_2d / b_sq_sarray_2d
        if not b_sq_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=beta_sarray_2d,
                param_name="<beta_sarray_2d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=beta_sarray_2d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return beta_sarray_2d, pressure_cell_width_2d


## } MODULE
