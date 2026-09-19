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
from ..._snapshot_readers.native_resolution import native_slice, native_values
from ..._snapshot_readers.uniform_resolution import expanded_boxes

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
        *,
        grad_order: int = 2,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        """
        Compute Lorentz force: `curl[vec(b)] x vec(b)`.

        By default, computes current density and reads `vec(b)` as two separate steps
        (the second is a cache hit, since both resolve to the same cached `vec(b)`).
        `use_chunked_reader=True` instead computes the Lorentz force box-by-box via
        `_derive_chunked_vfield_from_field_name`: `vec(b)` and current density each only ever
        exist as one box's worth of data, crossed immediately into that box's Lorentz
        force; only the returned field is ever a full-domain array. Composing the
        already-chunked `compute_current_density_vfield` with a second
        `load_3d_magnetic_vfield` call would both read `vec(b)` twice and hold `vec(b)`
        and current density as full arrays simultaneously for no reason, since the cross
        product never needs more than one box of each at once.
        """
        if not use_chunked_reader:
            j_vfield_3d = self.compute_current_density_vfield(
                grad_order=grad_order,
                amr_level=amr_level,
            )
            b_vfield_3d = self.load_3d_magnetic_vfield(amr_level=amr_level)
            return field_operators.compute_vfield_cross_product(
                f_vfield_3d=j_vfield_3d,
                g_vfield_3d=b_vfield_3d,
                field_name="lorentz_force",
                latex_label=r"(\nabla\times\vec{b})\times\vec{b}",
            )
        else:
            cell_widths_3d = self.load_3d_uniform_domain(amr_level=amr_level).cell_widths

            def derive_fn(
                expanded_b_varray: numpy.ndarray,
                num_extra_cells: int,
            ) -> numpy.ndarray:
                box_b_varray = expanded_boxes.trim_expanded_box(
                    expanded_farray=expanded_b_varray,
                    num_extra_cells=num_extra_cells,
                )
                local_curl_varray = farray_operators.compute_varray_curl(
                    varray_3d=expanded_b_varray,
                    cell_widths_3d=cell_widths_3d,
                    grad_order=grad_order,
                )
                box_j_varray = expanded_boxes.trim_expanded_box(
                    expanded_farray=local_curl_varray,
                    num_extra_cells=num_extra_cells,
                )
                return farray_operators.compute_varray_cross_product(
                    f_varray_3d=box_j_varray,
                    g_varray_3d=box_b_varray,
                )

            return self._derive_chunked_vfield_from_field_name(
                field_name="magnetic",
                grad_order=grad_order,
                amr_level=amr_level,
                derive_fn=derive_fn,
                output_field_name="lorentz_force",
                output_latex_label=r"(\nabla\times\vec{b})\times\vec{b}",
            )

    def compute_lorentz_force_sfield(
        self: FieldsProtocol,
        *,
        grad_order: int = 2,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        """Compute Lorentz force magnitude: `| curl[vec(b)] x vec(b) |`."""
        lorentz_force_vfield_3d = self.compute_lorentz_force_vfield(
            grad_order=grad_order,
            amr_level=amr_level,
        )
        return field_operators.compute_vfield_magnitude(
            vfield_3d=lorentz_force_vfield_3d,
            field_name="lorentz_force_magnitude",
            latex_label=r"|(\nabla\times\vec{b})\times\vec{b}|",
        )

    def compute_energy_ratio_sfield(
        self: FieldsProtocol,
        *,
        energy_prefactor: float = 0.5,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """Compute magnetic-to-kinetic energy ratio: `e_mag / e_kin`. See `_load_3d_sarray` for
        `use_chunked_reader`."""
        e_mag_sfield_3d = self.compute_magnetic_energy_sfield(
            energy_prefactor=energy_prefactor,
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        E_mag_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=e_mag_sfield_3d,
            param_name="<E_mag_sfield_3d>",
        )
        e_kin_sfield_3d = self.compute_kinetic_energy_sfield(
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        E_kin_sarray_3d = field_models.extract_3d_sarray(
            sfield_3d=e_kin_sfield_3d,
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
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=energy_ratio_sarray_3d,
            uniform_domain_3d=uniform_domain_3d,
            field_name="energy_ratio",
            latex_label=r"E_\mathrm{mag} / E_\mathrm{kin}",
            sim_time=self.sim_time,
        )

    def load_3d_energy_ratio_amr_leaves(
        self: FieldsProtocol,
        *,
        energy_prefactor: float = 0.5,
    ) -> read_fields.AMRLeaves:
        """Load magnetic-to-kinetic energy ratio `e_mag / e_kin` at every leaf cell across the full
        AMR hierarchy."""
        magnetic_energy_leaves = self.load_3d_magnetic_energy_amr_leaves(energy_prefactor=energy_prefactor)
        kinetic_energy_leaves = self.load_3d_kinetic_energy_amr_leaves()
        native_values.ensure_consistent_leaf_ordering(
            reference_leaves=magnetic_energy_leaves,
            other_leaves=kinetic_energy_leaves,
        )
        kinetic_energy_has_zeros = compute_array_stats.check_no_zero_values(
            array=kinetic_energy_leaves.values,
            param_name="<E_kin_leaves.values>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            energy_ratio_values = magnetic_energy_leaves.values / kinetic_energy_leaves.values
        if not kinetic_energy_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=energy_ratio_values,
                param_name="<energy_ratio_leaves.values>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=energy_ratio_values,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return read_fields.AMRLeaves(
            values=energy_ratio_values,
            cell_width=magnetic_energy_leaves.cell_width,
            positions=magnetic_energy_leaves.positions,
        )

    def load_3d_energy_ratio_native_slice(
        self: FieldsProtocol,
        *,
        energy_prefactor: float = 0.5,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Load magnetic-to-kinetic energy ratio `e_mag / e_kin`, and its per-pixel native `dx`,
        on a genuine AMR-native slice."""
        magnetic_energy_sarray_2d, magnetic_energy_cell_width_2d = self.load_3d_magnetic_energy_native_slice(
            energy_prefactor=energy_prefactor,
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )
        kinetic_energy_sarray_2d, kinetic_energy_cell_width_2d = self.load_3d_kinetic_energy_native_slice(
            axis_to_slice=axis_to_slice,
            slice_coordinate=slice_coordinate,
        )
        native_slice.ensure_consistent_slice_geometry(
            reference_cell_width_2d=magnetic_energy_cell_width_2d,
            other_cell_width_2d=kinetic_energy_cell_width_2d,
        )
        kinetic_energy_has_zeros = compute_array_stats.check_no_zero_values(
            array=kinetic_energy_sarray_2d,
            param_name="<E_kin_sarray_2d>",
            raise_error=False,
        )
        with compute_array_stats.suppress_divide_warnings():
            energy_ratio_sarray_2d = magnetic_energy_sarray_2d / kinetic_energy_sarray_2d
        if not kinetic_energy_has_zeros:
            compute_array_stats.check_no_nonfinite_values(
                array=energy_ratio_sarray_2d,
                param_name="<energy_ratio_sarray_2d>",
                raise_error=False,
            )
        compute_array_stats.make_nonfinites_zero(
            array=energy_ratio_sarray_2d,
            zero_nan=True,
            zero_posinf=True,
            zero_neginf=True,
        )
        return energy_ratio_sarray_2d, magnetic_energy_cell_width_2d

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
