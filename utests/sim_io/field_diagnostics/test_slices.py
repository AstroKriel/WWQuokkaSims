## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import pathlib
import tempfile
import unittest

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import domain_models

## local
from ww_quokka_sims.sim_io.field_diagnostics import slices

##
## === HELPERS
##

## distinct bounds per axis, so a swapped axis-pairing shows up as a wrong bounds pair
_UNIFORM_DOMAIN = domain_models.UniformDomain_3D(
    periodicity=(True, True, True),
    domain_bounds=((0.0, 1.0), (0.0, 2.0), (0.0, 3.0)),
    resolution=(3, 3, 3),
)

## every cell has a unique value, so a swapped axis in the slicing logic shows up as
## extracting the wrong 2D plane, not just a wrong-shaped one
_SARRAY_3D = numpy.arange(3 * 3 * 3, dtype=float).reshape(3, 3, 3)

##
## === TEST SUITE
##


class TestGetSliceBounds(unittest.TestCase):

    def test_slicing_x2_returns_x0_x1_bounds(
        self,
    ):
        bounds = slices.get_slice_bounds(
            uniform_domain=_UNIFORM_DOMAIN,
            axis_to_slice=cartesian_axes.CartesianAxis_3D.X2,
        )
        self.assertEqual(bounds, ((0.0, 1.0), (0.0, 2.0)))

    def test_slicing_x1_returns_x0_x2_bounds(
        self,
    ):
        bounds = slices.get_slice_bounds(
            uniform_domain=_UNIFORM_DOMAIN,
            axis_to_slice=cartesian_axes.CartesianAxis_3D.X1,
        )
        self.assertEqual(bounds, ((0.0, 1.0), (0.0, 3.0)))

    def test_slicing_x0_returns_x1_x2_bounds(
        self,
    ):
        bounds = slices.get_slice_bounds(
            uniform_domain=_UNIFORM_DOMAIN,
            axis_to_slice=cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertEqual(bounds, ((0.0, 2.0), (0.0, 3.0)))


class TestSliceField(unittest.TestCase):

    @staticmethod
    def _slice(
        *,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
    ) -> numpy.ndarray:
        return slices.slice_field(
            sarray_3d=_SARRAY_3D,
            axis_to_slice=axis_to_slice,
            uniform_domain=_UNIFORM_DOMAIN,
            comp_label=r"$\rho$",
            sim_time=0.0,
            step_index=0,
            amr_level=0,
        ).sarray_2d

    def test_slicing_x2_takes_the_x2_midplane(
        self,
    ):
        numpy.testing.assert_array_equal(
            self._slice(axis_to_slice=cartesian_axes.CartesianAxis_3D.X2),
            _SARRAY_3D[:, :, 1],
        )

    def test_slicing_x1_takes_the_x1_midplane(
        self,
    ):
        numpy.testing.assert_array_equal(
            self._slice(axis_to_slice=cartesian_axes.CartesianAxis_3D.X1),
            _SARRAY_3D[:, 1, :],
        )

    def test_slicing_x0_takes_the_x0_midplane(
        self,
    ):
        numpy.testing.assert_array_equal(
            self._slice(axis_to_slice=cartesian_axes.CartesianAxis_3D.X0),
            _SARRAY_3D[1, :, :],
        )


class TestSlicedFieldRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_fields(
        self,
    ):
        sliced_field = slices.SlicedField(
            sarray_2d=numpy.arange(4, dtype=float).reshape(2, 2),
            axis_bounds=((0.0, 1.0), (0.0, 2.0)),
            min_value=0.0,
            max_value=3.0,
            comp_label=r"$\rho$",
            sim_time=0.25,
            step_index=3,
            amr_level=1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "sliced_field.npz"
            sliced_field.save_to_file(file_path)
            loaded = slices.SlicedField.load_from_file(file_path)
        self.assertEqual(loaded.axis_bounds, sliced_field.axis_bounds)
        self.assertEqual(loaded.min_value, sliced_field.min_value)
        self.assertEqual(loaded.max_value, sliced_field.max_value)
        self.assertEqual(loaded.comp_label, sliced_field.comp_label)
        self.assertEqual(loaded.sim_time, sliced_field.sim_time)
        self.assertEqual(loaded.step_index, sliced_field.step_index)
        self.assertEqual(loaded.amr_level, sliced_field.amr_level)
        numpy.testing.assert_array_equal(loaded.sarray_2d, sliced_field.sarray_2d)


## } U-TEST
