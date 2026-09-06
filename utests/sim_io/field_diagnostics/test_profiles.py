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

## local
from ww_quokka_sims.sim_io.field_diagnostics import profiles

##
## === TEST SUITE
##


class TestScalarProfileRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_fields(
        self,
    ):
        scalar_profile = profiles.ScalarProfile(
            field_name="density",
            field_label=r"$\rho$",
            sim_time=0.25,
            step_index=3,
            profile_axis="x_0",
            position=numpy.array([0.0, 1.0, 2.0]),
            field_value=numpy.array([1.0, 2.0, 3.0]),
            amr_level=1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "scalar_profile.json"
            scalar_profile.save_to_file(file_path)
            loaded = profiles.ScalarProfile.load_from_file(file_path)
        self.assertEqual(loaded.field_name, scalar_profile.field_name)
        self.assertEqual(loaded.field_label, scalar_profile.field_label)
        self.assertEqual(loaded.sim_time, scalar_profile.sim_time)
        self.assertEqual(loaded.step_index, scalar_profile.step_index)
        self.assertEqual(loaded.profile_axis, scalar_profile.profile_axis)
        self.assertEqual(loaded.amr_level, scalar_profile.amr_level)
        numpy.testing.assert_array_equal(loaded.position, scalar_profile.position)
        numpy.testing.assert_array_equal(loaded.field_value, scalar_profile.field_value)


class TestVectorProfileRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_components(
        self,
    ):
        vector_profile = profiles.VectorProfile(
            field_name="velocity",
            sim_time=0.5,
            step_index=1,
            profile_axis="x_1",
            components={
                "x_0":
                profiles.ComponentArrays(
                    position=numpy.array([0.0, 1.0]),
                    field_value=numpy.array([1.0, 2.0]),
                    label=r"$v_x$",
                ),
                "x_1":
                profiles.ComponentArrays(
                    position=numpy.array([0.0, 1.0]),
                    field_value=numpy.array([3.0, 4.0]),
                    label=r"$v_y$",
                ),
            },
            amr_level=0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "vector_profile.json"
            vector_profile.save_to_file(file_path)
            loaded = profiles.VectorProfile.load_from_file(file_path)
        self.assertEqual(loaded.field_name, vector_profile.field_name)
        self.assertEqual(set(loaded.components.keys()), set(vector_profile.components.keys()))
        for key, comp_arrays in vector_profile.components.items():
            loaded_comp_arrays = loaded.components[key]
            self.assertEqual(loaded_comp_arrays.label, comp_arrays.label)
            numpy.testing.assert_array_equal(loaded_comp_arrays.position, comp_arrays.position)
            numpy.testing.assert_array_equal(loaded_comp_arrays.field_value, comp_arrays.field_value)


## } U-TEST
