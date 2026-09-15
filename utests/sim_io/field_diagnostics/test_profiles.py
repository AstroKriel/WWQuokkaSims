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
from jormi.ww_plots import latex_labels

## local
from ww_quokka_sims.sim_io.field_diagnostics import profiles
from ww_quokka_sims.sim_io.snapshots import find_snapshots

##
## === TEST SUITE
##


class TestScalarProfileRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_fields(
        self,
    ):
        scalar_field_profile = profiles.ScalarFieldProfile(
            field_name="density",
            field_latex_label=latex_labels.LatexLabel(content=r"\rho"),
            sim_time=0.25,
            step_index=find_snapshots.StepIndex.from_value(step_index_value=3),
            profile_axis="x_0",
            position=numpy.array([0.0, 1.0, 2.0]),
            field_value=numpy.array([1.0, 2.0, 3.0]),
            amr_level=1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "scalar_field_profile.json"
            scalar_field_profile.save_to_file(file_path=file_path)
            loaded = profiles.ScalarFieldProfile.load_from_file(file_path=file_path)
        self.assertEqual(loaded.field_name, scalar_field_profile.field_name)
        self.assertEqual(loaded.field_latex_label, scalar_field_profile.field_latex_label)
        self.assertEqual(loaded.sim_time, scalar_field_profile.sim_time)
        self.assertEqual(loaded.step_index, scalar_field_profile.step_index)
        self.assertEqual(loaded.profile_axis, scalar_field_profile.profile_axis)
        self.assertEqual(loaded.amr_level, scalar_field_profile.amr_level)
        numpy.testing.assert_array_equal(loaded.position, scalar_field_profile.position)
        numpy.testing.assert_array_equal(loaded.field_value, scalar_field_profile.field_value)


class TestVectorProfileRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_components(
        self,
    ):
        vector_field_profile = profiles.VectorFieldProfile(
            field_name="velocity",
            sim_time=0.5,
            step_index=find_snapshots.StepIndex.from_value(step_index_value=1),
            profile_axis="x_1",
            position=numpy.array([0.0, 1.0]),
            components={
                cartesian_axes.CartesianAxis_3D.X0:
                profiles.VectorComponent(
                    field_value=numpy.array([1.0, 2.0]),
                    latex_label=latex_labels.LatexLabel(content=r"v_x"),
                ),
                cartesian_axes.CartesianAxis_3D.X1:
                profiles.VectorComponent(
                    field_value=numpy.array([3.0, 4.0]),
                    latex_label=latex_labels.LatexLabel(content=r"v_y"),
                ),
            },
            amr_level=0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "vector_field_profile.json"
            vector_field_profile.save_to_file(file_path=file_path)
            loaded = profiles.VectorFieldProfile.load_from_file(file_path=file_path)
        self.assertEqual(loaded.field_name, vector_field_profile.field_name)
        numpy.testing.assert_array_equal(loaded.position, vector_field_profile.position)
        self.assertEqual(set(loaded.components.keys()), set(vector_field_profile.components.keys()))
        for key, component in vector_field_profile.components.items():
            loaded_component = loaded.components[key]
            self.assertEqual(loaded_component.latex_label, component.latex_label)
            numpy.testing.assert_array_equal(loaded_component.field_value, component.field_value)


## } U-TEST
