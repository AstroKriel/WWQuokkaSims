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
from jormi.ww_plots import latex_labels

## local
from ww_quokka_sims.sim_io.field_diagnostics import spectra
from ww_quokka_sims.sim_io.snapshots import find_snapshots

##
## === TEST SUITE
##


class TestSpectraDataRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_fields(
        self,
    ):
        field_spectrum = spectra.FieldSpectrum(
            sim_time=0.25,
            step_index=find_snapshots.StepIndex.from_value(3),
            latex_label=latex_labels.LatexLabel(content=r"\rho"),
            log10_k_bin_centers=numpy.array([0.0, 1.0, 2.0]),
            log10_power_spectrum=numpy.array([-1.0, -2.0, -3.0]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "spectra.json"
            field_spectrum.save_to_file(file_path)
            loaded = spectra.FieldSpectrum.load_from_file(file_path)
        self.assertEqual(loaded.sim_time, field_spectrum.sim_time)
        self.assertEqual(loaded.step_index, field_spectrum.step_index)
        self.assertEqual(loaded.latex_label, field_spectrum.latex_label)
        numpy.testing.assert_array_equal(loaded.log10_k_bin_centers, field_spectrum.log10_k_bin_centers)
        numpy.testing.assert_array_equal(loaded.log10_power_spectrum, field_spectrum.log10_power_spectrum)


## } U-TEST
