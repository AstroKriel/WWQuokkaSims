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
from ww_quokka_sims.sim_io.field_diagnostics import spectra

##
## === TEST SUITE
##


class TestSpectraDataRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_fields(
        self,
    ):
        spectra_data = spectra.SpectraData(
            sim_time=0.25,
            step_index=3,
            latex_label=r"\rho",
            log10_k_bin_centers=numpy.array([0.0, 1.0, 2.0]),
            log10_spectrum=numpy.array([-1.0, -2.0, -3.0]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "spectra.json"
            spectra_data.save_to_file(file_path)
            loaded = spectra.SpectraData.load_from_file(file_path)
        self.assertEqual(loaded.sim_time, spectra_data.sim_time)
        self.assertEqual(loaded.step_index, spectra_data.step_index)
        self.assertEqual(loaded.latex_label, spectra_data.latex_label)
        numpy.testing.assert_array_equal(loaded.log10_k_bin_centers, spectra_data.log10_k_bin_centers)
        numpy.testing.assert_array_equal(loaded.log10_spectrum, spectra_data.log10_spectrum)


## } U-TEST
