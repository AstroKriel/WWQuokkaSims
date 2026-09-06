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
from ww_quokka_sims.sim_io.field_diagnostics import pdfs

##
## === TEST SUITE
##


class TestEstimatePDF(unittest.TestCase):

    ## a mix of positive, zero, and negative values: only the positive ones survive
    ## log10-binning, so this distinguishes "binned in linear space" from "binned in
    ## log10 space, non-positive entries masked out" rather than just checking shapes
    _FIELD_DATA: numpy.ndarray = numpy.array([1.0, 2.0, 4.0, 8.0, -1.0, 0.0])

    def test_linear_bins_span_the_full_raw_range(
        self,
    ):
        bin_centers, densities = pdfs.ComputePDFs._estimate_pdf(
            field_data=self._FIELD_DATA,
            num_bins=4,
            use_log10_bins=False,
        )
        self.assertEqual(len(bin_centers), 4)
        self.assertEqual(len(densities), 4)
        self.assertAlmostEqual(float(bin_centers.min()), -1.0, places=6)
        self.assertAlmostEqual(float(bin_centers.max()), 8.0, places=6)

    def test_log10_bins_span_only_the_positive_values_in_log_space(
        self,
    ):
        bin_centers, densities = pdfs.ComputePDFs._estimate_pdf(
            field_data=self._FIELD_DATA,
            num_bins=4,
            use_log10_bins=True,
        )
        self.assertEqual(len(bin_centers), 4)
        self.assertEqual(len(densities), 4)
        ## non-positive entries (-1.0, 0.0) must be masked out before binning, so the
        ## range reflects only {1, 2, 4, 8} in log10-space, not the raw data's range
        self.assertAlmostEqual(float(bin_centers.min()), numpy.log10(1.0), places=6)
        self.assertAlmostEqual(float(bin_centers.max()), numpy.log10(8.0), places=6)


class TestPDFDataRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_scalar_field_data(
        self,
    ):
        pdf_data = pdfs.PDFData(
            sim_time=0.25,
            step_index=3,
            grouped_bin_centers=[numpy.array([0.0, 1.0, 2.0])],
            grouped_densities=[numpy.array([-1.0, -2.0, -3.0])],
            comp_labels=[r"$\rho$"],
            use_log10_bins=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "pdf.json"
            pdf_data.save_to_file(file_path)
            loaded = pdfs.PDFData.load_from_file(file_path)
        self.assertEqual(loaded.sim_time, pdf_data.sim_time)
        self.assertEqual(loaded.step_index, pdf_data.step_index)
        self.assertEqual(loaded.comp_labels, pdf_data.comp_labels)
        self.assertEqual(loaded.use_log10_bins, pdf_data.use_log10_bins)
        numpy.testing.assert_array_equal(loaded.grouped_bin_centers[0], pdf_data.grouped_bin_centers[0])
        numpy.testing.assert_array_equal(loaded.grouped_densities[0], pdf_data.grouped_densities[0])

    def test_save_and_load_preserves_multi_component_data(
        self,
    ):
        pdf_data = pdfs.PDFData(
            sim_time=0.5,
            step_index=1,
            grouped_bin_centers=[numpy.array([0.0, 1.0]), numpy.array([2.0, 3.0])],
            grouped_densities=[numpy.array([-1.0, -2.0]),
                               numpy.array([-3.0, -4.0])],
            comp_labels=[r"$v_x$", r"$v_y$"],
            use_log10_bins=False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "pdf.json"
            pdf_data.save_to_file(file_path)
            loaded = pdfs.PDFData.load_from_file(file_path)
        self.assertEqual(sorted(loaded.comp_labels), sorted(pdf_data.comp_labels))
        self.assertEqual(loaded.num_comps, pdf_data.num_comps)


## } U-TEST
