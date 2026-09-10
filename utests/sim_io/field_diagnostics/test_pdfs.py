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
from ww_quokka_sims.sim_io.field_diagnostics import pdfs
from ww_quokka_sims.sim_io.snapshots import find_snapshots

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
        pdf = pdfs.ComputePDFs._estimate_pdf(
            sarray_3d=self._FIELD_DATA,
            num_bins=4,
            use_log10_bins=False,
        )
        self.assertEqual(len(pdf.bin_centers), 4)
        self.assertEqual(len(pdf.densities), 4)
        self.assertAlmostEqual(float(pdf.bin_centers.min()), -1.0, places=6)
        self.assertAlmostEqual(float(pdf.bin_centers.max()), 8.0, places=6)

    def test_log10_bins_span_only_the_positive_values_in_log_space(
        self,
    ):
        pdf = pdfs.ComputePDFs._estimate_pdf(
            sarray_3d=self._FIELD_DATA,
            num_bins=4,
            use_log10_bins=True,
        )
        self.assertEqual(len(pdf.bin_centers), 4)
        self.assertEqual(len(pdf.densities), 4)
        ## non-positive entries (-1.0, 0.0) must be masked out before binning, so the
        ## range reflects only {1, 2, 4, 8} in log10-space, not the raw data's range
        self.assertAlmostEqual(float(pdf.bin_centers.min()), numpy.log10(1.0), places=6)
        self.assertAlmostEqual(float(pdf.bin_centers.max()), numpy.log10(8.0), places=6)


class TestPDFDataRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_scalar_field_data(
        self,
    ):
        field_pdf = pdfs.FieldPDF(
            sim_time=0.25,
            step_index=find_snapshots.StepIndex.from_value(3),
            grouped_bin_centers=[numpy.array([0.0, 1.0, 2.0])],
            grouped_densities=[numpy.array([-1.0, -2.0, -3.0])],
            comp_latex_labels=[latex_labels.LatexLabel(content=r"\rho")],
            use_log10_bins=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "pdf.json"
            field_pdf.save_to_file(file_path)
            loaded = pdfs.FieldPDF.load_from_file(file_path)
        self.assertEqual(loaded.sim_time, field_pdf.sim_time)
        self.assertEqual(loaded.step_index, field_pdf.step_index)
        self.assertEqual(loaded.comp_latex_labels, field_pdf.comp_latex_labels)
        self.assertEqual(loaded.use_log10_bins, field_pdf.use_log10_bins)
        numpy.testing.assert_array_equal(loaded.grouped_bin_centers[0], field_pdf.grouped_bin_centers[0])
        numpy.testing.assert_array_equal(loaded.grouped_densities[0], field_pdf.grouped_densities[0])

    def test_save_and_load_preserves_multi_component_data(
        self,
    ):
        field_pdf = pdfs.FieldPDF(
            sim_time=0.5,
            step_index=find_snapshots.StepIndex.from_value(1),
            grouped_bin_centers=[numpy.array([0.0, 1.0]), numpy.array([2.0, 3.0])],
            grouped_densities=[
                numpy.array([-1.0, -2.0]),
                numpy.array([-3.0, -4.0]),
            ],
            comp_latex_labels=[
                latex_labels.LatexLabel(content=r"v_x"),
                latex_labels.LatexLabel(content=r"v_y")
            ],
            use_log10_bins=False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "pdf.json"
            field_pdf.save_to_file(file_path)
            loaded = pdfs.FieldPDF.load_from_file(file_path)
        self.assertEqual(
            sorted(comp_latex_label.content for comp_latex_label in loaded.comp_latex_labels),
            sorted(comp_latex_label.content for comp_latex_label in field_pdf.comp_latex_labels),
        )
        self.assertEqual(loaded.num_comps, field_pdf.num_comps)


## } U-TEST
