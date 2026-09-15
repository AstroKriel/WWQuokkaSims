## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from ww_quokka_sims.sim_io.snapshots._snapshot_readers.uniform_resolution import expanded_boxes

##
## === TEST SUITE
##


class TestExpandedFArray(unittest.TestCase):

    def test_accepts_farray_matching_cell_range_resolution(
        self,
    ):
        farray = numpy.zeros((3, 6, 6, 6))
        expanded_farray = expanded_boxes.ExpandedFArray(
            farray=farray,
            cell_range=(slice(0, 6), slice(0, 6), slice(0, 6)),
        )
        self.assertIs(expanded_farray.farray, farray)

    def test_rejects_farray_missing_the_leading_component_axis(
        self,
    ):
        farray = numpy.zeros((6, 6, 6))
        with self.assertRaises(ValueError):
            expanded_boxes.ExpandedFArray(
                farray=farray,
                cell_range=(slice(0, 6), slice(0, 6), slice(0, 6)),
            )

    def test_rejects_farray_smaller_than_implied_resolution(
        self,
    ):
        farray = numpy.zeros((3, 6, 6, 4))
        with self.assertRaises(ValueError):
            expanded_boxes.ExpandedFArray(
                farray=farray,
                cell_range=(slice(0, 6), slice(0, 6), slice(0, 6)),
            )


class TestComputeNumExtraCells(unittest.TestCase):

    def test_returns_half_the_grad_order(
        self,
    ):
        self.assertEqual(expanded_boxes.compute_num_extra_cells(grad_order=2), 1)
        self.assertEqual(expanded_boxes.compute_num_extra_cells(grad_order=4), 2)
        self.assertEqual(expanded_boxes.compute_num_extra_cells(grad_order=6), 3)

    def test_rejects_unsupported_grad_order(
        self,
    ):
        with self.assertRaises(ValueError):
            expanded_boxes.compute_num_extra_cells(grad_order=3)


class TestTrimExpandedBox(unittest.TestCase):

    def test_drops_outer_cells_from_every_spatial_axis(
        self,
    ):
        expanded_farray = numpy.arange(3 * 6 * 6 * 6).reshape(3, 6, 6, 6).astype(numpy.float64)
        trimmed_farray = expanded_boxes.trim_expanded_box(
            expanded_farray=expanded_farray,
            num_extra_cells=1,
        )
        self.assertEqual(trimmed_farray.shape, (3, 4, 4, 4))
        numpy.testing.assert_array_equal(trimmed_farray, expanded_farray[:, 1:-1, 1:-1, 1:-1])

    def test_leaves_leading_component_axis_untouched_for_a_scalar(
        self,
    ):
        expanded_farray = numpy.zeros((5, 5, 5))
        trimmed_farray = expanded_boxes.trim_expanded_box(
            expanded_farray=expanded_farray,
            num_extra_cells=2,
        )
        self.assertEqual(trimmed_farray.shape, (1, 1, 1))


## } U-TEST
