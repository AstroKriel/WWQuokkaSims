## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes

## local
from ww_quokka_sims.sim_io.snapshots._snapshot_readers import read_fields

##
## === TEST SUITE
##


class TestAMRLeaves(unittest.TestCase):

    def test_accepts_matching_shapes(
        self,
    ):
        amr_leaves = read_fields.AMRLeaves(
            values=numpy.zeros(4),
            cell_widths=numpy.zeros((4, 3)),
            positions=numpy.zeros((4, 3)),
        )
        self.assertEqual(amr_leaves.values.shape, (4, ))

    def test_rejects_cell_widths_with_mismatched_shape(
        self,
    ):
        with self.assertRaises(ValueError):
            read_fields.AMRLeaves(
                values=numpy.zeros(4),
                cell_widths=numpy.zeros(4),
                positions=numpy.zeros((4, 3)),
            )

    def test_rejects_positions_with_mismatched_shape(
        self,
    ):
        with self.assertRaises(ValueError):
            read_fields.AMRLeaves(
                values=numpy.zeros(4),
                cell_widths=numpy.zeros((4, 3)),
                positions=numpy.zeros((4, 2)),
            )

    def test_cell_volumes_multiply_per_axis_widths(
        self,
    ):
        amr_leaves = read_fields.AMRLeaves(
            values=numpy.zeros(2),
            cell_widths=numpy.array([[0.5, 0.5, 0.125], [0.25, 0.25, 0.25]]),
            positions=numpy.zeros((2, 3)),
        )
        numpy.testing.assert_allclose(amr_leaves.get_cell_volumes(), [0.5 * 0.5 * 0.125, 0.25**3])

    def test_isotropic_cell_width_returns_single_dx_per_leaf(
        self,
    ):
        amr_leaves = read_fields.AMRLeaves(
            values=numpy.zeros(2),
            cell_widths=numpy.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]),
            positions=numpy.zeros((2, 3)),
        )
        numpy.testing.assert_array_equal(amr_leaves.get_isotropic_cell_width(), [0.5, 0.25])

    def test_isotropic_cell_width_rejects_anisotropic_leaves(
        self,
    ):
        amr_leaves = read_fields.AMRLeaves(
            values=numpy.zeros(2),
            cell_widths=numpy.array([[0.5, 0.5, 0.125], [0.25, 0.25, 0.25]]),
            positions=numpy.zeros((2, 3)),
        )
        with self.assertRaises(ValueError):
            amr_leaves.get_isotropic_cell_width()


class TestBoxlib3dAxesLabels(unittest.TestCase):

    def test_covers_every_default_3d_axis(
        self,
    ):
        self.assertEqual(
            set(read_fields.BOXLIB_3D_AXES_LABELS.keys()),
            set(cartesian_axes.DEFAULT_3D_AXES_ORDER),
        )

    def test_maps_axes_to_single_letter_labels(
        self,
    ):
        self.assertEqual(
            read_fields.BOXLIB_3D_AXES_LABELS[cartesian_axes.CartesianAxis_3D.X0],
            "x",
        )
        self.assertEqual(
            read_fields.BOXLIB_3D_AXES_LABELS[cartesian_axes.CartesianAxis_3D.X1],
            "y",
        )
        self.assertEqual(
            read_fields.BOXLIB_3D_AXES_LABELS[cartesian_axes.CartesianAxis_3D.X2],
            "z",
        )


## } U-TEST
