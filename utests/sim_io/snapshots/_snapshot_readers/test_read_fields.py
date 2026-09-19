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
            cell_width=numpy.zeros(4),
            positions=numpy.zeros((4, 3)),
        )
        self.assertEqual(amr_leaves.values.shape, (4, ))

    def test_rejects_cell_width_with_mismatched_shape(
        self,
    ):
        with self.assertRaises(ValueError):
            read_fields.AMRLeaves(
                values=numpy.zeros(4),
                cell_width=numpy.zeros(3),
                positions=numpy.zeros((4, 3)),
            )

    def test_rejects_positions_with_mismatched_shape(
        self,
    ):
        with self.assertRaises(ValueError):
            read_fields.AMRLeaves(
                values=numpy.zeros(4),
                cell_width=numpy.zeros(4),
                positions=numpy.zeros((4, 2)),
            )


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
