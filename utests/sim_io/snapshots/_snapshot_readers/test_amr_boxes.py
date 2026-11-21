## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from ww_quokka_sims.sim_io.snapshots._snapshot_readers import amr_boxes

##
## === TEST DOUBLES
##


class MockYtBox:

    Level: int
    LeftEdge: numpy.ndarray
    ActiveDimensions: numpy.ndarray

    def __init__(
        self,
        *,
        level: int,
        left_edge: tuple[float, float, float],
        active_dimensions: tuple[int, int, int],
    ):
        self.Level = level
        self.LeftEdge = numpy.array(left_edge)
        self.ActiveDimensions = numpy.array(active_dimensions)


class MockYtIndex:

    grids: list[MockYtBox]

    def __init__(
        self,
        *,
        grids: list[MockYtBox],
    ):
        self.grids = grids


class MockYtDataset:

    domain_left_edge: numpy.ndarray
    domain_right_edge: numpy.ndarray
    domain_dimensions: numpy.ndarray
    refine_by: int
    index: MockYtIndex

    def __init__(
        self,
        *,
        grids: list[MockYtBox],
    ):
        ## a 4x2x2 base grid on [-2, 2] x [-1, 1] x [-1, 1], so base cells are 1 wide
        self.domain_left_edge = numpy.array([-2.0, -1.0, -1.0])
        self.domain_right_edge = numpy.array([2.0, 1.0, 1.0])
        self.domain_dimensions = numpy.array([4, 2, 2])
        self.refine_by = 2
        self.index = MockYtIndex(grids=grids)


##
## === TEST SUITE
##


class TestReadAMRBoxes(unittest.TestCase):

    def test_reads_boxes_on_every_level_as_own_level_indices(
        self,
    ):
        yt_dataset = MockYtDataset(
            grids=[
                MockYtBox(
                    level=0,
                    left_edge=(-2.0, -1.0, -1.0),
                    active_dimensions=(4, 2, 2),
                ),
                MockYtBox(
                    level=1,
                    left_edge=(0.0, -0.5, -1.0),
                    active_dimensions=(2, 2, 4),
                ),
                MockYtBox(
                    level=2,
                    left_edge=(0.5, 0.0, -0.25),
                    active_dimensions=(2, 2, 2),
                ),
            ],
        )
        boxes = amr_boxes.read_amr_boxes(yt_dataset=yt_dataset)
        self.assertEqual(
            [(_box.amr_level, _box.start_index, _box.end_index) for _box in boxes],
            [
                (0, (0, 0, 0), (4, 2, 2)),
                (1, (4, 1, 0), (6, 3, 4)),
                (2, (10, 4, 3), (12, 6, 5)),
            ],
        )

    def test_rejects_box_off_its_level_grid(
        self,
    ):
        yt_dataset = MockYtDataset(
            grids=[
                MockYtBox(
                    level=1,
                    left_edge=(0.25, -1.0, -1.0),
                    active_dimensions=(2, 2, 2),
                ),
            ],
        )
        with self.assertRaises(ValueError):
            amr_boxes.read_amr_boxes(yt_dataset=yt_dataset)

    def test_rejects_empty_index(
        self,
    ):
        with self.assertRaises(ValueError):
            amr_boxes.read_amr_boxes(yt_dataset=MockYtDataset(grids=[]))


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
