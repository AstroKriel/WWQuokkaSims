## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from ww_quokka_sims.sim_io.snapshots._snapshot_readers.uniform_resolution import level0_boxes

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
    index: MockYtIndex

    def __init__(
        self,
        *,
        domain_left_edge: tuple[float, float, float],
        domain_right_edge: tuple[float, float, float],
        domain_dimensions: tuple[int, int, int],
        grids: list[MockYtBox],
    ):
        self.domain_left_edge = numpy.array(domain_left_edge)
        self.domain_right_edge = numpy.array(domain_right_edge)
        self.domain_dimensions = numpy.array(domain_dimensions)
        self.index = MockYtIndex(grids=grids)


##
## === TEST SUITE
##


class TestLevel0Box(unittest.TestCase):

    def test_accepts_cell_range_matching_box_resolution(
        self,
    ):
        mock_yt_box = MockYtBox(
            level=0,
            left_edge=(0.0, 0.0, 0.0),
            active_dimensions=(4, 4, 4),
        )
        level0_box = level0_boxes.Level0Box(
            box=mock_yt_box,
            cell_range=(slice(0, 4), slice(0, 4), slice(0, 4)),
        )
        self.assertIs(level0_box.box, mock_yt_box)

    def test_rejects_cell_range_with_mismatched_resolution(
        self,
    ):
        mock_yt_box = MockYtBox(
            level=0,
            left_edge=(0.0, 0.0, 0.0),
            active_dimensions=(4, 4, 4),
        )
        with self.assertRaises(ValueError):
            level0_boxes.Level0Box(
                box=mock_yt_box,
                cell_range=(slice(0, 4), slice(0, 4), slice(0, 3)),
            )


class TestIterateAmrLevel0Boxes(unittest.TestCase):

    def test_computes_cell_range_from_box_left_edge(
        self,
    ):
        mock_yt_boxes = [
            MockYtBox(
                level=0,
                left_edge=(0.0, 0.0, 0.0),
                active_dimensions=(2, 4, 4),
            ),
            MockYtBox(
                level=0,
                left_edge=(0.5, 0.0, 0.0),
                active_dimensions=(2, 4, 4),
            ),
        ]
        mock_yt_dataset = MockYtDataset(
            domain_left_edge=(0.0, 0.0, 0.0),
            domain_right_edge=(1.0, 1.0, 1.0),
            domain_dimensions=(4, 4, 4),
            grids=mock_yt_boxes,
        )
        level0_box_list = list(level0_boxes.iterate_amr_level_0_boxes(yt_dataset=mock_yt_dataset))
        self.assertEqual(len(level0_box_list), 2)
        self.assertEqual(level0_box_list[0].cell_range, (slice(0, 2), slice(0, 4), slice(0, 4)))
        self.assertEqual(level0_box_list[1].cell_range, (slice(2, 4), slice(0, 4), slice(0, 4)))

    def test_skips_boxes_above_amr_level_0(
        self,
    ):
        mock_yt_boxes = [
            MockYtBox(
                level=0,
                left_edge=(0.0, 0.0, 0.0),
                active_dimensions=(4, 4, 4),
            ),
            MockYtBox(
                level=1,
                left_edge=(0.0, 0.0, 0.0),
                active_dimensions=(8, 8, 8),
            ),
        ]
        mock_yt_dataset = MockYtDataset(
            domain_left_edge=(0.0, 0.0, 0.0),
            domain_right_edge=(1.0, 1.0, 1.0),
            domain_dimensions=(4, 4, 4),
            grids=mock_yt_boxes,
        )
        level0_box_list = list(level0_boxes.iterate_amr_level_0_boxes(yt_dataset=mock_yt_dataset))
        self.assertEqual(len(level0_box_list), 1)

    def test_rejects_box_left_edge_off_the_cell_grid(
        self,
    ):
        mock_yt_boxes = [
            MockYtBox(
                level=0,
                left_edge=(0.3, 0.0, 0.0),
                active_dimensions=(2, 4, 4),
            ),
        ]
        mock_yt_dataset = MockYtDataset(
            domain_left_edge=(0.0, 0.0, 0.0),
            domain_right_edge=(1.0, 1.0, 1.0),
            domain_dimensions=(4, 4, 4),
            grids=mock_yt_boxes,
        )
        with self.assertRaises(ValueError):
            list(level0_boxes.iterate_amr_level_0_boxes(yt_dataset=mock_yt_dataset))

    def test_rejects_snapshot_with_no_amr_level_0_boxes(
        self,
    ):
        mock_yt_dataset = MockYtDataset(
            domain_left_edge=(0.0, 0.0, 0.0),
            domain_right_edge=(1.0, 1.0, 1.0),
            domain_dimensions=(4, 4, 4),
            grids=[],
        )
        with self.assertRaises(ValueError):
            list(level0_boxes.iterate_amr_level_0_boxes(yt_dataset=mock_yt_dataset))


## } U-TEST
