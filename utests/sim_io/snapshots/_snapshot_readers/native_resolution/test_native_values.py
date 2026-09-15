## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from ww_quokka_sims.sim_io.snapshots._snapshot_readers.native_resolution import native_values

##
## === TEST DOUBLES
##


class MockYtBox:

    ActiveDimensions: numpy.ndarray
    child_mask: numpy.ndarray
    dds: numpy.ndarray
    _field_values: dict[tuple[str, str], numpy.ndarray]

    def __init__(
        self,
        *,
        active_dimensions: tuple[int, int, int],
        child_mask: numpy.ndarray,
        cell_widths: tuple[float, float, float],
        field_values: dict[tuple[str, str], numpy.ndarray],
    ):
        self.ActiveDimensions = numpy.array(active_dimensions)
        self.child_mask = child_mask
        self.dds = numpy.array(cell_widths)
        self._field_values = field_values

    def __getitem__(
        self,
        field_key: tuple[str, str],
    ) -> numpy.ndarray:
        return self._field_values[field_key]


class MockYtIndex:

    grids: list[MockYtBox]

    def __init__(
        self,
        *,
        grids: list[MockYtBox],
    ):
        self.grids = grids


class MockYtDataset:

    index: MockYtIndex

    def __init__(
        self,
        *,
        grids: list[MockYtBox],
    ):
        self.index = MockYtIndex(grids=grids)


def _make_mock_yt_box(
    *,
    shape: tuple[int, int, int],
    cell_width: float,
    value: float,
) -> MockYtBox:
    positions = numpy.zeros(shape, dtype=numpy.float64)
    field_key = ("boxlib", "gasDensity")
    return MockYtBox(
        active_dimensions=shape,
        child_mask=numpy.ones(shape, dtype=bool),
        cell_widths=(cell_width, cell_width, cell_width),
        field_values={
            field_key: numpy.full(shape, value, dtype=numpy.float64),
            ("index", "x"): positions,
            ("index", "y"): positions,
            ("index", "z"): positions,
        },
    )


##
## === TEST SUITE
##


class TestLeafBox(unittest.TestCase):

    def test_accepts_child_mask_matching_box_resolution(
        self,
    ):
        mock_yt_box = _make_mock_yt_box(
            shape=(2, 2, 2),
            cell_width=0.5,
            value=1.0,
        )
        leaf_box = native_values._LeafBox(
            box=mock_yt_box,
            child_mask=mock_yt_box.child_mask,
        )
        self.assertIs(leaf_box.box, mock_yt_box)

    def test_rejects_child_mask_with_mismatched_shape(
        self,
    ):
        mock_yt_box = _make_mock_yt_box(
            shape=(2, 2, 2),
            cell_width=0.5,
            value=1.0,
        )
        with self.assertRaises(ValueError):
            native_values._LeafBox(
                box=mock_yt_box,
                child_mask=numpy.ones((2, 2, 1), dtype=bool),
            )


class TestEnsureIsotropicCell(unittest.TestCase):

    def test_accepts_isotropic_cell_widths(
        self,
    ):
        native_values._ensure_isotropic_cell(cell_widths=numpy.array([0.5, 0.5, 0.5]))

    def test_rejects_anisotropic_cell_widths(
        self,
    ):
        with self.assertRaises(ValueError):
            native_values._ensure_isotropic_cell(cell_widths=numpy.array([0.5, 0.5, 0.125]))


class TestIterateAmrLeafBoxes(unittest.TestCase):

    def test_rejects_snapshot_with_no_boxes(
        self,
    ):
        mock_yt_dataset = MockYtDataset(grids=[])
        with self.assertRaises(ValueError):
            list(native_values._iterate_amr_leaf_boxes(yt_dataset=mock_yt_dataset))


class TestLoadAmrLeaves(unittest.TestCase):

    def test_pools_leaf_cells_across_boxes(
        self,
    ):
        mock_yt_boxes = [
            _make_mock_yt_box(
                shape=(2, 2, 2),
                cell_width=0.5,
                value=1.0,
            ),
            _make_mock_yt_box(
                shape=(3, 3, 3),
                cell_width=0.25,
                value=2.0,
            ),
        ]
        mock_yt_dataset = MockYtDataset(grids=mock_yt_boxes)
        amr_leaves = native_values.load_amr_leaves(
            yt_dataset=mock_yt_dataset,
            field_key=("boxlib", "gasDensity"),
        )
        self.assertEqual(amr_leaves.values.shape, (8 + 27, ))
        self.assertTrue(numpy.all(amr_leaves.values[:8] == 1.0))
        self.assertTrue(numpy.all(amr_leaves.values[8:] == 2.0))
        self.assertTrue(numpy.all(amr_leaves.cell_width[:8] == 0.5))
        self.assertTrue(numpy.all(amr_leaves.cell_width[8:] == 0.25))
        self.assertEqual(amr_leaves.positions.shape, (8 + 27, 3))

    def test_excludes_cells_masked_out_by_a_finer_box(
        self,
    ):
        mock_yt_box = _make_mock_yt_box(
            shape=(2, 2, 2),
            cell_width=0.5,
            value=1.0,
        )
        mock_yt_box.child_mask = numpy.zeros((2, 2, 2), dtype=bool)
        mock_yt_dataset = MockYtDataset(grids=[mock_yt_box])
        with self.assertRaises(ValueError):
            native_values.load_amr_leaves(
                yt_dataset=mock_yt_dataset,
                field_key=("boxlib", "gasDensity"),
            )

    def test_rejects_anisotropic_box_cells(
        self,
    ):
        mock_yt_box = _make_mock_yt_box(
            shape=(2, 2, 2),
            cell_width=0.5,
            value=1.0,
        )
        mock_yt_box.dds = numpy.array([0.5, 0.5, 0.25])
        mock_yt_dataset = MockYtDataset(grids=[mock_yt_box])
        with self.assertRaises(ValueError):
            native_values.load_amr_leaves(
                yt_dataset=mock_yt_dataset,
                field_key=("boxlib", "gasDensity"),
            )


## } U-TEST
