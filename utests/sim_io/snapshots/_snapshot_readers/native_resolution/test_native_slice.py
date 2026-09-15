## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from ww_quokka_sims.sim_io.snapshots._snapshot_readers.native_resolution import native_slice

##
## === TEST DOUBLES
##


class MockNativeSliceView:

    _field_values: dict[tuple[str, str], numpy.ndarray]

    def __init__(
        self,
        *,
        field_values: dict[tuple[str, str], numpy.ndarray],
    ):
        self._field_values = field_values

    def __getitem__(
        self,
        field_key: tuple[str, str],
    ) -> numpy.ndarray:
        return self._field_values[field_key]


##
## === TEST SUITE
##


class TestLoadSarray(unittest.TestCase):

    def test_reads_a_2d_field(
        self,
    ):
        sarray_2d = numpy.arange(9).reshape(3, 3).astype(numpy.float64)
        mock_native_slice_view = MockNativeSliceView(field_values={("boxlib", "gasDensity"): sarray_2d})
        loaded_sarray_2d = native_slice.load_sarray(
            native_slice_view=mock_native_slice_view,
            field_key=("boxlib", "gasDensity"),
        )
        numpy.testing.assert_array_equal(loaded_sarray_2d, sarray_2d)

    def test_rejects_a_field_that_is_not_2d(
        self,
    ):
        mock_native_slice_view = MockNativeSliceView(
            field_values={("boxlib", "gasDensity"): numpy.zeros((3, 3, 3))},
        )
        with self.assertRaises(ValueError):
            native_slice.load_sarray(
                native_slice_view=mock_native_slice_view,
                field_key=("boxlib", "gasDensity"),
            )


class TestLoadCellWidths(unittest.TestCase):

    def test_returns_shared_cell_width_for_isotropic_in_plane_axes(
        self,
    ):
        mock_native_slice_view = MockNativeSliceView(
            field_values={
                ("index", "dx"): numpy.full((2, 2), 0.5),
                ("index", "dy"): numpy.full((2, 2), 0.5),
            },
        )
        cell_width_2d = native_slice.load_cell_widths(
            native_slice_view=mock_native_slice_view,
            slice_axis_index=2,
        )
        numpy.testing.assert_array_equal(cell_width_2d, numpy.full((2, 2), 0.5))

    def test_rejects_anisotropic_in_plane_axes(
        self,
    ):
        mock_native_slice_view = MockNativeSliceView(
            field_values={
                ("index", "dx"): numpy.full((2, 2), 0.5),
                ("index", "dy"): numpy.full((2, 2), 0.25),
            },
        )
        with self.assertRaises(ValueError):
            native_slice.load_cell_widths(
                native_slice_view=mock_native_slice_view,
                slice_axis_index=2,
            )

    def test_reads_the_two_axes_that_are_not_the_slice_axis(
        self,
    ):
        mock_native_slice_view = MockNativeSliceView(
            field_values={
                ("index", "dy"): numpy.full((2, 2), 0.5),
                ("index", "dz"): numpy.full((2, 2), 0.5),
            },
        )
        cell_width_2d = native_slice.load_cell_widths(
            native_slice_view=mock_native_slice_view,
            slice_axis_index=0,
        )
        numpy.testing.assert_array_equal(cell_width_2d, numpy.full((2, 2), 0.5))


## } U-TEST
