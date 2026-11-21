## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import pathlib
import tempfile
import unittest

## local
from ww_quokka_sims.sim_io.snapshots import amr_box_layout, find_snapshots

##
## === HELPER FUNCTIONS
##


def make_two_level_layout() -> amr_box_layout.AMRBoxLayout:
    """A 4x2x2 base grid on [0, 4] x [0, 2] x [0, 2], refined by 2 over its first two base cells in x."""
    base_box = amr_box_layout.AMRBox(
        amr_level=0,
        start_index=(0, 0, 0),
        end_index=(4, 2, 2),
    )
    refined_box = amr_box_layout.AMRBox(
        amr_level=1,
        start_index=(0, 0, 0),
        end_index=(4, 4, 4),
    )
    return amr_box_layout.AMRBoxLayout(
        sim_time=0.5,
        step_index=find_snapshots.StepIndex.from_value(step_index_value=12),
        refinement_ratio=2,
        base_resolution=(4, 2, 2),
        domain_bounds=((0.0, 4.0), (0.0, 2.0), (0.0, 2.0)),
        boxes=[base_box, refined_box],
    )


##
## === TEST SUITE
##


class TestAMRBox_Validation(unittest.TestCase):

    def test_rejects_negative_level(
        self,
    ):
        with self.assertRaises(ValueError):
            amr_box_layout.AMRBox(
                amr_level=-1,
                start_index=(0, 0, 0),
                end_index=(1, 1, 1),
            )

    def test_rejects_empty_box(
        self,
    ):
        with self.assertRaises(ValueError):
            amr_box_layout.AMRBox(
                amr_level=0,
                start_index=(0, 0, 0),
                end_index=(1, 0, 1),
            )

    def test_resolution_is_end_minus_start(
        self,
    ):
        box = amr_box_layout.AMRBox(
            amr_level=0,
            start_index=(2, 0, 1),
            end_index=(5, 4, 2),
        )
        self.assertEqual(box.resolution, (3, 4, 1))


class TestAMRBoxLayout_Validation(unittest.TestCase):

    def test_rejects_box_outside_its_level(
        self,
    ):
        outside_box = amr_box_layout.AMRBox(
            amr_level=1,
            start_index=(6, 0, 0),
            end_index=(10, 4, 4),
        )
        with self.assertRaises(ValueError):
            amr_box_layout.AMRBoxLayout(
                sim_time=0.0,
                step_index=find_snapshots.StepIndex.from_value(step_index_value=0),
                refinement_ratio=2,
                base_resolution=(4, 2, 2),
                domain_bounds=((0.0, 4.0), (0.0, 2.0), (0.0, 2.0)),
                boxes=[outside_box],
            )

    def test_rejects_refinement_ratio_below_two(
        self,
    ):
        with self.assertRaises(ValueError):
            amr_box_layout.AMRBoxLayout(
                sim_time=0.0,
                step_index=find_snapshots.StepIndex.from_value(step_index_value=0),
                refinement_ratio=1,
                base_resolution=(4, 2, 2),
                domain_bounds=((0.0, 4.0), (0.0, 2.0), (0.0, 2.0)),
                boxes=[],
            )


class TestAMRBoxLayout_Geometry(unittest.TestCase):

    def test_level_resolution_scales_by_refinement_ratio(
        self,
    ):
        box_layout = make_two_level_layout()
        self.assertEqual(box_layout.get_level_resolution(amr_level=1), (8, 4, 4))
        self.assertEqual(box_layout.max_amr_level, 1)

    def test_refined_box_bounds_cover_first_two_base_cells(
        self,
    ):
        box_layout = make_two_level_layout()
        refined_box = box_layout.get_boxes_on_level(amr_level=1)[0]
        self.assertEqual(
            box_layout.get_box_bounds(refined_box),
            ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
        )

    def test_plane_crossing_excludes_boxes_that_end_at_the_plane(
        self,
    ):
        box_layout = make_two_level_layout()
        crossing_levels = sorted(
            _box.amr_level for _box in box_layout.get_boxes_crossing_plane(
                axis_index=0,
                coordinate=3.0,
            )
        )
        self.assertEqual(crossing_levels, [0])
        crossing_levels = sorted(
            _box.amr_level for _box in box_layout.get_boxes_crossing_plane(
                axis_index=0,
                coordinate=1.0,
            )
        )
        self.assertEqual(crossing_levels, [0, 1])


class TestAMRBoxLayout_RoundTrip(unittest.TestCase):

    def test_save_then_load_returns_equal_layout(
        self,
    ):
        box_layout = make_two_level_layout()
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = pathlib.Path(temp_dir) / "amr_boxes-index=12.json"
            box_layout.save_to_file(file_path=file_path)
            loaded_layout = amr_box_layout.AMRBoxLayout.load_from_file(file_path=file_path)
        self.assertEqual(loaded_layout, box_layout)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
