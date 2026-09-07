## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import pathlib
import unittest

## local
from ww_quokka_sims.sim_io.snapshots import find_snapshots

##
## === TEST SUITE
##


class TestStepIndex(unittest.TestCase):

    def test_value_returns_int(
        self,
    ):
        step_index = find_snapshots.StepIndex("0056218")
        self.assertEqual(step_index.value, 56218)

    def test_string_returns_original_string(
        self,
    ):
        step_index = find_snapshots.StepIndex("0056218")
        self.assertEqual(step_index.string, "0056218")

    def test_get_padded_string_pads_to_index_width(
        self,
    ):
        step_index = find_snapshots.StepIndex("218")
        self.assertEqual(step_index.get_padded_string(index_width=7), "0000218")

    def test_rejects_non_digit_string(
        self,
    ):
        with self.assertRaises(ValueError):
            find_snapshots.StepIndex("not_a_number")


class TestGetStepIndex(unittest.TestCase):

    def test_extracts_step_index_from_snapshot_name(
        self,
    ):
        step_index = find_snapshots.get_step_index(
            snapshot_dir=pathlib.Path("/some/dir/plt0056218"),
            snapshot_tag="plt",
        )
        self.assertEqual(step_index.value, 56218)

    def test_raises_when_tag_missing(
        self,
    ):
        with self.assertRaises(ValueError):
            find_snapshots.get_step_index(
                snapshot_dir=pathlib.Path("/some/dir/chk0056218"),
                snapshot_tag="plt",
            )

    def test_raises_when_suffix_is_not_digits(
        self,
    ):
        with self.assertRaises(ValueError):
            find_snapshots.get_step_index(
                snapshot_dir=pathlib.Path("/some/dir/plt_final"),
                snapshot_tag="plt",
            )


## } U-TEST
