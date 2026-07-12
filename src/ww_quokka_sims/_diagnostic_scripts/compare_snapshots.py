## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._script_tools import cli
from ww_quokka_sims.sim_io import (
    find_snapshots,
    load_snapshot,
)

##
## === DATA ACCESS
##


@final
class SnapshotView:

    def __init__(
        self,
        *,
        snapshot_dir: Path,
    ):
        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve()

    def get_field_keys(
        self,
    ) -> set[load_snapshot.FieldKey]:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=self.snapshot_dir,
                verbose=False,
        ) as snapshot:
            field_keys = snapshot.list_available_field_keys()
        return set(field_keys)

    def load_sfield(
        self,
        field_key: load_snapshot.FieldKey,
    ) -> numpy.ndarray:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=self.snapshot_dir,
                verbose=False,
        ) as snapshot:
            sarray_3d = snapshot._load_3d_sarray(field_key=field_key)
        return numpy.ascontiguousarray(sarray_3d, dtype=numpy.float64)


##
## === FIELD COMPARISON
##


@dataclass(frozen=True)
class FieldComparison:
    """Result of a field comparison; preview_* fields are empty when same_shape is False."""

    same_shape: bool
    shape_in: tuple[int, int, int]
    shape_ref: tuple[int, int, int]
    num_diffs: int = 0
    preview_diff_indices: list[tuple[int, int, int]] = field(default_factory=list)
    preview_values_in: list[float] = field(default_factory=list)
    preview_values_ref: list[float] = field(default_factory=list)
    preview_abs_diff_values: list[float] = field(default_factory=list)


@final
class CompareFields:

    def __init__(
        self,
        *,
        sarray_in: numpy.ndarray,
        sarray_ref: numpy.ndarray,
        preview_limit: int,
    ):
        self.sarray_in = sarray_in
        self.sarray_ref = sarray_ref
        self.preview_limit = preview_limit

    def _get_diff_mask(
        self,
    ) -> numpy.ndarray:
        """Return a boolean mask of cells that differ, including signed-zero mismatches."""
        ## Note: NaNs are naturally different (NaN == NaN -> False)
        diff_value_mask = numpy.not_equal(self.sarray_in, self.sarray_ref)
        both_zero_mask = (self.sarray_in == 0.0) & (self.sarray_ref == 0.0)
        diff_sign_mask = numpy.signbit(self.sarray_in) ^ numpy.signbit(self.sarray_ref)
        mismatched_signed_zeros_mask = both_zero_mask & diff_sign_mask
        return diff_value_mask | mismatched_signed_zeros_mask

    def _get_preview_diff_coords(
        self,
        diff_mask: numpy.ndarray,
    ) -> numpy.ndarray:
        if not numpy.any(diff_mask):
            return numpy.empty((0, 3), dtype=numpy.int64)
        diff_coords = numpy.argwhere(diff_mask)  # shape: (num_diffs, num_dims=3)
        assert diff_coords.shape[1] == 3
        return diff_coords[:self.preview_limit]  # shape: (limited[num_diffs], num_dims=3)

    @staticmethod
    def _get_diff_indices(
        diff_coords: numpy.ndarray,
    ) -> list[tuple[int, int, int]]:
        return [(int(diff_coord[0]), int(diff_coord[1]), int(diff_coord[2])) for diff_coord in diff_coords]

    @staticmethod
    def _get_preview_values(
        diff_coords: numpy.ndarray,
        sarray_values: numpy.ndarray,
    ) -> list[float]:
        if diff_coords.size == 0:
            return []
        coords_x, coords_y, coords_z = diff_coords.T
        return sarray_values[coords_x, coords_y, coords_z].astype(numpy.float64, copy=False).tolist()

    def run(
        self,
    ) -> FieldComparison:
        shape_in = self.sarray_in.shape
        shape_ref = self.sarray_ref.shape
        if shape_in != shape_ref:
            return FieldComparison(
                same_shape=False,
                shape_in=shape_in,
                shape_ref=shape_ref,
            )
        diff_mask = self._get_diff_mask()
        num_diffs = int(
            diff_mask.sum(
                dtype=numpy.int64,
            ),
        )
        preview_diff_coords = self._get_preview_diff_coords(diff_mask=diff_mask)
        preview_diff_indices = self._get_diff_indices(diff_coords=preview_diff_coords)
        preview_values_in = self._get_preview_values(
            sarray_values=self.sarray_in,
            diff_coords=preview_diff_coords,
        )
        preview_values_ref = self._get_preview_values(
            sarray_values=self.sarray_ref,
            diff_coords=preview_diff_coords,
        )
        preview_abs_diff_values = self._get_preview_values(
            sarray_values=numpy.abs(self.sarray_ref - self.sarray_in),
            diff_coords=preview_diff_coords,
        )
        return FieldComparison(
            same_shape=True,
            shape_in=shape_in,
            shape_ref=shape_ref,
            num_diffs=num_diffs,
            preview_diff_indices=preview_diff_indices,
            preview_values_in=preview_values_in,
            preview_values_ref=preview_values_ref,
            preview_abs_diff_values=preview_abs_diff_values,
        )


##
## === SNAPSHOT COMPARISON
##


@final
class CompareSnapshots:

    def __init__(
        self,
        *,
        snapshot_view_in: SnapshotView,
        snapshot_view_ref: SnapshotView,
        preview_limit: int,
    ):
        self.snapshot_view_in = snapshot_view_in
        self.snapshot_view_ref = snapshot_view_ref
        self.preview_limit = preview_limit

    def get_shared_field_keys(
        self,
    ) -> list[load_snapshot.FieldKey]:
        """Return keys present in both snapshots; exits with code 4 if no keys overlap."""
        field_keys_in = self.snapshot_view_in.get_field_keys()
        field_keys_ref = self.snapshot_view_ref.get_field_keys()
        keys_missing_from_in = sorted(field_keys_ref - field_keys_in)
        keys_missing_from_ref = sorted(field_keys_in - field_keys_ref)
        shared_keys = sorted(field_keys_in & field_keys_ref)
        if len(keys_missing_from_in) > 0:
            manage_log.log_items(
                title="Fields from dir-2 missing from dir-1",
                items=[str(field_key) for field_key in keys_missing_from_in],
                message=f"There are {len(keys_missing_from_in)} fields not found in dir-1.",
                message_position="bottom",
            )
        if len(keys_missing_from_ref) > 0:
            manage_log.log_items(
                title="Fields from dir-1 missing from dir-2",
                items=[str(field_key) for field_key in keys_missing_from_ref],
                message=f"There are {len(keys_missing_from_ref)} fields not found in dir-2.",
                message_position="bottom",
            )
        if len(shared_keys) == 0:
            manage_log.log_items(
                title="Available fields in dir-1",
                items=[str(field_key) for field_key in sorted(field_keys_in)],
                message=f"There are {len(field_keys_in)} available fields.",
                message_position="bottom",
            )
            manage_log.log_items(
                title="Available fields in dir-2",
                items=[str(field_key) for field_key in sorted(field_keys_ref)],
                message=f"There are {len(field_keys_ref)} available fields.",
                message_position="bottom",
            )
            manage_log.log_error(
                text="There are no shared fields to compare.",
                notes={
                    "dir-1": str(self.snapshot_view_in.snapshot_dir),
                    "dir-2": str(self.snapshot_view_ref.snapshot_dir),
                },
            )
            raise SystemExit(4)
        manage_log.log_summary(
            title="Fields in dir-1 and dir-2",
            notes={
                "Shared": len(shared_keys),
                "Missing from dir-1": len(keys_missing_from_in),
                "Missing from dir-2": len(keys_missing_from_ref),
            },
        )
        return shared_keys

    def _compare_snapshots(
        self,
        field_key: load_snapshot.FieldKey,
    ) -> None:
        sarray_in = self.snapshot_view_in.load_sfield(field_key=field_key)
        sarray_ref = self.snapshot_view_ref.load_sfield(field_key=field_key)
        compare_fields = CompareFields(
            sarray_in=sarray_in,
            sarray_ref=sarray_ref,
            preview_limit=self.preview_limit,
        )
        field_comparison = compare_fields.run()
        if not field_comparison.same_shape:
            manage_log.log_error(
                text=f"[{field_key}] Shape mismatch (dir-1 vs dir-2).",
                notes={
                    "dir-1-shape": field_comparison.shape_in,
                    "dir-2-shape": field_comparison.shape_ref,
                },
                message_position="bottom",
            )
            return
        if field_comparison.num_diffs == 0:
            manage_log.log_note(
                text=
                f"[{field_key}] dir-1 == dir-2 (no value differences over shape: {field_comparison.shape_in}).",
            )
            return
        num_cells = int(
            numpy.prod(
                field_comparison.shape_in,
            ),
        )
        warning_message = f"[{field_key}] There are {field_comparison.num_diffs}/{num_cells} cells that are different."
        if field_comparison.num_diffs > self.preview_limit:
            warning_message += f" (previewing {self.preview_limit}/{field_comparison.num_diffs})."
        manage_log.log_warning(
            text=warning_message,
            notes={
                "diff-indices": field_comparison.preview_diff_indices,
                "values dir-1": field_comparison.preview_values_in,
                "values dir-2": field_comparison.preview_values_ref,
                "abs-diff-values": field_comparison.preview_abs_diff_values,
                "shape": field_comparison.shape_in,
            },
            message_position="bottom",
        )

    def run(
        self,
    ) -> None:
        field_keys = self.get_shared_field_keys()
        for field_key in field_keys:
            self._compare_snapshots(field_key=field_key)


##
## === SCRIPT INTERFACE
##


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        snapshot_dir_in: Path,
        snapshot_dir_ref: Path,
        preview_limit: int,
    ):
        self.snapshot_dir_in = Path(snapshot_dir_in).expanduser().resolve()
        self.snapshot_dir_ref = Path(snapshot_dir_ref).expanduser().resolve()
        self.preview_limit = preview_limit
        self._validate_inputs()

    def _validate_inputs(
        self,
    ) -> None:
        validate_types.ensure_finite_int(param=self.preview_limit)
        assert self.preview_limit > 0
        if not find_snapshots.looks_like_boxlib_dir(snapshot_dir=self.snapshot_dir_in):
            raise ValueError(f"dir-1 does not look like a BoxLib directory: {self.snapshot_dir_in}.")
        if not find_snapshots.looks_like_boxlib_dir(snapshot_dir=self.snapshot_dir_ref):
            raise ValueError(f"dir-2 does not look like a BoxLib directory: {self.snapshot_dir_ref}.")

    def run(
        self,
    ) -> None:
        snapshot_view_in = SnapshotView(snapshot_dir=self.snapshot_dir_in)
        snapshot_view_ref = SnapshotView(snapshot_dir=self.snapshot_dir_ref)
        compare_snapshots = CompareSnapshots(
            snapshot_view_in=snapshot_view_in,
            snapshot_view_ref=snapshot_view_ref,
            preview_limit=self.preview_limit,
        )
        compare_snapshots.run()


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    parser = argparse.ArgumentParser(
        description="Compare two Quokka (BoxLib) data-directories.",
        parents=[
            cli.base_parser(
                num_dirs=2,
                allow_vfields=False,
                allow_slicing=False,
                allow_fields=False,
                produces_data=False,
            ),
        ],
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=100,
        help="Maximum number of index locations to preview.",
    )
    user_args = parser.parse_args()
    script_interface = ScriptInterface(
        snapshot_dir_in=user_args.input_dir_1,
        snapshot_dir_ref=user_args.input_dir_2,
        preview_limit=user_args.preview_limit,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
