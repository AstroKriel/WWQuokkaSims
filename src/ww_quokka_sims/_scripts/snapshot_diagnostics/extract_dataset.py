## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    NamedTuple,
    final,
)

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_fns import parallel_dispatch
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._scripts.snapshot_tools import (
    cli,
    field_registry,
)
from ww_quokka_sims.sim_io.snapshots import (
    find_snapshots,
    load_snapshot,
)

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class ResolvedFieldArgs:
    field_name: str
    field_loader: Callable
    amr_level: int = 0


class WorkerArgs(NamedTuple):
    """Flat, pickleable argument bundle passed to the parallel extraction worker."""

    snapshot_dir: str
    snapshot_tag: str
    field_name: str
    field_loader: Callable
    comps_to_extract: tuple[cartesian_axes.CartesianAxis_3D, ...]
    data_dir: str
    index_width: int
    overwrite: bool
    amr_level: int = 0


##
## === FIELD PROCESSING
##


def _axis_to_index(
    axis: cartesian_axes.CartesianAxis_3D,
) -> int:
    return cartesian_axes.get_axis_index(axis)


def _get_step_time(
    field: field_models.AnyField_3D,
) -> float:
    step_time = field.sim_time
    if (step_time is None) or (not numpy.isfinite(step_time)):
        msg = f"Invalid sim_time for field: {step_time!r}."
        manage_log.log_error(text=msg)
        raise RuntimeError(msg)
    return float(step_time)


##
## === DATASET EXTRACTION
##


@dataclass(frozen=True)
class FieldExtractor:
    snapshot_tag: str
    field_args: ResolvedFieldArgs
    comps_to_extract: tuple[cartesian_axes.CartesianAxis_3D, ...]
    overwrite: bool = False

    def _expected_file_name(
        self,
        *,
        step_index: int,
        index_width: int,
    ) -> str:
        field_name = self.field_args.field_name
        padded_index = f"{step_index:0{index_width}d}"
        return f"{field_name}-index={padded_index}-amr_level={self.field_args.amr_level}.npz"

    def _load_field(
        self,
        *,
        snapshot_dir: Path,
    ) -> field_models.AnyField_3D:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as snapshot:
            field = self.field_args.field_loader(
                snapshot,
                amr_level=self.field_args.amr_level,
            )  # ScalarField_3D or VectorField_3D
        return field

    def _save_field(
        self,
        *,
        field: field_models.AnyField_3D,
        step_time: float,
        step_index: int,
        index_width: int,
        data_dir: Path,
    ) -> None:
        field_name = self.field_args.field_name
        file_name = self._expected_file_name(step_index=step_index, index_width=index_width)
        if isinstance(field, field_models.ScalarField_3D):
            sarray_3d = field_models.extract_3d_sarray(
                sfield_3d=field,
                param_name=f"<{field_name}_sfield_3d>",
            )
            numpy.savez(
                data_dir / file_name,
                sarray_3d=sarray_3d,
                step_time=step_time,
                step_index=step_index,
                amr_level=self.field_args.amr_level,
            )
            return
        if not isinstance(field, field_models.VectorField_3D):
            raise ValueError(f"{field_name} is an unrecognised field type.")
        if not self.comps_to_extract:
            raise ValueError(
                f"Vector field `{field_name}` requires at least one component to extract; none provided.",
            )
        varray_3d = field_models.extract_3d_varray(
            vfield_3d=field,
            param_name=f"<{field_name}_vfield_3d>",
        )
        comp_indices = [_axis_to_index(comp_axis) for comp_axis in self.comps_to_extract]
        comp_labels = [comp_axis.axis_label for comp_axis in self.comps_to_extract]
        numpy.savez(
            data_dir / file_name,
            varray_3d=varray_3d[comp_indices, ...],
            comp_labels=numpy.array(comp_labels),
            step_time=step_time,
            step_index=step_index,
            amr_level=self.field_args.amr_level,
        )

    def extract_snapshot(
        self,
        *,
        snapshot_dir: Path,
        data_dir: Path,
        index_width: int,
    ) -> None:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            ),
        )
        file_path = data_dir / self._expected_file_name(step_index=step_index, index_width=index_width)
        if (not self.overwrite) and file_path.exists():
            return
        field = self._load_field(snapshot_dir=snapshot_dir)
        self._save_field(
            field=field,
            step_time=_get_step_time(field),
            step_index=step_index,
            index_width=index_width,
            data_dir=data_dir,
        )


def extract_fields_in_serial(
    *,
    snapshot_tag: str,
    fields_to_extract: tuple[str, ...],
    comps_to_extract: tuple[cartesian_axes.CartesianAxis_3D, ...],
    snapshot_dirs: list[Path],
    data_dir: Path,
    index_width: int,
    overwrite: bool = False,
    amr_level: int = 0,
) -> None:
    for field_name in fields_to_extract:
        registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
        field_extractor = FieldExtractor(
            snapshot_tag=snapshot_tag,
            field_args=ResolvedFieldArgs(
                field_name=field_name,
                field_loader=registered_field.loader,
                amr_level=amr_level,
            ),
            comps_to_extract=comps_to_extract,
            overwrite=overwrite,
        )
        for snapshot_dir in snapshot_dirs:
            field_extractor.extract_snapshot(
                snapshot_dir=snapshot_dir,
                data_dir=data_dir,
                index_width=index_width,
            )


def _extract_snapshot_worker(
    *user_args,
) -> None:
    """Positional-only signature required so WorkerArgs elements survive multiprocessing pickling."""
    worker_args = WorkerArgs(*user_args)
    field_extractor = FieldExtractor(
        snapshot_tag=worker_args.snapshot_tag,
        field_args=ResolvedFieldArgs(
            field_name=worker_args.field_name,
            field_loader=worker_args.field_loader,
            amr_level=worker_args.amr_level,
        ),
        comps_to_extract=worker_args.comps_to_extract,
        overwrite=worker_args.overwrite,
    )
    field_extractor.extract_snapshot(
        snapshot_dir=Path(worker_args.snapshot_dir),
        data_dir=Path(worker_args.data_dir),
        index_width=int(worker_args.index_width),
    )


def extract_fields_in_parallel(
    *,
    snapshot_tag: str,
    fields_to_extract: tuple[str, ...],
    comps_to_extract: tuple[cartesian_axes.CartesianAxis_3D, ...],
    snapshot_dirs: list[Path],
    data_dir: Path,
    index_width: int,
    overwrite: bool = False,
    amr_level: int = 0,
    num_workers: int | None = None,
) -> None:
    grouped_args: list[WorkerArgs] = []
    for field_name in fields_to_extract:
        registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
        for snapshot_dir in snapshot_dirs:
            grouped_args.append(
                WorkerArgs(
                    snapshot_dir=str(snapshot_dir),
                    snapshot_tag=snapshot_tag,
                    field_name=field_name,
                    field_loader=registered_field.loader,
                    comps_to_extract=comps_to_extract,
                    data_dir=str(data_dir),
                    index_width=index_width,
                    overwrite=overwrite,
                    amr_level=amr_level,
                ),
            )
    parallel_dispatch.run_in_parallel(
        worker_fn=_extract_snapshot_worker,
        grouped_args=grouped_args,
        num_workers=num_workers,
        timeout_seconds=120,
        show_progress=True,
        enable_plotting=False,
    )


##
## === DATASET PIPELINE
##


@final
class DatasetPipeline:

    def __init__(
        self,
        *,
        snapshot_args: cli.SnapshotArgs,
        field_comp_args: cli.FieldCompArgs,
        data_output_args: cli.DataOutputArgs,
        num_workers: int | None = None,
    ):
        field_registry.validate_fields(
            field_names=field_comp_args.fields,
            allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_extract = validate_types.as_tuple(param=field_comp_args.fields)
        self.comps_to_extract = cli.parse_axes(axes=field_comp_args.comps)
        self.amr_level = field_comp_args.amr_level
        self.data_output_args = data_output_args
        self.num_workers = num_workers

    def _extract_fields(
        self,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.index_width is not None
        if (self.num_workers != 1) and (len(resolved_inputs.snapshot_dirs) > 5):
            extract_fields_in_parallel(
                snapshot_tag=self.snapshot_args.snapshot_tag,
                fields_to_extract=self.fields_to_extract,
                comps_to_extract=self.comps_to_extract,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                index_width=resolved_inputs.index_width,
                overwrite=self.data_output_args.overwrite,
                amr_level=self.amr_level,
                num_workers=self.num_workers,
            )
        else:
            extract_fields_in_serial(
                snapshot_tag=self.snapshot_args.snapshot_tag,
                fields_to_extract=self.fields_to_extract,
                comps_to_extract=self.comps_to_extract,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                index_width=resolved_inputs.index_width,
                overwrite=self.data_output_args.overwrite,
                amr_level=self.amr_level,
            )

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.data_output_args,
        )
        if resolved_inputs is not None:
            self._extract_fields(resolved_inputs)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    parser = argparse.ArgumentParser(
        description="Extract and save full-domain Quokka field data.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_write=True,
                allow_parallel=True,
            ),
        ],
    )
    user_args = parser.parse_args()
    dataset_pipeline = DatasetPipeline(
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args),
        field_comp_args=cli.FieldCompArgs.from_user_args(user_args),
        data_output_args=cli.DataOutputArgs.from_user_args(user_args),
        num_workers=user_args.num_workers,
    )
    dataset_pipeline.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
