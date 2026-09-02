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
from ww_quokka_sims._script_tools import (
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
class FieldArgs:
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


def _parse_axes(
    *,
    axes: tuple[str, ...] | list[str] | None,
) -> tuple[cartesian_axes.CartesianAxis_3D, ...]:
    if axes is None:
        return tuple(cartesian_axes.DEFAULT_3D_AXES_ORDER)
    parsed_axes: list[cartesian_axes.CartesianAxis_3D] = []
    for axis_name in validate_types.as_tuple(param=axes):
        try:
            parsed_axes.append(
                cartesian_axes.as_axis(
                    axis=axis_name,
                ),
            )
        except (TypeError, ValueError):
            raise ValueError("Provide one or more components (via -c) from: x_0, x_1, x_2")
    return tuple(parsed_axes)


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
    field_args: FieldArgs
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
        field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
        field_extractor = FieldExtractor(
            snapshot_tag=snapshot_tag,
            field_args=FieldArgs(
                field_name=field_name,
                field_loader=field_meta.loader,
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
        field_args=FieldArgs(
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
        field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
        for snapshot_dir in snapshot_dirs:
            grouped_args.append(
                WorkerArgs(
                    snapshot_dir=str(snapshot_dir),
                    snapshot_tag=snapshot_tag,
                    field_name=field_name,
                    field_loader=field_meta.loader,
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
## === SCRIPT INTERFACE
##


@dataclass(frozen=True)
class ResolvedInputs:
    snapshot_dirs: list[Path]
    data_dir: Path
    index_width: int


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        snapshot_tag: str,
        fields_to_extract: tuple[str, ...] | list[str] | None,
        comps_to_extract: tuple[str, ...] | list[str] | None,
        data_dir: Path | None = None,
        num_workers: int | None = None,
        overwrite: bool = False,
        amr_level: int = 0,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        field_registry.validate_fields(
            field_names=fields_to_extract,
            allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
        )
        self.input_dir = Path(input_dir)
        self.snapshot_tag = snapshot_tag
        self.fields_to_extract = validate_types.as_tuple(param=fields_to_extract)
        self.comps_to_extract = _parse_axes(axes=comps_to_extract)
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.num_workers = num_workers
        self.overwrite = bool(overwrite)
        self.amr_level = amr_level

    def _resolve_inputs(
        self,
    ) -> ResolvedInputs | None:
        snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
            input_dir=self.input_dir,
            snapshot_tag=self.snapshot_tag,
        )
        if not snapshot_dirs:
            return None
        data_dir = cli.resolve_output_dir(
            output_dir=self.data_dir,
            default_dir=snapshot_dirs[0].parent,
        )
        index_width = find_snapshots.get_max_index_width(
            snapshot_dirs=snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
        )
        return ResolvedInputs(
            snapshot_dirs=snapshot_dirs,
            data_dir=data_dir,
            index_width=index_width,
        )

    def _extract_fields(
        self,
        resolved_inputs: ResolvedInputs,
    ) -> None:
        if (self.num_workers != 1) and (len(resolved_inputs.snapshot_dirs) > 5):
            extract_fields_in_parallel(
                snapshot_tag=self.snapshot_tag,
                fields_to_extract=self.fields_to_extract,
                comps_to_extract=self.comps_to_extract,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                index_width=resolved_inputs.index_width,
                overwrite=self.overwrite,
                amr_level=self.amr_level,
                num_workers=self.num_workers,
            )
        else:
            extract_fields_in_serial(
                snapshot_tag=self.snapshot_tag,
                fields_to_extract=self.fields_to_extract,
                comps_to_extract=self.comps_to_extract,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                index_width=resolved_inputs.index_width,
                overwrite=self.overwrite,
                amr_level=self.amr_level,
            )

    def run(
        self,
    ) -> None:
        resolved_inputs = self._resolve_inputs()
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
                allow_parallel=True,
            ),
        ],
    )
    parser.add_argument(
        "--data-dir",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Output directory for extracted data; defaults to the parent directory of the snapshot.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "Redo snapshots whose output already exists, instead of skipping them; "
            "default: False (resume where a prior run left off)."
        ),
    )
    user_args = parser.parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_extract=user_args.fields,
        comps_to_extract=user_args.comps,
        data_dir=user_args.data_dir,
        num_workers=user_args.num_workers,
        overwrite=user_args.overwrite,
        amr_level=user_args.amr_level,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
