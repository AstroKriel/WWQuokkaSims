## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
import pathlib

## personal
from jormi.ww_io import manage_log

## local
from ww_quokka_sims.sim_io.snapshots import find_snapshots, load_snapshot

##
## === GENERATE
##


def generate_amr_box_layouts(
    *,
    snapshot_dirs: list[pathlib.Path],
    snapshot_tag: str,
    index_width: int,
    data_dir: pathlib.Path,
    overwrite: bool,
) -> None:
    """Save one AMR box layout per snapshot."""
    for snapshot_dir in snapshot_dirs:
        step_index = find_snapshots.get_step_index(
            snapshot_dir=snapshot_dir,
            snapshot_tag=snapshot_tag,
        )
        padded_index = step_index.get_padded_string(index_width=index_width)
        layout_file_path = data_dir / f"amr_boxes-index={padded_index}.json"
        if layout_file_path.exists() and not overwrite:
            manage_log.log_note(text=f"skipping (already exists): {layout_file_path}")
            continue
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as snapshot:
            box_layout = snapshot.load_amr_box_layout(step_index=step_index)
        box_layout.save_to_file(file_path=layout_file_path)
        manage_log.log_note(text=f"saved: {layout_file_path} ({len(box_layout.boxes)} boxes)")


##
## === PROGRAM MAIN
##


def main() -> None:
    parser = argparse.ArgumentParser(description="Save where every AMR box sits, one JSON per snapshot.")
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--snapshot-tag",
        type=str,
        default="plt",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
        input_dir=args.input_dir,
        snapshot_tag=args.snapshot_tag,
    )
    if not snapshot_dirs:
        raise ValueError(f"no snapshots found under: {args.input_dir}")
    index_width = find_snapshots.get_max_index_width(
        snapshot_dirs=snapshot_dirs,
        snapshot_tag=args.snapshot_tag,
    )
    args.data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    generate_amr_box_layouts(
        snapshot_dirs=snapshot_dirs,
        snapshot_tag=args.snapshot_tag,
        index_width=index_width,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
