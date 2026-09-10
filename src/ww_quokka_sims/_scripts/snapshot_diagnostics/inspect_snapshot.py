## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

## personal
from jormi.ww_io import manage_log

## local
from ww_quokka_sims._scripts.snapshot_tools import cli
from ww_quokka_sims.sim_io.snapshots import find_snapshots, load_snapshot

##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    parser = argparse.ArgumentParser(
        description="Inspect a Quokka snapshot and list its available field keys.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_slicing=False,
                allow_fields=False,
            ),
        ],
    )
    user_args = parser.parse_args()
    if user_args.input_dir is None:
        raise ValueError("`--input-dir` is required.")
    snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
    )
    if not snapshot_dirs:
        raise ValueError(
            f"no snapshot directories found under `{user_args.input_dir}` matching tag `{user_args.tag}`.",
        )
    snapshot_dir = snapshot_dirs[-1]
    with load_snapshot.QuokkaSnapshot(
            snapshot_dir=snapshot_dir,
            verbose=True,
    ) as quokka_snapshot:
        quokka_snapshot.list_available_field_keys()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
