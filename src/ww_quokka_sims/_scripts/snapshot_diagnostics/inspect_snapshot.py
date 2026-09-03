## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from pathlib import Path

## personal
from jormi.ww_io import manage_log

## local
from ww_quokka_sims._scripts.snapshot_tools import cli
from ww_quokka_sims.sim_io.snapshots import load_snapshot

##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    user_args = argparse.ArgumentParser(
        description="Inspect a Quokka snapshot and list its available field keys.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_slicing=False,
                allow_fields=False,
            ),
        ],
    ).parse_args()
    snapshot_dir = Path(user_args.input_dir).expanduser().resolve()
    with load_snapshot.QuokkaSnapshot(
            snapshot_dir=snapshot_dir,
            verbose=True,
    ) as snapshot:
        snapshot.list_available_field_keys()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
