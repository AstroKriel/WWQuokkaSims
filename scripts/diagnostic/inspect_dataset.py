## { SCRIPT

##
## === DEPENDENCIES
##

import argparse

from pathlib import Path
from typing import final

from ww_quokka_sims.sim_io import load_snapshot
import quokka_fields

##
## === SCRIPT INTERFACE
##


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        snapshot_dir: Path,
        snapshot_tag: str,
    ):
        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve()
        self.snapshot_tag = snapshot_tag
        self._validate_inputs()

    def _validate_inputs(
        self,
    ) -> None:
        pass

    def run(
        self,
    ) -> None:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=self.snapshot_dir,
                verbose=True,
        ) as snapshot:
            snapshot.list_available_field_keys()


##
## === PROGRAM MAIN
##


def main():
    user_args = argparse.ArgumentParser(
        description="Inspect a Quokka snapshot and list available field keys.",
        parents=[
            quokka_fields.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_slicing=False,
                allow_fields=False,
                produces_data=False,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        snapshot_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        ## TODO: add field to check diagnostics for
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
