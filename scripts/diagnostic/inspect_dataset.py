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
        dataset_dir: Path,
        dataset_tag: str,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.dataset_tag = dataset_tag
        self._validate_inputs()

    def _validate_inputs(
        self,
    ) -> None:
        pass

    def run(
        self,
    ) -> None:
        with load_snapshot.QuokkaSnapshot(
                dataset_dir=self.dataset_dir,
                verbose=True,
        ) as dataset:
            dataset.list_available_field_keys()


##
## === PROGRAM MAIN
##


def main():
    user_args = argparse.ArgumentParser(
        description="Inspect a Quokka dataset and list available field keys.",
        parents=[
            quokka_fields.base_parser(
                num_dirs=1,
                allow_vfields=False,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        dataset_dir=user_args.dir,
        dataset_tag=user_args.tag,
        ## TODO: add field to check diagnostics for
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
