## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
import typing

## personal
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_io import manage_log
from jormi.ww_plots import style_figure
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._scripts.snapshot_tools import cli
from ww_quokka_sims.sim_io.field_diagnostics import spectra
from ww_quokka_sims.sim_io.snapshots import field_registry

##
## === DIAGNOSTIC PIPELINE
##


@typing.final
class DiagnosticPipeline:

    def __init__(
        self,
        *,
        snapshot_args: cli.SnapshotArgs,
        field_args: cli.FieldArgs,
        diagnostic_output_args: cli.DiagnosticOutputArgs,
    ):
        field_registry.validate_fields(
            field_names=field_args.fields,
            allowed_types=typing.get_args(field_models.AnyField_3D),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_args.fields)
        self.amr_level = field_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args

    def _pipeline(
        self,
        *,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        assert resolved_inputs.index_width is not None
        for field_name in self.fields_to_plot:
            registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
            generate_spectra = spectra.GenerateSpectra(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                snapshot_tag=self.snapshot_args.snapshot_tag,
                index_width=resolved_inputs.index_width,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                registered_field=registered_field,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                amr_level=self.amr_level,
            )
            generate_spectra.run()

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.diagnostic_output_args,
        )
        if resolved_inputs is not None:
            self._pipeline(resolved_inputs=resolved_inputs)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    parser = argparse.ArgumentParser(
        description="Generate power spectra of Quokka snapshots.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_write=True,
                allow_figures=True,
            ),
        ],
    )
    user_args = parser.parse_args()
    diagnostic_pipeline = DiagnosticPipeline(
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args=user_args),
        field_args=cli.FieldArgs.from_user_args(user_args=user_args),
        diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args=user_args),
    )
    diagnostic_pipeline.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
