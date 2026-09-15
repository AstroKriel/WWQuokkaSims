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
from ww_quokka_sims.sim_io.field_diagnostics import pdfs
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
        field_comp_args: cli.FieldCompArgs,
        diagnostic_output_args: cli.DiagnosticOutputArgs,
        num_bins: int = 20,
        use_log10_bins: bool = False,
    ):
        field_registry.validate_fields(
            field_names=field_comp_args.fields,
            allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_comp_args.fields)
        self.comps_to_plot = cli.parse_axes(axes=field_comp_args.comps)
        self.amr_level = field_comp_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args
        self.num_bins = int(num_bins)
        self.use_log10_bins = use_log10_bins

    def _pipeline(
        self,
        *,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        assert resolved_inputs.index_width is not None
        for field_name in self.fields_to_plot:
            registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
            generate_pdfs = pdfs.GeneratePDFs(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                snapshot_tag=self.snapshot_args.snapshot_tag,
                index_width=resolved_inputs.index_width,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                registered_field=registered_field,
                comps_to_plot=self.comps_to_plot,
                num_bins=self.num_bins,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                use_log10_bins=self.use_log10_bins,
                amr_level=self.amr_level,
            )
            generate_pdfs.run()

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
        description="Generate PDFs of Quokka snapshots.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_slicing=False,
                allow_write=True,
                allow_figures=True,
            ),
        ],
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Number of discrete histogram bins for the PDF estimate (default: 20).",
    )
    parser.add_argument(
        "--use-log10-bins",
        action="store_true",
        default=False,
        help="Bin the log10(|field|) values rather than the raw-field values (default: False).",
    )
    user_args = parser.parse_args()
    diagnostic_pipeline = DiagnosticPipeline(
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args=user_args),
        field_comp_args=cli.FieldCompArgs.from_user_args(user_args=user_args),
        diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args=user_args),
        num_bins=user_args.num_bins,
        use_log10_bins=user_args.use_log10_bins,
    )
    diagnostic_pipeline.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
