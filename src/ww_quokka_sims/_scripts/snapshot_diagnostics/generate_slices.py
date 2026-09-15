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
from ww_quokka_sims.sim_io.field_diagnostics import slices
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
        field_comp_axes_args: cli.FieldCompAxesArgs,
        diagnostic_output_args: cli.DiagnosticOutputArgs,
        num_workers: int | None = None,
        hide_annotations: bool = False,
        apply_log10_plot: bool = False,
    ):
        field_registry.validate_fields(
            field_names=field_comp_axes_args.fields,
            allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_comp_axes_args.fields)
        self.comps_to_plot = cli.parse_axes(axes=field_comp_axes_args.comps)
        self.axes_to_slice = cli.parse_axes(axes=field_comp_axes_args.axes)
        self.amr_level = field_comp_axes_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args
        self.num_workers = num_workers
        self.hide_annotations = hide_annotations
        self.apply_log10_plot = apply_log10_plot

    def _pipeline(
        self,
        *,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        assert resolved_inputs.index_width is not None
        if (self.num_workers != 1) and (len(resolved_inputs.snapshot_dirs) > 5):
            slices.generate_field_slices_in_parallel(
                snapshot_tag=self.snapshot_args.snapshot_tag,
                fields_to_plot=self.fields_to_plot,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                index_width=resolved_inputs.index_width,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                hide_annotations=self.hide_annotations,
                apply_log10_plot=self.apply_log10_plot,
                amr_level=self.amr_level,
                num_workers=self.num_workers,
            )
        else:
            slices.generate_field_slices_in_serial(
                snapshot_tag=self.snapshot_args.snapshot_tag,
                fields_to_plot=self.fields_to_plot,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                index_width=resolved_inputs.index_width,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                hide_annotations=self.hide_annotations,
                apply_log10_plot=self.apply_log10_plot,
                amr_level=self.amr_level,
            )

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.diagnostic_output_args,
            max_elems=100,
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
        description="Generate midplane slices of Quokka snapshots.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_slicing=True,
                allow_write=True,
                allow_figures=True,
                allow_parallel=True,
            ),
        ],
    )
    parser.add_argument(
        "--apply-log10-plot",
        action="store_true",
        default=False,
        help=
        "Apply log10 to the plotted field, abs-valued unless it is strictly positive (does not affect the saved `.npz` data slices).",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        default=False,
        help="Hide metadata annotations: min/max values, sim time, and field label (default: False).",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        default=False,
        help="Animate figures that exist under --figures-dir into an MP4 (default: False).",
    )
    user_args = parser.parse_args()
    if not (user_args.save_data or user_args.save_figure or user_args.animate):
        raise ValueError("must pass `--save-figure`, `--save-data`, and/or `--animate`; none was given.")
    field_registry.validate_fields(
        field_names=user_args.fields,
        allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
    )
    if user_args.save_data or user_args.save_figure:
        if user_args.input_dir is None:
            raise ValueError("`--input-dir` is required with `--save-data`/`--save-figure`.")
        diagnostic_pipeline = DiagnosticPipeline(
            snapshot_args=cli.SnapshotArgs.from_user_args(user_args=user_args),
            field_comp_axes_args=cli.FieldCompAxesArgs.from_user_args(user_args=user_args),
            diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args=user_args),
            num_workers=user_args.num_workers,
            hide_annotations=user_args.no_annotations,
            apply_log10_plot=user_args.apply_log10_plot,
        )
        diagnostic_pipeline.run()
    if user_args.animate:
        figures_dir = slices.resolve_figures_dir_to_animate(
            figures_dir=user_args.figures_dir,
            data_dir=user_args.data_dir,
            input_dir=user_args.input_dir,
        )
        slices.animate_saved_figures(
            figures_dir=figures_dir,
            fields_to_plot=validate_types.as_tuple(param=user_args.fields),
            apply_log10_plot=user_args.apply_log10_plot,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
