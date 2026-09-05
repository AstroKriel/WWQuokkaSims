## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
import typing

## personal
from jormi.ww_fields.fields_3d import (
    field_models,
    field_operators,
)
from jormi.ww_io import manage_log
from jormi.ww_plots import style_figure
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._scripts.snapshot_tools import cli
from ww_quokka_sims.sim_io.field_diagnostics import time_series
from ww_quokka_sims.sim_io.snapshots import field_registry

##
## === STATISTICS
##

_STATISTIC_LOOKUP: dict[str, time_series.Statistic] = {
    statistic.name: statistic
    for statistic in (
        time_series.Statistic(
            name="total",
            compute_fn=field_operators.compute_sfield_volume_integral,
        ),
        time_series.Statistic(
            name="rms",
            compute_fn=field_operators.compute_sfield_rms,
        ),
    )
}

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
        statistic_name: str,
        num_workers: int | None = None,
        apply_log10_plot: bool = False,
    ):
        if statistic_name not in _STATISTIC_LOOKUP:
            raise ValueError(
                f"unknown statistic `{statistic_name}`; expected one of {sorted(_STATISTIC_LOOKUP)}."
            )
        self.statistic = _STATISTIC_LOOKUP[statistic_name]
        field_registry.validate_fields(
            field_names=field_args.fields,
            allowed_types=(field_models.ScalarField_3D, ),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_args.fields)
        self.amr_level = field_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args
        self.num_workers = num_workers
        self.apply_log10_plot = apply_log10_plot

    def _pipeline(
        self,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        for field_name in self.fields_to_plot:
            registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
            generate_time_series = time_series.GenerateTimeSeries(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                registered_field=registered_field,
                statistic=self.statistic,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                num_workers=self.num_workers,
                overwrite=self.diagnostic_output_args.overwrite,
                amr_level=self.amr_level,
                apply_log10_plot=self.apply_log10_plot,
            )
            generate_time_series.run()

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.diagnostic_output_args,
            allow_index_width=False,
        )
        if resolved_inputs is not None:
            self._pipeline(resolved_inputs)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    parser = argparse.ArgumentParser(
        description="Generate a time-evolving statistic of Quokka snapshots.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
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
        "Apply log10 to the plotted field, abs-valued unless it is strictly positive (does not affect the saved `.json` datasets).",
    )
    parser.add_argument(
        "--statistic",
        type=str,
        choices=sorted(_STATISTIC_LOOKUP.keys()),
        default="total",
        help="Statistic applied to each snapshot's field before building its time series.",
    )
    user_args = parser.parse_args()
    diagnostic_pipeline = DiagnosticPipeline(
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args),
        field_args=cli.FieldArgs.from_user_args(user_args),
        diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args),
        statistic_name=user_args.statistic,
        num_workers=user_args.num_workers,
        apply_log10_plot=user_args.apply_log10_plot,
    )
    diagnostic_pipeline.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
