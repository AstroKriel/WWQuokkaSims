## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from pathlib import Path
from typing import final, get_args

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import compute_spectra, field_models
from jormi.ww_io import manage_log
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_validation import validate_types

## local
from ww_quokka_sims._scripts.snapshot_tools import (
    cli,
    field_registry,
)
from ww_quokka_sims.sim_io.field_diagnostics import spectra
from ww_quokka_sims.sim_io.snapshots import (
    find_snapshots,
    load_snapshot,
)

##
## === FIELD PROCESSING
##


@final
class ComputeSpectra:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        snapshot_tag: str,
        field_name: str,
        field_loader: Callable,
        index_width: int,
        save_data: bool,
        data_dir: Path,
        overwrite: bool = False,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.field_name = field_name
        self.field_loader = field_loader
        self.index_width = index_width
        self.save_data = save_data
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.amr_level = amr_level

    def _data_file_path(
        self,
        *,
        padded_index: str,
    ) -> Path:
        return self.data_dir / f"{self.field_name}-spectrum-index={padded_index}.json"

    def run(
        self,
    ) -> list[spectra.SpectraData]:
        field_spectra: list[spectra.SpectraData] = []
        for snapshot_dir in self.snapshot_dirs:
            step_index = int(
                find_snapshots.get_step_index_string(
                    snapshot_dir=snapshot_dir,
                    snapshot_tag=self.snapshot_tag,
                ),
            )
            padded_index = f"{step_index:0{self.index_width}d}"
            data_path = self._data_file_path(padded_index=padded_index)

            ## skip snapshots already computed in a prior (e.g. killed/interrupted) run, whether
            ## or not save_data is set this run, so a --save-figure-only run still gets the cheap
            ## reuse; each snapshot's file is independent, so a crash never risks earlier ones
            if (not self.overwrite) and data_path.exists():
                field_spectra.append(spectra.SpectraData.load_from_file(data_path))
                continue

            with load_snapshot.QuokkaSnapshot(
                    snapshot_dir=snapshot_dir,
                    verbose=False,
            ) as snapshot:
                field = self.field_loader(snapshot, amr_level=self.amr_level)
            spectrum = compute_spectra.compute_isotropic_power_spectrum_field(field)
            step_time = field.sim_time
            assert step_time is not None
            log10_k_bin_centers = numpy.ma.log10(
                numpy.ma.masked_less_equal(
                    x=spectrum.k_bin_centers_1d,
                    value=0.0,
                ),
            )
            log10_spectrum = numpy.ma.log10(
                numpy.ma.masked_less_equal(
                    x=spectrum.power_spectrum_1d,
                    value=0.0,
                ),
            )
            spectra_data = spectra.SpectraData(
                step_time=step_time,
                step_index=step_index,
                latex_label=field.latex_label,
                log10_k_bin_centers=log10_k_bin_centers,
                log10_spectrum=log10_spectrum,
            )
            field_spectra.append(spectra_data)
            ## save immediately, one file per snapshot, so a killed/interrupted run still
            ## leaves every already-completed snapshot independently usable and resumable
            if self.save_data:
                self.data_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                spectra_data.save_to_file(data_path)

        field_spectra.sort(key=lambda s: s.step_time)
        return field_spectra


##
## === FIGURE RENDERING
##


@final
class GenerateSpectra:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        snapshot_tag: str,
        index_width: int,
        data_dir: Path,
        figures_dir: Path,
        field_name: str,
        field_loader: Callable,
        cmap_name: str,
        save_data: bool,
        save_figure: bool,
        overwrite: bool = False,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.index_width = index_width
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.field_loader = field_loader
        self.cmap_name = cmap_name
        self.save_data = save_data
        self.save_figure = save_figure
        self.overwrite = overwrite
        self.amr_level = amr_level

    @staticmethod
    def _style_ax(
        *,
        ax: manage_figure.Panel,
        latex_label: str,
    ) -> None:
        ax.set_xlabel(r"$\log_{10}(k)$")
        ax.set_ylabel(rf"$\log_{{10}}\big(\mathcal{{P}}_{{{latex_label}}}(k)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        ax: manage_figure.Panel,
        spectra_data: spectra.SpectraData,
        color: annotate_panel.ColorType,
    ) -> None:
        ax.plot(
            spectra_data.log10_k_bin_centers,
            spectra_data.log10_spectrum,
            lw=2.0,
            color=color,
        )

    @staticmethod
    def _plot_series(
        *,
        ax: manage_figure.Panel,
        field_spectra: list[spectra.SpectraData],
        cmap_name: str,
    ) -> None:
        palette = add_color.make_palette(
            config=add_color.SequentialConfig(
                palette_name=cmap_name,
                palette_range=(0.25, 1.0),
            ),
            value_range=(
                0,
                max(
                    0,
                    len(field_spectra) - 1,
                ),
            ),
        )
        for series_index, spectra_data in enumerate(field_spectra):
            color = palette.mpl_cmap(
                palette.mpl_norm(
                    series_index,
                ),
            )
            GenerateSpectra._plot_snapshot(
                ax=ax,
                spectra_data=spectra_data,
                color=color,
            )
        add_color.add_colorbar(
            panels=ax,
            palette=palette,
            label=r"snapshot index",
        )

    def _snapshot_figure_file_path(
        self,
        *,
        figures_dir: Path,
        padded_index: str,
    ) -> Path:
        return figures_dir / f"{self.field_name}-spectrum-index={padded_index}.png"

    def _save_snapshot_figure(
        self,
        *,
        spectra_data: spectra.SpectraData,
        figure_path: Path,
    ) -> None:
        fig, ax = manage_figure.create_figure()
        self._plot_snapshot(
            ax=ax,
            spectra_data=spectra_data,
            color="black",
        )
        self._style_ax(
            ax=ax,
            latex_label=spectra_data.latex_label,
        )
        manage_figure.save_figure(
            figure=fig,
            figure_path=figure_path,
            verbose=False,
        )

    def _save_summary_figure(
        self,
        *,
        field_spectra: list[spectra.SpectraData],
        figures_dir: Path,
    ) -> None:
        """Combined overlay across every snapshot processed this run; always rebuilt fresh."""
        fig, ax = manage_figure.create_figure()
        if len(field_spectra) == 1:
            self._plot_snapshot(
                ax=ax,
                spectra_data=field_spectra[0],
                color="black",
            )
        else:
            self._plot_series(
                ax=ax,
                field_spectra=field_spectra,
                cmap_name=self.cmap_name,
            )
        self._style_ax(
            ax=ax,
            latex_label=field_spectra[0].latex_label,
        )
        fig_path = figures_dir / f"{self.field_name}-spectra-summary.png"
        manage_figure.save_figure(
            figure=fig,
            figure_path=fig_path,
            verbose=True,
        )

    def run(
        self,
    ) -> None:
        ## compute the isotropic power spectrum for each snapshot; saved incrementally as each completes
        compute = ComputeSpectra(
            snapshot_dirs=self.snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
            field_name=self.field_name,
            field_loader=self.field_loader,
            index_width=self.index_width,
            save_data=self.save_data,
            data_dir=self.data_dir,
            overwrite=self.overwrite,
            amr_level=self.amr_level,
        )
        field_spectra = compute.run()
        if not field_spectra:
            return
        if not self.save_figure:
            return
        ## one figure per snapshot, resumed like everything else; the combined summary always
        ## rebuilds since it's cheap relative to the per-snapshot compute above
        for spectra_data in field_spectra:
            padded_index = f"{spectra_data.step_index:0{self.index_width}d}"
            figure_path = self._snapshot_figure_file_path(figures_dir=self.figures_dir, padded_index=padded_index)
            if self.overwrite or not figure_path.exists():
                self._save_snapshot_figure(spectra_data=spectra_data, figure_path=figure_path)
        self._save_summary_figure(field_spectra=field_spectra, figures_dir=self.figures_dir)


##
## === DIAGNOSTIC PIPELINE
##


@final
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
            allowed_types=get_args(field_models.AnyField_3D),
        )
        self.snapshot_args = snapshot_args
        self.fields_to_plot = validate_types.as_tuple(param=field_args.fields)
        self.amr_level = field_args.amr_level
        self.diagnostic_output_args = diagnostic_output_args

    def _generate_fields(
        self,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        assert resolved_inputs.index_width is not None
        for field_name in self.fields_to_plot:
            registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
            generator = GenerateSpectra(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                snapshot_tag=self.snapshot_args.snapshot_tag,
                index_width=resolved_inputs.index_width,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                field_name=field_name,
                field_loader=registered_field.loader,
                cmap_name=registered_field.cmap,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                amr_level=self.amr_level,
            )
            generator.run()

    def run(
        self,
    ) -> None:
        resolved_inputs = cli.resolve_inputs(
            snapshot_args=self.snapshot_args,
            output_args=self.diagnostic_output_args,
        )
        if resolved_inputs is not None:
            self._generate_fields(resolved_inputs)


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    user_args = argparse.ArgumentParser(
        description="Generate power spectra of Quokka snapshots.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_write=True,
                allow_figures=True,
            ),
        ],
    ).parse_args()
    diagnostic_pipeline = DiagnosticPipeline(
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args),
        field_args=cli.FieldArgs.from_user_args(user_args),
        diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args),
    )
    diagnostic_pipeline.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
