## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import field_models
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
from ww_quokka_sims.sim_io.field_diagnostics import pdfs
from ww_quokka_sims.sim_io.snapshots import (
    find_snapshots,
    load_snapshot,
)

##
## === FIELD PROCESSING
##


@final
class ComputePDFs:

    def __init__(
        self,
        *,
        field_name: str,
        field_loader: Callable,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        num_bins: int,
        use_log10_bins: bool = False,
        amr_level: int = 0,
    ):
        self.field_name = field_name
        self.field_loader = field_loader
        self.comps_to_plot = comps_to_plot
        self.num_bins = num_bins
        self.use_log10_bins = use_log10_bins
        self.amr_level = amr_level

    @staticmethod
    def _estimate_pdf(
        *,
        field_data: numpy.ndarray,
        num_bins: int,
        use_log10_bins: bool,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Return (bin_centers, log10_densities); zero and negative bins are masked.

        When `use_log10_bins` is set, bins are placed in log10-space of the field itself (not
        just the density axis), since fields spanning orders of magnitude (eg. current density)
        get almost all of their linearly-spaced bins wasted on the rare, large-valued tail,
        leaving the bulk of the distribution unresolved in a single bin.
        """
        values = field_data.ravel()
        if use_log10_bins:
            ## non-positive entries become NaN (no divide-by-zero/invalid-value warning), and are
            ## then dropped by `estimate_pdf`'s own finite-value mask below
            values = compute_array_stats.compute_safe_log10(values)
        pdf = compute_array_stats.estimate_pdf(
            values=values,
            num_bins=num_bins,
        )
        log10_densities = numpy.ma.log10(
            numpy.ma.masked_less_equal(
                x=pdf.densities,
                value=0.0,
            ),
        )
        return (
            pdf.bin_centers,
            log10_densities,
        )

    def _compute_vfield_pdf(
        self,
        field: field_models.VectorField_3D,
        step_index: int,
    ) -> pdfs.PDFData:
        if len(self.comps_to_plot) == 0:
            raise ValueError(
                f"Vector field `{self.field_name}` requires at least one component to plot; none provided.",
            )
        field_models.ensure_3d_vfield(field)
        step_time = field.sim_time
        assert step_time is not None
        comp_names = sorted(self.comps_to_plot)
        comp_labels = [field_models.get_vcomp_label(field, comp_axis=comp_name) for comp_name in comp_names]
        grouped_bin_centers: list[numpy.ndarray] = []
        grouped_densities: list[numpy.ndarray] = []
        for comp_name in comp_names:
            comp_data = field.fdata.farray[cartesian_axes.get_axis_index(comp_name)]
            bin_centers, densities = self._estimate_pdf(
                field_data=comp_data,
                num_bins=self.num_bins,
                use_log10_bins=self.use_log10_bins,
            )
            grouped_bin_centers.append(bin_centers)
            grouped_densities.append(densities)
        return pdfs.PDFData(
            step_time=step_time,
            step_index=step_index,
            grouped_bin_centers=grouped_bin_centers,
            grouped_densities=grouped_densities,
            comp_labels=comp_labels,
            use_log10_bins=self.use_log10_bins,
        )

    def _compute_sfield_pdf(
        self,
        field: field_models.ScalarField_3D,
        step_index: int,
    ) -> pdfs.PDFData:
        field_models.ensure_3d_sfield(field)
        step_time = field.sim_time
        assert step_time is not None
        bin_centers, densities = self._estimate_pdf(
            field_data=field.fdata.farray,
            num_bins=self.num_bins,
            use_log10_bins=self.use_log10_bins,
        )
        return pdfs.PDFData(
            step_time=step_time,
            step_index=step_index,
            grouped_bin_centers=[bin_centers],
            grouped_densities=[densities],
            comp_labels=[field_models.get_label(field)],
            use_log10_bins=self.use_log10_bins,
        )

    def compute_snapshot(
        self,
        *,
        snapshot_dir: Path,
        snapshot_tag: str,
    ) -> pdfs.PDFData:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=snapshot_tag,
            ),
        )
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as snapshot:
            field = self.field_loader(snapshot, amr_level=self.amr_level)
        if isinstance(field, field_models.ScalarField_3D):
            return self._compute_sfield_pdf(
                field=field,
                step_index=step_index,
            )
        if isinstance(field, field_models.VectorField_3D):
            return self._compute_vfield_pdf(
                field=field,
                step_index=step_index,
            )
        raise ValueError(f"{self.field_name} is an unrecognised field type.")


##
## === FIGURE RENDERING
##


@final
class GeneratePDFs:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        snapshot_tag: str,
        index_width: int,
        data_dir: Path,
        figures_dir: Path,
        field_name: str,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        cmap_name: str,
        field_loader: Callable,
        num_bins: int,
        save_data: bool,
        save_figure: bool,
        overwrite: bool = False,
        use_log10_bins: bool = False,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.index_width = index_width
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.comps_to_plot = comps_to_plot
        self.cmap_name = cmap_name
        self.field_loader = field_loader
        self.num_bins = int(num_bins)
        self.save_data = save_data
        self.save_figure = save_figure
        self.overwrite = overwrite
        self.use_log10_bins = use_log10_bins
        self.amr_level = amr_level

    def _data_name(
        self,
    ) -> str:
        """Filename stem, tagged with `log10_` when bins are log10-spaced.

        The filename is a hint for humans browsing the directory, not the source of truth (it can
        be renamed); the saved `use_log10_bins` flag and `log10_bin_centers` key inside the file
        itself are what downstream code should actually check.
        """
        return f"log10_{self.field_name}" if self.use_log10_bins else self.field_name

    def _data_file_path(
        self,
        *,
        data_dir: Path,
        padded_index: str,
    ) -> Path:
        return data_dir / f"{self._data_name()}-pdf-index={padded_index}.json"

    def _snapshot_figure_file_path(
        self,
        *,
        figures_dir: Path,
        padded_index: str,
    ) -> Path:
        return figures_dir / f"{self._data_name()}-pdf-index={padded_index}.png"

    @staticmethod
    def _style_axs(
        *,
        axs_grid: manage_figure.PanelGrid,
        comp_labels: list[str],
        use_log10_bins: bool,
    ) -> None:
        for comp_index, label in enumerate(comp_labels):
            ax = axs_grid[0][comp_index]
            x_label = rf"$\log_{{10}}($ {label} $)$" if use_log10_bins else rf"$x \equiv$ {label}"
            ax.set_xlabel(x_label)
            if comp_index == 0:
                ax.set_ylabel(r"$\log_{10}\big(p(x)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        axs_grid: manage_figure.PanelGrid,
        pdf_data: pdfs.PDFData,
        color: annotate_panel.ColorType,
    ) -> None:
        for comp_index in range(pdf_data.num_comps):
            ax = axs_grid[0][comp_index]
            x_values, y_values = pdf_data.get_pdf(comp_index)
            ax.step(
                x_values,
                y_values,
                where="mid",
                lw=2.0,
                color=color,
                zorder=comp_index + 1,
            )

    @staticmethod
    def _plot_series(
        *,
        axs_grid: manage_figure.PanelGrid,
        field_pdfs: list[pdfs.PDFData],
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
                    len(field_pdfs) - 1,
                ),
            ),
        )
        for series_index, pdf_data in enumerate(field_pdfs):
            color = palette.mpl_cmap(
                palette.mpl_norm(
                    series_index,
                ),
            )
            GeneratePDFs._plot_snapshot(
                axs_grid=axs_grid,
                pdf_data=pdf_data,
                color=color,
            )
        add_color.add_colorbar(
            panels=axs_grid[-1][-1],
            palette=palette,
            label=r"snapshot index",
        )

    def _save_pdf(
        self,
        *,
        pdf_data: pdfs.PDFData,
        data_dir: Path,
    ) -> None:
        """Save one snapshot's PDF to its own file, mirroring `generate_slices.py`'s one-file-per-
        snapshot convention (rather than one file aggregating every snapshot) -- each file is
        self-contained (carries its own `step_time`/`use_log10_bins`), so results already on disk
        are immediately usable even if a later snapshot in the run fails or the job is cut off.
        """
        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        padded_index = f"{pdf_data.step_index:0{self.index_width}d}"
        pdf_data.save_to_file(self._data_file_path(data_dir=data_dir, padded_index=padded_index))

    def _save_snapshot_figure(
        self,
        *,
        pdf_data: pdfs.PDFData,
        figure_path: Path,
    ) -> None:
        fig, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=pdf_data.num_comps,
        )
        self._plot_snapshot(
            axs_grid=axs_grid,
            pdf_data=pdf_data,
            color="black",
        )
        self._style_axs(
            axs_grid=axs_grid,
            comp_labels=pdf_data.comp_labels,
            use_log10_bins=self.use_log10_bins,
        )
        manage_figure.save_figure(
            figure=fig,
            figure_path=figure_path,
            verbose=False,
        )

    def _process_snapshot(
        self,
        *,
        compute_pdfs: ComputePDFs,
        snapshot_dir: Path,
        data_dir: Path,
        figures_dir: Path,
        index_width: int,
    ) -> None:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            ),
        )
        padded_index = f"{step_index:0{index_width}d}"
        data_path = self._data_file_path(data_dir=data_dir, padded_index=padded_index)
        figure_path = self._snapshot_figure_file_path(figures_dir=figures_dir, padded_index=padded_index)
        data_exists = data_path.exists()
        data_needed = self.save_data and (self.overwrite or not data_exists)
        figure_needed = self.save_figure and (self.overwrite or not figure_path.exists())

        if not data_needed and not figure_needed:
            return

        if figure_needed and not data_needed and data_exists:
            ## cheap path: reconstruct the figure from already-saved data, skip the raw snapshot
            manage_log.log_hint(
                text=(
                    f"`{self.field_name}` at snapshot {step_index}: "
                    f"building figure from saved data, skipping the raw snapshot."
                ),
            )
            pdf_data = pdfs.PDFData.load_from_file(data_path)
            self._save_snapshot_figure(pdf_data=pdf_data, figure_path=figure_path)
            return

        pdf_data = compute_pdfs.compute_snapshot(snapshot_dir=snapshot_dir, snapshot_tag=self.snapshot_tag)
        if data_needed:
            self._save_pdf(pdf_data=pdf_data, data_dir=data_dir)
        if figure_needed:
            self._save_snapshot_figure(pdf_data=pdf_data, figure_path=figure_path)

    def _load_all_saved_pdfs(
        self,
        *,
        data_dir: Path,
    ) -> list[pdfs.PDFData]:
        paths = sorted(data_dir.glob(f"{self._data_name()}-pdf-index=*.json"))
        field_pdfs = [pdfs.PDFData.load_from_file(path) for path in paths]
        field_pdfs.sort(key=lambda pdf_data: pdf_data.step_time)
        return field_pdfs

    def _save_summary_figure(
        self,
        *,
        field_pdfs: list[pdfs.PDFData],
        figures_dir: Path,
    ) -> None:
        """Combined overlay across every saved snapshot; always rebuilt fresh from whatever is on
        disk (not from anything held in memory across the potentially-long per-snapshot loop above).
        """
        num_cols = field_pdfs[0].num_comps
        fig, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=num_cols,
        )
        if len(field_pdfs) == 1:
            self._plot_snapshot(
                axs_grid=axs_grid,
                pdf_data=field_pdfs[0],
                color="black",
            )
        else:
            self._plot_series(
                axs_grid=axs_grid,
                field_pdfs=field_pdfs,
                cmap_name=self.cmap_name,
            )
        self._style_axs(
            axs_grid=axs_grid,
            comp_labels=field_pdfs[0].comp_labels,
            use_log10_bins=self.use_log10_bins,
        )
        fig_path = figures_dir / f"{self._data_name()}-pdfs-summary.png"
        manage_figure.save_figure(
            figure=fig,
            figure_path=fig_path,
            verbose=True,
        )

    def run(
        self,
    ) -> None:
        if self.save_data or self.save_figure:
            compute_pdfs = ComputePDFs(
                field_name=self.field_name,
                field_loader=self.field_loader,
                comps_to_plot=self.comps_to_plot,
                num_bins=self.num_bins,
                use_log10_bins=self.use_log10_bins,
                amr_level=self.amr_level,
            )
            for snapshot_dir in self.snapshot_dirs:
                self._process_snapshot(
                    compute_pdfs=compute_pdfs,
                    snapshot_dir=snapshot_dir,
                    data_dir=self.data_dir,
                    figures_dir=self.figures_dir,
                    index_width=self.index_width,
                )
        if not self.save_figure:
            return
        ## the summary is only buildable from saved data; if none was ever saved for this field
        ## (eg. --save-figure was used without --save-data, ever), there's nothing to aggregate
        field_pdfs = self._load_all_saved_pdfs(data_dir=self.data_dir)
        if not field_pdfs:
            manage_log.log_hint(
                text=f"Skipping summary figure for `{self.field_name}`: no saved data found in {self.data_dir}.",
            )
            return
        self._save_summary_figure(field_pdfs=field_pdfs, figures_dir=self.figures_dir)


##
## === DIAGNOSTIC PIPELINE
##


@final
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

    def _generate_fields(
        self,
        resolved_inputs: cli.ResolvedInputs,
    ) -> None:
        assert resolved_inputs.figures_dir is not None
        assert resolved_inputs.index_width is not None
        for field_name in self.fields_to_plot:
            registered_field = field_registry.REGISTERED_FIELD_LOOKUP[field_name]
            generator = GeneratePDFs(
                snapshot_dirs=resolved_inputs.snapshot_dirs,
                snapshot_tag=self.snapshot_args.snapshot_tag,
                index_width=resolved_inputs.index_width,
                data_dir=resolved_inputs.data_dir,
                figures_dir=resolved_inputs.figures_dir,
                field_name=field_name,
                comps_to_plot=self.comps_to_plot,
                cmap_name=registered_field.cmap,
                field_loader=registered_field.loader,
                num_bins=self.num_bins,
                save_data=self.diagnostic_output_args.save_data,
                save_figure=self.diagnostic_output_args.save_figure,
                overwrite=self.diagnostic_output_args.overwrite,
                use_log10_bins=self.use_log10_bins,
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
        snapshot_args=cli.SnapshotArgs.from_user_args(user_args),
        field_comp_args=cli.FieldCompArgs.from_user_args(user_args),
        diagnostic_output_args=cli.DiagnosticOutputArgs.from_user_args(user_args),
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
