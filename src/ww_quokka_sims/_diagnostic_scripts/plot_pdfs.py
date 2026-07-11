## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_io import json_io, manage_log
from jormi.ww_plots import (
    add_color,
    annotate_axis,
    manage_plots,
    style_plots,
)
from jormi.ww_validation import (
    validate_arrays,
    validate_types,
)

## local
from ww_quokka_sims._script_tools import (
    cli,
    field_registry,
)
from ww_quokka_sims.sim_io import (
    find_snapshots,
    load_snapshot,
)

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class PDFData:
    step_time: float
    step_index: int
    grouped_bin_centers: list[numpy.ndarray]
    grouped_densities: list[numpy.ndarray]
    comp_labels: list[str]

    def __post_init__(
        self,
    ) -> None:
        ## container validation
        validate_types.ensure_sequence(
            param=self.grouped_bin_centers,
            valid_seq_types=(list, tuple),
            param_name="grouped_bin_centers",
            seq_length=len(self.comp_labels),
        )
        validate_types.ensure_sequence(
            param=self.grouped_densities,
            valid_seq_types=(list, tuple),
            param_name="grouped_densities",
            seq_length=len(self.comp_labels),
        )
        ## validate each comp-array
        for (bin_centers, densities) in zip(self.grouped_bin_centers, self.grouped_densities):
            validate_arrays.ensure_array(array=bin_centers)
            validate_arrays.ensure_array(array=densities)
            validate_arrays.ensure_1d(array=bin_centers)
            validate_arrays.ensure_1d(array=densities)
            validate_arrays.ensure_same_shape(
                array_a=bin_centers,
                array_b=densities,
            )

    @property
    def num_comps(
        self,
    ) -> int:
        return len(self.comp_labels)

    @property
    def is_scalar(
        self,
    ) -> bool:
        return self.num_comps == 1

    def get_pdf(
        self,
        comp_index: int = 0,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if (comp_index < 0) or (comp_index >= self.num_comps):
            raise IndexError(f"comp_index {comp_index} out of range [0, {self.num_comps - 1}]")
        return self.grouped_bin_centers[comp_index], self.grouped_densities[comp_index]


##
## === FIELD PROCESSING
##


@final
class ComputePDFs:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        snapshot_tag: str,
        field_name: str,
        field_loader: Callable,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        num_bins: int,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.field_name = field_name
        self.field_loader = field_loader
        self.comps_to_plot = comps_to_plot
        self.num_bins = num_bins

    @staticmethod
    def _estimate_pdf(
        *,
        field_data: numpy.ndarray,
        num_bins: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Returns (bin_centers, log10_densities); zero and negative bins are masked."""
        pdf = compute_array_stats.estimate_pdf(
            values=field_data.ravel(),
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
    ) -> PDFData:
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
            )
            grouped_bin_centers.append(bin_centers)
            grouped_densities.append(densities)
        return PDFData(
            step_time=step_time,
            step_index=step_index,
            grouped_bin_centers=grouped_bin_centers,
            grouped_densities=grouped_densities,
            comp_labels=comp_labels,
        )

    def _compute_sfield_pdf(
        self,
        field: field_models.ScalarField_3D,
        step_index: int,
    ) -> PDFData:
        field_models.ensure_3d_sfield(field)
        step_time = field.sim_time
        assert step_time is not None
        bin_centers, densities = self._estimate_pdf(
            field_data=field.fdata.farray,
            num_bins=self.num_bins,
        )
        return PDFData(
            step_time=step_time,
            step_index=step_index,
            grouped_bin_centers=[bin_centers],
            grouped_densities=[densities],
            comp_labels=[field_models.get_label(field)],
        )

    def run(
        self,
    ) -> list[PDFData]:
        field_pdfs: list[PDFData] = []
        for snapshot_dir in self.snapshot_dirs:
            step_index = int(
                find_snapshots.get_step_index_string(
                    snapshot_dir=snapshot_dir,
                    snapshot_tag=self.snapshot_tag,
                ),
            )
            with load_snapshot.QuokkaSnapshot(
                    snapshot_dir=snapshot_dir,
                    verbose=False,
            ) as snapshot:
                field = self.field_loader(snapshot)
            if isinstance(field, field_models.ScalarField_3D):
                pdf = self._compute_sfield_pdf(field=field, step_index=step_index)
            elif isinstance(field, field_models.VectorField_3D):
                pdf = self._compute_vfield_pdf(field=field, step_index=step_index)
            else:
                raise ValueError(f"{self.field_name} is an unrecognised field type.")
            field_pdfs.append(pdf)
        field_pdfs.sort(key=lambda pdf: pdf.step_time)
        return field_pdfs


##
## === FIGURE RENDERING
##


@final
class RenderPDFs:

    def __init__(
        self,
        *,
        snapshot_dirs: list[Path],
        snapshot_tag: str,
        index_width: int,
        extracted_dir: Path,
        figures_dir: Path,
        field_name: str,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        cmap_name: str,
        field_loader: Callable,
        num_bins: int,
        extract_data: bool,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.index_width = index_width
        self.extracted_dir = extracted_dir
        self.figures_dir = figures_dir
        self.field_name = field_name
        self.comps_to_plot = comps_to_plot
        self.cmap_name = cmap_name
        self.field_loader = field_loader
        self.num_bins = int(num_bins)
        self.extract_data = extract_data

    @staticmethod
    def _style_axs(
        *,
        axs_grid: manage_plots.PlotAxesGrid,
        comp_labels: list[str],
    ) -> None:
        for comp_index, label in enumerate(comp_labels):
            ax = axs_grid[0][comp_index]
            ax.set_xlabel(rf"$x \equiv$ {label}")
            if comp_index == 0:
                ax.set_ylabel(r"$\log_{10}\big(p(x)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        axs_grid: manage_plots.PlotAxesGrid,
        pdf_data: PDFData,
        color: annotate_axis.ColorType,
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
        axs_grid: manage_plots.PlotAxesGrid,
        field_pdfs: list[PDFData],
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
            RenderPDFs._plot_snapshot(
                axs_grid=axs_grid,
                pdf_data=pdf_data,
                color=color,
            )
        add_color.add_colorbar(
            ax=axs_grid[-1][-1],
            palette=palette,
            label=r"snapshot index",
        )

    def _save_pdfs(
        self,
        *,
        field_pdfs: list[PDFData],
        extracted_dir: Path,
    ) -> None:
        extracted_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_dict = {}
        for pdf_data in field_pdfs:
            snapshot_dict: dict = {"step_time": pdf_data.step_time}
            for comp_index, comp_label in enumerate(pdf_data.comp_labels):
                bin_centers, densities = pdf_data.get_pdf(comp_index)
                snapshot_dict[comp_label] = {
                    "bin_centers": bin_centers,
                    "log10_density": densities,
                }
            padded_index = f"{pdf_data.step_index:0{self.index_width}d}"
            output_dict[padded_index] = snapshot_dict
        json_io.save_dict_to_json_file(
            file_path=extracted_dir / f"{self.field_name}-pdfs.json",
            input_dict=output_dict,
            overwrite=True,
            verbose=False,
        )

    def run(
        self,
    ) -> None:
        ## compute PDFs for each snapshot and component
        compute_pdfs = ComputePDFs(
            snapshot_dirs=self.snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
            field_name=self.field_name,
            field_loader=self.field_loader,
            comps_to_plot=self.comps_to_plot,
            num_bins=self.num_bins,
        )
        field_pdfs = compute_pdfs.run()
        if not field_pdfs:
            return
        ## optionally write extracted PDF data to JSON
        if self.extract_data:
            self._save_pdfs(
                field_pdfs=field_pdfs,
                extracted_dir=self.extracted_dir,
            )
        ## figure layout: one col per field component; extra right margin for the colorbar if series
        num_cols = field_pdfs[0].num_comps
        add_cbar_space = len(field_pdfs) > 1
        fig, axs_grid = manage_plots.create_figure_grid(
            num_rows=1,
            num_cols=num_cols,
            x_spacing=0.75 if add_cbar_space else 0.25,
            y_spacing=0.25,
        )
        if add_cbar_space:
            fig.subplots_adjust(right=0.82)
        ## plot single snapshot in black, or a sequential color series across all snapshots
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
        )
        suffix = "pdf" if len(field_pdfs) == 1 else "pdfs"
        fig_path = self.figures_dir / f"{self.field_name}-{suffix}.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


##
## === SCRIPT INTERFACE
##


@final
class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        snapshot_tag: str,
        fields_to_plot: tuple[str, ...] | list[str] | None,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...] | list[cartesian_axes.AxisLike_3D] | None,
        extract_data: bool,
        num_bins: int = 15,
        extracted_dir: Path | None = None,
        figures_dir: Path | None = None,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        field_registry.validate_fields(field_names=fields_to_plot)
        if comps_to_plot is None:
            comps_to_plot = cartesian_axes.DEFAULT_3D_AXES_ORDER
        elif not set(comps_to_plot).issubset(set(cartesian_axes.DEFAULT_3D_AXES_ORDER)):
            raise ValueError("Provide one or more components (via -c) from: x_0, x_1, x_2")
        self.input_dir = Path(input_dir)
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = validate_types.as_tuple(param=fields_to_plot)
        self.comps_to_plot = validate_types.as_tuple(param=comps_to_plot)
        self.extract_data = extract_data
        self.num_bins = int(num_bins)
        self.extracted_dir = Path(extracted_dir) if extracted_dir is not None else None
        self.figures_dir = Path(figures_dir) if figures_dir is not None else None

    def run(
        self,
    ) -> None:
        ## find all snapshot dirs under input_dir whose names match snapshot_tag, sorted by index
        snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
            input_dir=self.input_dir,
            snapshot_tag=self.snapshot_tag,
        )
        if not snapshot_dirs:
            return
        extracted_dir = self.extracted_dir if self.extracted_dir is not None else snapshot_dirs[0].parent
        figures_dir = self.figures_dir if self.figures_dir is not None else extracted_dir
        extracted_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        index_width = find_snapshots.get_max_index_width(
            snapshot_dirs=snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
        )
        ## compute and render PDFs for each requested field
        for field_name in self.fields_to_plot:
            field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
            renderer = RenderPDFs(
                snapshot_dirs=snapshot_dirs,
                snapshot_tag=self.snapshot_tag,
                index_width=index_width,
                extracted_dir=extracted_dir,
                figures_dir=figures_dir,
                field_name=field_name,
                comps_to_plot=self.comps_to_plot,
                cmap_name=field_meta.cmap,
                field_loader=field_meta.loader,
                num_bins=self.num_bins,
                extract_data=self.extract_data,
            )
            renderer.run()


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    user_args = argparse.ArgumentParser(
        description="Plot PDFs of Quokka field components.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=True,
                allow_slicing=False,
                produces_data=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        comps_to_plot=user_args.comps,
        extract_data=user_args.save_data,
        num_bins=15,
        extracted_dir=user_args.extracted_dir,
        figures_dir=user_args.figures_dir,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
