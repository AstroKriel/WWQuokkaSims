## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
import dataclasses

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import (
    compute_spectra,
    field_models,
)
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
class SpectraData:
    step_time: float
    step_index: int
    latex_label: str
    log10_k_bin_centers: numpy.ndarray
    log10_spectrum: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        validate_arrays.ensure_array(array=self.log10_k_bin_centers)
        validate_arrays.ensure_array(array=self.log10_spectrum)
        validate_arrays.ensure_1d(array=self.log10_k_bin_centers)
        validate_arrays.ensure_1d(array=self.log10_spectrum)
        validate_arrays.ensure_same_shape(
            array_a=self.log10_k_bin_centers,
            array_b=self.log10_spectrum,
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
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.field_name = field_name
        self.field_loader = field_loader
        self.index_width = index_width
        self.save_data = save_data
        self.data_dir = data_dir

    def _save_incremental(
        self,
        *,
        output_dict: dict,
    ) -> None:
        self.data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_io.save_dict_to_json_file(
            file_path=self.data_dir / f"{self.field_name}-spectra.json",
            input_dict=output_dict,
            overwrite=True,
            verbose=False,
        )

    def run(
        self,
    ) -> list[SpectraData]:
        field_spectra: list[SpectraData] = []
        output_dict: dict = {}
        cached_entries: dict = {}
        if self.save_data:
            existing_path = self.data_dir / f"{self.field_name}-spectra.json"
            if existing_path.exists():
                cached_entries = json_io.read_json_file_into_dict(
                    file_path=existing_path,
                    verbose=False,
                )
                output_dict = dict(cached_entries)

        ## latex_label isn't stored per-snapshot in the cache, so track one from
        ## whichever snapshot actually gets freshly computed this run
        reference_latex_label: str | None = None

        for snapshot_dir in self.snapshot_dirs:
            step_index = int(
                find_snapshots.get_step_index_string(
                    snapshot_dir=snapshot_dir,
                    snapshot_tag=self.snapshot_tag,
                ),
            )
            padded_index = f"{step_index:0{self.index_width}d}"

            ## skip snapshots already computed in a prior (e.g. killed/interrupted) run
            if self.save_data and (padded_index in cached_entries):
                cached = cached_entries[padded_index]
                field_spectra.append(
                    SpectraData(
                        step_time=cached["step_time"],
                        step_index=step_index,
                        latex_label="",
                        log10_k_bin_centers=numpy.array(cached["log10_k_bin_centers"]),
                        log10_spectrum=numpy.array(cached["log10_spectrum"]),
                    ),
                )
                continue

            with load_snapshot.QuokkaSnapshot(
                    snapshot_dir=snapshot_dir,
                    verbose=False,
            ) as snapshot:
                field = self.field_loader(snapshot)
            if not isinstance(field, field_models.ScalarField_3D):
                raise TypeError(
                    f"`{self.field_name}` is not a scalar field; "
                    "power spectra are only supported for scalar fields.",
                )
            step_time = field.sim_time
            assert step_time is not None
            reference_latex_label = field.latex_label
            spectrum = compute_spectra.compute_isotropic_power_spectrum_sfield(field)
            log10_k_bin_centers = numpy.ma.log10(
                numpy.ma.masked_less_equal(
                    x=spectrum.k_bin_centers_1d,
                    value=0.0,
                ),
            )
            log10_spectrum = numpy.ma.log10(
                numpy.ma.masked_less_equal(
                    x=spectrum.spectrum_1d,
                    value=0.0,
                ),
            )
            spectra_data = SpectraData(
                step_time=step_time,
                step_index=step_index,
                latex_label=field.latex_label,
                log10_k_bin_centers=log10_k_bin_centers,
                log10_spectrum=log10_spectrum,
            )
            field_spectra.append(spectra_data)
            ## save after every snapshot, not just at the end, so a killed/interrupted
            ## run still leaves usable partial data on disk
            if self.save_data:
                output_dict[padded_index] = {
                    "step_time": spectra_data.step_time,
                    "log10_k_bin_centers": spectra_data.log10_k_bin_centers,
                    "log10_spectrum": spectra_data.log10_spectrum,
                }
                self._save_incremental(output_dict=output_dict)

        field_spectra.sort(key=lambda s: s.step_time)
        ## every entry needs a valid latex_label for plot styling, but cache-hit entries
        ## are given an empty one above; only field_spectra[0] is ever read, so fix that one
        if field_spectra and not field_spectra[0].latex_label:
            if reference_latex_label is None:
                ## every snapshot was a cache hit this run; do one minimal load just for the label
                with load_snapshot.QuokkaSnapshot(
                        snapshot_dir=self.snapshot_dirs[0],
                        verbose=False,
                ) as snapshot:
                    reference_latex_label = self.field_loader(snapshot).latex_label
            field_spectra[0] = dataclasses.replace(field_spectra[0], latex_label=reference_latex_label)
        return field_spectra


##
## === FIGURE RENDERING
##


@final
class RenderSpectra:

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

    @staticmethod
    def _style_ax(
        *,
        ax: manage_plots.PlotAxis,
        latex_label: str,
    ) -> None:
        ax.set_xlabel(r"$\log_{10}(k)$")
        ax.set_ylabel(rf"$\log_{{10}}\big(\mathcal{{P}}_{{{latex_label}}}(k)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        ax: manage_plots.PlotAxis,
        spectra_data: SpectraData,
        color: annotate_axis.ColorType,
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
        ax: manage_plots.PlotAxis,
        field_spectra: list[SpectraData],
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
            RenderSpectra._plot_snapshot(
                ax=ax,
                spectra_data=spectra_data,
                color=color,
            )
        add_color.add_colorbar(
            ax=ax,
            palette=palette,
            label=r"snapshot index",
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
        )
        field_spectra = compute.run()
        if not field_spectra:
            return
        if not self.save_figure:
            return
        ## plot single snapshot in black, or a sequential color series across all snapshots
        fig, ax = manage_plots.create_figure()
        if len(field_spectra) > 1:
            fig.subplots_adjust(right=0.82)
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
        ## include step index in the filename if there is only one snapshot
        if len(field_spectra) == 1:
            step_index = int(
                find_snapshots.get_step_index_string(
                    snapshot_dir=self.snapshot_dirs[0],
                    snapshot_tag=self.snapshot_tag,
                ),
            )
            padded_index = f"{step_index:0{self.index_width}d}"
            fig_path = self.figures_dir / f"{self.field_name}-spectrum-index={padded_index}.png"
        else:
            fig_path = self.figures_dir / f"{self.field_name}-spectra.png"
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
        save_data: bool,
        save_figure: bool,
        data_dir: Path | None = None,
        figures_dir: Path | None = None,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        cli.ensure_save_flag_selected(
            save_figure=save_figure,
            save_data=save_data,
        )
        field_registry.validate_fields(field_names=fields_to_plot)
        self.input_dir = Path(input_dir)
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = validate_types.as_tuple(param=fields_to_plot)
        self.save_data = save_data
        self.save_figure = save_figure
        self.data_dir = Path(data_dir) if data_dir is not None else None
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
        data_dir = cli.resolve_output_dir(
            output_dir=self.data_dir,
            default_dir=snapshot_dirs[0].parent,
        )
        figures_dir = cli.resolve_output_dir(
            output_dir=self.figures_dir,
            default_dir=data_dir,
        )
        index_width = find_snapshots.get_max_index_width(
            snapshot_dirs=snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
        )
        ## compute and render power spectra for each requested field
        for field_name in self.fields_to_plot:
            field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
            renderer = RenderSpectra(
                snapshot_dirs=snapshot_dirs,
                snapshot_tag=self.snapshot_tag,
                index_width=index_width,
                data_dir=data_dir,
                figures_dir=figures_dir,
                field_name=field_name,
                field_loader=field_meta.loader,
                cmap_name=field_meta.cmap,
                save_data=self.save_data,
                save_figure=self.save_figure,
            )
            renderer.run()


##
## === PROGRAM MAIN
##


def main():
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    user_args = argparse.ArgumentParser(
        description="Generate power spectra of Quokka scalar fields: figures and/or extracted data.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_output=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        save_data=user_args.save_data,
        save_figure=user_args.save_figure,
        data_dir=user_args.data_dir,
        figures_dir=user_args.figures_dir,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
