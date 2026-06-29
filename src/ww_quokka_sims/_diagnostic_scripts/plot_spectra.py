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
    sim_time: float
    latex_label: str
    k_bin_centers: numpy.ndarray
    log10_spectrum: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        validate_arrays.ensure_array(array=self.k_bin_centers)
        validate_arrays.ensure_array(array=self.log10_spectrum)
        validate_arrays.ensure_1d(array=self.k_bin_centers)
        validate_arrays.ensure_1d(array=self.log10_spectrum)
        validate_arrays.ensure_same_shape(
            array_a=self.k_bin_centers,
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
        field_name: str,
        field_loader: Callable,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.field_name = field_name
        self.field_loader = field_loader

    def run(
        self,
    ) -> list[SpectraData]:
        field_spectra: list[SpectraData] = []
        for snapshot_dir in self.snapshot_dirs:
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
            sim_time = field.sim_time
            assert sim_time is not None
            spectrum = compute_spectra.compute_isotropic_power_spectrum_sfield(field)
            log10_spectrum = numpy.ma.log10(
                numpy.ma.masked_less_equal(
                    x=spectrum.spectrum_1d,
                    value=0.0,
                ),
            )
            field_spectra.append(
                SpectraData(
                    sim_time=sim_time,
                    latex_label=field.latex_label,
                    k_bin_centers=spectrum.k_bin_centers_1d,
                    log10_spectrum=log10_spectrum,
                ),
            )
        field_spectra.sort(key=lambda s: s.sim_time)
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
        out_dir: Path,
        field_name: str,
        field_loader: Callable,
        cmap_name: str,
        extract_data: bool,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.out_dir = out_dir
        self.field_name = field_name
        self.field_loader = field_loader
        self.cmap_name = cmap_name
        self.extract_data = extract_data

    @staticmethod
    def _style_ax(
        *,
        ax: manage_plots.PlotAxis,
        latex_label: str,
    ) -> None:
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(rf"$\log_{{10}}\big(\mathcal{{P}}_{{{latex_label}}}(k)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        ax: manage_plots.PlotAxis,
        spectra_data: SpectraData,
        color: annotate_axis.ColorType,
    ) -> None:
        ax.plot(
            spectra_data.k_bin_centers,
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

    def _save_spectra(
        self,
        *,
        field_spectra: list[SpectraData],
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_dict = {}
        for snapshot_index, spectra_data in enumerate(field_spectra):
            output_dict[str(snapshot_index)] = {
                "sim_time": spectra_data.sim_time,
                "k_bin_centers": spectra_data.k_bin_centers,
                "log10_spectrum": spectra_data.log10_spectrum,
            }
        json_io.save_dict_to_json_file(
            file_path=out_dir / f"{self.field_name}-spectra.json",
            input_dict=output_dict,
            overwrite=True,
            verbose=False,
        )

    def run(
        self,
    ) -> None:
        ## compute the isotropic power spectrum for each snapshot
        compute = ComputeSpectra(
            snapshot_dirs=self.snapshot_dirs,
            field_name=self.field_name,
            field_loader=self.field_loader,
        )
        field_spectra = compute.run()
        if not field_spectra:
            return
        ## optionally write extracted spectrum data to JSON
        if self.extract_data:
            self._save_spectra(
                field_spectra=field_spectra,
                out_dir=self.out_dir,
            )
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
        ## include snapshot index in the filename if there is only one snapshot
        if len(field_spectra) == 1:
            snapshot_index = find_snapshots.get_snapshot_index_string(
                snapshot_dir=self.snapshot_dirs[0],
                snapshot_tag=self.snapshot_tag,
            )
            fig_path = self.out_dir / f"{self.field_name}-spectrum-index={snapshot_index}.png"
        else:
            fig_path = self.out_dir / f"{self.field_name}-spectra.png"
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
        extract_data: bool,
        out_dir: Path | None = None,
    ):
        validate_types.ensure_nonempty_string(
            param=snapshot_tag,
            param_name="snapshot_tag",
        )
        field_registry.validate_fields(field_names=fields_to_plot)
        self.input_dir = Path(input_dir)
        self.snapshot_tag = snapshot_tag
        self.fields_to_plot = validate_types.as_tuple(param=fields_to_plot)
        self.extract_data = extract_data
        self.out_dir = Path(out_dir) if out_dir is not None else None

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
        out_dir = self.out_dir if self.out_dir is not None else snapshot_dirs[0].parent
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        ## compute and render power spectra for each requested field
        for field_name in self.fields_to_plot:
            field_meta = field_registry.QUOKKA_FIELD_LOOKUP[field_name]
            renderer = RenderSpectra(
                snapshot_dirs=snapshot_dirs,
                snapshot_tag=self.snapshot_tag,
                out_dir=out_dir,
                field_name=field_name,
                field_loader=field_meta.loader,
                cmap_name=field_meta.cmap,
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
        description="Plot power spectra of Quokka scalar fields.",
        parents=[
            cli.base_parser(
                num_dirs=1,
                allow_vfields=False,
                produces_data=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.input_dir,
        snapshot_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        extract_data=user_args.save_data,
        out_dir=user_args.out_dir,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
