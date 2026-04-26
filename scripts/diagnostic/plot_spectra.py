## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import (
    compute_spectra,
    field_models,
)
from jormi.ww_io import json_io
from jormi.ww_plots import (
    add_color,
    manage_plots,
)
from jormi.ww_types import (
    check_arrays,
    check_types,
)

## local
from ww_quokka_sims.sim_io import (
    find_datasets,
    load_dataset,
)
import quokka_fields

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class SpectraData:
    sim_time: float
    field_label: str
    k_bin_centers: numpy.ndarray
    log10_spectrum: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        check_arrays.ensure_array(array=self.k_bin_centers)
        check_arrays.ensure_array(array=self.log10_spectrum)
        check_arrays.ensure_1d(array=self.k_bin_centers)
        check_arrays.ensure_1d(array=self.log10_spectrum)
        check_arrays.ensure_same_shape(
            array_a=self.k_bin_centers,
            array_b=self.log10_spectrum,
        )


##
## === FIELD PROCESSING
##


class ComputeSpectra:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        field_name: str,
        field_loader: Callable,
    ):
        self.dataset_dirs = dataset_dirs
        self.field_name = field_name
        self.field_loader = field_loader

    def run(
        self,
    ) -> list[SpectraData]:
        field_spectra: list[SpectraData] = []
        for dataset_dir in self.dataset_dirs:
            with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
                field = self.field_loader(ds)
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
                    field_label=field_models.get_label(field),
                    k_bin_centers=spectrum.k_bin_centers_1d,
                    log10_spectrum=log10_spectrum,
                ),
            )
        field_spectra.sort(key=lambda s: s.sim_time)
        return field_spectra


##
## === FIGURE RENDERING
##


class RenderSpectra:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        dataset_tag: str,
        out_dir: Path,
        field_name: str,
        field_loader: Callable,
        cmap_name: str,
        extract_data: bool,
    ):
        self.dataset_dirs = dataset_dirs
        self.dataset_tag = dataset_tag
        self.out_dir = out_dir
        self.field_name = field_name
        self.field_loader = field_loader
        self.cmap_name = cmap_name
        self.extract_data = extract_data

    @staticmethod
    def _style_ax(
        *,
        ax,
        field_label: str,
    ) -> None:
        ax.set_xlabel(r"$k$")
        raw_field_label = field_label.replace("$", "")
        ax.set_ylabel(rf"$\log_{{10}}\big(\mathcal{{P}}_{{{raw_field_label}}}(k)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        ax,
        spectra_data: SpectraData,
        color,
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
        ax,
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
            color = palette.mpl_cmap(palette.mpl_norm(series_index))
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
        out_dir.mkdir(parents=True, exist_ok=True)
        output_dict = {}
        for snapshot_index, spectra_data in enumerate(field_spectra):
            output_dict[str(snapshot_index)] = {
                "sim_time": spectra_data.sim_time,
                "k_bin_centers": spectra_data.k_bin_centers,
                "log10_spectrum": spectra_data.log10_spectrum,
            }
        json_io.save_dict_to_json_file(
            file_path=out_dir / f"{self.field_name}_spectra.json",
            input_dict=output_dict,
            overwrite=True,
            verbose=False,
        )

    def run(
        self,
    ) -> None:
        ## compute the isotropic power spectrum for each snapshot
        compute = ComputeSpectra(
            dataset_dirs=self.dataset_dirs,
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
            field_label=field_spectra[0].field_label,
        )
        ## include snapshot index in the filename if there is only one snapshot
        if len(field_spectra) == 1:
            snapshot_index = find_datasets.get_dataset_index_string(self.dataset_dirs[0], self.dataset_tag)
            fig_path = self.out_dir / f"{self.field_name}_spectrum_{snapshot_index}.png"
        else:
            fig_path = self.out_dir / f"{self.field_name}_spectra.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


##
## === SCRIPT INTERFACE
##


class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        dataset_tag: str,
        fields_to_plot: tuple[str, ...] | list[str] | None,
        extract_data: bool,
    ):
        check_types.ensure_nonempty_string(
            param=dataset_tag,
            param_name="dataset_tag",
        )
        quokka_fields.validate_fields(field_names=fields_to_plot)
        self.input_dir = Path(input_dir)
        self.dataset_tag = dataset_tag
        self.fields_to_plot = check_types.as_tuple(param=fields_to_plot)
        self.extract_data = extract_data

    def run(
        self,
    ) -> None:
        ## find all dataset dirs under input_dir whose names match dataset_tag, sorted by index
        dataset_dirs = find_datasets.resolve_dataset_dirs(
            input_dir=self.input_dir,
            dataset_tag=self.dataset_tag,
        )
        if not dataset_dirs:
            return
        ## output goes to the sim root (the shared parent of all dataset dirs)
        out_dir = dataset_dirs[0].parent
        ## compute and render power spectra for each requested field
        for field_name in self.fields_to_plot:
            field_meta = quokka_fields.QUOKKA_FIELD_LOOKUP[field_name]
            renderer = RenderSpectra(
                dataset_dirs=dataset_dirs,
                dataset_tag=self.dataset_tag,
                out_dir=out_dir,
                field_name=field_name,
                field_loader=field_meta["loader"],
                cmap_name=field_meta["cmap"],
                extract_data=self.extract_data,
            )
            renderer.run()


##
## === PROGRAM MAIN
##


def main():
    user_args = argparse.ArgumentParser(
        description="Plot power spectra of Quokka scalar fields.",
        parents=[
            quokka_fields.base_parser(
                num_dirs=1,
                allow_vfields=False,
                allow_extract=True,
            ),
        ],
    ).parse_args()
    script_interface = ScriptInterface(
        input_dir=user_args.dir,
        dataset_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        extract_data=user_args.extract,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
