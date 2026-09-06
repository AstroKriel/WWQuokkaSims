## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import pathlib
import typing

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import compute_spectra
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import add_color, annotate_panel, manage_figure
from jormi.ww_validation import validate_arrays, validate_types

## local
from ww_quokka_sims.sim_io.field_diagnostics import field_palettes
from ww_quokka_sims.sim_io.snapshots import field_registry, find_snapshots, load_snapshot

##
## === SPECTRA DATA
##


@dataclasses.dataclass(frozen=True)
class SpectraData:
    sim_time: float
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

    def save_to_file(
        self,
        file_path: pathlib.Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "sim_time": self.sim_time,
                "step_index": self.step_index,
                "latex_label": self.latex_label,
                "log10_k_bin_centers": self.log10_k_bin_centers,
                "log10_spectrum": self.log10_spectrum,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: pathlib.Path,
    ) -> "SpectraData":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={
                "sim_time",
                "step_index",
                "latex_label",
                "log10_k_bin_centers",
                "log10_spectrum",
            },
            param_name="<SpectraData JSON>",
        )
        return cls(
            sim_time=float(data["sim_time"]),
            step_index=int(data["step_index"]),
            latex_label=data["latex_label"],
            log10_k_bin_centers=numpy.asarray(data["log10_k_bin_centers"]),
            log10_spectrum=numpy.asarray(data["log10_spectrum"]),
        )


##
## === FIELD PROCESSING
##


@typing.final
class ComputeSpectra:

    def __init__(
        self,
        *,
        snapshot_dirs: list[pathlib.Path],
        snapshot_tag: str,
        registered_field: field_registry.RegisteredField,
        index_width: int,
        save_data: bool,
        data_dir: pathlib.Path,
        overwrite: bool = False,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.registered_field = registered_field
        self.index_width = index_width
        self.save_data = save_data
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.amr_level = amr_level

    def _data_file_path(
        self,
        *,
        padded_index: str,
    ) -> pathlib.Path:
        return self.data_dir / f"{self.registered_field.name}-spectrum-index={padded_index}.json"

    def run(
        self,
    ) -> list[SpectraData]:
        field_spectra: list[SpectraData] = []
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
                field_spectra.append(SpectraData.load_from_file(data_path))
                continue
            with load_snapshot.QuokkaSnapshot(
                    snapshot_dir=snapshot_dir,
                    verbose=False,
            ) as quokka_snapshot:
                field = self.registered_field.load(quokka_snapshot=quokka_snapshot, amr_level=self.amr_level)
            spectrum = compute_spectra.compute_isotropic_power_spectrum_field(field)
            sim_time = field.sim_time
            assert sim_time is not None
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
            spectra_data = SpectraData(
                sim_time=sim_time,
                step_index=step_index,
                latex_label=field.latex_label,
                log10_k_bin_centers=log10_k_bin_centers,
                log10_spectrum=log10_spectrum,
            )
            field_spectra.append(spectra_data)
            ## save immediately, one file per snapshot, so a killed/interrupted run still
            ## leaves every already-completed snapshot independently usable and resumable
            if self.save_data:
                manage_io.create_directory(
                    directory=self.data_dir,
                    verbose=False,
                )
                spectra_data.save_to_file(data_path)
        field_spectra.sort(key=lambda s: s.sim_time)
        return field_spectra


##
## === FIGURE RENDERING
##


@typing.final
class GenerateSpectra:

    def __init__(
        self,
        *,
        snapshot_dirs: list[pathlib.Path],
        snapshot_tag: str,
        index_width: int,
        data_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        registered_field: field_registry.RegisteredField,
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
        self.registered_field = registered_field
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
        spectra_data: SpectraData,
        color: annotate_panel.ColorType,
    ) -> None:
        ax.plot(
            spectra_data.log10_k_bin_centers,
            spectra_data.log10_spectrum,
            color=color,
        )

    @staticmethod
    def _plot_series(
        *,
        ax: manage_figure.Panel,
        field_spectra: list[SpectraData],
    ) -> None:
        palette = add_color.make_palette(
            config=add_color.SequentialPaletteConfig(
                palette_name=field_palettes.SEQUENTIAL_PALETTE_NAME,
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
            colorbar_gap_pt=5.0,
            label_gap_pt=10.0,
        )

    def _snapshot_figure_file_path(
        self,
        *,
        figures_dir: pathlib.Path,
        padded_index: str,
    ) -> pathlib.Path:
        return figures_dir / f"{self.registered_field.name}-spectrum-index={padded_index}.png"

    def _save_snapshot_figure(
        self,
        *,
        spectra_data: SpectraData,
        figure_path: pathlib.Path,
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
        field_spectra: list[SpectraData],
        figures_dir: pathlib.Path,
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
            )
        self._style_ax(
            ax=ax,
            latex_label=field_spectra[0].latex_label,
        )
        fig_path = figures_dir / f"{self.registered_field.name}-spectra-summary.png"
        manage_figure.save_figure(
            figure=fig,
            figure_path=fig_path,
            verbose=True,
        )

    def run(
        self,
    ) -> None:
        ## compute the isotropic power spectrum for each snapshot; saved incrementally as each completes
        compute_spectra_pipeline = ComputeSpectra(
            snapshot_dirs=self.snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
            registered_field=self.registered_field,
            index_width=self.index_width,
            save_data=self.save_data,
            data_dir=self.data_dir,
            overwrite=self.overwrite,
            amr_level=self.amr_level,
        )
        field_spectra = compute_spectra_pipeline.run()
        if not field_spectra:
            return
        if not self.save_figure:
            return
        ## one figure per snapshot, resumed like everything else; the combined summary always
        ## rebuilds since it's cheap relative to the per-snapshot compute above
        for spectra_data in field_spectra:
            padded_index = f"{spectra_data.step_index:0{self.index_width}d}"
            figure_path = self._snapshot_figure_file_path(
                figures_dir=self.figures_dir,
                padded_index=padded_index,
            )
            if self.overwrite or not figure_path.exists():
                self._save_snapshot_figure(
                    spectra_data=spectra_data,
                    figure_path=figure_path,
                )
        self._save_summary_figure(
            field_spectra=field_spectra,
            figures_dir=self.figures_dir,
        )


## } MODULE
