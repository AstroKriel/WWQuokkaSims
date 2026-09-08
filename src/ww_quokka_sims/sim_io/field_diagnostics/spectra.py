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
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields.fields_3d import compute_spectra
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import add_color, annotate_panel, latex_labels, manage_figure
from jormi.ww_validation import validate_arrays, validate_types

## local
from ww_quokka_sims.sim_io.field_diagnostics import field_palettes
from ww_quokka_sims.sim_io.snapshots import field_registry, find_snapshots, load_snapshot

##
## === SPECTRA DATA
##


@dataclasses.dataclass(frozen=True)
class FieldSpectrum:
    sim_time: float
    step_index: find_snapshots.StepIndex
    latex_label: latex_labels.LatexLabel
    log10_k_bin_centers: numpy.ndarray
    log10_power_spectrum: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        validate_arrays.ensure_array(array=self.log10_k_bin_centers)
        validate_arrays.ensure_array(array=self.log10_power_spectrum)
        validate_arrays.ensure_1d(array=self.log10_k_bin_centers)
        validate_arrays.ensure_1d(array=self.log10_power_spectrum)
        validate_arrays.ensure_same_shape(
            array_a=self.log10_k_bin_centers,
            array_b=self.log10_power_spectrum,
        )

    def save_to_file(
        self,
        file_path: pathlib.Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "sim_time": self.sim_time,
                "step_index": self.step_index.value,
                "latex_label": self.latex_label.content,
                "log10_k_bin_centers": self.log10_k_bin_centers,
                "log10_power_spectrum": self.log10_power_spectrum,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: pathlib.Path,
    ) -> "FieldSpectrum":
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
                "log10_power_spectrum",
            },
            param_name="<FieldSpectrum JSON>",
        )
        return cls(
            sim_time=float(data["sim_time"]),
            step_index=find_snapshots.StepIndex.from_value(int(data["step_index"])),
            latex_label=latex_labels.LatexLabel(content=data["latex_label"]),
            log10_k_bin_centers=numpy.asarray(data["log10_k_bin_centers"]),
            log10_power_spectrum=numpy.asarray(data["log10_power_spectrum"]),
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

    def _get_data_path(
        self,
        *,
        padded_step_index_string: str,
    ) -> pathlib.Path:
        return self.data_dir / f"{self.registered_field.name}-spectrum-index={padded_step_index_string}.json"

    def _compute_spectrum(
        self,
        *,
        snapshot_dir: pathlib.Path,
        step_index: find_snapshots.StepIndex,
    ) -> FieldSpectrum:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as quokka_snapshot:
            field_3d = self.registered_field.load(
                quokka_snapshot=quokka_snapshot,
                amr_level=self.amr_level,
            )
        spectrum = compute_spectra.compute_isotropic_power_spectrum_field(field_3d)
        sim_time = field_3d.sim_time
        assert sim_time is not None
        log10_k_bin_centers = compute_array_stats.compute_safe_log10(spectrum.k_bin_centers_1d)
        log10_power_spectrum = compute_array_stats.compute_safe_log10(spectrum.power_spectrum_1d)
        return FieldSpectrum(
            sim_time=sim_time,
            step_index=step_index,
            latex_label=latex_labels.LatexLabel(content=field_3d.latex_label),
            log10_k_bin_centers=log10_k_bin_centers,
            log10_power_spectrum=log10_power_spectrum,
        )

    def run(
        self,
    ) -> list[FieldSpectrum]:
        field_spectra: list[FieldSpectrum] = []
        for snapshot_dir in self.snapshot_dirs:
            step_index = find_snapshots.get_step_index(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            )
            padded_step_index_string = step_index.get_padded_string(index_width=self.index_width)
            data_path = self._get_data_path(padded_step_index_string=padded_step_index_string)
            if (not self.overwrite) and data_path.exists():
                field_spectrum = FieldSpectrum.load_from_file(data_path)
            else:
                field_spectrum = self._compute_spectrum(
                    snapshot_dir=snapshot_dir,
                    step_index=step_index,
                )
                if self.save_data:
                    manage_io.create_directory(
                        directory=self.data_dir,
                        verbose=False,
                    )
                    field_spectrum.save_to_file(data_path)
            field_spectra.append(field_spectrum)
        field_spectra.sort(key=lambda _field_spectrum: _field_spectrum.sim_time)
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
        field_label: latex_labels.LatexLabel,
    ) -> None:
        ylabel = latex_labels.LatexLabel(
            content=rf"\log_{{10}}\big(\mathcal{{P}}_{{{field_label.content}}}(k)\big)",
        ).label
        ax.set_xlabel(r"$\log_{10}(k)$")
        ax.set_ylabel(ylabel)

    @staticmethod
    def _plot_snapshot(
        *,
        ax: manage_figure.Panel,
        field_spectrum: FieldSpectrum,
        color: annotate_panel.ColorType,
    ) -> None:
        ax.plot(
            field_spectrum.log10_k_bin_centers,
            field_spectrum.log10_power_spectrum,
            color=color,
        )

    @staticmethod
    def _plot_series(
        *,
        ax: manage_figure.Panel,
        field_spectra: list[FieldSpectrum],
    ) -> None:
        last_series_index = max(0, len(field_spectra) - 1)
        palette = add_color.make_palette(
            config=add_color.SequentialPaletteConfig(
                palette_name=field_palettes.SEQUENTIAL_PALETTE_NAME,
                palette_range=(0.25, 1.0),
            ),
            value_range=(0, last_series_index),
        )
        for series_index, field_spectrum in enumerate(field_spectra):
            color = palette.get_color(series_index)
            GenerateSpectra._plot_snapshot(
                ax=ax,
                field_spectrum=field_spectrum,
                color=color,
            )
        add_color.add_colorbar(
            panels=ax,
            palette=palette,
            label=r"snapshot index",
            colorbar_gap_pt=5.0,
            label_gap_pt=10.0,
        )

    def _save_snapshot_figure(
        self,
        *,
        field_spectrum: FieldSpectrum,
        figure_path: pathlib.Path,
    ) -> None:
        figure, ax = manage_figure.create_figure()
        self._plot_snapshot(
            ax=ax,
            field_spectrum=field_spectrum,
            color="black",
        )
        self._style_ax(
            ax=ax,
            field_label=field_spectrum.latex_label,
        )
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
            verbose=False,
        )

    def _save_summary_figure(
        self,
        *,
        field_spectra: list[FieldSpectrum],
        figures_dir: pathlib.Path,
    ) -> None:
        """Combined overlay across every snapshot processed this run; always rebuilt fresh."""
        figure, ax = manage_figure.create_figure()
        if len(field_spectra) == 1:
            self._plot_snapshot(
                ax=ax,
                field_spectrum=field_spectra[0],
                color="black",
            )
        else:
            self._plot_series(
                ax=ax,
                field_spectra=field_spectra,
            )
        self._style_ax(
            ax=ax,
            field_label=field_spectra[0].latex_label,
        )
        figure_path = figures_dir / f"{self.registered_field.name}-spectra-summary.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
            verbose=True,
        )

    def run(
        self,
    ) -> None:
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
        if field_spectra and self.save_figure:
            for field_spectrum in field_spectra:
                padded_step_index_string = field_spectrum.step_index.get_padded_string(index_width=self.index_width)
                figure_path = self.figures_dir / f"{self.registered_field.name}-spectrum-index={padded_step_index_string}.png"
                if self.overwrite or not figure_path.exists():
                    self._save_snapshot_figure(
                        field_spectrum=field_spectrum,
                        figure_path=figure_path,
                    )
            self._save_summary_figure(
                field_spectra=field_spectra,
                figures_dir=self.figures_dir,
            )


## } MODULE
