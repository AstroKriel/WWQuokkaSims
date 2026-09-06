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
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_io import json_io, manage_io, manage_log
from jormi.ww_plots import add_color, annotate_panel, manage_figure
from jormi.ww_validation import validate_arrays, validate_types

## local
from ww_quokka_sims.sim_io.field_diagnostics import field_palettes
from ww_quokka_sims.sim_io.snapshots import field_registry, find_snapshots, load_snapshot

##
## === PDF DATA
##


@dataclasses.dataclass(frozen=True)
class PDFData:
    step_time: float
    step_index: int
    grouped_bin_centers: list[numpy.ndarray]
    grouped_densities: list[numpy.ndarray]
    comp_labels: list[str]
    use_log10_bins: bool = False

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_sequence(
            param=self.grouped_bin_centers,
            valid_seq_types=(list, tuple),
            param_name="<grouped_bin_centers>",
            seq_length=len(self.comp_labels),
        )
        validate_types.ensure_sequence(
            param=self.grouped_densities,
            valid_seq_types=(list, tuple),
            param_name="<grouped_densities>",
            seq_length=len(self.comp_labels),
        )
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

    def save_to_file(
        self,
        file_path: pathlib.Path,
    ) -> None:
        bin_centers_key = "log10_bin_centers" if self.use_log10_bins else "bin_centers"
        output_dict: dict = {
            "step_time": self.step_time,
            "step_index": self.step_index,
            "use_log10_bins": self.use_log10_bins,
        }
        for comp_index, comp_label in enumerate(self.comp_labels):
            bin_centers, densities = self.get_pdf(comp_index)
            output_dict[comp_label] = {
                bin_centers_key: bin_centers,
                "log10_density": densities,
            }
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict=output_dict,
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: pathlib.Path,
    ) -> "PDFData":
        input_dict = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=input_dict,
            required_keys={"step_time", "step_index", "use_log10_bins"},
            param_name="<PDFData JSON>",
        )
        use_log10_bins = bool(input_dict["use_log10_bins"])
        bin_centers_key = "log10_bin_centers" if use_log10_bins else "bin_centers"
        comp_labels = [key for key in input_dict if key not in ("step_time", "step_index", "use_log10_bins")]
        return cls(
            step_time=float(input_dict["step_time"]),
            step_index=int(input_dict["step_index"]),
            grouped_bin_centers=[numpy.array(input_dict[label][bin_centers_key]) for label in comp_labels],
            grouped_densities=[numpy.array(input_dict[label]["log10_density"]) for label in comp_labels],
            comp_labels=comp_labels,
            use_log10_bins=use_log10_bins,
        )


##
## === FIELD PROCESSING
##


@typing.final
class ComputePDFs:

    def __init__(
        self,
        *,
        registered_field: field_registry.RegisteredField,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        num_bins: int,
        use_log10_bins: bool = False,
        amr_level: int = 0,
    ):
        self.registered_field = registered_field
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
    ) -> PDFData:
        if len(self.comps_to_plot) == 0:
            raise ValueError(
                f"Vector field `{self.registered_field.name}` requires at least one component to plot; none provided.",
            )
        field_models.ensure_3d_vfield(field)
        step_time = field.sim_time
        assert step_time is not None
        comp_names = sorted(self.comps_to_plot)
        comp_labels = [field_models.get_vcomp_label(vfield_3d=field, comp_axis=comp_name) for comp_name in comp_names]
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
        return PDFData(
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
    ) -> PDFData:
        field_models.ensure_3d_sfield(field)
        step_time = field.sim_time
        assert step_time is not None
        bin_centers, densities = self._estimate_pdf(
            field_data=field.fdata.farray,
            num_bins=self.num_bins,
            use_log10_bins=self.use_log10_bins,
        )
        return PDFData(
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
        snapshot_dir: pathlib.Path,
        snapshot_tag: str,
    ) -> PDFData:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=snapshot_tag,
            ),
        )
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as quokka_snapshot:
            field = self.registered_field.load(quokka_snapshot=quokka_snapshot, amr_level=self.amr_level)
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
        raise ValueError(f"{self.registered_field.name} is an unrecognised field type.")


##
## === FIGURE RENDERING
##


@typing.final
class GeneratePDFs:

    def __init__(
        self,
        *,
        snapshot_dirs: list[pathlib.Path],
        snapshot_tag: str,
        index_width: int,
        data_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        registered_field: field_registry.RegisteredField,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
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
        self.registered_field = registered_field
        self.comps_to_plot = comps_to_plot
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
        return f"log10_{self.registered_field.name}" if self.use_log10_bins else self.registered_field.name

    def _data_file_path(
        self,
        *,
        data_dir: pathlib.Path,
        padded_index: str,
    ) -> pathlib.Path:
        return data_dir / f"{self._data_name()}-pdf-index={padded_index}.json"

    def _snapshot_figure_file_path(
        self,
        *,
        figures_dir: pathlib.Path,
        padded_index: str,
    ) -> pathlib.Path:
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
        pdf_data: PDFData,
        color: annotate_panel.ColorType,
    ) -> None:
        for comp_index in range(pdf_data.num_comps):
            ax = axs_grid[0][comp_index]
            x_values, y_values = pdf_data.get_pdf(comp_index)
            ax.step(
                x_values,
                y_values,
                where="mid",
                color=color,
                zorder=comp_index + 1,
            )

    @staticmethod
    def _plot_series(
        *,
        axs_grid: manage_figure.PanelGrid,
        field_pdfs: list[PDFData],
    ) -> None:
        palette = add_color.make_palette(
            config=add_color.SequentialConfig(
                palette_name=field_palettes.SEQUENTIAL_PALETTE_NAME,
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
            colorbar_gap_pt=15.0,
        )

    def _save_pdf(
        self,
        *,
        pdf_data: PDFData,
        data_dir: pathlib.Path,
    ) -> None:
        """Save one snapshot's PDF to its own file, mirroring `generate_slices.py`'s one-file-per-
        snapshot convention (rather than one file aggregating every snapshot) -- each file is
        self-contained (carries its own `step_time`/`use_log10_bins`), so results already on disk
        are immediately usable even if a later snapshot in the run fails or the job is cut off.
        """
        manage_io.create_directory(
            directory=data_dir,
            verbose=False,
        )
        padded_index = f"{pdf_data.step_index:0{self.index_width}d}"
        pdf_data.save_to_file(
            self._data_file_path(
                data_dir=data_dir,
                padded_index=padded_index,
            ),
        )

    def _save_snapshot_figure(
        self,
        *,
        pdf_data: PDFData,
        figure_path: pathlib.Path,
    ) -> None:
        fig, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=pdf_data.num_comps,
            panel_col_gap_pt=30.0,
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
        snapshot_dir: pathlib.Path,
        data_dir: pathlib.Path,
        figures_dir: pathlib.Path,
        index_width: int,
    ) -> None:
        step_index = int(
            find_snapshots.get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            ),
        )
        padded_index = f"{step_index:0{index_width}d}"
        data_path = self._data_file_path(
            data_dir=data_dir,
            padded_index=padded_index,
        )
        figure_path = self._snapshot_figure_file_path(
            figures_dir=figures_dir,
            padded_index=padded_index,
        )
        data_exists = data_path.exists()
        data_needed = self.save_data and (self.overwrite or not data_exists)
        figure_needed = self.save_figure and (self.overwrite or not figure_path.exists())

        if not data_needed and not figure_needed:
            return

        if figure_needed and not data_needed and data_exists:
            ## cheap path: reconstruct the figure from already-saved data, skip the raw snapshot
            manage_log.log_hint(
                text=(
                    f"`{self.registered_field.name}` at snapshot {step_index}: "
                    f"building figure from saved data, skipping the raw snapshot."
                ),
            )
            pdf_data = PDFData.load_from_file(data_path)
            self._save_snapshot_figure(
                pdf_data=pdf_data,
                figure_path=figure_path,
            )
            return

        pdf_data = compute_pdfs.compute_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot_tag=self.snapshot_tag,
        )
        if data_needed:
            self._save_pdf(
                pdf_data=pdf_data,
                data_dir=data_dir,
            )
        if figure_needed:
            self._save_snapshot_figure(
                pdf_data=pdf_data,
                figure_path=figure_path,
            )

    def _load_all_saved_pdfs(
        self,
        *,
        data_dir: pathlib.Path,
    ) -> list[PDFData]:
        paths = sorted(data_dir.glob(f"{self._data_name()}-pdf-index=*.json"))
        field_pdfs = [PDFData.load_from_file(path) for path in paths]
        field_pdfs.sort(key=lambda pdf_data: pdf_data.step_time)
        return field_pdfs

    def _save_summary_figure(
        self,
        *,
        field_pdfs: list[PDFData],
        figures_dir: pathlib.Path,
    ) -> None:
        """Combined overlay across every saved snapshot; always rebuilt fresh from whatever is on
        disk (not from anything held in memory across the potentially-long per-snapshot loop above).
        """
        num_cols = field_pdfs[0].num_comps
        fig, axs_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=num_cols,
            panel_col_gap_pt=30.0,
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
                registered_field=self.registered_field,
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
                text=f"Skipping summary figure for `{self.registered_field.name}`: no saved data found in {self.data_dir}.",
            )
            return
        self._save_summary_figure(
            field_pdfs=field_pdfs,
            figures_dir=self.figures_dir,
        )


## } MODULE
