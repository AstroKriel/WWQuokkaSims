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
from jormi.ww_io import json_io, manage_io
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    latex_labels,
    manage_figure,
)
from jormi.ww_validation import validate_arrays, validate_types

## local
from ww_quokka_sims.sim_io.field_diagnostics import field_palettes
from ww_quokka_sims.sim_io.snapshots import (
    field_registry,
    find_snapshots,
    load_snapshot,
)

##
## === PDF DATA
##

## every other key in the saved JSON is a per-component latex-label string
_METADATA_KEYS: set[str] = {"sim_time", "step_index", "use_log10_bins"}


@dataclasses.dataclass(frozen=True)
class FieldPDF:
    sim_time: float
    step_index: find_snapshots.StepIndex
    grouped_bin_centers: list[numpy.ndarray]
    grouped_densities: list[numpy.ndarray]
    comp_latex_labels: list[latex_labels.LatexLabel]
    use_log10_bins: bool = False

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_sequence(
            param=self.grouped_bin_centers,
            valid_seq_types=(list, tuple),
            param_name="<grouped_bin_centers>",
            seq_length=len(self.comp_latex_labels),
        )
        validate_types.ensure_sequence(
            param=self.grouped_densities,
            valid_seq_types=(list, tuple),
            param_name="<grouped_densities>",
            seq_length=len(self.comp_latex_labels),
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
        return len(self.comp_latex_labels)

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
        if self.use_log10_bins:
            bin_centers_key = "log10_bin_centers"
        else:
            bin_centers_key = "bin_centers"
        output_dict: dict = {
            "sim_time": self.sim_time,
            "step_index": self.step_index.value,
            "use_log10_bins": self.use_log10_bins,
        }
        for comp_index, comp_latex_label in enumerate(self.comp_latex_labels):
            bin_centers, densities = self.get_pdf(comp_index)
            output_dict[comp_latex_label.content] = {
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
    ) -> "FieldPDF":
        input_dict = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=input_dict,
            required_keys=_METADATA_KEYS,
            param_name="<FieldPDF JSON>",
        )
        sim_time = float(input_dict["sim_time"])
        step_index = find_snapshots.StepIndex.from_value(int(input_dict["step_index"]))
        comp_label_strings = [_key for _key in input_dict if _key not in _METADATA_KEYS]
        use_log10_bins = bool(input_dict["use_log10_bins"])
        if use_log10_bins:
            bin_centers_key = "log10_bin_centers"
        else:
            bin_centers_key = "bin_centers"
        grouped_bin_centers = [
            numpy.array(input_dict[_comp_label_string][bin_centers_key]) for _comp_label_string in comp_label_strings
        ]
        grouped_densities = [
            numpy.array(input_dict[_comp_label_string]["log10_density"]) for _comp_label_string in comp_label_strings
        ]
        comp_latex_labels = [
            latex_labels.LatexLabel(content=_comp_label_string) for _comp_label_string in comp_label_strings
        ]
        return cls(
            sim_time=sim_time,
            step_index=step_index,
            grouped_bin_centers=grouped_bin_centers,
            grouped_densities=grouped_densities,
            comp_latex_labels=comp_latex_labels,
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
        snapshot_dirs: list[pathlib.Path],
        snapshot_tag: str,
        registered_field: field_registry.RegisteredField,
        index_width: int,
        comps_to_plot: tuple[cartesian_axes.AxisLike_3D, ...],
        num_bins: int,
        save_data: bool,
        data_dir: pathlib.Path,
        overwrite: bool = False,
        use_log10_bins: bool = False,
        amr_level: int = 0,
    ):
        self.snapshot_dirs = snapshot_dirs
        self.snapshot_tag = snapshot_tag
        self.registered_field = registered_field
        self.index_width = index_width
        self.comps_to_plot = comps_to_plot
        self.num_bins = int(num_bins)
        self.save_data = save_data
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.use_log10_bins = use_log10_bins
        self.amr_level = amr_level

    def _get_data_tag(
        self,
    ) -> str:
        """Filename stem, tagged with `log10_` when bins are log10-spaced.

        The filename is a hint for humans browsing the directory, not the source of truth (it can
        be renamed); the saved `use_log10_bins` flag and `log10_bin_centers` key inside the file
        itself are what downstream code should actually check.
        """
        if self.use_log10_bins:
            return f"log10_{self.registered_field.name}"
        else:
            return self.registered_field.name

    def _get_data_path(
        self,
        *,
        padded_step_index_string: str,
    ) -> pathlib.Path:
        data_tag = self._get_data_tag()
        return self.data_dir / f"{data_tag}-pdf-index={padded_step_index_string}.json"

    @staticmethod
    def _estimate_pdf(
        *,
        sarray_3d: numpy.ndarray,
        num_bins: int,
        use_log10_bins: bool,
    ) -> compute_array_stats.EstimatedPDF:
        """Estimate a 1D PDF of `sarray_3d`, optionally binned in log10-space.

        When `use_log10_bins` is set, bins are placed in log10-space of the field itself (not
        just the density axis), since fields spanning orders of magnitude (eg. current density)
        get almost all of their linearly-spaced bins wasted on the rare, large-valued tail,
        leaving the bulk of the distribution unresolved in a single bin.
        """
        values = sarray_3d.ravel()
        if use_log10_bins:
            values = compute_array_stats.compute_safe_log10(values)
        return compute_array_stats.estimate_pdf(
            values=values,
            num_bins=num_bins,
        )

    def _compute_vfield_pdf(
        self,
        vfield_3d: field_models.VectorField_3D,
        step_index: find_snapshots.StepIndex,
    ) -> FieldPDF:
        if len(self.comps_to_plot) == 0:
            raise ValueError(
                f"Vector field `{self.registered_field.name}` requires at least one component to plot; none provided.",
            )
        field_models.ensure_3d_vfield(vfield_3d)
        sim_time = vfield_3d.sim_time
        assert sim_time is not None
        comp_names = sorted(self.comps_to_plot)
        comp_latex_labels = [
            field_models.get_vcomp_label(
                vfield_3d=vfield_3d,
                comp_axis=comp_name,
            ) for comp_name in comp_names
        ]
        grouped_bin_centers: list[numpy.ndarray] = []
        grouped_densities: list[numpy.ndarray] = []
        for comp_name in comp_names:
            sarray_3d = vfield_3d.fdata.farray[cartesian_axes.get_axis_index(comp_name)]
            pdf = self._estimate_pdf(
                sarray_3d=sarray_3d,
                num_bins=self.num_bins,
                use_log10_bins=self.use_log10_bins,
            )
            grouped_bin_centers.append(pdf.bin_centers)
            grouped_densities.append(pdf.log10_densities)
        return FieldPDF(
            sim_time=sim_time,
            step_index=step_index,
            grouped_bin_centers=grouped_bin_centers,
            grouped_densities=grouped_densities,
            comp_latex_labels=comp_latex_labels,
            use_log10_bins=self.use_log10_bins,
        )

    def _compute_sfield_pdf(
        self,
        sfield_3d: field_models.ScalarField_3D,
        step_index: find_snapshots.StepIndex,
    ) -> FieldPDF:
        field_models.ensure_3d_sfield(sfield_3d)
        sim_time = sfield_3d.sim_time
        assert sim_time is not None
        pdf = self._estimate_pdf(
            sarray_3d=sfield_3d.fdata.farray,
            num_bins=self.num_bins,
            use_log10_bins=self.use_log10_bins,
        )
        return FieldPDF(
            sim_time=sim_time,
            step_index=step_index,
            grouped_bin_centers=[pdf.bin_centers],
            grouped_densities=[pdf.log10_densities],
            comp_latex_labels=[field_models.get_label(sfield_3d)],
            use_log10_bins=self.use_log10_bins,
        )

    def _compute_snapshot(
        self,
        *,
        snapshot_dir: pathlib.Path,
        step_index: find_snapshots.StepIndex,
    ) -> FieldPDF:
        with load_snapshot.QuokkaSnapshot(
                snapshot_dir=snapshot_dir,
                verbose=False,
        ) as quokka_snapshot:
            field_3d = self.registered_field.load(
                quokka_snapshot=quokka_snapshot,
                amr_level=self.amr_level,
            )
        if isinstance(field_3d, field_models.ScalarField_3D):
            return self._compute_sfield_pdf(
                sfield_3d=field_3d,
                step_index=step_index,
            )
        elif isinstance(field_3d, field_models.VectorField_3D):
            return self._compute_vfield_pdf(
                vfield_3d=field_3d,
                step_index=step_index,
            )
        else:
            raise ValueError(f"{self.registered_field.name} is an unrecognised field type.")

    def run(
        self,
    ) -> list[FieldPDF]:
        field_pdfs: list[FieldPDF] = []
        for snapshot_dir in self.snapshot_dirs:
            step_index = find_snapshots.get_step_index(
                snapshot_dir=snapshot_dir,
                snapshot_tag=self.snapshot_tag,
            )
            padded_step_index_string = step_index.get_padded_string(index_width=self.index_width)
            data_path = self._get_data_path(padded_step_index_string=padded_step_index_string)
            if (not self.overwrite) and data_path.exists():
                field_pdf = FieldPDF.load_from_file(data_path)
            else:
                field_pdf = self._compute_snapshot(
                    snapshot_dir=snapshot_dir,
                    step_index=step_index,
                )
                if self.save_data:
                    manage_io.create_directory(
                        directory=self.data_dir,
                        verbose=False,
                    )
                    field_pdf.save_to_file(data_path)
            field_pdfs.append(field_pdf)
        field_pdfs.sort(key=lambda _field_pdf: _field_pdf.sim_time)
        return field_pdfs


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

    def _get_data_tag(
        self,
    ) -> str:
        """Filename stem, tagged with `log10_` when bins are log10-spaced.

        The filename is a hint for humans browsing the directory, not the source of truth (it can
        be renamed); the saved `use_log10_bins` flag and `log10_bin_centers` key inside the file
        itself are what downstream code should actually check.
        """
        if self.use_log10_bins:
            return f"log10_{self.registered_field.name}"
        else:
            return self.registered_field.name

    @staticmethod
    def _style_panel_grid(
        *,
        panel_grid: manage_figure.PanelGrid,
        comp_latex_labels: list[latex_labels.LatexLabel],
        use_log10_bins: bool,
    ) -> None:
        for comp_index, comp_latex_label in enumerate(comp_latex_labels):
            panel = panel_grid[0][comp_index]
            if use_log10_bins:
                x_latex_label = latex_labels.LatexLabel(content=rf"x \equiv \log_{{10}}({comp_latex_label.content})")
            else:
                x_latex_label = latex_labels.LatexLabel(content=rf"x \equiv {comp_latex_label.content}")
            panel.set_xlabel(x_latex_label.label)
            if comp_index == 0:
                panel.set_ylabel(r"$\log_{10}\big(p(x)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        panel_grid: manage_figure.PanelGrid,
        field_pdf: FieldPDF,
        color: annotate_panel.ColorType,
    ) -> None:
        for comp_index in range(field_pdf.num_comps):
            panel = panel_grid[0][comp_index]
            x_values, y_values = field_pdf.get_pdf(comp_index)
            panel.step(
                x_values,
                y_values,
                where="mid",
                color=color,
                zorder=comp_index + 1,
            )

    @staticmethod
    def _plot_series(
        *,
        panel_grid: manage_figure.PanelGrid,
        field_pdfs: list[FieldPDF],
    ) -> None:
        last_series_index = max(0, len(field_pdfs) - 1)
        palette = add_color.make_palette(
            config=add_color.SequentialPaletteConfig(
                palette_name=field_palettes.SEQUENTIAL_PALETTE_NAME,
                palette_range=(0.25, 1.0),
            ),
            value_range=(0, last_series_index),
        )
        for series_index, field_pdf in enumerate(field_pdfs):
            color = palette.get_color(series_index)
            GeneratePDFs._plot_snapshot(
                panel_grid=panel_grid,
                field_pdf=field_pdf,
                color=color,
            )
        add_color.add_colorbar(
            panels=panel_grid[-1][-1],
            palette=palette,
            label=r"snapshot index",
            colorbar_gap_pt=15.0,
            label_gap_pt=10.0,
        )

    def _save_snapshot_figure(
        self,
        *,
        field_pdf: FieldPDF,
        figure_path: pathlib.Path,
    ) -> None:
        figure, panel_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=field_pdf.num_comps,
            panel_col_gap_pt=30.0,
        )
        self._plot_snapshot(
            panel_grid=panel_grid,
            field_pdf=field_pdf,
            color="black",
        )
        self._style_panel_grid(
            panel_grid=panel_grid,
            comp_latex_labels=field_pdf.comp_latex_labels,
            use_log10_bins=self.use_log10_bins,
        )
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
            verbose=False,
        )

    def _save_summary_figure(
        self,
        *,
        field_pdfs: list[FieldPDF],
        figures_dir: pathlib.Path,
    ) -> None:
        """Combined overlay across every snapshot processed this run; always rebuilt fresh."""
        num_cols = field_pdfs[0].num_comps
        figure, panel_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=num_cols,
            panel_col_gap_pt=30.0,
        )
        if len(field_pdfs) == 1:
            self._plot_snapshot(
                panel_grid=panel_grid,
                field_pdf=field_pdfs[0],
                color="black",
            )
        else:
            self._plot_series(
                panel_grid=panel_grid,
                field_pdfs=field_pdfs,
            )
        self._style_panel_grid(
            panel_grid=panel_grid,
            comp_latex_labels=field_pdfs[0].comp_latex_labels,
            use_log10_bins=self.use_log10_bins,
        )
        data_tag = self._get_data_tag()
        figure_path = figures_dir / f"{data_tag}-pdfs-summary.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
            verbose=True,
        )

    def run(
        self,
    ) -> None:
        compute_pdfs_pipeline = ComputePDFs(
            snapshot_dirs=self.snapshot_dirs,
            snapshot_tag=self.snapshot_tag,
            registered_field=self.registered_field,
            index_width=self.index_width,
            comps_to_plot=self.comps_to_plot,
            num_bins=self.num_bins,
            save_data=self.save_data,
            data_dir=self.data_dir,
            overwrite=self.overwrite,
            use_log10_bins=self.use_log10_bins,
            amr_level=self.amr_level,
        )
        field_pdfs = compute_pdfs_pipeline.run()
        if field_pdfs and self.save_figure:
            data_tag = self._get_data_tag()
            for field_pdf in field_pdfs:
                padded_step_index_string = field_pdf.step_index.get_padded_string(index_width=self.index_width)
                figure_path = self.figures_dir / f"{data_tag}-pdf-index={padded_step_index_string}.png"
                if self.overwrite or not figure_path.exists():
                    self._save_snapshot_figure(
                        field_pdf=field_pdf,
                        figure_path=figure_path,
                    )
            self._save_summary_figure(
                field_pdfs=field_pdfs,
                figures_dir=self.figures_dir,
            )


## } MODULE
