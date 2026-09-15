## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
import dataclasses
import pathlib

## third-party
import numpy

## personal
from jormi.ww_arrays import compute_array_stats
from jormi.ww_fields import cartesian_axes
from jormi.ww_io import manage_log
from jormi.ww_plots import latex_labels

## local
from ww_quokka_sims.sim_io.field_diagnostics import pdfs, slices
from ww_quokka_sims.sim_io.snapshots import find_snapshots, load_snapshot

##
## === FIELD PROCESSING
##


@dataclasses.dataclass(frozen=True)
class LabelledLog10ScaledDivB:
    sim_time: float
    step_index: find_snapshots.StepIndex
    log10_scaled_div_b: numpy.ndarray


def load_log10_scaled_div_b(
    *,
    snapshot_dir: pathlib.Path,
    snapshot_tag: str,
) -> LabelledLog10ScaledDivB:
    """
    Load `log10(dx * |div(b)|)` pooled over every leaf cell in the AMR hierarchy.

    Unlike a covering-grid-based read, this never resamples onto one resolution: `dx` is
    each leaf cell's own native cell width (see `load_amr_leaves`), so a coarse cell
    and a fine cell inside a refined region each contribute their own correct `dx`, not one
    shared across the whole domain. Not normalised by `|b|`: since div(b) is only preserved
    to roundoff (not a fixed truncation order), `|b|` decaying to its own roundoff floor
    (e.g. outside the field loop) makes a `/|b|` ratio divide noise by noise, producing
    spurious order-unity values with no relation to divergence preservation; `dx * |div(b)|`
    alone already removes the resolution-dependence a bare `|div(b)|` would have (its
    roundoff scales as `~epsilon * |b| / dx`, so the `dx` factor cancels that level
    dependence) without dividing by anything that can vanish.
    """
    step_index = find_snapshots.get_step_index(
        snapshot_dir=snapshot_dir,
        snapshot_tag=snapshot_tag,
    )
    with load_snapshot.QuokkaSnapshot(
            snapshot_dir=snapshot_dir,
            verbose=False,
    ) as snapshot:
        sim_time = snapshot.sim_time
        div_b_leaves = snapshot.load_3d_magnetic_divergence_amr_leaves()
    scaled_div_b = div_b_leaves.cell_width * numpy.abs(div_b_leaves.values)
    return LabelledLog10ScaledDivB(
        sim_time=sim_time,
        step_index=step_index,
        log10_scaled_div_b=compute_array_stats.compute_safe_log10(scaled_div_b),
    )


def bin_dimensionless_divb_pdf(
    *,
    labelled_values: LabelledLog10ScaledDivB,
    bin_centers: numpy.ndarray,
) -> pdfs.FieldPDF:
    """Bin one snapshot's `log10(dx * |div(b)|)` values against a fixed, shared `bin_centers`."""
    estimated_pdf = compute_array_stats.estimate_pdf(
        values=labelled_values.log10_scaled_div_b,
        bin_centers=bin_centers,
    )
    log10_densities = numpy.ma.log10(
        numpy.ma.masked_less_equal(
            x=estimated_pdf.densities,
            value=0.0,
        ),
    )
    return pdfs.FieldPDF(
        sim_time=labelled_values.sim_time,
        step_index=labelled_values.step_index,
        grouped_bin_centers=[estimated_pdf.bin_centers],
        grouped_densities=[log10_densities],
        comp_latex_labels=[latex_labels.LatexLabel(content=r"\log_{10}(\Delta x\,|\nabla\cdot\vec{b}|)")],
        use_log10_bins=False,
    )


def compute_dimensionless_divb_native_slice(
    *,
    snapshot_dir: pathlib.Path,
    snapshot_tag: str,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> slices.FieldSlice:
    """
    Compute a genuine AMR-native 2D slice of the raw, signed `div(b)`.

    Unlike a covering-grid-based slice, each pixel is read from whichever box actually
    covers it (see `load_native_slice_sarray`), so the refined region is shown at its own
    true resolution rather than resampled to match the rest of the plane. Deliberately not
    scaled by `dx` or `|b|` (unlike `compute_dimensionless_divb_pdf`): keeping the raw,
    signed value lets a reader see both its absolute scale directly off the colorbar and
    its sign noise (evidence it is roundoff, not a systematic drift) at a glance.
    """
    step_index = find_snapshots.get_step_index(
        snapshot_dir=snapshot_dir,
        snapshot_tag=snapshot_tag,
    )
    with load_snapshot.QuokkaSnapshot(
            snapshot_dir=snapshot_dir,
            verbose=False,
    ) as snapshot:
        sim_time = snapshot.sim_time
        div_b_2d, _ = snapshot.load_3d_magnetic_divergence_native_slice(axis_to_slice=axis_to_slice)
        domain_bounds = snapshot.load_3d_uniform_domain(amr_level=0).domain_bounds
        plane_axes = [axis for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER if axis != axis_to_slice]
        axis_bounds: slices.AxisBounds = (
            domain_bounds[cartesian_axes.get_axis_index(plane_axes[0])],
            domain_bounds[cartesian_axes.get_axis_index(plane_axes[1])],
        )
    ## `load_native_slice_sarray` returns [row=height_axis, col=width_axis] ("ij"); the
    ## `FieldSlice`/`plot_2d_array` convention this feeds into is [x, y] ("xy")
    div_b_2d = div_b_2d.T
    return slices.FieldSlice(
        sarray_2d=div_b_2d,
        axis_bounds=axis_bounds,
        min_value=float(div_b_2d.min()),
        max_value=float(div_b_2d.max()),
        comp_latex_label=latex_labels.LatexLabel(content=r"\nabla\cdot\vec{b}"),
        sim_time=sim_time,
        step_index=step_index,
    )


##
## === GENERATE
##


def generate_divb_pdfs(
    *,
    snapshot_dirs: list[pathlib.Path],
    snapshot_tag: str,
    index_width: int,
    data_dir: pathlib.Path,
    num_bins: int,
    overwrite: bool,
) -> None:
    """
    Save one PDF per snapshot, all binned against the same, shared `bin_centers`.

    Two passes: first load every snapshot's `log10(dx * |div(b)|)` values and track the
    global min/max across the whole series, then bin each snapshot against one shared set
    of `bin_centers` spanning that global range. A per-snapshot bin range (each snapshot
    auto-ranged to its own min/max) would place bins differently from one snapshot to the
    next, making the resulting PDFs impossible to compare directly across time.
    """
    labelled_values_by_snapshot = [
        load_log10_scaled_div_b(
            snapshot_dir=snapshot_dir,
            snapshot_tag=snapshot_tag,
        ) for snapshot_dir in snapshot_dirs
    ]
    ## a snapshot with div(b) exactly zero everywhere (e.g. before any evolution) has an
    ## all-NaN `log10_scaled_div_b`; exclude it rather than let it poison the global range
    finite_extrema = [
        (
            float(numpy.nanmin(labelled_values.log10_scaled_div_b)),
            float(numpy.nanmax(labelled_values.log10_scaled_div_b))
        )
        for labelled_values in labelled_values_by_snapshot
        if numpy.any(numpy.isfinite(labelled_values.log10_scaled_div_b))
    ]
    if not finite_extrema:
        raise ValueError("every snapshot's `log10(dx * |div(b)|)` was entirely non-finite.")
    global_min = min(extremum[0] for extremum in finite_extrema)
    global_max = max(extremum[1] for extremum in finite_extrema)
    bin_edges = numpy.linspace(global_min, global_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for labelled_values in labelled_values_by_snapshot:
        padded_index = labelled_values.step_index.get_padded_string(index_width=index_width)
        pdf_file_path = data_dir / f"divb_dimensionless-pdf-index={padded_index}.json"
        if pdf_file_path.exists() and not overwrite:
            manage_log.log_note(text=f"skipping (already exists): {pdf_file_path}")
            continue
        pdf_data = bin_dimensionless_divb_pdf(
            labelled_values=labelled_values,
            bin_centers=bin_centers,
        )
        pdf_data.save_to_file(file_path=pdf_file_path)
        manage_log.log_note(text=f"saved: {pdf_file_path}")


def generate_divb_native_slices(
    *,
    snapshot_dirs: list[pathlib.Path],
    snapshot_tag: str,
    index_width: int,
    data_dir: pathlib.Path,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
    overwrite: bool,
) -> None:
    for snapshot_dir in snapshot_dirs:
        step_index = find_snapshots.get_step_index(
            snapshot_dir=snapshot_dir,
            snapshot_tag=snapshot_tag,
        )
        padded_index = step_index.get_padded_string(index_width=index_width)
        slice_file_path = (data_dir / f"divb-slice={axis_to_slice.value}-index={padded_index}.npz")
        if slice_file_path.exists() and not overwrite:
            manage_log.log_note(text=f"skipping (already exists): {slice_file_path}")
            continue
        sliced_field = compute_dimensionless_divb_native_slice(
            snapshot_dir=snapshot_dir,
            snapshot_tag=snapshot_tag,
            axis_to_slice=axis_to_slice,
        )
        sliced_field.save_to_file(file_path=slice_file_path)
        manage_log.log_note(text=f"saved: {slice_file_path}")


def generate_divb_dimensionless(
    *,
    snapshot_dirs: list[pathlib.Path],
    snapshot_tag: str,
    index_width: int,
    data_dir: pathlib.Path,
    num_bins: int,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
    overwrite: bool,
) -> None:
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    generate_divb_pdfs(
        snapshot_dirs=snapshot_dirs,
        snapshot_tag=snapshot_tag,
        index_width=index_width,
        data_dir=data_dir,
        num_bins=num_bins,
        overwrite=overwrite,
    )
    generate_divb_native_slices(
        snapshot_dirs=snapshot_dirs,
        snapshot_tag=snapshot_tag,
        index_width=index_width,
        data_dir=data_dir,
        axis_to_slice=axis_to_slice,
        overwrite=overwrite,
    )


##
## === ENTRY POINT
##


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--snapshot-tag",
        type=str,
        default="plt",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--axis-to-slice",
        type=str,
        default="x_2",
        choices=["x_0", "x_1", "x_2"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    snapshot_dirs = find_snapshots.resolve_snapshot_dirs(
        input_dir=args.input_dir,
        snapshot_tag=args.snapshot_tag,
    )
    if not snapshot_dirs:
        raise ValueError(f"no snapshots found under: {args.input_dir}")
    index_width = find_snapshots.get_max_index_width(
        snapshot_dirs=snapshot_dirs,
        snapshot_tag=args.snapshot_tag,
    )
    generate_divb_dimensionless(
        snapshot_dirs=snapshot_dirs,
        snapshot_tag=args.snapshot_tag,
        axis_to_slice=cartesian_axes.CartesianAxis_3D(args.axis_to_slice),
        index_width=index_width,
        data_dir=args.data_dir,
        num_bins=args.num_bins,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

## } SCRIPT
