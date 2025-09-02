import os
import sys
import numpy
from pathlib import Path

from yt.loaders import load as yt_load
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, plot_data
from jormi.parallelism import independent_tasks

DEFAULT_DATA_DIR = Path(
    "/scratch/jh2/nk7952/quokka/sims/weak/new_scheme/OrszagTang/cfl=0.3_rk=2_ro=5_ld04/N=160_Nbo=32_Nbl=32_bopr=1_mpir=144",
).expanduser()

FIELD_NAME = ("boxlib", "x-BField")
SLICE_AXIS = 2  # 0:x, 1:y, 2:z
available_procs = (os.cpu_count() or 1)
capped_procs = min(available_procs, 20)
NUM_PROCS = max(1, capped_procs - 1)
ONLY_ANIMATE = True
USE_TEX = False


def find_data_paths(directory: Path) -> list[Path]:
    data_paths = [
        dir for dir in directory.iterdir() if all([
            dir.is_dir(),
            "plt" in dir.name,
            "old" not in dir.name,
        ])
    ]
    data_paths.sort(key=lambda dir: int(dir.name.split("plt")[1]))
    return data_paths


def get_mid_slice(
    arr: numpy.ndarray,
    axis: int,
) -> numpy.ndarray:
    return numpy.take(arr, arr.shape[axis] // 2, axis=axis)


def worker_extract_slice(
    plotfile_path: str,
    npy_out: str,
) -> tuple[float, float, float]:
    ## force a headless backend per process
    import matplotlib
    matplotlib.use("Agg", force=True)
    ds = yt_load(plotfile_path)
    sim_time = float(ds.current_time)
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    data3d = numpy.asarray(cg[FIELD_NAME], dtype=numpy.float32)
    slice2d = get_mid_slice(data3d, SLICE_AXIS)
    numpy.save(npy_out, slice2d)
    try:
        ds.close()
    except Exception:
        pass
    return sim_time, float(slice2d.min()), float(slice2d.max())


def worker_render_frame(
    npy_path: str,
    frame_png: str,
    title: str,
    vmin: float,
    vmax: float,
    slice_plane: str,
    use_tex: bool = False,
) -> str:
    import matplotlib
    matplotlib.use("Agg", force=True)
    if use_tex:
        ## isolate latex cache per process
        import os, tempfile
        os.environ["TEXMFOUTPUT"] = tempfile.mkdtemp(prefix="mpltex_")
        matplotlib.rcParams["text.usetex"] = True
    else:
        matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    field_slice = numpy.load(npy_path, mmap_mode="r")
    fig, ax = plot_manager.create_figure(fig_scale=1.25)
    plot_data.plot_sfield_slice(
        ax=ax,
        field_slice=field_slice,
        axis_bounds=(-1, 1, -1, 1),
        cbar_bounds=(vmin, vmax),
        cmap_name="cmr.iceburn",
        add_colorbar=True,
        cbar_label="",
        cbar_side="right",
    )
    ax.set_title(title)
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(str(v) for v in ticks)
    ax.set_yticklabels(str(v) for v in ticks)
    ax.set_xlabel(f"{slice_plane[0]} axis")
    ax.set_ylabel(f"{slice_plane[1]} axis")
    fig.savefig(frame_png, dpi=150)
    plt.close(fig)
    return frame_png


def build_frames_with_parallel(data_dir: Path):
    print(f"Looking at: {data_dir}")
    output_dir = data_dir / "frames"
    tmp_dir = output_dir / "_tmp_slices"
    io_manager.init_directory(output_dir)
    io_manager.init_directory(tmp_dir)
    data_paths = find_data_paths(data_dir)
    if not data_paths:
        raise SystemExit(f"No data_paths found in {data_dir}")
    slice_plane = ["yz", "xz", "xy"][SLICE_AXIS]
    print(f"Slice plane: {slice_plane}")
    frame_pngs = [output_dir / f"frame_{i:05d}_{slice_plane}_plane.png" for i in range(len(data_paths))]
    have_all_frames = all(dir.exists() for dir in frame_pngs)
    if not ONLY_ANIMATE or not have_all_frames:
        slice_npys = [tmp_dir / f"slice_{i:05d}.npy" for i in range(len(data_paths))]
        extract_args = [(str(dir), str(npy)) for dir, npy in zip(data_paths, slice_npys)]
        print(f"[Phase 1] Extracting slices...")
        extract_results = independent_tasks.run_in_parallel(
            func=worker_extract_slice,
            grouped_args=extract_args,
            timeout_seconds=90,
            show_progress=True,
        )
        sim_times = [t for (t, _, _) in extract_results]
        local_mins = [mn for (_, mn, _) in extract_results]
        local_maxs = [mx for (_, _, mx) in extract_results]
        vmin, vmax = float(min(local_mins)), float(max(local_maxs))
        print(f"Global color limits: vmin={vmin:.6g}, vmax={vmax:.6g}")
        titles = [f"{dir.name}: t = {t:.3f}" for dir, t in zip(data_paths, sim_times)]
        render_args = [
            (str(npy), str(png), title, vmin, vmax, slice_plane, USE_TEX)
            for npy, png, title in zip(slice_npys, frame_pngs, titles)
        ]
        print(f"[Phase 2] Rendering frames...")
        _ = independent_tasks.run_in_parallel(
            func=worker_render_frame,
            grouped_args=render_args,
            timeout_seconds=90,
            show_progress=True,
            enable_plotting=True,
        )
        print(f"Frames saved under: {output_dir}")
    mp4_path = output_dir / f"animated_{slice_plane}_plane.mp4"
    print(f"[Phase 3] Writing MP4...")
    plot_manager.animate_pngs_to_mp4(
        frames_dir=output_dir,
        mp4_path=mp4_path,
        pattern=f"frame_%05d_{slice_plane}_plane.png",
        fps=30,
    )


def main():
    data_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    if not data_dir.exists():
        print(f"Error: data directory does not exist: {data_dir}")
        sys.exit(1)
    build_frames_with_parallel(data_dir)


if __name__ == "__main__":
    main()
