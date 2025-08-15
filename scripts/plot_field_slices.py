import sys
import numpy
import imageio.v3 as iio
import matplotlib.pyplot as mpl_plot
from pathlib import Path
from yt.loaders import load as yt_load
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, plot_data

DEFAULT_DATA_DIR = Path(
  # "/scratch/jh2/nk7952/quokka/sims/weak/new_scheme/OrszagTang/cfl=0.3_rk=2_ro=5_ld04/N=32_Nbo=32_Nbl=32_bopr=1_mpir=1"
  # "/scratch/jh2/nk7952/quokka/sims/weak/new_scheme/OrszagTang/cfl=0.3_rk=2_ro=5_ld04/N=64_Nbo=32_Nbl=32_bopr=1_mpir=8"
  # "/scratch/jh2/nk7952/quokka/sims/weak/new_scheme/OrszagTang/cfl=0.3_rk=2_ro=5_ld04/N=96_Nbo=32_Nbl=32_bopr=1_mpir=27"
  # "/scratch/jh2/nk7952/quokka/sims/weak/new_scheme/OrszagTang/cfl=0.3_rk=2_ro=5_ld04/N=128_Nbo=32_Nbl=32_bopr=1_mpir=96"
  "/scratch/jh2/nk7952/quokka/sims/weak/new_scheme/OrszagTang/cfl=0.3_rk=2_ro=5_ld04/N=160_Nbo=32_Nbl=32_bopr=1_mpir=144"
).expanduser()
FIELD_NAME = ("boxlib", "x-BField")
SLICE_AXIS = 2 # 0:x, 1:y, 2:z
ONLY_ANIMATE = False

def find_data_paths(directory: Path):
  data_paths = [
    file_path
    for file_path in directory.iterdir()
    if all([
      file_path.is_dir(),
      "plt" in file_path.name,
      "old" not in file_path.name,
    ])
  ]
  data_paths.sort(key=lambda p: int(p.name.split("plt")[1]))
  return data_paths

def get_slice(array, axis):
  position = array.shape[axis] // 2
  return numpy.take(array, position, axis=axis)

def main():
  if len(sys.argv) > 1:
    data_dir = Path(sys.argv[1]).expanduser()
  else:
    data_dir = DEFAULT_DATA_DIR
  if not data_dir.exists():
    print(f"Error: data directory does not exist: {data_dir}")
    sys.exit(1)
  print(f"Looking at: {data_dir}")
  output_dir = data_dir / "frames"
  io_manager.init_directory(output_dir)
  data_paths = find_data_paths(data_dir)
  if not data_paths:
    raise SystemExit(f"No data_paths found in {data_dir}")
  slice_plane = ["yz", "xz", "xy"][SLICE_AXIS]
  print(f"Plotting slice through the {slice_plane}-plane.")
  data_slices = []
  sim_times = []
  if not ONLY_ANIMATE:
    for data_path in data_paths:
      ds = yt_load(str(data_path))
      print(f"Read in: {data_path}")
      sim_time = ds.current_time
      grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
      data = numpy.asarray(grid[FIELD_NAME])
      data_slice = get_slice(data, SLICE_AXIS)
      data_slices.append(data_slice)
      sim_times.append(sim_time)
    min_value = numpy.min(data_slices)
    max_value = numpy.max(data_slices)
  frame_paths = []
  for plot_index, data_path in enumerate(data_paths):
    frame_path = output_dir / f"frame_{plot_index:05d}_{slice_plane}_plane.png"
    if not ONLY_ANIMATE:
      fig, ax = plot_manager.create_figure(fig_scale=1.25)
      ax = plot_manager.cast_to_axis(ax)
      data_slice = data_slices[plot_index]
      sim_time = sim_times[plot_index]
      plot_data.plot_sfield_slice(
        ax           = ax,
        field_slice  = data_slice,
        axis_bounds  = (-1, 1, -1, 1),
        cbar_bounds  = (min_value, max_value),
        cmap_name    = "cmr.iceburn",
        add_colorbar = True,
        cbar_label   = "",
        cbar_side    = "right",
      )
      ax.set_title(f"{data_path.name}: t = {float(sim_time):.3f}")
      tick_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
      ax.set_xticks(tick_values)
      ax.set_yticks(tick_values)
      ax.set_xticklabels(str(val) for val in tick_values)
      ax.set_yticklabels(str(val) for val in tick_values)
      ax.set_xlabel(f"{slice_plane[0]} axis")
      ax.set_ylabel(f"{slice_plane[1]} axis")
      fig.savefig(frame_path, dpi=150)
      mpl_plot.close(fig)
      print(f"Saved {frame_path}")
    frame_paths.append(iio.imread(frame_path))
  if not ONLY_ANIMATE: print("\nFrames saved under :", output_dir)
  gif_path = output_dir / f"animated_{slice_plane}_plane.gif"
  iio.imwrite(gif_path, frame_paths, fps=30)
  print(f"\nGIF saved to {gif_path}")

if __name__ == "__main__":
  main()
