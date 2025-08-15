import numpy
import matplotlib.pyplot as mpl_plot
import imageio.v3 as iio
from pathlib import Path
from yt.loaders import load as yt_load
from jormi.ww_plots import plot_manager, plot_data

DATA_DIR = Path(
  "/Users/necoturb/Documents/Codes/asgard/mimir/kriel_2025_quokka_mhd/sims/weak/new_scheme/OrszagTang/cfl=0.3_rk=2_ro=2_ld04/N=64_Nbo=32_Nbl=32_bopr=1_mpir=8"
).expanduser()
FIELD_NAME = ("boxlib", "x-BField")
SLICE_AXIS = 2 # 0:x, 1:y, 2:z
OUTPUT_DIR = DATA_DIR / "frames"
ONLY_ANIMATE = False

def find_data_paths(directory: Path):
  """Return sorted list of plotfile directories."""
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
  OUTPUT_DIR.mkdir(exist_ok=True)
  data_paths = find_data_paths(DATA_DIR)
  slice_plane = ["yz", "xz", "xy"][SLICE_AXIS]
  if not data_paths:
    raise SystemExit(f"No data_paths found in {DATA_DIR}")
  all_min, all_max = numpy.inf, -numpy.inf
  if not ONLY_ANIMATE:
    for data_path in data_paths:
      ds = yt_load(str(data_path))
      grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
      data = numpy.asarray(grid[FIELD_NAME])
      slice_data = get_slice(data, SLICE_AXIS)
      all_min = min(all_min, slice_data.min())
      all_max = max(all_max, slice_data.max())
  frame_paths = []
  for plot_index, data_path in enumerate(data_paths):
    frame_path = OUTPUT_DIR / f"frame_{plot_index:05d}_{slice_plane}_plane.png"
    if not ONLY_ANIMATE:
      ds = yt_load(str(data_path))
      grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
      data = numpy.asarray(grid[FIELD_NAME])
      slice_data = get_slice(data, SLICE_AXIS)
      fig, ax = plot_manager.create_figure(fig_scale=1.25)
      ax = plot_manager.cast_to_axis(ax)
      plot_data.plot_sfield_slice(
        ax           = ax,
        field_slice  = slice_data,
        axis_bounds  = (-1, 1, -1, 1),
        cbar_bounds  = (all_min, all_max),
        cmap_name    = "cmr.iceburn",
        add_colorbar = True,
        cbar_label   = "",
        cbar_side    = "right",
      )
      ax.set_title(f"{data_path.name}: t = {float(ds.current_time):.3f}")
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
  if not ONLY_ANIMATE: print("\nFrames saved under :", OUTPUT_DIR)
  gif_path = OUTPUT_DIR / f"animated_{slice_plane}_plane.png"
  iio.imwrite(gif_path, frame_paths, fps=30)
  print(f"\nGIF saved to {gif_path}")

if __name__ == "__main__":
  main()
