import yt
import numpy
from pathlib import Path
from matplotlib import pyplot as mpl_plt

def get_field_data(ds, field_name):
  ds_data = ds.covering_grid(
    level     = 0,
    left_edge = ds.domain_left_edge,
    dims      = ds.domain_dimensions
  )
  return numpy.squeeze(ds_data[("boxlib", field_name)])

directory = "/Users/necoturb/Documents/Codes/quokka/tests"
file_name = "plt00000"
file_path = Path(directory).resolve() / file_name

ds = yt.load(file_path)
# print(ds.field_list)

field_names = [
  "x-BField",
  "y-BField",
  "z-BField",
]

fig, axs = mpl_plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
for field_index, field_name in enumerate(field_names):
  bi_field = get_field_data(ds, field_name)
  print(bi_field.shape)
  axs[field_index].imshow(
    bi_field[:, :, 0].T,
    extent = [0, 1, 0, 1],
    origin = "lower",
    aspect = "auto",
)
mpl_plt.tight_layout()
mpl_plt.show()

# for field_index, field in enumerate(fields):
#   slc = yt.SlicePlot(ds, "z", field, origin="native")
#   slc.set_log(field, False)
#   slc.set_cmap(field, "viridis")
#   slc.set_zlim(field, -1e-6, 1e-6)
#   plot = slc.plots[field]
#   plot.render(figure=fig, axes=axs[field_index])
#   axs[field_index].set_title(field[1].upper())
#   axs[field_index].set_xlabel("X")
#   axs[field_index].set_ylabel("Y")
