import yt
import yt
import numpy
from pathlib import Path

Field = tuple[str, str]


def load_plt_file(file_path: Path, show_fields: bool = False) -> yt.Dataset:
    """
  Load an AMReX/BoxLib plotfile with yt.

  Args:
    file_path: Path to the plotfile directory (e.g. 'plt0042').
    show_fields: If True, print available field names.

  Returns:
    A yt Dataset object.
  """
    ds = yt.load(str(file_path))
    if show_fields:
        print("Available fields:")
        for ftype, fname in ds.field_list:
            print(f"\t- ({ftype}, {fname})")
    return ds


def extract_data_array(
    ds: yt.Dataset,
    field: str | Field,
    level: int = 0,
) -> numpy.ndarray:
    """
  Extract a field from the dataset as a NumPy array at a given AMR level.
  """
    cg = _covering_grid(ds, level)
    f = resolve_field_name(ds, field)
    return numpy.asarray(cg[f])


def _covering_grid(
    ds: yt.Dataset,
    level: int = 0,
):
    """Return a uniform covering grid for the given AMR level."""
    return ds.covering_grid(
        level=level,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions * (2**level),
    )


def resolve_field_name(ds: yt.Dataset, field: str | Field) -> Field:
    """
  Resolve a field string or tuple into a concrete (type, name) tuple.
  """
    if isinstance(field, tuple):
        return field
    if ":" in field:
        parts = field.split(":", 1)
        tup = (parts[0], parts[1])
        if tup in ds.field_list:
            return tup
    for f in ds.field_list:
        if f[1] == field:
            return f
    raise KeyError(f"Field `{field}` was not found in the dataset.")


## .
