## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import domain_models

##
## === FUNCTIONS
##


def get_domain_file_name(
    *,
    amr_level: int,
) -> str:
    """Deterministic file name for cached domain metadata at `amr_level`."""
    return f"domain-amr_level={amr_level}.npz"


def save_domain(
    *,
    uniform_domain_3d: domain_models.UniformDomain_3D,
    amr_level: int,
    data_dir: Path,
    overwrite: bool = False,
) -> None:
    """Save `uniform_domain_3d` once per `(data_dir, amr_level)`; skip if already saved unless `overwrite`."""
    file_path = data_dir / get_domain_file_name(amr_level=amr_level)
    if (not overwrite) and file_path.exists():
        return
    numpy.savez(
        file_path,
        periodicity=numpy.array(uniform_domain_3d.periodicity),
        resolution=numpy.array(uniform_domain_3d.resolution),
        domain_bounds=numpy.array(uniform_domain_3d.domain_bounds),
    )


def load_domain(
    *,
    data_dir: Path,
    amr_level: int,
) -> domain_models.UniformDomain_3D:
    """
    Load cached domain metadata for `amr_level` from `data_dir`.

    Raises if no cache file is present; there is no fallback reconstruction, so data extracted
    before this cache existed must be re-extracted with `quokka-extract-dataset`.
    """
    file_path = data_dir / get_domain_file_name(amr_level=amr_level)
    if not file_path.exists():
        raise FileNotFoundError(
            f"no cached domain metadata for amr_level={amr_level} in {data_dir}; "
            "re-run `quokka-extract-dataset` for this amr_level to generate it.",
        )
    with numpy.load(file_path) as domain_npz:
        periodicity_array = domain_npz["periodicity"]
        resolution_array = domain_npz["resolution"]
        domain_bounds_array = domain_npz["domain_bounds"]
    return domain_models.UniformDomain_3D(
        periodicity=tuple(bool(flag) for flag in periodicity_array),
        resolution=tuple(int(num_cells) for num_cells in resolution_array),
        domain_bounds=tuple((float(lower), float(upper)) for lower, upper in domain_bounds_array),
    )


## } MODULE
