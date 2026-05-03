## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## personal
from jormi import ww_lists
from jormi.ww_validation import validate_types

##
## === FUNCTIONS
##


def looks_like_boxlib_dir(
    dataset_dir: Path,
) -> bool:
    """Return `True` iff `dataset_dir` contains a boxlib `Header` file and `Level_0` subdirectory."""
    validate_types.ensure_type(
        param=dataset_dir,
        valid_types=Path,
    )
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return False
    has_header = (dataset_dir / "Header").is_file()
    has_level0 = (dataset_dir / "Level_0").is_dir()
    return has_header and has_level0


def get_snapshot_index_string(
    dataset_dir: Path,
    dataset_tag: str,
) -> str:
    """Extract the index string from a snapshot directory named `<dataset_tag><index_string>`."""
    dataset_name = dataset_dir.name
    if dataset_tag not in dataset_name:
        raise ValueError(f"Dataset tag `{dataset_tag}` was not found in `{dataset_name}`.")
    name_parts = dataset_name.split(dataset_tag)
    if len(name_parts) < 2:
        raise ValueError(f"Unexpected dataset name format: {dataset_name}")
    digits_string = name_parts[1].split(".")[0]
    if not digits_string.isdigit():
        raise ValueError(f"Expected digits after `{dataset_tag}` in {dataset_name}")
    return digits_string


def get_latest_snapshot_dirs(
    sim_dir: Path,
    dataset_tag: str,
) -> list[Path]:
    """Return all snapshot directories under `sim_dir` matching `dataset_tag`; sorted by ascending index."""
    dataset_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and (dataset_tag in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    dataset_dirs.sort(
        key=lambda dataset_dir: int(
            get_snapshot_index_string(
                dataset_dir,
                dataset_tag,
            ),
        ),
    )
    return dataset_dirs


def resolve_snapshot_dirs(
    input_dir: Path,
    dataset_tag: str,
    max_elems: int | None = None,
) -> list[Path]:
    """
    Resolve `input_dir` to an ordered list of snapshot directories.

    Returns `[input_dir]` directly if it is itself a snapshot directory; otherwise scans for all
    `dataset_tag`-matched directories under `input_dir` and subsamples to `max_elems` if provided.
    """
    if (dataset_tag in input_dir.name) or looks_like_boxlib_dir(input_dir):
        return [input_dir]
    dataset_dirs = get_latest_snapshot_dirs(
        sim_dir=input_dir,
        dataset_tag=dataset_tag,
    )
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found using tag `{dataset_tag}` under: {input_dir}")
    if max_elems is not None:
        dataset_dirs = ww_lists.sample_list(
            elems=dataset_dirs,
            max_elems=max_elems,
        )
    return dataset_dirs


def get_max_index_width(
    dataset_dirs: list[Path],
    dataset_tag: str,
) -> int:
    """Return the character width of the widest index string across `dataset_dirs`."""
    if not dataset_dirs:
        return 1
    index_widths: list[int] = []
    for dataset_dir in dataset_dirs:
        dataset_index_string = get_snapshot_index_string(
            dataset_dir=dataset_dir,
            dataset_tag=dataset_tag,
        )
        index_widths.append(
            len(
                dataset_index_string,
            ),
        )
    return max(index_widths) if len(index_widths) > 0 else 1


## } MODULE
