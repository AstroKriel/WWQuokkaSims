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
    *,
    snapshot_dir: Path,
) -> bool:
    """Return `True` iff `snapshot_dir` contains a boxlib `Header` file and `Level_0` subdirectory."""
    validate_types.ensure_type(
        param=snapshot_dir,
        valid_types=Path,
    )
    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        return False
    has_header = (snapshot_dir / "Header").is_file()
    has_level0 = (snapshot_dir / "Level_0").is_dir()
    return has_header and has_level0


def get_step_index_string(
    *,
    snapshot_dir: Path,
    snapshot_tag: str,
) -> str:
    """Extract the step-index string from a snapshot directory named `<snapshot_tag><step_index_string>`."""
    snapshot_name = snapshot_dir.name
    if snapshot_tag not in snapshot_name:
        raise ValueError(f"snapshot tag `{snapshot_tag}` not found in snapshot name `{snapshot_name}`.")
    name_parts = snapshot_name.split(snapshot_tag)
    if len(name_parts) < 2:
        raise ValueError(f"unexpected format for snapshot name: {snapshot_name}.")
    digits_string = name_parts[1].split(".")[0]
    if not digits_string.isdigit():
        raise ValueError(f"expected digits after `{snapshot_tag}` in snapshot name {snapshot_name}.")
    return digits_string


def get_latest_snapshot_dirs(
    *,
    sim_dir: Path,
    snapshot_tag: str,
) -> list[Path]:
    """Return all snapshot directories under `sim_dir` matching `snapshot_tag`; sorted by ascending step index."""
    snapshot_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and (snapshot_tag in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    snapshot_dirs.sort(
        key=lambda snapshot_dir: int(
            get_step_index_string(
                snapshot_dir=snapshot_dir,
                snapshot_tag=snapshot_tag,
            ),
        ),
    )
    return snapshot_dirs


def resolve_snapshot_dirs(
    *,
    input_dir: Path,
    snapshot_tag: str,
    max_elems: int | None = None,
) -> list[Path]:
    """
    Resolve `input_dir` to an ordered list of snapshot directories.

    Returns `[input_dir]` directly if it is itself a snapshot directory; otherwise scans for all
    `snapshot_tag`-matched directories under `input_dir` and subsamples to `max_elems` if provided.
    """
    if (snapshot_tag in input_dir.name) or looks_like_boxlib_dir(snapshot_dir=input_dir):
        return [input_dir]
    snapshot_dirs = get_latest_snapshot_dirs(
        sim_dir=input_dir,
        snapshot_tag=snapshot_tag,
    )
    if not snapshot_dirs:
        raise ValueError(
            f"no snapshot directories found using tag `{snapshot_tag}`; searched in {input_dir}.",
        )
    if max_elems is not None:
        snapshot_dirs = ww_lists.sample_list(
            elems=snapshot_dirs,
            max_elems=max_elems,
        )
    return snapshot_dirs


def get_max_index_width(
    *,
    snapshot_dirs: list[Path],
    snapshot_tag: str,
) -> int:
    """Return the character width of the widest step-index string across `snapshot_dirs`."""
    if not snapshot_dirs:
        raise ValueError("`snapshot_dirs` must be non-empty.")
    index_widths: list[int] = []
    for snapshot_dir in snapshot_dirs:
        step_index_string = get_step_index_string(
            snapshot_dir=snapshot_dir,
            snapshot_tag=snapshot_tag,
        )
        index_widths.append(len(step_index_string))
    return max(index_widths)


## } MODULE
