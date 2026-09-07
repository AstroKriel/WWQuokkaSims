## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import pathlib

## third-party
import numpy

## personal
from jormi import ww_lists
from jormi.ww_validation import validate_types

##
## === STEP INDEX
##


@dataclasses.dataclass(frozen=True)
class StepIndex:
    _string: str

    def __post_init__(
        self,
    ) -> None:
        if not self._string.isdigit():
            raise ValueError(f"expected a digit-only step-index string, got: {self._string!r}")

    def get_value(
        self,
    ) -> int:
        return int(self._string)

    def get_string(
        self,
    ) -> str:
        return self._string

    def get_padded_string(
        self,
        *,
        index_width: int,
    ) -> str:
        return f"{self.get_value():0{index_width}d}"


##
## === FUNCTIONS
##


def looks_like_boxlib_dir(
    *,
    snapshot_dir: pathlib.Path,
) -> bool:
    """Return `True` iff `snapshot_dir` contains a boxlib `Header` file and `Level_0` subdirectory."""
    validate_types.ensure_type(
        param=snapshot_dir,
        valid_types=pathlib.Path,
    )
    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        return False
    has_header = (snapshot_dir / "Header").is_file()
    has_level0 = (snapshot_dir / "Level_0").is_dir()
    return has_header and has_level0


def get_step_index(
    *,
    snapshot_dir: pathlib.Path,
    snapshot_tag: str,
) -> StepIndex:
    """Extract the step index from a snapshot directory named `<snapshot_tag><step_index_string>`."""
    snapshot_name = snapshot_dir.name
    if snapshot_tag not in snapshot_name:
        raise ValueError(f"snapshot tag `{snapshot_tag}` not found in snapshot name `{snapshot_name}`.")
    name_parts = snapshot_name.split(snapshot_tag)
    if len(name_parts) < 2:
        raise ValueError(f"unexpected format for snapshot name: {snapshot_name}.")
    digits_string = name_parts[1].split(".")[0]
    if not digits_string.isdigit():
        raise ValueError(f"expected digits after `{snapshot_tag}` in snapshot name {snapshot_name}.")
    return StepIndex(digits_string)


def get_latest_snapshot_dirs(
    *,
    sim_dir: pathlib.Path,
    snapshot_tag: str,
) -> list[pathlib.Path]:
    """Return all snapshot directories under `sim_dir` matching `snapshot_tag`; sorted by ascending step index."""
    snapshot_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and (snapshot_tag in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    snapshot_dirs.sort(
        key=lambda snapshot_dir: get_step_index(
            snapshot_dir=snapshot_dir,
            snapshot_tag=snapshot_tag,
        ).get_value(),
    )
    return snapshot_dirs


def resolve_snapshot_dirs(
    *,
    input_dir: pathlib.Path,
    snapshot_tag: str,
    max_elems: int | None = None,
) -> list[pathlib.Path]:
    """
    Resolve `input_dir` to an ordered list of snapshot directories.

    Returns `[input_dir]` directly if it is itself a snapshot directory; otherwise scans for all
    `snapshot_tag`-matched directories under `input_dir` and subsamples to `max_elems` if provided.
    Returns an empty list if none are found.
    """
    if (snapshot_tag in input_dir.name) or looks_like_boxlib_dir(snapshot_dir=input_dir):
        return [input_dir]
    snapshot_dirs = get_latest_snapshot_dirs(
        sim_dir=input_dir,
        snapshot_tag=snapshot_tag,
    )
    if max_elems is not None:
        snapshot_dirs = ww_lists.sample_list(
            elems=snapshot_dirs,
            max_elems=max_elems,
        )
    return snapshot_dirs


def get_max_index_width(
    *,
    snapshot_dirs: list[pathlib.Path],
    snapshot_tag: str,
) -> int:
    """Return the character width of the widest step-index string across `snapshot_dirs`."""
    if not snapshot_dirs:
        raise ValueError("`snapshot_dirs` must be non-empty.")
    index_widths: list[int] = []
    for snapshot_dir in snapshot_dirs:
        step_index = get_step_index(
            snapshot_dir=snapshot_dir,
            snapshot_tag=snapshot_tag,
        )
        index_widths.append(len(step_index.get_string()))
    return max(index_widths)


def find_npz_near_time(
    *,
    extracted_dir: pathlib.Path,
    glob_pattern: str,
    target_time: float,
) -> pathlib.Path:
    """Return the `extracted_dir` file matching `glob_pattern` whose `sim_time` is nearest `target_time`."""
    npz_paths = sorted(extracted_dir.glob(glob_pattern))
    if not npz_paths:
        raise FileNotFoundError(f"no file matching `{glob_pattern}` found in: {extracted_dir}.")
    return min(
        npz_paths,
        key=lambda npz_path: abs(float(numpy.load(npz_path)["sim_time"]) - target_time),
    )


## } MODULE
