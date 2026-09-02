## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_validation import validate_types

##
## === CONSTANTS
##

_AXES = ("x", "y", "z")
_FORMATTABLE_TYPES = (bool, str, int, float, list, tuple)

##
## === VALUE FORMATTING
##


def format_value(
    value: object,
) -> str:
    """Format one scalar/list/tuple value the way AMReX's `sim_params.toml` files are hand-written."""
    validate_types.ensure_type(
        param=value,
        param_name="<value>",
        valid_types=_FORMATTABLE_TYPES,
    )
    if isinstance(value, bool):
        ## AMReX booleans are written as bare 0/1, never Python's True/False
        return "1" if value else "0"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(elem) for elem in value) + "]"
    return str(value)


def format_key_value(
    *,
    key: str,
    value: object,
) -> str:
    """Format one `key = value` line."""
    return f"{key} = {format_value(value)}"


##
## === PER-AXIS EXPANSION
##


def expand_per_axis(
    value: int | tuple[int, int, int],
    *,
    key_prefix: str,
) -> list[str]:
    """
    Expand a scalar-or-per-axis `value` into three explicit `<key_prefix>_x/_y/_z` lines.

    Never emits a bare `<key_prefix> = ...` fallback key: AMReX interprets that array
    form as one value per AMR *level*, not per axis, which silently produces the wrong
    grid decomposition at any level other than 0.
    """
    if isinstance(value, tuple):
        validate_types.ensure_tuple_of_ints(
            param=value,
            param_name="<value>",
            seq_length=3,
        )
        per_axis = value
    else:
        validate_types.ensure_finite_int(
            param=value,
            param_name="<value>",
            require_positive=True,
            allow_zero=False,
        )
        per_axis = (value, value, value)
    return [
        format_key_value(
            key=f"{key_prefix}_{axis}",
            value=per_axis[axis_index],
        ) for axis_index, axis in enumerate(_AXES)
    ]


##
## === PARAM GROUP FORMATTING
##


def format_param_group(
    *,
    group_title: str,
    assignment_lines: list[str],
) -> str:
    """Format one `## <group_title>` param group from already-formatted `key = value` assignment lines."""
    return "\n".join([f"## {group_title}", *assignment_lines])


def format_param_groups(
    param_groups: list[tuple[str, list[str]]],
) -> str:
    """Join `(group_title, assignment_lines)` param groups in order, blank-line separated, trailing newline."""
    formatted = [
        format_param_group(
            group_title=group_title,
            assignment_lines=assignment_lines,
        ) for group_title, assignment_lines in param_groups
    ]
    return "\n\n".join(formatted) + "\n"


## } MODULE
