## { MODULE

##
## === CONSTANTS
##

_AXES = ("x", "y", "z")

##
## === VALUE FORMATTING
##


def format_value(
    value: object,
) -> str:
    """Render one scalar/list/tuple value the way AMReX's `sim_params.toml` files are hand-written."""
    if isinstance(value, bool):
        ## AMReX booleans are written as bare 0/1, never Python's True/False
        return "1" if value else "0"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(elem) for elem in value) + "]"
    return str(value)


def render_key_value(
    *,
    key: str,
    value: object,
) -> str:
    """Render one `key = value` line."""
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
        per_axis = value
    else:
        per_axis = (value, value, value)
    return [
        render_key_value(key=f"{key_prefix}_{axis}", value=per_axis[axis_index])
        for axis_index, axis in enumerate(_AXES)
    ]


##
## === SECTION RENDERING
##


def render_section(
    *,
    title: str,
    lines: list[str],
) -> str:
    """Render one `## <title>` section from already-formatted `key = value` lines."""
    return "\n".join([f"## {title}", *lines])


def render_sections(
    sections: list[tuple[str, list[str]]],
) -> str:
    """Join `(title, lines)` sections in order, blank-line separated, trailing newline."""
    rendered = [render_section(title=title, lines=lines) for title, lines in sections]
    return "\n\n".join(rendered) + "\n"


## } MODULE
