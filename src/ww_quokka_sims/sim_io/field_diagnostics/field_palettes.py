## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_plots import add_color

##
## === DEFAULT PALETTES
##

SEQUENTIAL_PALETTE_NAME = "cmr.lavender"
DIVERGING_PALETTE_NAME = "cmr.iceburn"

##
## === PALETTE
##


def resolve_palette_config(
    *,
    pivot_value: float | None,
    value_range: tuple[float, float],
) -> add_color.PaletteConfig:
    """Choose a sequential or diverging palette, centred at `pivot_value` when there is one.

    `pivot_value` is the pivot for whatever is actually being displayed, not necessarily the
    field's own declared pivot: eg. log10 of a strictly-positive field diverges around 0 even
    though the raw field has no pivot, so the caller resolves which pivot applies to this view.

    Falls back to sequential when `value_range` does not straddle `pivot_value`: a signed field
    can still have an instance (eg. one component of one slice) that comes out one-sided, which
    a diverging palette cannot render regardless of what the field is expected to look like overall.
    """
    min_value, max_value = value_range
    if (pivot_value is None) or not (min_value < pivot_value < max_value):
        return add_color.SequentialConfig(palette_name=SEQUENTIAL_PALETTE_NAME)
    return add_color.DivergingConfig(
        mid_value=pivot_value,
        palette_name=DIVERGING_PALETTE_NAME,
    )


## } MODULE
