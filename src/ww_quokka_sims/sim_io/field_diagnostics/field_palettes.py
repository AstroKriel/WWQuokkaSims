## { MODULE

##
## === DEPENDENCIES
##

## personal
from jormi.ww_plots import add_color

## local
from ww_quokka_sims.sim_io.snapshots import field_registry

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
    expected_properties: field_registry.ExpectedProperties,
    value_range: tuple[float, float],
) -> add_color.PaletteConfig:
    """Choose a sequential or diverging palette, centred at `pivot_value` when the field has one.

    Falls back to sequential when `value_range` does not straddle `pivot_value`: a signed field
    can still have an instance (eg. one component of one slice) that comes out one-sided, which
    a diverging palette cannot render regardless of what the field is expected to look like overall.
    """
    pivot_value = expected_properties.pivot_value
    min_value, max_value = value_range
    if (pivot_value is None) or not (min_value < pivot_value < max_value):
        return add_color.SequentialConfig(palette_name=SEQUENTIAL_PALETTE_NAME)
    return add_color.DivergingConfig(
        mid_value=pivot_value,
        palette_name=DIVERGING_PALETTE_NAME,
    )


## } MODULE
