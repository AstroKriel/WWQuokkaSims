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
) -> add_color.PaletteConfig:
    """Choose a sequential or diverging palette, centred at `pivot_value` when the field has one."""
    if expected_properties.pivot_value is None:
        return add_color.SequentialConfig(palette_name=SEQUENTIAL_PALETTE_NAME)
    return add_color.DivergingConfig(
        mid_value=expected_properties.pivot_value,
        palette_name=DIVERGING_PALETTE_NAME,
    )


## } MODULE
