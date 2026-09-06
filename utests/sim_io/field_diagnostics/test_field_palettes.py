## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## personal
from jormi.ww_plots import add_color

## local
from ww_quokka_sims.sim_io.field_diagnostics import field_palettes
from ww_quokka_sims.sim_io.snapshots import field_registry

##
## === TEST SUITE
##


class TestResolvePaletteConfig(unittest.TestCase):

    def test_no_pivot_value_gives_sequential_config(
        self,
    ):
        config = field_palettes.resolve_palette_config(
            expected_properties=field_registry.ExpectedProperties(pivot_value=None, is_strictly_positive=True),
            value_range=(0.0, 1.0),
        )
        self.assertIsInstance(config, add_color.SequentialConfig)
        self.assertEqual(config.palette_name, field_palettes.SEQUENTIAL_PALETTE_NAME)

    def test_pivot_value_gives_diverging_config_centred_there(
        self,
    ):
        config = field_palettes.resolve_palette_config(
            expected_properties=field_registry.ExpectedProperties(pivot_value=1.0, is_strictly_positive=True),
            value_range=(0.0, 2.0),
        )
        self.assertIsInstance(config, add_color.DivergingConfig)
        assert isinstance(config, add_color.DivergingConfig)
        self.assertEqual(config.mid_value, 1.0)
        self.assertEqual(config.palette_name, field_palettes.DIVERGING_PALETTE_NAME)

    def test_zero_pivot_value_still_gives_diverging_config(
        self,
    ):
        ## `0.0` is falsy but must not be treated the same as `None`
        config = field_palettes.resolve_palette_config(
            expected_properties=field_registry.ExpectedProperties(pivot_value=0.0, is_strictly_positive=False),
            value_range=(-1.0, 1.0),
        )
        self.assertIsInstance(config, add_color.DivergingConfig)
        assert isinstance(config, add_color.DivergingConfig)
        self.assertEqual(config.mid_value, 0.0)

    def test_one_sided_value_range_falls_back_to_sequential(
        self,
    ):
        ## a signed field can still have one instance (eg. one comp of one slice) that comes
        ## out one-sided; a diverging palette cannot render a range that misses its own pivot
        config = field_palettes.resolve_palette_config(
            expected_properties=field_registry.ExpectedProperties(pivot_value=0.0, is_strictly_positive=False),
            value_range=(-6.4e-4, -2.96e-9),
        )
        self.assertIsInstance(config, add_color.SequentialConfig)

## } U-TEST
