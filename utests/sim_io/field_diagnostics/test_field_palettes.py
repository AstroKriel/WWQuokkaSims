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
        )
        self.assertIsInstance(config, add_color.SequentialConfig)
        self.assertEqual(config.palette_name, field_palettes.SEQUENTIAL_PALETTE_NAME)

    def test_pivot_value_gives_diverging_config_centred_there(
        self,
    ):
        config = field_palettes.resolve_palette_config(
            expected_properties=field_registry.ExpectedProperties(pivot_value=1.0, is_strictly_positive=True),
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
        )
        self.assertIsInstance(config, add_color.DivergingConfig)
        assert isinstance(config, add_color.DivergingConfig)
        self.assertEqual(config.mid_value, 0.0)


## } U-TEST
