## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from pathlib import Path

## local
from ww_quokka_sims.sim_io.sim_params import (
    _format_params,
    _param_groups,
    _scheme_lookup,
    sim_types,
    _save_params,
)

##
## === TEST RENDER PRIMITIVES
##


class FormatTests(unittest.TestCase):

    def test_expand_per_axis_scalar(
        self,
    ):
        lines = _format_params.expand_per_axis(16, key_prefix="amr.blocking_factor")
        self.assertEqual(
            lines,
            [
                "amr.blocking_factor_x = 16",
                "amr.blocking_factor_y = 16",
                "amr.blocking_factor_z = 16",
            ],
        )

    def test_expand_per_axis_tuple(
        self,
    ):
        lines = _format_params.expand_per_axis((16, 8, 8), key_prefix="amr.blocking_factor")
        self.assertEqual(
            lines,
            [
                "amr.blocking_factor_x = 16",
                "amr.blocking_factor_y = 8",
                "amr.blocking_factor_z = 8",
            ],
        )

    def test_expand_per_axis_never_emits_bare_key(
        self,
    ):
        for value in (16, (16, 8, 8)):
            lines = _format_params.expand_per_axis(value, key_prefix="amr.blocking_factor")
            self.assertTrue(all("_x" in line or "_y" in line or "_z" in line for line in lines))
            self.assertFalse(any(line.startswith("amr.blocking_factor =") for line in lines))

    def test_expand_per_axis_rejects_wrong_length_tuple(
        self,
    ):
        with self.assertRaises(ValueError):
            _format_params.expand_per_axis(
                (16, 8),  # pyright: ignore[reportArgumentType]
                key_prefix="amr.blocking_factor",
            )
        with self.assertRaises(ValueError):
            _format_params.expand_per_axis(
                (16, 8, 8, 4),  # pyright: ignore[reportArgumentType]
                key_prefix="amr.blocking_factor",
            )

    def test_expand_per_axis_rejects_non_positive_scalar(
        self,
    ):
        with self.assertRaises(ValueError):
            _format_params.expand_per_axis(0, key_prefix="amr.blocking_factor")
        with self.assertRaises(ValueError):
            _format_params.expand_per_axis(-1, key_prefix="amr.blocking_factor")

    def test_format_value_list_of_floats(
        self,
    ):
        self.assertEqual(_format_params.format_value([0.0, 1.0, 1.0]), "[0.0, 1.0, 1.0]")

    def test_format_value_string(
        self,
    ):
        self.assertEqual(_format_params.format_value("Quokka2026"), '"Quokka2026"')

    def test_format_value_int_not_python_bool(
        self,
    ):
        ## AMReX booleans are bare 0/1, never Python's True/False
        self.assertEqual(_format_params.format_value(True), "1")
        self.assertEqual(_format_params.format_value(0), "0")


##
## === TEST SCHEME LOOKUP
##


class SchemeLookupTests(unittest.TestCase):

    def test_reconstruction_resolves_by_key(
        self,
    ):
        self.assertEqual(_scheme_lookup.resolve_reconstruction_scheme("ppm_ep"), 5)
        self.assertEqual(_scheme_lookup.resolve_reconstruction_scheme("pcm"), 1)

    def test_emf_compute_scheme_resolves_by_key(
        self,
    ):
        self.assertEqual(_scheme_lookup.resolve_emf_compute_scheme("q26"), "Quokka2026")

    def test_invalid_key_raises_with_valid_options_listed(
        self,
    ):
        with self.assertRaises(ValueError) as ctx:
            _scheme_lookup.resolve_reconstruction_scheme("pmm")  # typo: transposed letters
        self.assertIn("PPM", str(ctx.exception))


##
## === TEST PARAM GROUPS VALIDATION
## only rendering-ambiguity and degenerate inputs; Quokka validates everything else itself
##


class ModelValidationTests(unittest.TestCase):

    def test_resolution_rejects_num_cells_below_minimum(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.ResolutionParams(
                num_cells=(4, 8, 8),
                blocking_factor=8,
                max_grid_size=128,
            )

    def test_resolution_accepts_a_sensible_combination(
        self,
    ):
        _param_groups.ResolutionParams(
            num_cells=(256, 8, 8),
            blocking_factor=(256, 8, 8),
            max_grid_size=256,
        )

    def test_resolution_rejects_blocking_factor_not_power_of_two(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.ResolutionParams(
                num_cells=(24, 8, 8),
                blocking_factor=(24, 8, 8),
                max_grid_size=128,
            )

    def test_resolution_rejects_blocking_factor_above_max_grid_size(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=128,
                max_grid_size=64,
            )

    def test_resolution_rejects_num_cells_not_divisible_by_blocking_factor(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.ResolutionParams(
                num_cells=(129, 8, 8),
                blocking_factor=16,
                max_grid_size=128,
            )

    def test_resolution_rejects_max_grid_size_not_divisible_by_blocking_factor_with_amr(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=16,
                max_grid_size=100,
                max_amr_levels=1,
            )

    def test_resolution_allows_max_grid_size_not_divisible_by_blocking_factor_without_amr(
        self,
    ):
        ## AMReX itself only enforces this relationship once AMR is enabled
        _param_groups.ResolutionParams(
            num_cells=(8, 8, 8),
            blocking_factor=8,
            max_grid_size=100,
        )

    def test_resolution_rejects_num_refinement_buffer_cells_without_amr(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=8,
                max_grid_size=128,
                max_amr_levels=0,
                num_refinement_buffer_cells=1,
            )

    def test_geometry_requires_exactly_one_boundary_condition_style(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
            )  # neither set
        with self.assertRaises(ValueError):
            _param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                is_boundary_periodic=(1, 1, 1),
                boundary_conditions=("ext_dir", "periodic", "periodic"),
            )  # both set

    def test_geometry_rejects_domain_lo_not_less_than_domain_hi(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 1.0),
                domain_hi=(1.0, 1.0, 1.0),
                is_boundary_periodic=(1, 1, 1),
            )  # equal on axis 2
        with self.assertRaises(ValueError):
            _param_groups.GeometryParams(
                domain_lo=(0.0, 1.0, 0.0),
                domain_hi=(1.0, 0.5, 1.0),
                is_boundary_periodic=(1, 1, 1),
            )  # inverted on axis 1

    def test_output_requires_at_least_one_interval_style(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.OutputFileParams()  # neither snapshot_index_interval nor snapshot_time_interval

    def test_output_allows_both_interval_styles_simultaneously(
        self,
    ):
        ## Quokka accepts index- and time-based cadence together; whichever fires first wins
        _param_groups.OutputFileParams(
            snapshot_index_interval=100,
            snapshot_time_interval=0.5,
        )

    def test_output_rejects_empty_derived_vars(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.OutputFileParams(
                snapshot_index_interval=100,
                derived_vars=(),
            )

    def test_time_integration_rejects_cfl_out_of_bounds(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.TimeIntegrationParams(cfl=1.5)
        with self.assertRaises(ValueError):
            _param_groups.TimeIntegrationParams(cfl=-0.1)

    def test_setup_rejects_empty_param_values(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.SetupParams(param_values={})

    def test_setup_rejects_empty_group_title(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.SetupParams(
                param_values={"nx_max": 128},
                group_title="",
            )

    def test_setup_rejects_empty_key_prefix(
        self,
    ):
        with self.assertRaises(ValueError):
            _param_groups.SetupParams(
                param_values={"nx_max": 128},
                key_prefix="",
            )


##
## === SHARED FIXTURES
##


class _SavesSimParamsFileTestCase(unittest.TestCase):
    test_file_path: Path  # pyright: ignore[reportUninitializedInstanceVariable]

    def setUp(
        self,
    ):
        self.test_file_path = Path("sim_params.toml")
        if self.test_file_path.exists():
            self.test_file_path.unlink()

    def tearDown(
        self,
    ):
        if self.test_file_path.exists():
            self.test_file_path.unlink()


class _DefaultSaveKwargsTestCase(_SavesSimParamsFileTestCase):

    def _base_kwargs(
        self,
        **override_kwargs: object,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "output_path":
            self.test_file_path,
            "geometry_params":
            _param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                is_boundary_periodic=(1, 1, 1),
            ),
            "resolution_params":
            _param_groups.ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=(16, 8, 8),
                max_grid_size=128,
            ),
            "output_file_params":
            _param_groups.OutputFileParams(snapshot_index_interval=-1),
            "time_integration_params":
            _param_groups.TimeIntegrationParams(
                cfl=0.3,
                use_subcycle=0,
            ),
            "hydro_params":
            _param_groups.HydroParams(
                integrator_order=2,
                reconstruction_order=5,
            ),
            "mhd_params":
            _param_groups.MHDParams(
                emf_compute_scheme="Quokka2026",
                emf_averaging_scheme="Balsara2025",
                reconstruction_order=5,
            ),
            "verbose":
            False,
        }
        kwargs.update(override_kwargs)
        return kwargs


##
## === TEST WRITE GUARDRAILS
##


class GuardrailTests(_DefaultSaveKwargsTestCase):

    def test_refuses_to_overwrite_by_default(
        self,
    ):
        _save_params.save_sim_params_toml(
            **self._base_kwargs(),  # pyright: ignore[reportArgumentType]
        )
        with self.assertRaises(FileExistsError):
            _save_params.save_sim_params_toml(
                **self._base_kwargs(),  # pyright: ignore[reportArgumentType]
            )

    def test_rejects_non_sim_params_filename(
        self,
    ):
        kwargs = self._base_kwargs(output_path=Path("not_sim_params.toml"))
        with self.assertRaises(ValueError):
            _save_params.save_sim_params_toml(
                **kwargs,  # pyright: ignore[reportArgumentType]
            )

    def test_param_group_order_and_setup_last(
        self,
    ):
        file_path = _save_params.save_sim_params_toml(
            **self._base_kwargs(),  # pyright: ignore[reportArgumentType]
        )
        rendered_group_titles = [line for line in file_path.read_text().splitlines() if line.startswith("##")]
        expected_group_titles = (
            _save_params._ParamGroupTitle.GEOMETRY,
            _save_params._ParamGroupTitle.RESOLUTION,
            _save_params._ParamGroupTitle.VERBOSITY,
            _save_params._ParamGroupTitle.OUTPUT,
            _save_params._ParamGroupTitle.TIME_INTEGRATION,
            _save_params._ParamGroupTitle.HYDRO,
            _save_params._ParamGroupTitle.MHD,
        )
        self.assertEqual(rendered_group_titles, [f"## {title}" for title in expected_group_titles])


##
## === TEST WRITTEN TOML CONTENT
##


class SaveContentTests(_DefaultSaveKwargsTestCase):

    def test_writes_ext_dir_boundary_conditions(
        self,
    ):
        kwargs = self._base_kwargs(
            geometry_params=_param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                boundary_conditions=("ext_dir", "periodic", "periodic"),
            ),
        )
        file_path = _save_params.save_sim_params_toml(
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
        file_content = file_path.read_text()
        self.assertIn('quokka.bc = ["ext_dir", "periodic", "periodic"]', file_content)
        self.assertNotIn("geometry.is_periodic", file_content)

    def test_writes_checkpoint_and_plottime_settings(
        self,
    ):
        kwargs = self._base_kwargs(
            output_file_params=_param_groups.OutputFileParams(
                snapshot_time_interval=0.5,
                checkpoint_index_interval=100,
                checkpoint_prefix="checkpoints/chk",
            ),
        )
        file_path = _save_params.save_sim_params_toml(
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
        file_content = file_path.read_text()
        self.assertIn("plottime_interval = 0.5", file_content)
        self.assertIn("checkpoint_interval = 100", file_content)
        self.assertIn('checkpoint_prefix = "checkpoints/chk"', file_content)
        self.assertNotIn("plotfile_interval", file_content)

    def test_writes_both_interval_styles_and_derived_vars(
        self,
    ):
        kwargs = self._base_kwargs(
            output_file_params=_param_groups.OutputFileParams(
                snapshot_index_interval=100,
                snapshot_time_interval=0.5,
                checkpoint_index_interval=10,
                checkpoint_time_interval=1.0,
                derived_vars=("magnetic_divergence", ),
            ),
        )
        file_path = _save_params.save_sim_params_toml(
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
        file_content = file_path.read_text()
        self.assertIn("plotfile_interval = 100", file_content)
        self.assertIn("plottime_interval = 0.5", file_content)
        self.assertIn("checkpoint_interval = 10", file_content)
        self.assertIn("checkpointtime_interval = 1.0", file_content)
        self.assertIn('derived_vars = ["magnetic_divergence"]', file_content)

    def test_writes_amr_refinement_settings(
        self,
    ):
        kwargs = self._base_kwargs(
            resolution_params=_param_groups.ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=(16, 8, 8),
                max_grid_size=128,
                max_amr_levels=1,
                num_refinement_buffer_cells=2,
            ),
        )
        file_path = _save_params.save_sim_params_toml(
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
        file_content = file_path.read_text()
        self.assertIn("amr.max_level = 1", file_content)
        self.assertIn("amr.n_error_buf = 2", file_content)

    def test_writes_amr_verbosity(
        self,
    ):
        kwargs = self._base_kwargs(amr_verbosity=0)
        file_path = _save_params.save_sim_params_toml(
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
        file_content = file_path.read_text()
        self.assertIn("amr.v = 0", file_content)


##
## === TEST SIM TYPES CAPABILITY
## doesn't verify any real Quokka problem's domain values, only the tool's own mechanics
##


class SimTypesTests(unittest.TestCase):

    def test_resolve_sim_params_builder_returns_registered_function(
        self,
    ):
        self.assertIs(
            sim_types.resolve_sim_params_builder(sim_types.ProblemSetup.BLAST_WAVE),
            sim_types.blast_wave.build_sim_params,
        )

    def test_profile_minimal_call_uses_documented_defaults(
        self,
    ):
        sim_params = sim_types.blast_wave.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction_order_key="ppm",
            num_cells=(128, 128, 128),
            blocking_factor=16,
            max_grid_size=16,
            max_time_steps=4000,
            snapshot_index_interval=25,
        )
        self.assertEqual(sim_params.time_integration_params.cfl, 0.3)
        self.assertEqual(
            sim_params.geometry_params.domain_lo,
            (-0.5, -0.5, -0.5),
        )

    def test_profile_override_threads_through(
        self,
    ):
        sim_params = sim_types.blast_wave.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction_order_key="ppm",
            num_cells=(128, 128, 128),
            blocking_factor=16,
            max_grid_size=16,
            max_time_steps=4000,
            snapshot_index_interval=25,
            cfl=0.7,
        )
        self.assertEqual(sim_params.time_integration_params.cfl, 0.7)

    def test_convergence_profile_omits_run_sim_key(
        self,
    ):
        sim_params = sim_types.alfven_wave_linear_convergence.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction_order_key="ppm",
            num_modes_x=1,
            num_modes_y=0,
            num_modes_z=0,
            angle_between_k_b0=0.0,
        )
        assert sim_params.setup_params is not None
        self.assertNotIn("run_sim", sim_params.setup_params.param_values)

    def test_correctness_profile_renders_run_sim_setup_keys(
        self,
    ):
        sim_params = sim_types.alfven_wave_linear_correctness.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction_order_key="ppm",
            num_modes_x=1,
            num_modes_y=0,
            num_modes_z=0,
            angle_between_k_b0=0.0,
            stop_time=5.0,
            max_time_steps=100_000,
        )
        assert sim_params.setup_params is not None
        self.assertEqual(sim_params.setup_params.param_values["run_sim"], True)
        self.assertEqual(sim_params.setup_params.param_values["run_convergence"], False)

    def test_correctness_profile_requires_stop_time(
        self,
    ):
        with self.assertRaises(TypeError):
            sim_types.alfven_wave_linear_correctness.build_sim_params(  # pyright: ignore[reportCallIssue]
                compute_scheme_key="q26",
                averaging_scheme_key="b25",
                reconstruction_order_key="ppm",
                num_modes_x=1,
                num_modes_y=0,
                num_modes_z=0,
                angle_between_k_b0=0.0,
                max_time_steps=100_000,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
