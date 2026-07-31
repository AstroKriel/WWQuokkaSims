## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import tomllib
import unittest
from pathlib import Path

## local
from ww_quokka_sims.sim_io.sim_params import (
    _render_param_groups,
    param_groups,
    sim_types,
    write_param_groups,
)
from ww_quokka_sims.sim_io.sim_params.sim_types import scheme_lookup

##
## === REFERENCE FILES
## hand-audited, known-good `sim_params.toml` files, frozen here as fixtures
##

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FAST_WAVE_CONVERGENCE_REFERENCE_FILE = _FIXTURES_DIR / "fast_wave-convergence.toml"
_BLAST_WAVE_NCELLS128_REFERENCE_FILE = _FIXTURES_DIR / "blast_wave-ncells=128.toml"
_BLAST_WAVE_NCELLS1024_REFERENCE_FILE = _FIXTURES_DIR / "blast_wave-ncells=1024.toml"
_BRIO_WU_SHOCK_TUBE_NCELLS256_REFERENCE_FILE = _FIXTURES_DIR / "brio_wu_shock_tube-ncells=256.toml"
_BRIO_WU_SHOCK_TUBE_NCELLS8192_REFERENCE_FILE = _FIXTURES_DIR / "brio_wu_shock_tube-ncells=8192.toml"
_CURRENT_SHEET_REFERENCE_FILE = _FIXTURES_DIR / "current_sheet.toml"
_ORSZAG_TANG_NCELLS1024_REFERENCE_FILE = _FIXTURES_DIR / "orszag_tang-ncells=1024.toml"
_ORSZAG_TANG_NCELLS8192_REFERENCE_FILE = _FIXTURES_DIR / "orszag_tang-ncells=8192.toml"
_MHD_QUIRK_REFERENCE_FILE = _FIXTURES_DIR / "mhd_quirk.toml"
_RYU_JONES_2A_SHOCK_TUBE_REFERENCE_FILE = _FIXTURES_DIR / "ryu_jones_2a_shock_tube.toml"
_FIELD_LOOP_REFERENCE_FILE = _FIXTURES_DIR / "field_loop.toml"

##
## === TEST SUITE: _render_param_groups internals
##


class RenderTests(unittest.TestCase):

    def test_expand_per_axis_scalar(
        self,
    ):
        lines = _render_param_groups.expand_per_axis(16, key_prefix="amr.blocking_factor")
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
        lines = _render_param_groups.expand_per_axis((16, 8, 8), key_prefix="amr.blocking_factor")
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
            lines = _render_param_groups.expand_per_axis(value, key_prefix="amr.blocking_factor")
            self.assertTrue(all("_x" in line or "_y" in line or "_z" in line for line in lines))
            self.assertFalse(any(line.startswith("amr.blocking_factor =") for line in lines))

    def test_expand_per_axis_rejects_wrong_length_tuple(
        self,
    ):
        with self.assertRaises(ValueError):
            _render_param_groups.expand_per_axis((16, 8), key_prefix="amr.blocking_factor")  # pyright: ignore[reportArgumentType]
        with self.assertRaises(ValueError):
            _render_param_groups.expand_per_axis((16, 8, 8, 4), key_prefix="amr.blocking_factor")  # pyright: ignore[reportArgumentType]

    def test_expand_per_axis_rejects_non_positive_scalar(
        self,
    ):
        with self.assertRaises(ValueError):
            _render_param_groups.expand_per_axis(0, key_prefix="amr.blocking_factor")
        with self.assertRaises(ValueError):
            _render_param_groups.expand_per_axis(-1, key_prefix="amr.blocking_factor")

    def test_format_value_list_of_floats(
        self,
    ):
        self.assertEqual(_render_param_groups.format_value([0.0, 1.0, 1.0]), "[0.0, 1.0, 1.0]")

    def test_format_value_string(
        self,
    ):
        self.assertEqual(_render_param_groups.format_value("Quokka2026"), '"Quokka2026"')

    def test_format_value_int_not_python_bool(
        self,
    ):
        ## AMReX booleans are bare 0/1, never Python's True/False
        self.assertEqual(_render_param_groups.format_value(True), "1")
        self.assertEqual(_render_param_groups.format_value(0), "0")


##
## === TEST SUITE: scheme_lookup
##


class SchemeLookupTests(unittest.TestCase):

    def test_reconstruction_resolves_by_key(
        self,
    ):
        self.assertEqual(scheme_lookup.resolve_reconstruction_scheme("ppm_ep").value, 5)
        self.assertEqual(scheme_lookup.resolve_reconstruction_scheme("pcm").value, 1)

    def test_emf_compute_scheme_resolves_by_key(
        self,
    ):
        self.assertEqual(scheme_lookup.resolve_emf_compute_scheme("q26").value, "Quokka2026")

    def test_invalid_key_raises_with_valid_options_listed(
        self,
    ):
        ## the whole point of Enum + resolve_member over a plain dict: an invalid key is a
        ## clear ValueError naming the valid options, not a bare KeyError
        with self.assertRaises(ValueError) as ctx:
            scheme_lookup.resolve_reconstruction_scheme("pmm")  # typo: transposed letters
        self.assertIn("PPM", str(ctx.exception))


##
## === TEST SUITE: param_groups/scheme_lookup consistency
## `param_groups.py` validates independently of `scheme_lookup.py`'s enums, so the two can
## silently drift; these tests catch it
##


class SchemaConsistencyTests(unittest.TestCase):

    def test_emf_compute_scheme_values_match_enum(
        self,
    ):
        self.assertEqual(
            set(param_groups.VALID_EMF_COMPUTE_SCHEMES),
            {member.value for member in scheme_lookup.EMFComputeScheme},
        )

    def test_emf_averaging_scheme_values_match_enum(
        self,
    ):
        self.assertEqual(
            set(param_groups.VALID_EMF_AVERAGING_SCHEMES),
            {member.value for member in scheme_lookup.EMFAveragingScheme},
        )

    def test_reconstruction_order_values_match_enum(
        self,
    ):
        self.assertEqual(
            set(param_groups.VALID_RECONSTRUCTION_ORDERS),
            {member.value for member in scheme_lookup.ReconstructionScheme},
        )


##
## === TEST SUITE: sim_params.param_groups validation
##


class ModelValidationTests(unittest.TestCase):

    def test_resolution_rejects_num_cells_below_minimum(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.ResolutionParams(num_cells=(4, 8, 8), blocking_factor=8, max_grid_size=128)

    def test_resolution_rejects_num_refinement_buffer_cells_without_amr(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.ResolutionParams(
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
            param_groups.GeometryParams(domain_lo=(0.0, 0.0, 0.0), domain_hi=(1.0, 1.0, 1.0))  # neither set
        with self.assertRaises(ValueError):
            param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                is_boundary_periodic=(1, 1, 1),
                boundary_conditions=("ext_dir", "periodic", "periodic"),
            )  # both set

    def test_output_requires_at_least_one_interval_style(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.OutputFileParams()  # neither snapshot_index_interval nor snapshot_time_interval

    def test_output_allows_both_interval_styles_simultaneously(
        self,
    ):
        ## Quokka accepts index- and time-based cadence together; whichever fires first wins
        param_groups.OutputFileParams(snapshot_index_interval=100, snapshot_time_interval=0.5)

    def test_output_rejects_empty_derived_vars(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.OutputFileParams(snapshot_index_interval=100, derived_vars=())

    def test_hydro_rejects_invalid_reconstruction_order(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.HydroParams(integrator_order=2, reconstruction_order=4)

    def test_time_integration_rejects_cfl_out_of_bounds(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.TimeIntegrationParams(cfl=1.5)
        with self.assertRaises(ValueError):
            param_groups.TimeIntegrationParams(cfl=-0.1)

    def test_mhd_rejects_invalid_emf_compute_scheme(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.MHDParams(emf_compute_scheme="NotAScheme", emf_averaging_scheme="Balsara2025", reconstruction_order=5)

    def test_mhd_rejects_invalid_emf_averaging_scheme(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.MHDParams(emf_compute_scheme="Quokka2026", emf_averaging_scheme="NotAScheme", reconstruction_order=5)

    def test_mhd_rejects_invalid_reconstruction_order(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.MHDParams(emf_compute_scheme="Quokka2026", emf_averaging_scheme="Balsara2025", reconstruction_order=4)

    def test_setup_rejects_empty_param_values(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.SetupParams(param_values={})

    def test_setup_rejects_empty_group_title(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.SetupParams(param_values={"nx_max": 128}, group_title="")

    def test_setup_rejects_empty_key_prefix(
        self,
    ):
        with self.assertRaises(ValueError):
            param_groups.SetupParams(param_values={"nx_max": 128}, key_prefix="")


##
## === SHARED FIXTURES: write_sim_params_toml tests
##


class _WritesSimParamsFileTestCase(unittest.TestCase):

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


class _DefaultWriteKwargsTestCase(_WritesSimParamsFileTestCase):

    def _base_kwargs(
        self,
        **overrides: object,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "output_path": self.test_file_path,
            "geometry_params": param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                is_boundary_periodic=(1, 1, 1),
            ),
            "resolution_params": param_groups.ResolutionParams(num_cells=(128, 8, 8), blocking_factor=(16, 8, 8), max_grid_size=128),
            "output_file_params": param_groups.OutputFileParams(snapshot_index_interval=-1),
            "time_integration_params": param_groups.TimeIntegrationParams(cfl=0.3, use_subcycle=0),
            "hydro_params": param_groups.HydroParams(integrator_order=2, reconstruction_order=5),
            "mhd_params": param_groups.MHDParams(
                emf_compute_scheme="Quokka2026",
                emf_averaging_scheme="Balsara2025",
                reconstruction_order=5,
            ),
            "verbose": False,
        }
        kwargs.update(overrides)
        return kwargs


##
## === TEST SUITE: write_sim_params_toml guardrails
##


class GuardrailTests(_DefaultWriteKwargsTestCase):

    def test_resistivity_requires_use_subcycle_zero(
        self,
    ):
        kwargs = self._base_kwargs(
            time_integration_params=param_groups.TimeIntegrationParams(cfl=0.3, use_subcycle=1),
            mhd_params=param_groups.MHDParams(
                emf_compute_scheme="Quokka2026",
                emf_averaging_scheme="Balsara2025",
                reconstruction_order=5,
                resistivity=0.001,
            ),
        )
        with self.assertRaises(ValueError):
            write_param_groups.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]

    def test_refuses_to_overwrite_by_default(
        self,
    ):
        write_param_groups.write_sim_params_toml(**self._base_kwargs())  # pyright: ignore[reportArgumentType]
        with self.assertRaises(FileExistsError):
            write_param_groups.write_sim_params_toml(**self._base_kwargs())  # pyright: ignore[reportArgumentType]

    def test_rejects_non_sim_params_filename(
        self,
    ):
        kwargs = self._base_kwargs(output_path=Path("not_sim_params.toml"))
        with self.assertRaises(ValueError):
            write_param_groups.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]

    def test_param_group_order_and_setup_last(
        self,
    ):
        path = write_param_groups.write_sim_params_toml(**self._base_kwargs())  # pyright: ignore[reportArgumentType]
        headers = [line for line in path.read_text().splitlines() if line.startswith("##")]
        expected_titles = (
            write_param_groups._ParamGroupTitle.GEOMETRY,
            write_param_groups._ParamGroupTitle.RESOLUTION,
            write_param_groups._ParamGroupTitle.VERBOSITY,
            write_param_groups._ParamGroupTitle.OUTPUT,
            write_param_groups._ParamGroupTitle.TIME_INTEGRATION,
            write_param_groups._ParamGroupTitle.HYDRO,
            write_param_groups._ParamGroupTitle.MHD,
        )
        self.assertEqual(headers, [f"## {title}" for title in expected_titles])


##
## === TEST SUITE: write_sim_params_toml render paths not covered by the FastWaveConvergence fixture
##


class WriteContentTests(_DefaultWriteKwargsTestCase):

    def test_writes_ext_dir_boundary_conditions(
        self,
    ):
        kwargs = self._base_kwargs(
            geometry_params=param_groups.GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                boundary_conditions=("ext_dir", "periodic", "periodic"),
            ),
        )
        path = write_param_groups.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]
        content = path.read_text()
        self.assertIn('quokka.bc = ["ext_dir", "periodic", "periodic"]', content)
        self.assertNotIn("geometry.is_periodic", content)

    def test_writes_checkpoint_and_plottime_settings(
        self,
    ):
        kwargs = self._base_kwargs(
            output_file_params=param_groups.OutputFileParams(
                snapshot_time_interval=0.5,
                checkpoint_index_interval=100,
                checkpoint_prefix="checkpoints/chk",
            ),
        )
        path = write_param_groups.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]
        content = path.read_text()
        self.assertIn("plottime_interval = 0.5", content)
        self.assertIn("checkpoint_interval = 100", content)
        self.assertIn('checkpoint_prefix = "checkpoints/chk"', content)
        self.assertNotIn("plotfile_interval", content)

    def test_writes_both_interval_styles_and_derived_vars(
        self,
    ):
        kwargs = self._base_kwargs(
            output_file_params=param_groups.OutputFileParams(
                snapshot_index_interval=100,
                snapshot_time_interval=0.5,
                checkpoint_index_interval=10,
                checkpoint_time_interval=1.0,
                derived_vars=("magnetic_divergence",),
            ),
        )
        path = write_param_groups.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]
        content = path.read_text()
        self.assertIn("plotfile_interval = 100", content)
        self.assertIn("plottime_interval = 0.5", content)
        self.assertIn("checkpoint_interval = 10", content)
        self.assertIn("checkpointtime_interval = 1.0", content)
        self.assertIn('derived_vars = ["magnetic_divergence"]', content)

    def test_writes_amr_refinement_settings(
        self,
    ):
        kwargs = self._base_kwargs(
            resolution_params=param_groups.ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=(16, 8, 8),
                max_grid_size=128,
                max_amr_levels=1,
                num_refinement_buffer_cells=2,
            ),
        )
        path = write_param_groups.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]
        content = path.read_text()
        self.assertIn("amr.max_level = 1", content)
        self.assertIn("amr.n_error_buf = 2", content)


##
## === TEST SUITE: round-trip against a real, already-audited reference file
##


class RoundTripTests(_WritesSimParamsFileTestCase):

    def _assert_matches_reference(
        self,
        *,
        bundle: write_param_groups.SimParams,
        reference_file: Path,
    ) -> None:
        bundle.write(
            output_path=self.test_file_path,
            verbose=False,
        )
        with open(self.test_file_path, "rb") as file_pointer:
            generated = tomllib.load(file_pointer)
        with open(reference_file, "rb") as file_pointer:
            reference = tomllib.load(file_pointer)
        self.assertEqual(generated, reference)

    def test_fast_wave_convergence_matches_real_file(
        self,
    ):
        bundle = sim_types.fast_wave_convergence.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm_ep",
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_FAST_WAVE_CONVERGENCE_REFERENCE_FILE)

    def test_blast_wave_ncells128_matches_real_file(
        self,
    ):
        bundle = sim_types.blast_wave.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm",
            num_cells=(128, 128, 128),
            blocking_factor=16,
            max_grid_size=16,
            max_time_steps=4000,
            snapshot_index_interval=25,
            checkpoint_index_interval=-1,
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_BLAST_WAVE_NCELLS128_REFERENCE_FILE)

    def test_blast_wave_ncells1024_matches_real_file(
        self,
    ):
        bundle = sim_types.blast_wave.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm",
            num_cells=(1024, 1024, 1024),
            blocking_factor=32,
            max_grid_size=128,
            max_time_steps=10000,
            snapshot_time_interval=0.0025,
            checkpoint_index_interval=-1,
            checkpoint_time_interval=0.005,
            checkpoint_prefix="checkpoints/chk",
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_BLAST_WAVE_NCELLS1024_REFERENCE_FILE)

    def test_brio_wu_shock_tube_ncells256_matches_real_file(
        self,
    ):
        bundle = sim_types.brio_wu_shock_tube.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm_ep",
            num_cells=(256, 8, 8),
            blocking_factor=(16, 8, 8),
            max_grid_size=128,
            max_time_steps=4000,
            snapshot_index_interval=200,
            checkpoint_index_interval=-1,
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_BRIO_WU_SHOCK_TUBE_NCELLS256_REFERENCE_FILE)

    def test_brio_wu_shock_tube_ncells8192_matches_real_file(
        self,
    ):
        bundle = sim_types.brio_wu_shock_tube.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm_ep",
            num_cells=(8192, 8, 8),
            blocking_factor=(16, 8, 8),
            max_grid_size=64,
            max_time_steps=100000,
            snapshot_index_interval=1600,
            checkpoint_index_interval=-1,
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_BRIO_WU_SHOCK_TUBE_NCELLS8192_REFERENCE_FILE)

    def test_current_sheet_matches_real_file(
        self,
    ):
        bundle = sim_types.current_sheet.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm",
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_CURRENT_SHEET_REFERENCE_FILE)

    def test_orszag_tang_ncells1024_matches_real_file(
        self,
    ):
        bundle = sim_types.orszag_tang.build_sim_params(
            compute_scheme_key="b25",
            averaging_scheme_key="b25",
            reconstruction="plm",
            checkpoint_index_interval=-1,
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_ORSZAG_TANG_NCELLS1024_REFERENCE_FILE)

    def test_orszag_tang_ncells8192_matches_real_file(
        self,
    ):
        bundle = sim_types.orszag_tang.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm",
            num_cells=(8192, 8192, 8),
            max_time_steps=200_000,
            use_reflux=0,
            use_subcycle=0,
            use_dual_energy=0,
            checkpoint_time_interval=0.05,
            checkpoint_prefix="checkpoints/chk",
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_ORSZAG_TANG_NCELLS8192_REFERENCE_FILE)

    def test_mhd_quirk_matches_real_file(
        self,
    ):
        ## `cfl`/`stop_time`/`max_timesteps` in the fixture are the true runtime-effective values,
        ## not what kriel-quokka-mhd's checked-in file has (see threads/dead-toml-params/)
        bundle = sim_types.mhd_quirk.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm_ep",
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_MHD_QUIRK_REFERENCE_FILE)

    def test_ryu_jones_2a_shock_tube_matches_real_file(
        self,
    ):
        bundle = sim_types.ryu_jones_2a_shock_tube.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="ld04",
            reconstruction="ppm_ep",
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_RYU_JONES_2A_SHOCK_TUBE_REFERENCE_FILE)

    def test_field_loop_matches_real_file(
        self,
    ):
        bundle = sim_types.field_loop.build_sim_params(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            reconstruction="ppm_ep",
            domain_lo=(-1.5, -1.0, -0.5),
            domain_hi=(1.5, 1.0, 0.5),
            num_cells=(96, 64, 32),
            blocking_factor=16,
            max_grid_size=32,
            max_time_steps=5000,
            stop_time=3.6055512755,
            loop_radius=0.5,
            loop_center_x=0.0,
            loop_center_y=0.0,
            advection_vz=0.2773500981126146,
            advection_angle_deg=56.309932474020215,
            refine_based_on="Region",
            region_lo_x=-1.25,
            region_hi_x=-0.75,
            region_lo_y=-0.75,
            region_hi_y=0.75,
            region_lo_z=-0.25,
            region_hi_z=0.25,
        )
        self._assert_matches_reference(bundle=bundle, reference_file=_FIELD_LOOP_REFERENCE_FILE)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
