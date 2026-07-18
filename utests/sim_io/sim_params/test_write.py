## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import tomllib
import unittest
from pathlib import Path

## local
from ww_quokka_sims.sim_io.sim_params import _render, sim_types, write
from ww_quokka_sims.sim_io.sim_params.models import (
    VALID_EMF_AVERAGING_SCHEMES,
    VALID_EMF_COMPUTE_SCHEMES,
    VALID_RECONSTRUCTION_ORDERS,
    GeometryParams,
    HydroParams,
    MHDParams,
    OutputParams,
    ResolutionParams,
    SetupParams,
    TimeIntegrationParams,
)
from ww_quokka_sims.sim_io.sim_params.sim_types import scheme_lookup

##
## === REFERENCE FILE
## a frozen copy of a real, already-audited `sim_params.toml` from the downstream paper repo
## (kriel-quokka-mhd), used to validate that `FastWaveConvergence`'s profile reproduces a
## known-good file's parameter values exactly. Copied in as a fixture (not read from that repo
## on disk) so this test is portable: it must pass for anyone who clones `ww-quokka-sims` alone.
##

_REFERENCE_FILE = Path(__file__).parent / "fixtures" / "fast_wave_convergence_reference.toml"

##
## === TEST SUITE: _render internals
##


class RenderTests(unittest.TestCase):

    def test_expand_per_axis_scalar(
        self,
    ):
        lines = _render.expand_per_axis(16, key_prefix="amr.blocking_factor")
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
        lines = _render.expand_per_axis((16, 8, 8), key_prefix="amr.blocking_factor")
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
            lines = _render.expand_per_axis(value, key_prefix="amr.blocking_factor")
            self.assertTrue(all("_x" in line or "_y" in line or "_z" in line for line in lines))
            self.assertFalse(any(line.startswith("amr.blocking_factor =") for line in lines))

    def test_expand_per_axis_rejects_wrong_length_tuple(
        self,
    ):
        with self.assertRaises(ValueError):
            _render.expand_per_axis((16, 8), key_prefix="amr.blocking_factor")  # pyright: ignore[reportArgumentType]
        with self.assertRaises(ValueError):
            _render.expand_per_axis((16, 8, 8, 4), key_prefix="amr.blocking_factor")  # pyright: ignore[reportArgumentType]

    def test_expand_per_axis_rejects_non_positive_scalar(
        self,
    ):
        with self.assertRaises(ValueError):
            _render.expand_per_axis(0, key_prefix="amr.blocking_factor")
        with self.assertRaises(ValueError):
            _render.expand_per_axis(-1, key_prefix="amr.blocking_factor")

    def test_format_value_list_of_floats(
        self,
    ):
        self.assertEqual(_render.format_value([0.0, 1.0, 1.0]), "[0.0, 1.0, 1.0]")

    def test_format_value_string(
        self,
    ):
        self.assertEqual(_render.format_value("Quokka2026"), '"Quokka2026"')

    def test_format_value_int_not_python_bool(
        self,
    ):
        ## AMReX booleans are bare 0/1, never Python's True/False
        self.assertEqual(_render.format_value(True), "1")
        self.assertEqual(_render.format_value(0), "0")


##
## === TEST SUITE: scheme_lookup
##


class SchemeLookupTests(unittest.TestCase):

    def test_interpolation_resolves_by_key(
        self,
    ):
        self.assertEqual(scheme_lookup.interpolation_by_key("ppm_ep").value, 5)
        self.assertEqual(scheme_lookup.interpolation_by_key("pcm").value, 1)

    def test_emf_compute_scheme_resolves_by_key(
        self,
    ):
        self.assertEqual(scheme_lookup.emf_compute_scheme_by_key("q26").value, "Quokka2026")

    def test_invalid_key_raises_with_valid_options_listed(
        self,
    ):
        ## the whole point of Enum + resolve_member over a plain dict: an invalid key is a
        ## clear ValueError naming the valid options, not a bare KeyError
        with self.assertRaises(ValueError) as ctx:
            scheme_lookup.interpolation_by_key("pmm")  # typo: transposed letters
        self.assertIn("PPM", str(ctx.exception))


##
## === TEST SUITE: models/scheme_lookup consistency
## `models.py` validates against its own literal value sets, independent of `scheme_lookup.py`'s
## enums, since a model must reject an invalid Quokka value even when constructed directly,
## bypassing `scheme_lookup` entirely. That independence means the two can silently drift; these
## tests catch it.
##


class SchemaConsistencyTests(unittest.TestCase):

    def test_emf_compute_scheme_values_match_enum(
        self,
    ):
        self.assertEqual(
            set(VALID_EMF_COMPUTE_SCHEMES),
            {member.value for member in scheme_lookup.EMFComputeScheme},
        )

    def test_emf_averaging_scheme_values_match_enum(
        self,
    ):
        self.assertEqual(
            set(VALID_EMF_AVERAGING_SCHEMES),
            {member.value for member in scheme_lookup.EMFAveragingScheme},
        )

    def test_reconstruction_order_values_match_enum(
        self,
    ):
        self.assertEqual(
            set(VALID_RECONSTRUCTION_ORDERS),
            {member.value for member in scheme_lookup.InterpolationScheme},
        )


##
## === TEST SUITE: sim_params.models validation
##


class ModelValidationTests(unittest.TestCase):

    def test_resolution_rejects_num_cells_below_minimum(
        self,
    ):
        with self.assertRaises(ValueError):
            ResolutionParams(num_cells=(4, 8, 8), blocking_factor=8, max_grid_size=128)

    def test_resolution_rejects_num_refinement_buffer_cells_without_amr(
        self,
    ):
        with self.assertRaises(ValueError):
            ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=8,
                max_grid_size=128,
                max_level=0,
                num_refinement_buffer_cells=1,
            )

    def test_geometry_requires_exactly_one_boundary_condition_style(
        self,
    ):
        with self.assertRaises(ValueError):
            GeometryParams(domain_lo=(0.0, 0.0, 0.0), domain_hi=(1.0, 1.0, 1.0))  # neither set
        with self.assertRaises(ValueError):
            GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                is_boundary_periodic=(1, 1, 1),
                boundary_conditions=("ext_dir", "periodic", "periodic"),
            )  # both set

    def test_output_requires_exactly_one_interval_style(
        self,
    ):
        with self.assertRaises(ValueError):
            OutputParams()  # neither plotfile_interval nor plottime_interval

    def test_hydro_rejects_invalid_reconstruction_order(
        self,
    ):
        with self.assertRaises(ValueError):
            HydroParams(rk_integrator_order=2, reconstruction_order=4)

    def test_time_integration_rejects_cfl_out_of_bounds(
        self,
    ):
        with self.assertRaises(ValueError):
            TimeIntegrationParams(cfl=1.5)
        with self.assertRaises(ValueError):
            TimeIntegrationParams(cfl=-0.1)

    def test_mhd_rejects_invalid_emf_compute_scheme(
        self,
    ):
        with self.assertRaises(ValueError):
            MHDParams(emf_compute_scheme="NotAScheme", emf_averaging_scheme="Balsara2025", emf_reconstruction_order=5)

    def test_mhd_rejects_invalid_emf_averaging_scheme(
        self,
    ):
        with self.assertRaises(ValueError):
            MHDParams(emf_compute_scheme="Quokka2026", emf_averaging_scheme="NotAScheme", emf_reconstruction_order=5)

    def test_mhd_rejects_invalid_emf_reconstruction_order(
        self,
    ):
        with self.assertRaises(ValueError):
            MHDParams(emf_compute_scheme="Quokka2026", emf_averaging_scheme="Balsara2025", emf_reconstruction_order=4)

    def test_setup_rejects_empty_values(
        self,
    ):
        with self.assertRaises(ValueError):
            SetupParams(values={})

    def test_setup_rejects_empty_group_title(
        self,
    ):
        with self.assertRaises(ValueError):
            SetupParams(values={"nx_max": 128}, group_title="")

    def test_setup_rejects_empty_key_prefix(
        self,
    ):
        with self.assertRaises(ValueError):
            SetupParams(values={"nx_max": 128}, key_prefix="")


##
## === TEST SUITE: write_sim_params_toml guardrails
##


class GuardrailTests(unittest.TestCase):

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

    def _base_kwargs(
        self,
        **overrides: object,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "output_path": self.test_file_path,
            "geometry": GeometryParams(domain_lo=(0.0, 0.0, 0.0), domain_hi=(1.0, 1.0, 1.0), is_boundary_periodic=(1, 1, 1)),
            "resolution": ResolutionParams(num_cells=(128, 8, 8), blocking_factor=(16, 8, 8), max_grid_size=128),
            "output": OutputParams(plotfile_interval=-1),
            "time_integration": TimeIntegrationParams(cfl=0.3, do_subcycle=0),
            "hydro": HydroParams(rk_integrator_order=2, reconstruction_order=5),
            "mhd": MHDParams(emf_compute_scheme="Quokka2026", emf_averaging_scheme="Balsara2025", emf_reconstruction_order=5),
            "verbose": False,
        }
        kwargs.update(overrides)
        return kwargs

    def test_resistivity_requires_do_subcycle_zero(
        self,
    ):
        kwargs = self._base_kwargs(
            time_integration=TimeIntegrationParams(cfl=0.3, do_subcycle=1),
            mhd=MHDParams(
                emf_compute_scheme="Quokka2026",
                emf_averaging_scheme="Balsara2025",
                emf_reconstruction_order=5,
                resistivity=0.001,
            ),
        )
        with self.assertRaises(ValueError):
            write.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]

    def test_resistivity_disallowed_for_fast_wave_convergence(
        self,
    ):
        kwargs = self._base_kwargs(
            mhd=MHDParams(
                emf_compute_scheme="Quokka2026",
                emf_averaging_scheme="Balsara2025",
                emf_reconstruction_order=5,
                resistivity=0.001,
            ),
            problem_type="FastWaveConvergence",
        )
        with self.assertRaises(ValueError):
            write.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]

    def test_refuses_to_overwrite_by_default(
        self,
    ):
        write.write_sim_params_toml(**self._base_kwargs())  # pyright: ignore[reportArgumentType]
        with self.assertRaises(FileExistsError):
            write.write_sim_params_toml(**self._base_kwargs())  # pyright: ignore[reportArgumentType]

    def test_rejects_non_sim_params_filename(
        self,
    ):
        kwargs = self._base_kwargs(output_path=Path("not_sim_params.toml"))
        with self.assertRaises(ValueError):
            write.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]

    def test_param_group_order_and_setup_last(
        self,
    ):
        path = write.write_sim_params_toml(**self._base_kwargs())  # pyright: ignore[reportArgumentType]
        headers = [line for line in path.read_text().splitlines() if line.startswith("##")]
        self.assertEqual(
            headers,
            ["## geometry", "## resolution", "## verbosity", "## output", "## time integration", "## hydro", "## mhd"],
        )


##
## === TEST SUITE: write_sim_params_toml render paths not covered by the FastWaveConvergence fixture
##


class WriteContentTests(unittest.TestCase):

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

    def _base_kwargs(
        self,
        **overrides: object,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "output_path": self.test_file_path,
            "geometry": GeometryParams(domain_lo=(0.0, 0.0, 0.0), domain_hi=(1.0, 1.0, 1.0), is_boundary_periodic=(1, 1, 1)),
            "resolution": ResolutionParams(num_cells=(128, 8, 8), blocking_factor=(16, 8, 8), max_grid_size=128),
            "output": OutputParams(plotfile_interval=-1),
            "time_integration": TimeIntegrationParams(cfl=0.3, do_subcycle=0),
            "hydro": HydroParams(rk_integrator_order=2, reconstruction_order=5),
            "mhd": MHDParams(emf_compute_scheme="Quokka2026", emf_averaging_scheme="Balsara2025", emf_reconstruction_order=5),
            "verbose": False,
        }
        kwargs.update(overrides)
        return kwargs

    def test_writes_ext_dir_boundary_conditions(
        self,
    ):
        kwargs = self._base_kwargs(
            geometry=GeometryParams(
                domain_lo=(0.0, 0.0, 0.0),
                domain_hi=(1.0, 1.0, 1.0),
                boundary_conditions=("ext_dir", "periodic", "periodic"),
            ),
        )
        path = write.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]
        content = path.read_text()
        self.assertIn('quokka.bc = ["ext_dir", "periodic", "periodic"]', content)
        self.assertNotIn("geometry.is_periodic", content)

    def test_writes_checkpoint_and_plottime_settings(
        self,
    ):
        kwargs = self._base_kwargs(
            output=OutputParams(
                plottime_interval=0.5,
                checkpoint_interval=100,
                checkpoint_prefix="checkpoints/chk",
            ),
        )
        path = write.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]
        content = path.read_text()
        self.assertIn("plottime_interval = 0.5", content)
        self.assertIn("checkpoint_interval = 100", content)
        self.assertIn('checkpoint_prefix = "checkpoints/chk"', content)
        self.assertNotIn("plotfile_interval", content)

    def test_writes_amr_refinement_settings(
        self,
    ):
        kwargs = self._base_kwargs(
            resolution=ResolutionParams(
                num_cells=(128, 8, 8),
                blocking_factor=(16, 8, 8),
                max_grid_size=128,
                max_level=1,
                num_refinement_buffer_cells=2,
            ),
        )
        path = write.write_sim_params_toml(**kwargs)  # pyright: ignore[reportArgumentType]
        content = path.read_text()
        self.assertIn("amr.max_level = 1", content)
        self.assertIn("amr.n_error_buf = 2", content)


##
## === TEST SUITE: round-trip against a real, already-audited kriel-quokka-mhd file
##


class RoundTripTests(unittest.TestCase):

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

    def test_fast_wave_convergence_matches_real_file(
        self,
    ):
        bundle = sim_types.fast_wave_convergence.build_combo(
            compute_scheme_key="q26",
            averaging_scheme_key="b25",
            interpolation="ppm_ep",
        )
        bundle.write(
            output_path=self.test_file_path,
            verbose=False,
        )
        with open(self.test_file_path, "rb") as file_pointer:
            generated = tomllib.load(file_pointer)
        with open(_REFERENCE_FILE, "rb") as file_pointer:
            reference = tomllib.load(file_pointer)
        self.assertEqual(generated, reference)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
