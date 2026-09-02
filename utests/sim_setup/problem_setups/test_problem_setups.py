## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from ww_quokka_sims.sim_setup import problem_setups

##
## === TEST PROBLEM SETUPS CAPABILITY
## doesn't verify any real Quokka problem's domain values, only the tool's own mechanics
##


class ProblemSetupsTests(unittest.TestCase):

    def test_profile_minimal_call_uses_documented_defaults(
        self,
    ):
        sim_params = problem_setups.blast_wave.build_sim_params(
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
        sim_params = problem_setups.blast_wave.build_sim_params(
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
        sim_params = problem_setups.alfven_wave_linear_convergence.build_sim_params(
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
        sim_params = problem_setups.alfven_wave_linear_correctness.build_sim_params(
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
            problem_setups.alfven_wave_linear_correctness.build_sim_params(  # pyright: ignore[reportCallIssue]
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
