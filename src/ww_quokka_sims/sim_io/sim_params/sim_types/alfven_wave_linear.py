## { MODULE

##
## === DEPENDENCIES
##

## local
from .. import param_groups
from .. import write_param_groups

## fully-dotted, not `from . import scheme_lookup`: that form triggers `reportImportCycles`,
## since `sim_types/__init__.py` imports this module before `scheme_lookup` resolves
import ww_quokka_sims.sim_io.sim_params.sim_types.scheme_lookup as scheme_lookup

##
## === CONSTANTS
##

PROBLEM_NAME = "AlfvenWaveLinear"

## `problem_main()` has two mutually exclusive modes, both under `setup.*`:
## - `run_convergence` (default `true`): a Richardson sweep that overrides `amr.n_cell`/
##   `geometry.*`/`stop_time`/`max_timesteps` per iteration, ignoring the toml. `cfl`/etc. ARE
##   respected but are fixed here too: the sweep's pass/fail check (`expected_rate=2.0`/
##   `tolerance=0.3`, hardcoded, no toml key) can fail from temporal error alone if `cfl` is too
##   large, unrelated to reconstruction/EMF-scheme accuracy.
## - `run_sim` (must explicitly set `run_convergence=false` too): a single fixed-resolution
##   run at real user-chosen geometry/resolution/output/time-integration, and the only one
##   of the 11 problems here with real resistive-MHD support (`mhd.resistivity`).
_CONVERGENCE_DOMAIN_LO = (0.0, 0.0, 0.0)
_CONVERGENCE_DOMAIN_HI = (1.0, 1.0, 1.0)
_CONVERGENCE_NUM_CELLS = (128, 8, 8)
_NX_MAX = 2048

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction: str,
    num_modes_x: int,
    num_modes_y: int,
    num_modes_z: int,
    angle_between_k_b0: float,
    run_sim: bool = False,
    error_tol: float = 0.005,
    resistivity: float | None = None,
    domain_lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_cells: tuple[int, int, int] = (256, 8, 8),
    blocking_factor: int | tuple[int, int, int] = (256, 8, 8),
    max_grid_size: int | tuple[int, int, int] = 256,
    cfl: float = 0.3,
    stop_time: float | None = None,
    max_time_steps: int | None = None,
    use_subcycle: int = 0,
    snapshot_index_interval: int = 100,
    checkpoint_index_interval: int = -1,
) -> write_param_groups.SimParams:
    """
    Build the full parameter set for one `AlfvenWaveLinear` run.

    `run_sim=False` (default) builds a Richardson-convergence-sweep combo (e.g. `q26-b25-pcm`),
    matching `FastWaveConvergence`'s shape. `run_sim=True` builds a single fixed-resolution,
    optionally resistive, correctness run. `stop_time`/`max_time_steps` are required (not
    defaulted): the correct `stop_time` is one full wave period, which depends on
    `num_modes_x/y/z`/`angle_between_k_b0`, so defaulting it would silently be wrong for any
    combo other than the one it was measured for.
    """
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction).value
    emf_compute_scheme = scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value
    emf_averaging_scheme = scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value

    if run_sim:
        if stop_time is None or max_time_steps is None:
            raise ValueError("`run_sim=True` requires `stop_time`/`max_time_steps`.")
        return write_param_groups.SimParams(
            geometry_params=param_groups.GeometryParams(
                domain_lo=domain_lo,
                domain_hi=domain_hi,
                boundary_conditions=("periodic", "periodic", "periodic"),
            ),
            resolution_params=param_groups.ResolutionParams(
                num_cells=num_cells,
                blocking_factor=blocking_factor,
                max_grid_size=max_grid_size,
            ),
            output_file_params=param_groups.OutputFileParams(
                snapshot_prefix="snapshots/plt",
                snapshot_index_interval=snapshot_index_interval,
                checkpoint_index_interval=checkpoint_index_interval,
            ),
            time_integration_params=param_groups.TimeIntegrationParams(
                cfl=cfl,
                use_subcycle=use_subcycle,
                stop_time=stop_time,
                max_time_steps=max_time_steps,
            ),
            hydro_params=param_groups.HydroParams(
                integrator_order=2,
                reconstruction_order=reconstruction_order,
                use_dual_energy=0,
            ),
            mhd_params=param_groups.MHDParams(
                emf_compute_scheme=emf_compute_scheme,
                emf_averaging_scheme=emf_averaging_scheme,
                reconstruction_order=reconstruction_order,
                resistivity=resistivity,
            ),
            setup_params=param_groups.SetupParams(
                param_values={
                    "run_sim": True,
                    "run_convergence": False,
                    "angle_between_k_b0": angle_between_k_b0,
                    "num_modes_x": num_modes_x,
                    "num_modes_y": num_modes_y,
                    "num_modes_z": num_modes_z,
                    "error_tol": error_tol,
                },
            ),
        )

    return write_param_groups.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=_CONVERGENCE_DOMAIN_LO,
            domain_hi=_CONVERGENCE_DOMAIN_HI,
            is_boundary_periodic=(1, 1, 1),
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=_CONVERGENCE_NUM_CELLS,
            blocking_factor=(16, 8, 8),
            max_grid_size=128,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_index_interval=-1,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=0.3,
            use_reflux=0,
            use_subcycle=0,
            use_tracers=1,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=2,
            reconstruction_order=reconstruction_order,
            use_dual_energy=0,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=emf_compute_scheme,
            emf_averaging_scheme=emf_averaging_scheme,
            reconstruction_order=reconstruction_order,
        ),
        setup_params=param_groups.SetupParams(
            group_title="wave setup",
            param_values={
                "num_modes_x": num_modes_x,
                "num_modes_y": num_modes_y,
                "num_modes_z": num_modes_z,
                "angle_between_k_b0": angle_between_k_b0,
                "machine_precision_target": 0,
                "nx_max": _NX_MAX,
            },
        ),
    )


## } MODULE
