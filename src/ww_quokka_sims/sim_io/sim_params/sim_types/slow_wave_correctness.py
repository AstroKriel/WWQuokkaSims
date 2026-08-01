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

PROBLEM_NAME = "SlowWave"

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
    stop_time: float,
    max_time_steps: int,
    error_tol: float = 0.002,
    domain_lo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_hi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_cells: tuple[int, int, int] = (128, 128, 128),
    blocking_factor: int | tuple[int, int, int] = 128,
    max_grid_size: int | tuple[int, int, int] = 128,
    cfl: float = 0.3,
    snapshot_index_interval: int = 25,
    checkpoint_index_interval: int = -1,
) -> write_param_groups.SimParams:
    """
    Build the full parameter set for one `SlowWaveConvergence` fixed-resolution correctness run.

    `stop_time`/`max_time_steps` have no default: the right `stop_time` (one wave period) depends
    on `num_modes_x/y/z`/`angle_between_k_b0`, so a fixed default would silently be wrong.
    """
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction).value
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
            stop_time=stop_time,
            max_time_steps=max_time_steps,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=2,
            reconstruction_order=reconstruction_order,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
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


## } MODULE
