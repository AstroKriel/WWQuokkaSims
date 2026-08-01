## { MODULE

##
## === DEPENDENCIES
##

## local
from .. import param_groups
from .. import write_param_groups

## import scheme_lookup directly from its module file, not via the package __init__, to avoid a static import cycle
import ww_quokka_sims.sim_io.sim_params.sim_types.scheme_lookup as scheme_lookup

##
## === CONSTANTS
##

PROBLEM_KEY = "FieldLoop"

## `stop_time`'s default is tuned for one full domain-diagonal crossing at the other defaults;
## overriding those without reconsidering `stop_time` still runs, just not one clean crossing.

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction: str,
    domain_lo: tuple[float, float, float] = (-1.5, -1.0, -0.5),
    domain_hi: tuple[float, float, float] = (1.5, 1.0, 0.5),
    num_cells: tuple[int, int, int] = (96, 64, 32),
    blocking_factor: int | tuple[int, int, int] = 16,
    max_grid_size: int | tuple[int, int, int] = 32,
    max_time_steps: int = 5000,
    stop_time: float = 3.6055512755,
    loop_radius: float = 0.5,
    loop_center_x: float = 0.0,
    loop_center_y: float = 0.0,
    advection_vz: float = 0.2773500981126146,
    advection_angle_deg: float = 56.309932474020215,
    refine_based_on: str = "Region",
    region_lo_x: float = -1.25,
    region_hi_x: float = -0.75,
    region_lo_y: float = -0.75,
    region_hi_y: float = 0.75,
    region_lo_z: float = -0.25,
    region_hi_z: float = 0.25,
    max_amr_levels: int = 1,
    cfl: float = 0.2,
    use_reflux: int = 1,
    use_subcycle: int = 0,
    use_dual_energy: int | None = 0,
    snapshot_prefix: str = "snapshots/plt",
    snapshot_index_interval: int | None = 100,
    snapshot_time_interval: float | None = None,
    checkpoint_index_interval: int | None = 500,
    checkpoint_time_interval: float | None = None,
    checkpoint_prefix: str | None = "checkpoints/chk",
    derived_vars: tuple[str, ...] | None = ("magnetic_divergence", ),
) -> write_param_groups.SimParams:
    """
    Build the full parameter set for one `FieldLoop` run.

    Not validated here: `loop_radius > 0`, `region_lo_* < region_hi_*`, and `refine_based_on`
    validity are all checked by Quokka itself, with clear abort messages.
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
            max_amr_levels=max_amr_levels,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_prefix=snapshot_prefix,
            snapshot_index_interval=snapshot_index_interval,
            snapshot_time_interval=snapshot_time_interval,
            checkpoint_index_interval=checkpoint_index_interval,
            checkpoint_time_interval=checkpoint_time_interval,
            checkpoint_prefix=checkpoint_prefix,
            derived_vars=derived_vars,
        ),
        time_integration_params=param_groups.TimeIntegrationParams(
            cfl=cfl,
            use_reflux=use_reflux,
            use_subcycle=use_subcycle,
            stop_time=stop_time,
            max_time_steps=max_time_steps,
        ),
        hydro_params=param_groups.HydroParams(
            integrator_order=2,
            reconstruction_order=reconstruction_order,
            use_dual_energy=use_dual_energy,
        ),
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
        ),
        setup_params=param_groups.SetupParams(
            group_title="field loop setup",
            param_values={
                "loop_radius": loop_radius,
                "loop_center_x": loop_center_x,
                "loop_center_y": loop_center_y,
                "advection_vz": advection_vz,
                "advection_angle_deg": advection_angle_deg,
                "refine_based_on": refine_based_on,
                "region_lo_x": region_lo_x,
                "region_hi_x": region_hi_x,
                "region_lo_y": region_lo_y,
                "region_hi_y": region_hi_y,
                "region_lo_z": region_lo_z,
                "region_hi_z": region_hi_z,
            },
        ),
    )


## } MODULE
