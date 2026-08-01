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

PROBLEM_NAME = "CurrentSheet"

## perturbation/reversal position are hardcoded relative to the domain; domain must stay fixed,
## and B-field BCs are hardcoded periodic, so cell BCs must match
_DOMAIN_LO = (-0.5, -0.5, -0.5)
_DOMAIN_HI = (0.5, 0.5, 0.5)
_BOUNDARY_CONDITIONS = ("periodic", "periodic", "periodic")

##
## === PROGRAM MAIN
##


def build_sim_params(
    *,
    compute_scheme_key: str,
    averaging_scheme_key: str,
    reconstruction: str,
    num_cells: tuple[int, int, int] = (1024, 1024, 8),
    blocking_factor: int | tuple[int, int, int] = (16, 16, 8),
    max_grid_size: int | tuple[int, int, int] = 128,
    max_time_steps: int = 1_000_000,
    cfl: float = 0.2,
    stop_time: float = 10.0,
    use_reflux: int = 0,
    use_subcycle: int = 0,
    snapshot_prefix: str = "snapshots/plt",
    snapshot_index_interval: int | None = None,
    snapshot_time_interval: float | None = 0.5,
    checkpoint_index_interval: int | None = None,
    checkpoint_time_interval: float | None = 1.0,
    checkpoint_prefix: str | None = "chk",
) -> write_param_groups.SimParams:
    """Build the full parameter set for one `CurrentSheet` run."""
    reconstruction_order = scheme_lookup.resolve_reconstruction_scheme(reconstruction).value
    return write_param_groups.SimParams(
        geometry_params=param_groups.GeometryParams(
            domain_lo=_DOMAIN_LO,
            domain_hi=_DOMAIN_HI,
            boundary_conditions=_BOUNDARY_CONDITIONS,
        ),
        resolution_params=param_groups.ResolutionParams(
            num_cells=num_cells,
            blocking_factor=blocking_factor,
            max_grid_size=max_grid_size,
        ),
        output_file_params=param_groups.OutputFileParams(
            snapshot_prefix=snapshot_prefix,
            snapshot_index_interval=snapshot_index_interval,
            snapshot_time_interval=snapshot_time_interval,
            checkpoint_index_interval=checkpoint_index_interval,
            checkpoint_time_interval=checkpoint_time_interval,
            checkpoint_prefix=checkpoint_prefix,
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
        ),
        ## no `resistivity`: `CurrentSheet` doesn't opt into resistive physics (default `ResistivityModel::none`)
        mhd_params=param_groups.MHDParams(
            emf_compute_scheme=scheme_lookup.resolve_emf_compute_scheme(compute_scheme_key).value,
            emf_averaging_scheme=scheme_lookup.resolve_emf_averaging_scheme(averaging_scheme_key).value,
            reconstruction_order=reconstruction_order,
        ),
    )


## } MODULE
