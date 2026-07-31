## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## personal
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_types

## local
from . import _render_param_groups
from . import param_groups

##
## === PARAMETER BUNDLE
##


@dataclass(frozen=True)
class SimParams:
    """
    One full `write_sim_params_toml` call's worth of dataclasses, plus a `write` convenience method.

    Fields
    ---
    - `geometry_params`, `resolution_params`, `output_file_params`, `time_integration_params`,
      `hydro_params`, `mhd_params`:
        See the matching dataclass in `sim_params.param_groups`.

    - `setup_params`:
        Problem-specific parameters; omit for problem types with none.

    - `amr_verbosity`:
        AMReX's own `amr.v` logging verbosity; omit for Quokka's default.
    """

    geometry_params: param_groups.GeometryParams
    resolution_params: param_groups.ResolutionParams
    output_file_params: param_groups.OutputFileParams
    time_integration_params: param_groups.TimeIntegrationParams
    hydro_params: param_groups.HydroParams
    mhd_params: param_groups.MHDParams
    setup_params: param_groups.SetupParams | None = None
    amr_verbosity: int = 1

    def write(
        self,
        *,
        output_path: str | Path,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> Path:
        return write_sim_params_toml(
            output_path=output_path,
            geometry_params=self.geometry_params,
            resolution_params=self.resolution_params,
            output_file_params=self.output_file_params,
            time_integration_params=self.time_integration_params,
            hydro_params=self.hydro_params,
            mhd_params=self.mhd_params,
            setup_params=self.setup_params,
            amr_verbosity=self.amr_verbosity,
            overwrite=overwrite,
            verbose=verbose,
        )


##
## === FILE PATH VALIDATION
##


def _ensure_path_is_valid(
    output_path: str | Path,
) -> Path:
    """Ensure `output_path` is a valid `sim_params.toml` path and return it as an absolute Path."""
    validate_types.ensure_not_none(
        param=output_path,
        param_name="output_path",
    )
    output_path = Path(output_path).absolute()
    if output_path.name != "sim_params.toml":
        raise ValueError(
            f"file must be named `sim_params.toml` (matches `tools/run_sim.sh`'s bare-filename convention), "
            f"got {output_path}.",
        )
    return output_path


##
## === PARAM GROUP TITLES
##


class _ParamGroupTitle:
    GEOMETRY: str = "geometry"
    RESOLUTION: str = "resolution"
    VERBOSITY: str = "verbosity"
    OUTPUT: str = "output"
    TIME_INTEGRATION: str = "time integration"
    HYDRO: str = "hydro"
    MHD: str = "mhd"


##
## === SECTION BUILDERS
##


def _build_geometry_lines(
    geometry_params: param_groups.GeometryParams,
) -> list[str]:
    assignment_lines = [
        _render_param_groups.render_key_value(key="geometry.prob_lo", value=list(geometry_params.domain_lo)),
        _render_param_groups.render_key_value(key="geometry.prob_hi", value=list(geometry_params.domain_hi)),
    ]
    if geometry_params.is_boundary_periodic is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(
                key="geometry.is_periodic",
                value=list(geometry_params.is_boundary_periodic),
            ),
        )
    else:
        assignment_lines.append(
            _render_param_groups.render_key_value(
                key="quokka.bc",
                value=list(geometry_params.boundary_conditions),  # pyright: ignore[reportArgumentType]
            ),
        )
    return assignment_lines


def _build_resolution_lines(
    resolution_params: param_groups.ResolutionParams,
) -> list[str]:
    assignment_lines = [
        _render_param_groups.render_key_value(key="amr.n_cell", value=list(resolution_params.num_cells)),
        _render_param_groups.render_key_value(key="amr.max_level", value=resolution_params.max_amr_levels),
        *_render_param_groups.expand_per_axis(resolution_params.blocking_factor, key_prefix="amr.blocking_factor"),
        *_render_param_groups.expand_per_axis(resolution_params.max_grid_size, key_prefix="amr.max_grid_size"),
    ]
    if resolution_params.num_refinement_buffer_cells is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(
                key="amr.n_error_buf",
                value=resolution_params.num_refinement_buffer_cells,
            ),
        )
    return assignment_lines


def _build_output_lines(
    output_file_params: param_groups.OutputFileParams,
) -> list[str]:
    assignment_lines: list[str] = []
    if output_file_params.checkpoint_index_interval is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="checkpoint_interval", value=output_file_params.checkpoint_index_interval),
        )
    if output_file_params.checkpoint_time_interval is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="checkpointtime_interval", value=output_file_params.checkpoint_time_interval),
        )
    if output_file_params.checkpoint_prefix is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="checkpoint_prefix", value=output_file_params.checkpoint_prefix),
        )
    if output_file_params.snapshot_index_interval is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="plotfile_interval", value=output_file_params.snapshot_index_interval),
        )
    if output_file_params.snapshot_time_interval is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="plottime_interval", value=output_file_params.snapshot_time_interval),
        )
    if output_file_params.snapshot_prefix is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="plotfile_prefix", value=output_file_params.snapshot_prefix),
        )
    if output_file_params.derived_vars is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="derived_vars", value=list(output_file_params.derived_vars)),
        )
    return assignment_lines


def _build_time_integration_lines(
    time_integration_params: param_groups.TimeIntegrationParams,
) -> list[str]:
    assignment_lines = [_render_param_groups.render_key_value(key="cfl", value=time_integration_params.cfl)]
    optional_keys = (
        ("do_reflux", time_integration_params.use_reflux),
        ("do_subcycle", time_integration_params.use_subcycle),
        ("do_tracers", time_integration_params.use_tracers),
        ("stop_time", time_integration_params.stop_time),
        ("max_timesteps", time_integration_params.max_time_steps),
    )
    for key, value in optional_keys:
        if value is not None:
            assignment_lines.append(_render_param_groups.render_key_value(key=key, value=value))
    return assignment_lines


def _build_hydro_lines(
    hydro_params: param_groups.HydroParams,
) -> list[str]:
    assignment_lines = [
        _render_param_groups.render_key_value(key="hydro.rk_integrator_order", value=hydro_params.integrator_order),
        _render_param_groups.render_key_value(key="hydro.reconstruction_order", value=hydro_params.reconstruction_order),
    ]
    if hydro_params.use_dual_energy is not None:
        assignment_lines.append(
            _render_param_groups.render_key_value(key="hydro.use_dual_energy", value=hydro_params.use_dual_energy),
        )
    return assignment_lines


def _build_mhd_lines(
    mhd_params: param_groups.MHDParams,
) -> list[str]:
    assignment_lines = [
        _render_param_groups.render_key_value(key="mhd.emf_compute_scheme", value=mhd_params.emf_compute_scheme),
        _render_param_groups.render_key_value(key="mhd.emf_averaging_scheme", value=mhd_params.emf_averaging_scheme),
        _render_param_groups.render_key_value(key="mhd.emf_reconstruction_order", value=mhd_params.reconstruction_order),
    ]
    if mhd_params.resistivity is not None:
        assignment_lines.append(_render_param_groups.render_key_value(key="mhd.resistivity", value=mhd_params.resistivity))
    return assignment_lines


def _build_setup_lines(
    setup_params: param_groups.SetupParams,
) -> list[str]:
    return [
        _render_param_groups.render_key_value(key=f"{setup_params.key_prefix}.{key}", value=value)
        for key, value in setup_params.param_values.items()
    ]


##
## === PROGRAM MAIN
##


def write_sim_params_toml(
    *,
    output_path: str | Path,
    geometry_params: param_groups.GeometryParams,
    resolution_params: param_groups.ResolutionParams,
    output_file_params: param_groups.OutputFileParams,
    time_integration_params: param_groups.TimeIntegrationParams,
    hydro_params: param_groups.HydroParams,
    mhd_params: param_groups.MHDParams,
    setup_params: param_groups.SetupParams | None = None,
    amr_verbosity: int = 1,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Write a Quokka `sim_params.toml` file.

    Not real nested TOML: it is AMReX ParmParse-style flat dotted keys grouped under
    `## <group_title>` comments, in a fixed param group order (`geometry, resolution,
    verbosity, output, time integration, hydro, mhd`, with an optional `setup` param
    group always last).

    Parameters
    ---
    - `amr_verbosity`:
        AMReX's own `amr.v` logging verbosity.

    - `overwrite`:
        Raise if `output_path` already exists and this is `False`.
    """
    validate_types.ensure_finite_int(param=amr_verbosity, param_name="amr_verbosity")
    validate_types.ensure_bool(param=overwrite, param_name="overwrite")
    validate_types.ensure_bool(param=verbose, param_name="verbose")
    output_path = _ensure_path_is_valid(output_path)
    if output_path.is_file() and not overwrite:
        raise FileExistsError(f"sim_params.toml already exists (pass `overwrite=True` to replace it): {output_path}.")

    param_group_entries: list[tuple[str, list[str]]] = [
        (_ParamGroupTitle.GEOMETRY, _build_geometry_lines(geometry_params)),
        (_ParamGroupTitle.RESOLUTION, _build_resolution_lines(resolution_params)),
        (_ParamGroupTitle.VERBOSITY, [_render_param_groups.render_key_value(key="amr.v", value=amr_verbosity)]),
        (_ParamGroupTitle.OUTPUT, _build_output_lines(output_file_params)),
        (_ParamGroupTitle.TIME_INTEGRATION, _build_time_integration_lines(time_integration_params)),
        (_ParamGroupTitle.HYDRO, _build_hydro_lines(hydro_params)),
        (_ParamGroupTitle.MHD, _build_mhd_lines(mhd_params)),
    ]
    if setup_params is not None:
        param_group_entries.append((setup_params.group_title, _build_setup_lines(setup_params)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_param_groups.render_param_groups(param_group_entries))
    if verbose:
        manage_log.log_action(
            title="Write sim_params.toml",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="Saved sim_params.toml.",
            notes={"file": str(output_path)},
        )
    return output_path


## } MODULE
