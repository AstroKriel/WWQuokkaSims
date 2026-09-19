## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import pathlib
import types
import typing

from collections import abc as collections_abc

## third-party
import numpy

from yt import loaders as yt_loaders
from yt.utilities import logger as yt_logger_module

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
)
from jormi import ww_lists
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_types

## local
from . import _snapshot_fields
from ._snapshot_readers.read_fields import AMRLeaves as AMRLeaves  # explicit re-export so pyright treats it as public API
from ._snapshot_readers.read_fields import FieldKey as FieldKey  # explicit re-export so pyright treats it as public API
from ._snapshot_readers.native_resolution import native_slice, native_values
from ._snapshot_readers.uniform_resolution import chunked_domain, expanded_boxes, whole_domain

##
## === SNAPSHOT OPERATOR CLASS
##


class QuokkaSnapshot(
        _snapshot_fields._LoadStoredFields,
        _snapshot_fields._DeriveVelocityFields,
        _snapshot_fields._DeriveEnergyFields,
        _snapshot_fields._DeriveMagneticFields,
        _snapshot_fields._DeriveMHDFields,
):
    """Interface for loading Quokka snapshots with yt."""

    snapshot_dir: pathlib.Path
    verbose: bool
    _yt_dataset: typing.Any | None
    _in_context: bool
    _sim_time: float | None
    _whole_domain_view_cache: dict[int, typing.Any]
    _uniform_domain_3d_cache: dict[int, domain_models.UniformDomain_3D]
    _field_cache: _snapshot_fields.LRUCache

    ##
    ## --- SNAPSHOT LIFECYCLE
    ##

    def __init__(
        self,
        *,
        snapshot_dir: str | pathlib.Path,
        verbose: bool = True,
    ):
        """Initialise a snapshot handle without opening the underlying yt dataset."""
        validate_types.ensure_bool(
            param=verbose,
            param_name="verbose",
        )
        self.snapshot_dir = pathlib.Path(snapshot_dir)
        self.verbose = verbose
        self._yt_dataset = None
        self._in_context = False
        self._sim_time = None
        self._whole_domain_view_cache = {}
        self._uniform_domain_3d_cache = {}
        ## cached fields: density, momentum, magnetic, total_energy, and magnetic_divergence, each per amr_level
        self._field_cache = _snapshot_fields.LRUCache(max_size=10)

    def __enter__(
        self,
    ):
        """Enter the context; open the yt dataset if needed; validate simulation time."""
        self._in_context = True
        self._open_if_needed()
        _ = self.sim_time  # force this property to validate itself
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: types.TracebackType | None,
    ) -> None:
        """Exit the context; close the yt dataset."""
        self._in_context = False
        self._close()

    def _open_if_needed(
        self,
    ) -> None:
        """Open the yt dataset if not already open; cache the simulation time."""
        if self._yt_dataset is None:
            if not self.verbose:
                ## reduce yt verbosity: only print warnings, errors and critical messages
                yt_logger_module.ytLogger.setLevel("WARNING")
            yt_dataset = yt_loaders.load(str(self.snapshot_dir))
            self._sim_time = float(yt_dataset.current_time)
            self._yt_dataset = yt_dataset

    def _close_if_needed(
        self,
    ) -> None:
        """Close the yt dataset unless currently inside a context manager."""
        if not self._in_context:
            self._close()

    def _close(
        self,
    ) -> None:
        """Close the yt dataset; clear cached grid objects; keep simulation time cached."""
        if self._yt_dataset is not None:
            self._yt_dataset.close()
            self._yt_dataset = None
            self._whole_domain_view_cache = {}
            self._uniform_domain_3d_cache = {}
            self._field_cache.clear_cache()

    @property
    def is_open(
        self,
    ) -> bool:
        """`True` iff the yt dataset is currently open."""
        return self._yt_dataset is not None

    def close(
        self,
    ) -> None:
        """Close the yt dataset; exit any active context."""
        self._in_context = False
        self._close()

    ##
    ## --- PROBE SNAPSHOT
    ##

    @property
    def sim_time(
        self,
    ) -> float:
        """Simulation time in code units."""
        if self._sim_time is None:
            self._open_if_needed()
        sim_time = self._sim_time
        if (sim_time is None) or not numpy.isfinite(sim_time):
            msg = f"invalid simulation time in {self.snapshot_dir}: {sim_time!r}."
            manage_log.log_error(text=msg)
            raise RuntimeError(msg)
        return float(sim_time)

    def _validate_amr_level(
        self,
        *,
        amr_level: int,
    ) -> None:
        """Raise if `amr_level` exceeds the finest level actually present in this snapshot."""
        assert self._yt_dataset is not None
        max_amr_level = int(self._yt_dataset.index.max_level)
        if not (0 <= amr_level <= max_amr_level):
            msg = (
                f"requested AMR level {amr_level} but {self.snapshot_dir} only has "
                f"levels 0-{max_amr_level} (max_amr_level={max_amr_level})."
            )
            manage_log.log_error(text=msg)
            raise ValueError(msg)

    def _get_whole_domain_view(
        self,
        *,
        amr_level: int = 0,
    ) -> typing.Any:
        """
        Return a covering grid spanning the whole domain at `amr_level`'s resolution.

        For `amr_level > 0`, this is the composite of the finest data available up to and
        including `amr_level` (coarser regions are filled with a piecewise-constant copy of
        the value from the highest level that does cover them, not interpolated/blended);
        `amr_level=0` (the default) always reads the base level.
        """
        self._open_if_needed()
        assert self._yt_dataset is not None
        self._validate_amr_level(amr_level=amr_level)
        if amr_level not in self._whole_domain_view_cache:
            self._whole_domain_view_cache[amr_level] = whole_domain.initialize_whole_domain_view(
                yt_dataset=self._yt_dataset,
                amr_level=amr_level,
            )
        return self._whole_domain_view_cache[amr_level]

    def _get_available_field_keys(
        self,
    ) -> list[FieldKey]:
        """Return all (field-group, field-name) yt keys available in the snapshot."""
        self._open_if_needed()
        assert self._yt_dataset is not None
        field_keys = sorted(set(self._yt_dataset.field_list))
        self._close_if_needed()
        return field_keys

    def list_available_field_keys(
        self,
    ) -> list[FieldKey]:
        """List all available yt field keys in this snapshot."""
        field_keys = self._get_available_field_keys()
        manage_log.log_items(
            title="Available Fields",
            items=field_keys,
            message=f"Stored under: {self.snapshot_dir}",
            message_position="bottom",
            show_time=False,
        )
        return field_keys

    def is_field_key_available(
        self,
        *,
        field_key: FieldKey,
    ) -> bool:
        """Return `True` iff `field_key` exists in the snapshot."""
        available_keys = set(self._get_available_field_keys())
        return field_key in available_keys

    ##
    ## --- RESOLVE FIELD
    ##

    def _resolve_sfield_key(
        self,
        *,
        field_name: str,
    ) -> FieldKey:
        """Resolve the yt key associated with a named scalar field."""
        if field_name not in _snapshot_fields.YT_SFIELD_KEYS:
            valid_string = ww_lists.as_quoted_string(list(_snapshot_fields.YT_SFIELD_KEYS.keys()))
            msg = f"unknown scalar field `{field_name}`; valid options: {valid_string}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return _snapshot_fields.YT_SFIELD_KEYS[field_name]["key"]

    def _get_sfield_key(
        self,
        *,
        field_name: str,
    ) -> FieldKey:
        """Resolve and validate the yt key associated with a scalar field."""
        field_key = self._resolve_sfield_key(field_name=field_name)
        if not self.is_field_key_available(field_key=field_key):
            msg = f"scalar field `{field_name}` ({field_key[0]}:{field_key[1]}) not found; searched in {self.snapshot_dir}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return field_key

    def _resolve_vfield_key_lookup(
        self,
        *,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        """Return the component yt keys associated with a named vector field."""
        if field_name not in _snapshot_fields.YT_VFIELD_KEYS:
            valid_string = ww_lists.as_quoted_string(list(_snapshot_fields.YT_VFIELD_KEYS.keys()))
            msg = f"unknown vector field `{field_name}`; valid options: {valid_string}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return _snapshot_fields.YT_VFIELD_KEYS[field_name]["keys"]

    def _get_missing_vfield_keys(
        self,
        *,
        field_name: str,
    ) -> list[FieldKey]:
        """Return missing component yt keys for `field_name`."""
        vfield_key_lookup = self._resolve_vfield_key_lookup(field_name=field_name)
        available_keys = set(self._get_available_field_keys())
        return [comp_key for comp_key in vfield_key_lookup.values() if comp_key not in available_keys]

    def _get_vfield_key_lookup(
        self,
        *,
        field_name: str,
    ) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
        """Resolve and validate component keys associated with a named vector field."""
        missing_keys = self._get_missing_vfield_keys(field_name=field_name)
        if missing_keys:
            missing_string = ww_lists.as_quoted_string(
                [f"{yt_group}:{yt_field}" for yt_group, yt_field in missing_keys],
            )
            msg = f"vector field `{field_name}` is incomplete in {self.snapshot_dir}; missing components: {missing_string}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        return self._resolve_vfield_key_lookup(field_name=field_name)

    def _is_vfield_keys_available(
        self,
        *,
        field_name: str,
    ) -> bool:
        """Return `True` iff all components associated with a named vector field exist."""
        return len(self._get_missing_vfield_keys(field_name=field_name)) == 0

    def _load_3d_sarray(
        self,
        *,
        field_key: FieldKey,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> numpy.ndarray:
        """
        Load a scalar field as a 3D `ndarray`.

        By default, reads via yt's `covering_grid` (materializes the whole domain in one
        call; the only option that supports `amr_level > 0`, since compositing across AMR
        levels is delegated to yt). `use_chunked_reader=True` instead reads each
        amr_level=0 box individually, holding only one box plus the output array in
        memory at once, at the cost of only supporting amr_level=0. See `chunked_domain` for
        why: yt's whole-domain grid carries a documented ~6x memory overhead on top of
        the output array's own size.
        """
        if use_chunked_reader and (amr_level != 0):
            raise ValueError(
                "use_chunked_reader=True only supports amr_level=0 (no cross-level"
                f" compositing implemented); got amr_level={amr_level}.",
            )
        self._open_if_needed()
        assert self._yt_dataset is not None
        if field_key not in self._yt_dataset.field_list:
            self._close_if_needed()
            raise KeyError(f"field {field_key} not found; searched in {self.snapshot_dir}.")
        if use_chunked_reader:
            sarray_3d = chunked_domain.load_sarray(
                yt_dataset=self._yt_dataset,
                field_key=field_key,
            )
        else:
            whole_domain_view = self._get_whole_domain_view(amr_level=amr_level)
            sarray_3d = whole_domain.load_sarray(
                whole_domain_view=whole_domain_view,
                field_key=field_key,
            )
        self._close_if_needed()
        return sarray_3d

    def load_amr_leaves(
        self,
        *,
        field_key: FieldKey,
    ) -> AMRLeaves:
        """
        Load one scalar field at every leaf cell across the full AMR hierarchy.

        Each returned value keeps its own box's native `dx` and position (see
        `native_values`); cells covered by a finer box are excluded, so every physical
        cell in the hierarchy contributes exactly once, at whichever resolution it
        actually exists. There is no `amr_level`/`use_chunked_reader` choice here: this
        reader always walks every level natively, since resampling onto one resolution
        (what both of those options do) is exactly what this reader exists to avoid.
        """
        self._open_if_needed()
        assert self._yt_dataset is not None
        if field_key not in self._yt_dataset.field_list:
            self._close_if_needed()
            raise KeyError(f"field {field_key} not found; searched in {self.snapshot_dir}.")
        amr_leaves = native_values.load_amr_leaves(
            yt_dataset=self._yt_dataset,
            field_key=field_key,
        )
        self._close_if_needed()
        return amr_leaves

    def _load_amr_leaves_of_derived_field(
        self,
        *,
        field_keys: tuple[FieldKey, ...],
        derive_fn: collections_abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> AMRLeaves:
        """
        Derive a scalar field at every leaf cell, box-by-box, without ever holding a
        full-domain array: only the returned `AMRLeaves` is ever a full-hierarchy result.
        See `native_values.load_amr_leaves_of_derived_field` for `derive_fn`.
        """
        self._open_if_needed()
        assert self._yt_dataset is not None
        amr_leaves = native_values.load_amr_leaves_of_derived_field(
            yt_dataset=self._yt_dataset,
            field_keys=field_keys,
            derive_fn=derive_fn,
        )
        self._close_if_needed()
        return amr_leaves

    def load_native_slice_sarray(
        self,
        *,
        field_key: FieldKey,
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Read one scalar field, and its per-pixel native cell width, on a genuine AMR-native slice.

        Unlike `_load_3d_sarray`, this never resamples onto one resolution: each pixel is
        read from whichever box actually covers it, at the finest level present, so a
        pixel inside a refined region keeps its true (finer) cell width and one outside
        keeps its true (coarser) cell width, rather than both being forced to share one value.
        """
        self._open_if_needed()
        assert self._yt_dataset is not None
        if field_key not in self._yt_dataset.field_list:
            self._close_if_needed()
            raise KeyError(f"field {field_key} not found; searched in {self.snapshot_dir}.")
        slice_axis_index = cartesian_axes.get_axis_index(axis_to_slice)
        native_slice_view = native_slice.initialize_native_slice_view(
            yt_dataset=self._yt_dataset,
            slice_axis_index=slice_axis_index,
            slice_coordinate=slice_coordinate,
        )
        sarray_2d = native_slice.load_sarray(
            native_slice_view=native_slice_view,
            field_key=field_key,
        )
        cell_width_2d = native_slice.load_cell_widths(
            native_slice_view=native_slice_view,
            slice_axis_index=slice_axis_index,
        )
        self._close_if_needed()
        return sarray_2d, cell_width_2d

    def _load_native_slice_of_derived_field(
        self,
        *,
        field_keys: tuple[FieldKey, ...],
        derive_fn: collections_abc.Callable[[numpy.ndarray], numpy.ndarray],
        axis_to_slice: cartesian_axes.CartesianAxis_3D,
        slice_coordinate: float,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Derive a scalar field on a genuine AMR-native slice. See
        `native_slice.load_native_slice_of_derived_field` for `derive_fn`.
        """
        self._open_if_needed()
        assert self._yt_dataset is not None
        slice_axis_index = cartesian_axes.get_axis_index(axis_to_slice)
        sarray_2d, cell_width_2d = native_slice.load_native_slice_of_derived_field(
            yt_dataset=self._yt_dataset,
            field_keys=field_keys,
            derive_fn=derive_fn,
            slice_axis_index=slice_axis_index,
            slice_coordinate=slice_coordinate,
        )
        self._close_if_needed()
        return sarray_2d, cell_width_2d

    def _iterate_expanded_boxes_of_vfield(
        self,
        *,
        field_name: str,
        num_extra_cells: int,
        amr_level: int = 0,
    ) -> collections_abc.Iterator[expanded_boxes.ExpandedFArray]:
        """
        Yield, for each amr_level=0 box, an expanded raw vector-field block (components
        stacked along axis 0, in x/y/z order) and the domain-index slices its own cells
        belong to. Only reads raw field values via yt's `retrieve_ghost_zones`: no
        derivative or other computation happens here, and the caller is responsible for
        trimming the outer `num_extra_cells` layer of whatever it computes before
        placing a result at the yielded slices. See `expanded_boxes` for details.

        Only amr_level=0 is supported (boxes tile the domain with no cross-level
        compositing). Periodicity is forced so the expanded region near a domain edge
        wraps correctly, matching `load_3d_uniform_domain`'s default.
        """
        if amr_level != 0:
            raise ValueError(
                "expanded-box chunked reading only supports amr_level=0 (no cross-level"
                f" compositing implemented); got amr_level={amr_level}.",
            )
        self._open_if_needed()
        assert self._yt_dataset is not None
        vfield_key_lookup = self._get_vfield_key_lookup(field_name=field_name)
        self._yt_dataset.force_periodicity()
        try:
            yield from expanded_boxes.iterate_expanded_boxes_of_vfield(
                yt_dataset=self._yt_dataset,
                vfield_key_lookup=vfield_key_lookup,
                num_extra_cells=num_extra_cells,
            )
        finally:
            self._close_if_needed()

    def _iterate_expanded_boxes_of_derived_vfield(
        self,
        *,
        field_keys: tuple[FieldKey, ...],
        derive_fn: collections_abc.Callable[[numpy.ndarray], numpy.ndarray],
        num_extra_cells: int,
        amr_level: int = 0,
    ) -> collections_abc.Iterator[expanded_boxes.ExpandedFArray]:
        """
        Yield, for each amr_level=0 box, a derived vector field's own expanded block
        (`field_keys` read raw and in lockstep for the same box, then combined via
        `derive_fn(raw_expanded_farray) -> derived_expanded_varray`) and the domain-index
        slices its own cells belong to.

        For a derived field (not itself stored) that needs differentiating box-by-box:
        reading its raw ingredients via independent `_iterate_expanded_boxes_of_vfield`-style
        calls would not guarantee the same box in the same order from each, so they must
        be read in lockstep here instead.
        """
        if amr_level != 0:
            raise ValueError(
                "expanded-box chunked reading only supports amr_level=0 (no cross-level"
                f" compositing implemented); got amr_level={amr_level}.",
            )
        self._open_if_needed()
        assert self._yt_dataset is not None
        self._yt_dataset.force_periodicity()
        try:
            for expanded_farray in expanded_boxes.iterate_expanded_boxes(
                    yt_dataset=self._yt_dataset,
                    field_keys=field_keys,
                    num_extra_cells=num_extra_cells,
            ):
                derived_expanded_varray = derive_fn(expanded_farray.farray)
                yield expanded_boxes.ExpandedFArray(
                    farray=derived_expanded_varray,
                    cell_range=expanded_farray.cell_range,
                )
        finally:
            self._close_if_needed()

    def _iterate_expanded_boxes_of_velocity_vfield(
        self,
        *,
        num_extra_cells: int,
        amr_level: int = 0,
    ) -> collections_abc.Iterator[expanded_boxes.ExpandedFArray]:
        """Velocity's own expanded block: momentum's raw expanded box divided elementwise by
        density's raw expanded box. See `_iterate_expanded_boxes_of_derived_vfield`."""
        momentum_key_lookup = self._get_vfield_key_lookup(field_name="momentum")
        density_key = self._get_sfield_key(field_name="density")
        field_keys = tuple(
            momentum_key_lookup[comp_axis] for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER
        ) + (density_key, )

        def derive_velocity_fn(
            raw_expanded_farray: numpy.ndarray,
        ) -> numpy.ndarray:
            expanded_mom_varray = raw_expanded_farray[:3]
            expanded_rho_sarray = raw_expanded_farray[3]
            return expanded_mom_varray / expanded_rho_sarray[numpy.newaxis, ...]

        yield from self._iterate_expanded_boxes_of_derived_vfield(
            field_keys=field_keys,
            derive_fn=derive_velocity_fn,
            num_extra_cells=num_extra_cells,
            amr_level=amr_level,
        )

    def _derive_chunked_vfield_from_source(
        self,
        *,
        expanded_box_source: collections_abc.Iterator[expanded_boxes.ExpandedFArray],
        num_extra_cells: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.VectorField_3D:
        """
        Derive a 3-component field box-by-box from `expanded_box_source`, without ever
        holding a full-domain array of the source data or of any intermediate: only the
        returned field is ever a full-domain array. See `_derive_chunked_vfield_from_field_name` for the
        `field_name`-based convenience wrapper around this.
        """
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        out_varray_3d = numpy.full((3, *uniform_domain_3d.resolution), numpy.nan, dtype=numpy.float64)
        for expanded_farray in expanded_box_source:
            out_varray_3d[(slice(None), *expanded_farray.cell_range)
                          ] = derive_fn(expanded_farray.farray, num_extra_cells)
        if numpy.isnan(out_varray_3d).any():
            raise ValueError(
                f"some cells were never written by any amr_level=0 box while computing"
                f" {output_field_name}; the boxes do not fully tile the domain.",
            )
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=out_varray_3d,
            uniform_domain_3d=uniform_domain_3d,
            sim_time=self.sim_time,
            field_name=output_field_name,
            latex_label=output_latex_label,
        )

    def _derive_chunked_vfield_from_field_name(
        self,
        *,
        field_name: str,
        grad_order: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.VectorField_3D:
        """
        Derive a 3-component field from `field_name`, box-by-box, without ever holding a
        full-domain array of `field_name` or of any intermediate: only the returned field
        is ever a full-domain array.

        `derive_fn(expanded_varray, num_extra_cells)` receives one amr_level=0 box's raw
        `field_name` data, expanded by `num_extra_cells` cells, and must return the
        already-trimmed local result for that box's own cells (see
        `expanded_boxes.trim_expanded_box`): whatever `derive_fn` does internally
        (differentiate, combine with other locally-derived quantities, ...),
        memory-boundedness only holds if it never retains more than one box's worth of
        data itself.
        """
        num_extra_cells = expanded_boxes.compute_num_extra_cells(grad_order=grad_order)
        expanded_box_source = self._iterate_expanded_boxes_of_vfield(
            field_name=field_name,
            num_extra_cells=num_extra_cells,
            amr_level=amr_level,
        )
        return self._derive_chunked_vfield_from_source(
            expanded_box_source=expanded_box_source,
            num_extra_cells=num_extra_cells,
            amr_level=amr_level,
            derive_fn=derive_fn,
            output_field_name=output_field_name,
            output_latex_label=output_latex_label,
        )

    def _derive_chunked_sfield_from_source(
        self,
        *,
        expanded_box_source: collections_abc.Iterator[expanded_boxes.ExpandedFArray],
        num_extra_cells: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.ScalarField_3D:
        """Scalar-output counterpart of `_derive_chunked_vfield_from_source`."""
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        out_sarray_3d = numpy.full(uniform_domain_3d.resolution, numpy.nan, dtype=numpy.float64)
        for expanded_farray in expanded_box_source:
            out_sarray_3d[expanded_farray.cell_range] = derive_fn(expanded_farray.farray, num_extra_cells)
        if numpy.isnan(out_sarray_3d).any():
            raise ValueError(
                f"some cells were never written by any amr_level=0 box while computing"
                f" {output_field_name}; the boxes do not fully tile the domain.",
            )
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=out_sarray_3d,
            uniform_domain_3d=uniform_domain_3d,
            sim_time=self.sim_time,
            field_name=output_field_name,
            latex_label=output_latex_label,
        )

    def _derive_chunked_r2tfield_from_source(
        self,
        *,
        expanded_box_source: collections_abc.Iterator[expanded_boxes.ExpandedFArray],
        num_extra_cells: int,
        amr_level: int,
        derive_fn: collections_abc.Callable[[numpy.ndarray, int], numpy.ndarray],
        output_field_name: str,
        output_latex_label: str,
    ) -> field_models.RankTwoTensorField_3D:
        """Rank-2-tensor-output counterpart of `_derive_chunked_vfield_from_source`."""
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        out_r2tarray_3d = numpy.full((3, 3, *uniform_domain_3d.resolution), numpy.nan, dtype=numpy.float64)
        for expanded_farray in expanded_box_source:
            out_r2tarray_3d[(slice(None), slice(None), *expanded_farray.cell_range)
                            ] = derive_fn(expanded_farray.farray, num_extra_cells)
        if numpy.isnan(out_r2tarray_3d).any():
            raise ValueError(
                f"some cells were never written by any amr_level=0 box while computing"
                f" {output_field_name}; the boxes do not fully tile the domain.",
            )
        return field_models.RankTwoTensorField_3D.from_3d_r2tarray(
            r2tarray_3d=out_r2tarray_3d,
            uniform_domain_3d=uniform_domain_3d,
            sim_time=self.sim_time,
            field_name=output_field_name,
            latex_label=output_latex_label,
        )

    ##
    ## --- GENERIC READ ENGINE
    ##

    def _extract_3d_sarray(
        self,
        *,
        sfield_3d: field_models.ScalarField_3D,
        param_name: str,
    ) -> numpy.ndarray:
        return field_models.extract_3d_sarray(
            sfield_3d=sfield_3d,
            param_name=param_name,
        )

    def _extract_3d_varray(
        self,
        *,
        vfield_3d: field_models.VectorField_3D,
        param_name: str,
    ) -> numpy.ndarray:
        return field_models.extract_3d_varray(
            vfield_3d=vfield_3d,
            param_name=param_name,
        )

    def load_3d_sfield(
        self,
        *,
        field_key: FieldKey,
        field_name: str,
        latex_label: str,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        """
        Wrap a scalar array as `ScalarField_3D` with a `field_name`, `latex_label`, and
        `sim_time`. See `_load_3d_sarray` for `use_chunked_reader`.
        """
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        validate_types.ensure_nonempty_string(
            param=latex_label,
            param_name="latex_label",
        )
        sarray_3d = self._load_3d_sarray(
            field_key=field_key,
            amr_level=amr_level,
            use_chunked_reader=use_chunked_reader,
        )
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray_3d,
            uniform_domain_3d=uniform_domain_3d,
            field_name=field_name,
            latex_label=latex_label,
            sim_time=self.sim_time,
        )

    def load_3d_vfield(
        self,
        *,
        vfield_key_lookup: dict[cartesian_axes.CartesianAxis_3D, FieldKey],
        field_name: str,
        latex_label: str,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.VectorField_3D:
        """
        Load and stack 3 components into a `VectorField_3D` with a `field_name`,
        `latex_label`, and `sim_time`. See `_load_3d_sarray` for `use_chunked_reader`,
        applied per component.
        """
        if set(vfield_key_lookup) != set(cartesian_axes.DEFAULT_3D_AXES_ORDER):
            received_axes = [
                axis.value for axis in sorted(vfield_key_lookup.keys(), key=lambda _axis: _axis.value)
            ]
            expected_axes = [axis.value for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER]
            msg = f"`vfield_key_lookup` must contain all 3 components {expected_axes}; got {received_axes}."
            manage_log.log_error(text=msg)
            raise KeyError(msg)
        validate_types.ensure_nonempty_string(
            param=field_name,
            param_name="field_name",
        )
        validate_types.ensure_nonempty_string(
            param=latex_label,
            param_name="latex_label",
        )
        if use_chunked_reader and (amr_level != 0):
            raise ValueError(
                "use_chunked_reader=True only supports amr_level=0 (no cross-level"
                f" compositing implemented); got amr_level={amr_level}.",
            )
        self._open_if_needed()
        assert self._yt_dataset is not None
        if use_chunked_reader:
            whole_domain_view = None
        else:
            whole_domain_view = self._get_whole_domain_view(amr_level=amr_level)
        grouped_sarrays: dict[cartesian_axes.CartesianAxis_3D, numpy.ndarray] = {}
        for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER:
            comp_key = vfield_key_lookup[comp_axis]
            if comp_key not in self._yt_dataset.field_list:
                self._close_if_needed()
                raise KeyError(f"field {comp_key} not found; searched in {self.snapshot_dir}.")
            if use_chunked_reader:
                comp_sarray = chunked_domain.load_sarray(
                    yt_dataset=self._yt_dataset,
                    field_key=comp_key,
                )
            else:
                assert whole_domain_view is not None
                comp_sarray = numpy.asarray(whole_domain_view[comp_key], dtype=numpy.float64)
            if comp_sarray.ndim != 3:
                self._close_if_needed()
                raise ValueError(f"expected a 3D array for {comp_key}; got shape {comp_sarray.shape}.")
            grouped_sarrays[comp_axis] = comp_sarray
        self._close_if_needed()
        sim_time = self.sim_time
        varray_3d = numpy.stack(
            [grouped_sarrays[comp_axis] for comp_axis in cartesian_axes.DEFAULT_3D_AXES_ORDER],
            axis=0,
        )
        uniform_domain_3d = self.load_3d_uniform_domain(amr_level=amr_level)
        return field_models.VectorField_3D.from_3d_varray(
            varray_3d=varray_3d,
            uniform_domain_3d=uniform_domain_3d,
            sim_time=sim_time,
            field_name=field_name,
            latex_label=latex_label,
        )

    ##
    ## --- DOMAIN
    ##

    def load_3d_uniform_domain(
        self,
        *,
        force_periodicity: bool = True,
        amr_level: int = 0,
    ) -> domain_models.UniformDomain_3D:
        """
        Return uniform domain metadata: bounds, resolution, and periodicity; result is cached per `amr_level`.

        `resolution` is the base-level `domain_dimensions` scaled by `refinement_ratio**amr_level`, matching the
        resolution `_get_whole_domain_view(amr_level=...)` actually returns, so the two stay consistent.
        `force_periodicity` only takes effect on the first call for a given `amr_level`; yt cannot read
        periodicity reliably.
        """
        validate_types.ensure_bool(
            param=force_periodicity,
            param_name="force_periodicity",
        )
        self._open_if_needed()
        assert self._yt_dataset is not None
        self._validate_amr_level(amr_level=amr_level)
        if amr_level in self._uniform_domain_3d_cache:
            cached_uniform_domain_3d = self._uniform_domain_3d_cache[amr_level]
            self._close_if_needed()
            return cached_uniform_domain_3d
        else:
            x_min, y_min, z_min = (float(value) for value in self._yt_dataset.domain_left_edge)
            x_max, y_max, z_max = (float(value) for value in self._yt_dataset.domain_right_edge)
            refinement_ratio = int(self._yt_dataset.refine_by)
            num_cells_x, num_cells_y, num_cells_z = (
                int(num_cells) * refinement_ratio**amr_level
                for num_cells in self._yt_dataset.domain_dimensions
            )
            is_periodic_x, is_periodic_y, is_periodic_z = (
                (bool(is_periodic) or force_periodicity) for is_periodic in self._yt_dataset.periodicity
            )
            self._close_if_needed()
            uniform_domain_3d = domain_models.UniformDomain_3D(
                periodicity=(
                    is_periodic_x,
                    is_periodic_y,
                    is_periodic_z,
                ),
                resolution=(
                    num_cells_x,
                    num_cells_y,
                    num_cells_z,
                ),
                domain_bounds=(
                    (x_min, x_max),
                    (y_min, y_max),
                    (z_min, z_max),
                ),
            )
            self._uniform_domain_3d_cache[amr_level] = uniform_domain_3d
            return uniform_domain_3d


## } MODULE
