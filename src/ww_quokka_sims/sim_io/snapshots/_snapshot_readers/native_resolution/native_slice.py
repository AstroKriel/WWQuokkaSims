## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import typing

from collections import abc as collections_abc

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes

## local
from .. import read_fields

##
## === INTERNAL HELPERS
##


def ensure_consistent_slice_geometry(
    *,
    reference_cell_width_2d: numpy.ndarray,
    other_cell_width_2d: numpy.ndarray,
) -> None:
    """
    Raise if two independently-read native slices do not share the same pixel geometry.

    Composing multiple native-slice reads elementwise (eg. `internal_energy = total_energy -
    kinetic_energy - magnetic_energy`) is only valid if pixel `[i, j]` refers to the same
    physical location in each: two calls with the same `axis_to_slice`/`slice_coordinate`
    on the same open dataset always build the same fixed-resolution frame buffer, so
    independent reads agree, but that is an assumption about yt's behaviour, not something
    the type system enforces.
    """
    if not numpy.array_equal(reference_cell_width_2d, other_cell_width_2d):
        raise ValueError(
            "native slices from independent reads do not share the same pixel geometry;"
            " cannot combine their values elementwise.",
        )


##
## === PUBLIC FUNCTIONS
##


def initialize_native_slice_view(
    *,
    yt_dataset: typing.Any,
    slice_axis_index: int,
    slice_coordinate: float,
) -> typing.Any:
    """
    Build a view through a genuine AMR-native slice of the domain, at the resolution
    of the finest AMR level actually present.

    Unlike `whole_domain`'s covering grid, each output pixel is sampled from
    whichever box actually covers it there; reading a cell width through the
    returned view therefore gives each pixel's own true native cell width, not one
    value shared across the whole domain.
    """
    refinement_ratio = int(yt_dataset.refine_by)
    max_amr_level = int(yt_dataset.index.max_level)
    base_resolution = numpy.array([int(num_cells) for num_cells in yt_dataset.domain_dimensions])
    finest_resolution = base_resolution * (refinement_ratio**max_amr_level)
    in_plane_axis_indices = [
        axis_index for axis_index in range(len(cartesian_axes.DEFAULT_3D_AXES_ORDER))
        if axis_index != slice_axis_index
    ]
    domain_left_edge = numpy.array([float(value) for value in yt_dataset.domain_left_edge])
    domain_right_edge = numpy.array([float(value) for value in yt_dataset.domain_right_edge])
    center_position = numpy.array([float(value) for value in yt_dataset.domain_center])
    center_position[slice_axis_index] = slice_coordinate
    yt_slice = yt_dataset.slice(slice_axis_index, slice_coordinate)
    return yt_slice.to_frb(
        width=(
            domain_right_edge[in_plane_axis_indices[0]] - domain_left_edge[in_plane_axis_indices[0]],
            "code_length",
        ),
        height=(
            domain_right_edge[in_plane_axis_indices[1]] - domain_left_edge[in_plane_axis_indices[1]],
            "code_length",
        ),
        resolution=(
            int(finest_resolution[in_plane_axis_indices[0]]),
            int(finest_resolution[in_plane_axis_indices[1]]),
        ),
        center=tuple(center_position),
    )


def load_sarray(
    *,
    native_slice_view: typing.Any,
    field_key: read_fields.FieldKey,
) -> numpy.ndarray:
    """Read one field out of an already-built native-slice view as a plain 2D array."""
    sarray_2d = numpy.asarray(native_slice_view[field_key], dtype=numpy.float64)
    if sarray_2d.ndim != 2:
        raise ValueError(f"expected a 2D array for {field_key}; got shape {sarray_2d.shape}.")
    return numpy.ascontiguousarray(sarray_2d)


def load_cell_widths(
    *,
    native_slice_view: typing.Any,
    slice_axis_index: int,
) -> numpy.ndarray:
    """
    Read each pixel's true native cell width on an already-built native-slice view.

    Reads both in-plane axes' cell widths (not just the out-of-plane one, which is only
    correct when `slice_axis_index == 2`) and requires them to match: a single scalar
    cell width per pixel is only well-defined when the two in-plane axes are isotropic,
    regardless of whatever the out-of-plane axis's width happens to be.
    """
    in_plane_axis_indices = [
        axis_index for axis_index in range(len(cartesian_axes.DEFAULT_3D_AXES_ORDER))
        if axis_index != slice_axis_index
    ]
    in_plane_axis_names = [
        read_fields.BOXLIB_3D_AXES_LABELS[cartesian_axes.DEFAULT_3D_AXES_ORDER[axis_index]]
        for axis_index in in_plane_axis_indices
    ]
    first_in_plane_cell_width_2d = load_sarray(
        native_slice_view=native_slice_view,
        field_key=("index", f"d{in_plane_axis_names[0]}"),
    )
    second_in_plane_cell_width_2d = load_sarray(
        native_slice_view=native_slice_view,
        field_key=("index", f"d{in_plane_axis_names[1]}"),
    )
    if not numpy.allclose(
            first_in_plane_cell_width_2d,
            second_in_plane_cell_width_2d,
            rtol=1e-6,
            atol=0.0,
    ):
        raise ValueError(
            "pixel has anisotropic in-plane cells; a single native cell width is only"
            " well-defined when the two in-plane axes match.",
        )
    return first_in_plane_cell_width_2d


def load_native_slice_of_derived_field(
    *,
    yt_dataset: typing.Any,
    field_keys: tuple[read_fields.FieldKey, ...],
    derive_fn: collections_abc.Callable[[numpy.ndarray], numpy.ndarray],
    slice_axis_index: int,
    slice_coordinate: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Read a derived scalar field on a genuine AMR-native slice, combining `field_keys`
    pointwise via `derive_fn`. See `load_sarray`/`load_cell_widths` for the single,
    already-stored-field case this generalises.

    `derive_fn(raw_sarray_stack) -> derived_sarray_2d` receives `field_keys` stacked along
    axis 0, each already sampled from the same native-slice view (so pixel-for-pixel
    aligned), and must return a same-shaped 2D array. `derive_fn` must be pointwise (no
    spatial derivative): neighbouring pixels can be sampled from different native
    resolutions, so there is no well-defined stencil between them.
    """
    native_slice_view = initialize_native_slice_view(
        yt_dataset=yt_dataset,
        slice_axis_index=slice_axis_index,
        slice_coordinate=slice_coordinate,
    )
    raw_sarray_stack = numpy.stack(
        [load_sarray(
            native_slice_view=native_slice_view,
            field_key=field_key,
        ) for field_key in field_keys],
        axis=0,
    )
    derived_sarray_2d = derive_fn(raw_sarray_stack)
    cell_width_2d = load_cell_widths(
        native_slice_view=native_slice_view,
        slice_axis_index=slice_axis_index,
    )
    return derived_sarray_2d, cell_width_2d


## } MODULE
