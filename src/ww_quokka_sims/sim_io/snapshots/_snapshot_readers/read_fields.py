## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import typing

## third-party
import numpy

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_validation import validate_arrays

##
## === DATA STRUCTURES
##

FieldKey: typing.TypeAlias = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class AMRLeaves:
    """One field's values at every leaf cell in the AMR hierarchy, each paired with its own
    native cell width and physical position; unordered and not resampled onto a common resolution."""

    values: numpy.ndarray
    cell_width: numpy.ndarray
    positions: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        validate_arrays.ensure_same_shape(
            array_a=self.values,
            array_b=self.cell_width,
            param_name_a="<values>",
            param_name_b="<cell_width>",
        )
        validate_arrays.ensure_shape(
            array=self.positions,
            expected_shape=(*self.values.shape, 3),
            param_name="<positions>",
        )


##
## === YT FIELD MAPPINGS
##

## boxlib uses "x-", "y-", "z-" prefixes for vector component field names
BOXLIB_3D_AXES_LABELS: dict[cartesian_axes.CartesianAxis_3D, str] = {
    cartesian_axes.CartesianAxis_3D.X0: "x",
    cartesian_axes.CartesianAxis_3D.X1: "y",
    cartesian_axes.CartesianAxis_3D.X2: "z",
}


def create_boxlib_vkeys(
    *,
    field_name: str,
) -> dict[cartesian_axes.CartesianAxis_3D, FieldKey]:
    """Map `CartesianAxis_3D` to yt field keys using the pattern `("boxlib", "<axis>-<field_name>")` for each axis."""
    return {
        axis: ("boxlib", f"{BOXLIB_3D_AXES_LABELS[axis]}-{field_name}")
        for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER
    }


## } MODULE
