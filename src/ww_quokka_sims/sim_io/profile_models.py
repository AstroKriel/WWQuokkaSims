## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import re

from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
from numpy.typing import NDArray

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_io import json_io
from jormi.ww_validation import validate_types

##
## === HELPERS
##


def _ensure_field_name(
    field_name: object,
) -> None:
    validate_types.ensure_nonempty_string(
        param=field_name,
        param_name="<field_name>",
    )
    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(field_name)):
        raise ValueError(
            f"`<field_name>` must be snake_case, got: {field_name!r}",
        )


def _ensure_profile_axis(
    profile_axis: object,
) -> None:
    valid = cartesian_axes.VALID_3D_AXIS_LABELS
    if profile_axis not in valid:
        raise ValueError(
            f"`<profile_axis>` must be one of {valid}, got: {profile_axis!r}",
        )


def _ensure_profile_arrays(
    position: object,
    field_value: object,
) -> None:
    validate_types.ensure_ndarray_ndim(
        param=position,
        ndim=1,
        param_name="<position>",
    )
    validate_types.ensure_ndarray_ndim(
        param=field_value,
        ndim=1,
        param_name="<field_value>",
    )
    if len(position) == 0:  # type: ignore[arg-type]
        raise ValueError("`<position>` must be non-empty.")
    if len(position) != len(field_value):  # type: ignore[arg-type]
        raise ValueError(
            f"`<position>` and `<field_value>` must have the same length, "
            f"got {len(position)} and {len(field_value)}.",  # type: ignore[arg-type]
        )


##
## === COMPONENT ARRAYS
##


@dataclass(frozen=True)
class ComponentArrays:
    position: NDArray[numpy.floating]
    field_value: NDArray[numpy.floating]

    def __post_init__(self) -> None:
        _ensure_profile_arrays(
            position=self.position,
            field_value=self.field_value,
        )


##
## === SCALAR PROFILE
##


@dataclass(frozen=True)
class ScalarProfile:
    field_name: str
    sim_time: float
    profile_axis: str
    position: NDArray[numpy.floating]
    field_value: NDArray[numpy.floating]

    def __post_init__(self) -> None:
        _ensure_field_name(self.field_name)
        validate_types.ensure_finite_float(
            param=self.sim_time,
            param_name="<sim_time>",
            allow_none=False,
        )
        _ensure_profile_axis(self.profile_axis)
        _ensure_profile_arrays(
            position=self.position,
            field_value=self.field_value,
        )

    def save_to_file(
        self,
        file_path: Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "field_name": self.field_name,
                "sim_time": self.sim_time,
                "profile_axis": self.profile_axis,
                "position": self.position,
                "field_value": self.field_value,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: Path,
    ) -> "ScalarProfile":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={"field_name", "sim_time", "profile_axis", "position", "field_value"},
            param_name="<ScalarProfile JSON>",
        )
        return cls(
            field_name=data["field_name"],
            sim_time=float(data["sim_time"]),
            profile_axis=data["profile_axis"],
            position=numpy.asarray(data["position"]),
            field_value=numpy.asarray(data["field_value"]),
        )


##
## === VECTOR PROFILE
##


@dataclass(frozen=True)
class VectorProfile:
    field_name: str
    sim_time: float
    profile_axis: str
    components: dict[str, ComponentArrays]

    def __post_init__(self) -> None:
        _ensure_field_name(self.field_name)
        validate_types.ensure_finite_float(
            param=self.sim_time,
            param_name="<sim_time>",
            allow_none=False,
        )
        _ensure_profile_axis(self.profile_axis)
        if not self.components:
            raise ValueError("`<components>` must be non-empty.")
        valid = cartesian_axes.VALID_3D_AXIS_LABELS
        for key in self.components:
            if key not in valid:
                raise ValueError(
                    f"`<components>` key must be one of {valid}, got: {key!r}",
                )

    def save_to_file(
        self,
        file_path: Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "field_name": self.field_name,
                "sim_time": self.sim_time,
                "profile_axis": self.profile_axis,
                "components": {
                    comp_axis: {
                        "position": comp.position,
                        "field_value": comp.field_value,
                    }
                    for comp_axis, comp in self.components.items()
                },
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: Path,
    ) -> "VectorProfile":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={"field_name", "sim_time", "profile_axis", "components"},
            param_name="<VectorProfile JSON>",
        )
        components = {
            comp_axis: ComponentArrays(
                position=numpy.asarray(comp_data["position"]),
                field_value=numpy.asarray(comp_data["field_value"]),
            )
            for comp_axis, comp_data in data["components"].items()
        }
        return cls(
            field_name=data["field_name"],
            sim_time=float(data["sim_time"]),
            profile_axis=data["profile_axis"],
            components=components,
        )


## } MODULE
