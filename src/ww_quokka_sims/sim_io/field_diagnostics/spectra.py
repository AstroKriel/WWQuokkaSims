## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

## personal
from jormi.ww_io import json_io
from jormi.ww_validation import validate_arrays, validate_types

##
## === SPECTRA DATA
##


@dataclass(frozen=True)
class SpectraData:
    step_time: float
    step_index: int
    latex_label: str
    log10_k_bin_centers: numpy.ndarray
    log10_spectrum: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        validate_arrays.ensure_array(array=self.log10_k_bin_centers)
        validate_arrays.ensure_array(array=self.log10_spectrum)
        validate_arrays.ensure_1d(array=self.log10_k_bin_centers)
        validate_arrays.ensure_1d(array=self.log10_spectrum)
        validate_arrays.ensure_same_shape(
            array_a=self.log10_k_bin_centers,
            array_b=self.log10_spectrum,
        )

    def save_to_file(
        self,
        file_path: Path,
    ) -> None:
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict={
                "step_time": self.step_time,
                "step_index": self.step_index,
                "latex_label": self.latex_label,
                "log10_k_bin_centers": self.log10_k_bin_centers,
                "log10_spectrum": self.log10_spectrum,
            },
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: Path,
    ) -> "SpectraData":
        data = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=data,
            required_keys={
                "step_time",
                "step_index",
                "latex_label",
                "log10_k_bin_centers",
                "log10_spectrum",
            },
            param_name="<SpectraData JSON>",
        )
        return cls(
            step_time=float(data["step_time"]),
            step_index=int(data["step_index"]),
            latex_label=data["latex_label"],
            log10_k_bin_centers=numpy.asarray(data["log10_k_bin_centers"]),
            log10_spectrum=numpy.asarray(data["log10_spectrum"]),
        )


## } MODULE
