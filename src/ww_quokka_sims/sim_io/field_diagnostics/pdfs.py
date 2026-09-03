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
## === PDF DATA
##


@dataclass(frozen=True)
class PDFData:
    step_time: float
    step_index: int
    grouped_bin_centers: list[numpy.ndarray]
    grouped_densities: list[numpy.ndarray]
    comp_labels: list[str]
    use_log10_bins: bool = False

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_sequence(
            param=self.grouped_bin_centers,
            valid_seq_types=(list, tuple),
            param_name="<grouped_bin_centers>",
            seq_length=len(self.comp_labels),
        )
        validate_types.ensure_sequence(
            param=self.grouped_densities,
            valid_seq_types=(list, tuple),
            param_name="<grouped_densities>",
            seq_length=len(self.comp_labels),
        )
        for (bin_centers, densities) in zip(self.grouped_bin_centers, self.grouped_densities):
            validate_arrays.ensure_array(array=bin_centers)
            validate_arrays.ensure_array(array=densities)
            validate_arrays.ensure_1d(array=bin_centers)
            validate_arrays.ensure_1d(array=densities)
            validate_arrays.ensure_same_shape(
                array_a=bin_centers,
                array_b=densities,
            )

    @property
    def num_comps(
        self,
    ) -> int:
        return len(self.comp_labels)

    @property
    def is_scalar(
        self,
    ) -> bool:
        return self.num_comps == 1

    def get_pdf(
        self,
        comp_index: int = 0,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if (comp_index < 0) or (comp_index >= self.num_comps):
            raise IndexError(f"comp_index {comp_index} out of range [0, {self.num_comps - 1}]")
        return self.grouped_bin_centers[comp_index], self.grouped_densities[comp_index]

    def save_to_file(
        self,
        file_path: Path,
    ) -> None:
        bin_centers_key = "log10_bin_centers" if self.use_log10_bins else "bin_centers"
        output_dict: dict = {
            "step_time": self.step_time,
            "step_index": self.step_index,
            "use_log10_bins": self.use_log10_bins,
        }
        for comp_index, comp_label in enumerate(self.comp_labels):
            bin_centers, densities = self.get_pdf(comp_index)
            output_dict[comp_label] = {
                bin_centers_key: bin_centers,
                "log10_density": densities,
            }
        json_io.save_dict_to_json_file(
            file_path=file_path,
            input_dict=output_dict,
            overwrite=True,
            verbose=False,
        )

    @classmethod
    def load_from_file(
        cls,
        file_path: Path,
    ) -> "PDFData":
        input_dict = json_io.read_json_file_into_dict(
            file_path=file_path,
            verbose=False,
        )
        validate_types.ensure_dict_has_keys(
            param=input_dict,
            required_keys={"step_time", "step_index", "use_log10_bins"},
            param_name="<PDFData JSON>",
        )
        use_log10_bins = bool(input_dict["use_log10_bins"])
        bin_centers_key = "log10_bin_centers" if use_log10_bins else "bin_centers"
        comp_labels = [
            key for key in input_dict if key not in ("step_time", "step_index", "use_log10_bins")
        ]
        return cls(
            step_time=float(input_dict["step_time"]),
            step_index=int(input_dict["step_index"]),
            grouped_bin_centers=[numpy.array(input_dict[label][bin_centers_key]) for label in comp_labels],
            grouped_densities=[numpy.array(input_dict[label]["log10_density"]) for label in comp_labels],
            comp_labels=comp_labels,
            use_log10_bins=use_log10_bins,
        )


## } MODULE
