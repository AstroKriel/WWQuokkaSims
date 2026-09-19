## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import pathlib
import tempfile
import unittest
import unittest.mock

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import domain_models, field_models
from jormi.ww_plots import latex_labels

## local
from ww_quokka_sims.sim_io.field_diagnostics import time_series
from ww_quokka_sims.sim_io.snapshots import field_registry, load_snapshot

##
## === HELPERS
##


def _make_scalar_field(
    *,
    latex_label: str = r"\rho",
    value: float = 1.0,
    sim_time: float = 0.5,
) -> field_models.ScalarField_3D:
    uniform_domain_3d = domain_models.UniformDomain_3D(
        periodicity=(True, True, True),
        domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        resolution=(2, 2, 2),
    )
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=numpy.full((2, 2, 2), value),
        uniform_domain_3d=uniform_domain_3d,
        field_name="density",
        latex_label=latex_label,
        sim_time=sim_time,
    )


def _stub_loader(
    _quokka_snapshot: load_snapshot.QuokkaSnapshot,
    *,
    amr_level: int = 0,
) -> field_models.ScalarField_3D:
    _ = amr_level
    return _make_scalar_field()


##
## === TEST SUITE
##


class TestComputeTimePointLabel(unittest.TestCase):

    def test_label_combines_statistic_and_field(
        self,
    ):
        registered_field = field_registry.RegisteredField(
            name="density",
            field_loader_fn=_stub_loader,
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        )
        field_statistic = time_series.FieldStatistic(
            name="rms",
            _compute_fn=lambda field_3d: 1.0,
        )
        time_point_args = time_series.TimePointArgs(
            snapshot_dir=pathlib.Path("/unused"),
            registered_field=registered_field,
            field_statistic=field_statistic,
        )
        with unittest.mock.patch.object(load_snapshot, "QuokkaSnapshot") as mock_snapshot_cls:
            mock_snapshot_cls.return_value.__enter__.return_value = unittest.mock.Mock()
            time_point = time_series.GenerateTimeSeries._compute_time_point(time_point_args)
        self.assertEqual(time_point.latex_label.label, r"$\mathrm{rms}\big(\rho\big)$")
        self.assertEqual(time_point.field_name, "density")
        self.assertEqual(time_point.statistic_name, "rms")

    def test_label_reflects_the_specific_statistic_used(
        self,
    ):
        registered_field = field_registry.RegisteredField(
            name="density",
            field_loader_fn=_stub_loader,
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        )
        field_statistic = time_series.FieldStatistic(
            name="total",
            _compute_fn=lambda field_3d: 1.0,
        )
        time_point_args = time_series.TimePointArgs(
            snapshot_dir=pathlib.Path("/unused"),
            registered_field=registered_field,
            field_statistic=field_statistic,
        )
        with unittest.mock.patch.object(load_snapshot, "QuokkaSnapshot") as mock_snapshot_cls:
            mock_snapshot_cls.return_value.__enter__.return_value = unittest.mock.Mock()
            time_point = time_series.GenerateTimeSeries._compute_time_point(time_point_args)
        ## a regression here (e.g. always saying "rms" regardless of which statistic ran) is exactly
        ## the bug this suite exists to catch
        self.assertEqual(time_point.latex_label.label, r"$\mathrm{total}\big(\rho\big)$")


class TestTimePointRoundTrip(unittest.TestCase):

    def test_save_and_load_preserves_all_fields(
        self,
    ):
        time_point = time_series.TimePoint(
            sim_time=0.25,
            value=3.5,
            field_name="density",
            statistic_name="rms",
            latex_label=latex_labels.LatexLabel(content=r"\mathrm{rms}\big(\rho\big)"),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = pathlib.Path(tmp_dir) / "time_point.json"
            time_point.save_to_file(file_path=file_path)
            loaded = time_series.TimePoint.load_from_file(file_path=file_path)
        self.assertEqual(loaded, time_point)


## } U-TEST
