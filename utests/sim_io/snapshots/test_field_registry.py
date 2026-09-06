## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import typing
import unittest
import unittest.mock

from collections import abc as collections_abc

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import domain_models, field_models
from jormi.ww_io import manage_log

## local
from ww_quokka_sims.sim_io.snapshots import field_registry, load_snapshot

##
## === HELPERS
##


def _make_scalar_field(
    *,
    value: float,
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
        latex_label=r"\rho",
        sim_time=0.0,
    )


def _make_stub_loader(
    *,
    value: float,
) -> collections_abc.Callable:

    def stub_loader(
        _quokka_snapshot: load_snapshot.QuokkaSnapshot,
        *,
        amr_level: int = 0,
    ) -> field_models.ScalarField_3D:
        _ = amr_level
        return _make_scalar_field(value=value)

    return stub_loader


##
## === TEST SUITES
##


class TestGetFieldType(unittest.TestCase):

    def test_resolves_scalar_field(
        self,
    ):
        self.assertEqual(
            field_registry.get_field_type("density"),
            field_models.ScalarField_3D,
        )

    def test_resolves_vector_field(
        self,
    ):
        self.assertEqual(
            field_registry.get_field_type("velocity"),
            field_models.VectorField_3D,
        )

    def test_resolves_rank2_tensor_field(
        self,
    ):
        self.assertEqual(
            field_registry.get_field_type("velocity_gradient"),
            field_models.RankTwoTensorField_3D,
        )

    def test_raises_for_unregistered_name(
        self,
    ):
        with self.assertRaises(KeyError):
            field_registry.get_field_type("not_a_real_field")

    def test_every_registered_field_resolves_to_a_known_type(
        self,
    ):
        ## if a new rank is ever added to a loader without adding the matching
        ## Field type to field_models.AnyField_3D, this catches it here rather
        ## than at some downstream script's runtime dispatch
        known_types = typing.get_args(field_models.AnyField_3D)
        for field_name in field_registry.REGISTERED_FIELD_LOOKUP:
            with self.subTest(field_name=field_name):
                self.assertIn(
                    field_registry.get_field_type(field_name),
                    known_types,
                )


class TestValidateFieldsAllowedTypes(unittest.TestCase):

    def test_accepts_field_matching_allowed_type(
        self,
    ):
        field_registry.validate_fields(
            field_names=["velocity"],
            allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
        )

    def test_rejects_field_outside_allowed_types(
        self,
    ):
        with self.assertRaises(ValueError):
            field_registry.validate_fields(
                field_names=["velocity_gradient"],
                allowed_types=(field_models.ScalarField_3D, field_models.VectorField_3D),
            )

    def test_accepts_field_once_type_is_included(
        self,
    ):
        field_registry.validate_fields(
            field_names=["velocity_gradient"],
            allowed_types=(
                field_models.ScalarField_3D,
                field_models.VectorField_3D,
                field_models.RankTwoTensorField_3D,
            ),
        )

    def test_none_allowed_types_skips_rank_check(
        self,
    ):
        ## matches the pre-existing behaviour: only membership in the registry is checked
        field_registry.validate_fields(field_names=["velocity_gradient"])

    def test_still_rejects_unregistered_names_with_allowed_types_set(
        self,
    ):
        with self.assertRaises(ValueError):
            field_registry.validate_fields(
                field_names=["not_a_real_field"],
                allowed_types=(field_models.ScalarField_3D, ),
            )


class TestRegisteredFieldLoad(unittest.TestCase):

    def test_warns_when_strictly_positive_is_violated(
        self,
    ):
        registered_field = field_registry.RegisteredField(
            name="density",
            loader_fn=_make_stub_loader(value=-1.0),
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        )
        with unittest.mock.patch.object(manage_log, "log_warning") as mock_log_warning:
            registered_field.load(quokka_snapshot=unittest.mock.Mock(spec=load_snapshot.QuokkaSnapshot))
        mock_log_warning.assert_called_once()

    def test_no_warning_when_values_are_non_negative(
        self,
    ):
        registered_field = field_registry.RegisteredField(
            name="density",
            loader_fn=_make_stub_loader(value=1.0),
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
        )
        with unittest.mock.patch.object(manage_log, "log_warning") as mock_log_warning:
            registered_field.load(quokka_snapshot=unittest.mock.Mock(spec=load_snapshot.QuokkaSnapshot))
        mock_log_warning.assert_not_called()

    def test_no_check_when_not_declared_strictly_positive(
        self,
    ):
        registered_field = field_registry.RegisteredField(
            name="velocity_divergence",
            loader_fn=_make_stub_loader(value=-1.0),
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        )
        with unittest.mock.patch.object(manage_log, "log_warning") as mock_log_warning:
            registered_field.load(quokka_snapshot=unittest.mock.Mock(spec=load_snapshot.QuokkaSnapshot))
        mock_log_warning.assert_not_called()


## } U-TEST
