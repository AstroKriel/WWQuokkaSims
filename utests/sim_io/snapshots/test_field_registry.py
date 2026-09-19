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
from jormi.ww_fields import cartesian_axes
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


def _make_stub_loader_with_chunked_reader() -> collections_abc.Callable:

    def stub_loader(
        _quokka_snapshot: load_snapshot.QuokkaSnapshot,
        *,
        amr_level: int = 0,
        use_chunked_reader: bool = False,
    ) -> field_models.ScalarField_3D:
        _ = amr_level
        _ = use_chunked_reader
        return _make_scalar_field(value=0.0)

    return stub_loader


def _make_stub_registered_field(
    *,
    name: str,
    field_loader_fn: collections_abc.Callable,
) -> field_registry.RegisteredField:
    return field_registry.RegisteredField(
        name=name,
        field_loader_fn=field_loader_fn,
        expected_properties=field_registry.ExpectedProperties(
            pivot_value=None,
            is_strictly_positive=False,
        ),
    )


##
## === TEST SUITES
##


class TestGetFieldType(unittest.TestCase):

    def test_resolves_scalar_field(
        self,
    ):
        self.assertEqual(
            field_registry.get_field_type(field_name="density"),
            field_models.ScalarField_3D,
        )

    def test_resolves_vector_field(
        self,
    ):
        self.assertEqual(
            field_registry.get_field_type(field_name="velocity"),
            field_models.VectorField_3D,
        )

    def test_resolves_rank2_tensor_field(
        self,
    ):
        self.assertEqual(
            field_registry.get_field_type(field_name="velocity_gradient"),
            field_models.RankTwoTensorField_3D,
        )

    def test_raises_for_unregistered_name(
        self,
    ):
        with self.assertRaises(KeyError):
            field_registry.get_field_type(field_name="not_a_real_field")

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
                    field_registry.get_field_type(field_name=field_name),
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


class TestValidateFieldsSupportAmrLeaves(unittest.TestCase):

    def test_accepts_field_with_amr_leaves_loader(
        self,
    ):
        field_registry.validate_fields_support_amr_leaves(field_names=["density"])

    def test_rejects_field_without_amr_leaves_loader(
        self,
    ):
        with self.assertRaises(ValueError):
            field_registry.validate_fields_support_amr_leaves(field_names=["velocity_divergence"])

    def test_lists_every_unsupported_field_at_once(
        self,
    ):
        with self.assertRaises(ValueError) as raised:
            field_registry.validate_fields_support_amr_leaves(
                field_names=["density", "velocity_divergence", "vorticity"],
            )
        self.assertIn("velocity_divergence", str(raised.exception))
        self.assertIn("vorticity", str(raised.exception))


class TestFieldSupportsChunkedReader(unittest.TestCase):

    def test_true_when_loader_accepts_use_chunked_reader(
        self,
    ):
        with unittest.mock.patch.dict(
                field_registry.REGISTERED_FIELD_LOOKUP,
            {
                "_test_stub_with_chunking": _make_stub_registered_field(
                    name="_test_stub_with_chunking",
                    field_loader_fn=_make_stub_loader_with_chunked_reader(),
                ),
            },
        ):
            self.assertTrue(
                field_registry.field_supports_chunked_reader(field_name="_test_stub_with_chunking"),
            )

    def test_false_when_loader_lacks_use_chunked_reader(
        self,
    ):
        with unittest.mock.patch.dict(
                field_registry.REGISTERED_FIELD_LOOKUP,
            {
                "_test_stub_without_chunking": _make_stub_registered_field(
                    name="_test_stub_without_chunking",
                    field_loader_fn=_make_stub_loader(value=1.0),
                ),
            },
        ):
            self.assertFalse(
                field_registry.field_supports_chunked_reader(field_name="_test_stub_without_chunking"),
            )

    def test_every_registered_field_currently_supports_it(
        self,
    ):
        ## documents present-day state (every loader was threaded with use_chunked_reader); not a
        ## structural guarantee -- a future field's loader may legitimately not support it
        for field_name in field_registry.REGISTERED_FIELD_LOOKUP:
            with self.subTest(field_name=field_name):
                self.assertTrue(field_registry.field_supports_chunked_reader(field_name=field_name))


class TestValidateFieldsSupportChunkedReader(unittest.TestCase):

    def test_accepts_when_every_field_supports_it(
        self,
    ):
        field_registry.validate_fields_support_chunked_reader(field_names=["density", "velocity"])

    def test_rejects_and_lists_every_unsupported_field(
        self,
    ):
        with unittest.mock.patch.dict(
                field_registry.REGISTERED_FIELD_LOOKUP,
            {
                "_test_stub_without_chunking": _make_stub_registered_field(
                    name="_test_stub_without_chunking",
                    field_loader_fn=_make_stub_loader(value=1.0),
                ),
            },
        ):
            with self.assertRaises(ValueError) as raised:
                field_registry.validate_fields_support_chunked_reader(
                    field_names=["density", "_test_stub_without_chunking"],
                )
            self.assertIn("_test_stub_without_chunking", str(raised.exception))


class TestValidateFieldsSupportNativeSlice(unittest.TestCase):

    def test_accepts_field_with_native_slice_loader(
        self,
    ):
        field_registry.validate_fields_support_native_slice(field_names=["density"])

    def test_rejects_field_without_native_slice_loader(
        self,
    ):
        with self.assertRaises(ValueError):
            field_registry.validate_fields_support_native_slice(field_names=["velocity_divergence"])

    def test_lists_every_unsupported_field_at_once(
        self,
    ):
        with self.assertRaises(ValueError) as raised:
            field_registry.validate_fields_support_native_slice(
                field_names=["density", "velocity_divergence", "vorticity"],
            )
        self.assertIn("velocity_divergence", str(raised.exception))
        self.assertIn("vorticity", str(raised.exception))


class TestRegisteredFieldLoadNativeSlice(unittest.TestCase):

    def test_raises_when_native_slice_unsupported(
        self,
    ):
        registered_field = field_registry.RegisteredField(
            name="velocity_divergence",
            field_loader_fn=_make_stub_loader(value=1.0),
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        )
        with self.assertRaises(ValueError):
            registered_field.load_native_slice(
                quokka_snapshot=unittest.mock.Mock(spec=load_snapshot.QuokkaSnapshot),
                axis_to_slice=cartesian_axes.CartesianAxis_3D.X2,
                slice_coordinate=0.5,
            )

    def test_forwards_axis_and_coordinate_to_loader(
        self,
    ):
        recorded_calls: list[dict] = []

        def stub_native_slice_loader(
            _quokka_snapshot: load_snapshot.QuokkaSnapshot,
            *,
            axis_to_slice: cartesian_axes.CartesianAxis_3D,
            slice_coordinate: float,
        ) -> tuple[numpy.ndarray, numpy.ndarray]:
            recorded_calls.append({
                "axis_to_slice": axis_to_slice,
                "slice_coordinate": slice_coordinate,
            }, )
            return numpy.zeros((2, 2)), numpy.zeros((2, 2))

        registered_field = field_registry.RegisteredField(
            name="density",
            field_loader_fn=_make_stub_loader(value=1.0),
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=None,
                is_strictly_positive=True,
            ),
            native_slice_loader_fn=stub_native_slice_loader,
        )
        registered_field.load_native_slice(
            quokka_snapshot=unittest.mock.Mock(spec=load_snapshot.QuokkaSnapshot),
            axis_to_slice=cartesian_axes.CartesianAxis_3D.X1,
            slice_coordinate=0.25,
        )
        self.assertEqual(len(recorded_calls), 1)
        self.assertEqual(recorded_calls[0]["axis_to_slice"], cartesian_axes.CartesianAxis_3D.X1)
        self.assertEqual(recorded_calls[0]["slice_coordinate"], 0.25)


class TestRegisteredFieldLoad(unittest.TestCase):

    def test_warns_when_strictly_positive_is_violated(
        self,
    ):
        registered_field = field_registry.RegisteredField(
            name="density",
            field_loader_fn=_make_stub_loader(value=-1.0),
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
            field_loader_fn=_make_stub_loader(value=1.0),
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
            field_loader_fn=_make_stub_loader(value=-1.0),
            expected_properties=field_registry.ExpectedProperties(
                pivot_value=0.0,
                is_strictly_positive=False,
            ),
        )
        with unittest.mock.patch.object(manage_log, "log_warning") as mock_log_warning:
            registered_field.load(quokka_snapshot=unittest.mock.Mock(spec=load_snapshot.QuokkaSnapshot))
        mock_log_warning.assert_not_called()


## } U-TEST
