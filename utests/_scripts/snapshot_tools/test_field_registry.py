## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from typing import get_args

## personal
from jormi.ww_fields.fields_3d import field_models

## local
from ww_quokka_sims._scripts.snapshot_tools import field_registry

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
        known_types = get_args(field_models.AnyField_3D)
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


## } U-TEST
