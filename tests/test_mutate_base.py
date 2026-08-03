import dataclasses

from autoevolve.mutate.base import OperatorContext, OperatorError


def test_operator_context_has_exact_public_field_order():
    assert [field.name for field in dataclasses.fields(OperatorContext)] == [
        "contract",
        "rng",
        "endpoint_cheap",
        "endpoint_strong",
        "evaluate_locally",
        "workdir",
    ]


def test_operator_error_carries_reason():
    error = OperatorError("no usable proposal")
    assert str(error) == "no usable proposal"
    assert error.reason == "no usable proposal"
