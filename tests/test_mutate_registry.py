import pytest

from autoevolve.mutate.registry import OPERATORS, get_operator


def test_registry_constructs_all_public_operators():
    assert set(OPERATORS) == {"agentic", "crossover", "diff", "rewrite"}
    assert {name: get_operator(name).name for name in OPERATORS} == {
        name: name for name in OPERATORS
    }


def test_registry_unknown_name_lists_valid_choices():
    with pytest.raises(KeyError) as caught:
        get_operator("missing")
    message = str(caught.value)
    assert "missing" in message
    assert all(name in message for name in OPERATORS)
