"""Mutation operator registry."""

from __future__ import annotations

from collections.abc import Callable

from autoevolve.mutate.agentic import AgenticOperator
from autoevolve.mutate.base import Operator
from autoevolve.mutate.crossover import CrossoverOperator
from autoevolve.mutate.diff import DiffOperator
from autoevolve.mutate.rewrite import RewriteOperator

OPERATORS: dict[str, Callable[[], Operator]] = {
    "diff": DiffOperator,
    "rewrite": RewriteOperator,
    "agentic": AgenticOperator,
    "crossover": CrossoverOperator,
}


def get_operator(name: str) -> Operator:
    """Construct a registered operator by name."""

    try:
        factory = OPERATORS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(OPERATORS))
        raise KeyError(f"unknown operator {name!r}; valid operators: {valid}") from exc
    return factory()
