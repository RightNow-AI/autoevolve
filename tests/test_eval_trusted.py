"""The builtins table survives whatever a candidate does to it."""

from __future__ import annotations

import builtins
import inspect

import pytest

from autoevolve.eval.trusted import restored_builtins


def test_replaced_builtin_is_restored() -> None:
    original = builtins.len

    with restored_builtins():
        builtins.len = lambda value: 99  # noqa: ARG005
        assert builtins.len([]) == 99

    assert builtins.len is original


def test_deleted_builtin_is_restored() -> None:
    original = builtins.callable

    with restored_builtins():
        del builtins.callable

    assert builtins.callable is original


def test_invented_builtin_is_removed() -> None:
    with restored_builtins():
        builtins.smuggled = "a name the gate never defined"

    assert not hasattr(builtins, "smuggled")


def test_restoration_survives_an_exception() -> None:
    original = builtins.len

    with pytest.raises(RuntimeError):
        with restored_builtins():
            builtins.len = lambda value: 0  # noqa: ARG005
            raise RuntimeError("candidate raised partway through")

    assert builtins.len is original


def test_standard_library_works_again_after_restore() -> None:
    """The case that motivated the helper.

    inspect.signature calls callable() internally, resolving it from builtins
    at call time rather than from anything an evaluator bound at import time.
    A candidate that poisons callable therefore breaks signature detection in
    every pack that inspects a candidate entrypoint, and binding trusted
    primitives does not prevent it.
    """

    def solve(n: int) -> int:
        return n

    def explode(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("candidate replaced a builtin")

    with restored_builtins():
        builtins.callable = explode
        with pytest.raises(RuntimeError):
            inspect.signature(solve)

    assert list(inspect.signature(solve).parameters) == ["n"]
