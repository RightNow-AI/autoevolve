"""Gate-failing symbolic regression mutant."""

from __future__ import annotations


def predict(x: float) -> float:
    del x
    return float("nan")

