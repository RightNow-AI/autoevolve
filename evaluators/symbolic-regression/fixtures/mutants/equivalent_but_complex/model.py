"""Scripted mutant with baseline-equivalent predictions and extra AST nodes."""

from __future__ import annotations


# EVOLVE-BLOCK-START
def predict(x: float) -> float:
    value = 0.5 + 0.9 * x
    if x >= 0.0:
        return value
    return value
# EVOLVE-BLOCK-END

