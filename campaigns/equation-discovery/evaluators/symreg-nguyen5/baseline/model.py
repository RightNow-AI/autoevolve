"""Weak seed for Nguyen-5 rediscovery."""

from __future__ import annotations


# EVOLVE-BLOCK-START
def predict(x: float) -> float:
    """Return a weak affine approximation."""

    return -0.95 + 0.1 * x
# EVOLVE-BLOCK-END

