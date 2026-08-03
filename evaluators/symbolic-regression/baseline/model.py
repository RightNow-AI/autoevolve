"""Deliberately weak linear seed for the Nguyen-7 rediscovery task.

Candidate contract:
predict(x: float) returns one finite float for every input in the fixture domain.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def predict(x: float) -> float:
    """Return a weak linear approximation with substantial held-out headroom."""
    return 0.5 + 0.9 * x
# EVOLVE-BLOCK-END

