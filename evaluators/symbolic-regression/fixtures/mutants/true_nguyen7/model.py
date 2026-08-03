"""Scripted rediscovery of the published Nguyen-7 target expression."""

from __future__ import annotations

import math


# EVOLVE-BLOCK-START
def predict(x: float) -> float:
    return math.log(x + 1.0) + math.log(x * x + 1.0)
# EVOLVE-BLOCK-END

