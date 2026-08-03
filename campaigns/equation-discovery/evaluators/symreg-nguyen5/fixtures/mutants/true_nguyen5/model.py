"""Exact Nguyen-5 rediscovery fixture."""

from __future__ import annotations

import math


def predict(x: float) -> float:
    return math.sin(x * x) * math.cos(x) - 1.0

