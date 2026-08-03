"""Deterministic goal-domain classification."""

from __future__ import annotations

import re


def classify_domain(goal_text: str) -> str:
    """Classify a goal using a small ordered keyword map."""

    lowered = goal_text.lower()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    if words & {"kernel", "gpu", "triton"}:
        return "triton-kernel"
    if words & {"speed", "speedup", "faster"} or {"optimize", "python"} <= words:
        return "python-speedup"
    if words & {"tour", "route", "routing", "tsp"}:
        return "routing-heuristic"
    if words & {"equation", "formula", "fit"}:
        return "symbolic-regression"
    return "general"
