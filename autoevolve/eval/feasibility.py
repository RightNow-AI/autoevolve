"""Pre-compute evaluator target feasibility from an optional ceiling."""

from __future__ import annotations

import math
from typing import Any

from autoevolve.core.types import Contract

FeasibilityResult = dict[str, bool | str | float | None]


def _ceiling_value(ceiling: dict[str, Any]) -> float | None:
    value = ceiling.get("value")
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def check_feasibility(
    contract: Contract,
    ceiling: dict[str, Any] | None,
) -> FeasibilityResult:
    """Report whether a target is inside the evaluator's plausible bound."""

    if ceiling is None:
        return {
            "feasible": True,
            "reason": "evaluator defines no ceiling",
            "max_plausible": None,
        }

    plausible = _ceiling_value(ceiling)
    if plausible is None:
        return {
            "feasible": True,
            "reason": "evaluator ceiling has no numeric value",
            "max_plausible": None,
        }
    if contract.target is None:
        return {
            "feasible": True,
            "reason": "contract defines no target",
            "max_plausible": plausible,
        }

    infeasible = (
        contract.target > plausible if contract.maximize else contract.target < plausible
    )
    if infeasible:
        direction = "exceeds" if contract.maximize else "is below"
        return {
            "feasible": False,
            "reason": f"target {contract.target} {direction} plausible bound {plausible}",
            "max_plausible": plausible,
        }
    return {
        "feasible": True,
        "reason": "target is within the evaluator ceiling",
        "max_plausible": plausible,
    }
