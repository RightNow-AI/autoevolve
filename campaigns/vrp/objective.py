"""Shared hierarchical objective helpers for the VRPTW campaign."""

from __future__ import annotations

import json
import math

DISTANCE_EPSILON = 1e-9
BOUND_CLAIM_PREFIX = "best known VRPTW solution for "


def is_better_result(
    candidate: tuple[int, float],
    incumbent: tuple[int, float],
    *,
    distance_epsilon: float = DISTANCE_EPSILON,
) -> bool:
    """Return whether candidate wins by vehicles first, then distance.

    SINTEF computes distance in double precision and reports totals to two decimal
    places. The epsilon only absorbs binary floating-point noise, so a reported
    difference of 0.01 remains decisive. SINTEF also warns that many best known
    solutions have no peer-reviewed publication behind them. Any apparent win must
    be rechecked against that instance's current source row before it is claimed.
    """

    candidate_vehicles, candidate_distance = candidate
    incumbent_vehicles, incumbent_distance = incumbent
    if candidate_vehicles != incumbent_vehicles:
        return candidate_vehicles < incumbent_vehicles
    return candidate_distance < incumbent_distance - distance_epsilon


def encode_objective_value(vehicle_count: int, total_distance: float) -> str:
    """Encode both objective components inside the bounds schema's string value."""

    if isinstance(vehicle_count, bool) or vehicle_count < 1:
        raise ValueError("vehicle_count must be a positive integer")
    if not math.isfinite(total_distance) or total_distance < 0.0:
        raise ValueError("total_distance must be a non-negative finite number")
    return json.dumps(
        {"vehicle_count": vehicle_count, "total_distance": total_distance},
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_objective_value(value: str) -> tuple[int, float]:
    """Decode a VRP bounds value produced by :func:`encode_objective_value`."""

    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("bound value must be a JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"vehicle_count", "total_distance"}:
        raise ValueError("bound value must contain vehicle_count and total_distance")
    vehicle_count = payload["vehicle_count"]
    total_distance = payload["total_distance"]
    if isinstance(vehicle_count, bool) or not isinstance(vehicle_count, int):
        raise ValueError("bound vehicle_count must be an integer")
    if isinstance(total_distance, bool) or not isinstance(total_distance, int | float):
        raise ValueError("bound total_distance must be numeric")
    objective = (vehicle_count, float(total_distance))
    encode_objective_value(*objective)
    return objective
