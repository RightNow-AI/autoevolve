"""Nearest-neighbor seed for Euclidean tour construction.

Candidate contract:
build_tour(points) returns a list containing each index from zero through n - 1 once.
The tour starts at index zero. The evaluator closes the cycle for scoring.
"""

from __future__ import annotations

import math


# EVOLVE-BLOCK-START
def build_tour(points: list[tuple[float, float]]) -> list[int]:
    """Build a tour by repeatedly choosing the nearest unvisited point."""
    if not points:
        return []
    tour = [0]
    unvisited = set(range(1, len(points)))
    while unvisited:
        current = tour[-1]
        current_x, current_y = points[current]
        next_index = min(
            unvisited,
            key=lambda index, x=current_x, y=current_y: (
                math.hypot(points[index][0] - x, points[index][1] - y),
                index,
            ),
        )
        tour.append(next_index)
        unvisited.remove(next_index)
    return tour
# EVOLVE-BLOCK-END
