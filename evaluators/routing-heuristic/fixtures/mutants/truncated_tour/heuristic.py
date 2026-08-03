"""Scripted mutant that omits the final point from every nonempty tour."""

from __future__ import annotations


# EVOLVE-BLOCK-START
def build_tour(points: list[tuple[float, float]]) -> list[int]:
    return list(range(max(0, len(points) - 1)))
# EVOLVE-BLOCK-END

