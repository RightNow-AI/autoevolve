"""First-principles greedy seed for generated kidney exchange graphs."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence


# EVOLVE-BLOCK-START
def solve(
    instance: Mapping[str, object],
    deadline: float | None = None,
) -> dict[str, object]:
    """Repeatedly take the shortest lexicographically first available cycle."""

    pair_count = int(instance["pair_count"])
    cycle_cap = int(instance["cycle_cap"])
    raw_edges = instance["edges"]
    if not isinstance(raw_edges, Sequence):
        raise TypeError("edges must be a sequence")
    edges = tuple(frozenset(int(target) for target in row) for row in raw_edges)
    used: set[int] = set()
    cycles: list[list[int]] = []

    def find_cycle(length: int) -> list[int] | None:
        def extend(path: list[int]) -> list[int] | None:
            if len(path) == length:
                return path if path[0] in edges[path[-1]] else None
            for target in sorted(edges[path[-1]]):
                if target >= pair_count or target in used or target in path:
                    continue
                found = extend([*path, target])
                if found is not None:
                    return found
            return None

        for start in range(pair_count):
            if start in used:
                continue
            found = extend([start])
            if found is not None:
                return found
        return None

    while deadline is None or time.monotonic() < deadline:
        chosen = None
        for length in range(2, cycle_cap + 1):
            chosen = find_cycle(length)
            if chosen is not None:
                break
        if chosen is None:
            break
        cycles.append(chosen)
        used.update(chosen)
    return {"cycles": cycles, "chains": []}
# EVOLVE-BLOCK-END
