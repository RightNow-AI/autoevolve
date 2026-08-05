"""Invalid certificate whose complete graph is entirely blue."""

from __future__ import annotations


def construct(n_cap: int, deadline: float | None = None) -> dict[str, object]:
    del deadline
    n = min(n_cap, 5)
    if n < 5:
        raise ValueError("blue K4 mutant needs at least five vertices")
    return {"n": n, "edge_colors": [1] * (n * (n - 1) // 2)}
