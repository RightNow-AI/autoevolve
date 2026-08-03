"""Invalid Ramsey certificate with every edge colored blue."""

from __future__ import annotations


def construct(s: int, n_cap: int, deadline: float | None = None) -> dict[str, object]:
    del deadline
    if n_cap < s:
        raise ValueError("n_cap is smaller than the required clique size")
    return {"form": "circulant", "n": s, "red_diffs": []}
