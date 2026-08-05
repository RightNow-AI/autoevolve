"""Invalid colors hidden behind a list whose iterations disagree."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations


class HostileColors(list[int]):
    """Expose an invalid all-blue snapshot first and a valid one later."""

    def __init__(self, valid: list[int]) -> None:
        super().__init__([1] * len(valid))
        self.valid = valid
        self.reads = 0

    def __iter__(self) -> Iterator[int]:
        self.reads += 1
        if self.reads == 1:
            return super().__iter__()
        return iter(self.valid)


def construct(n_cap: int, deadline: float | None = None) -> dict[str, object]:
    del deadline
    n = 5
    if n_cap < n:
        raise ValueError("hostile fixture needs five vertices")
    red_edges = {(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)}
    valid = [0 if edge in red_edges else 1 for edge in combinations(range(n), 2)]
    return {"n": n, "edge_colors": HostileColors(valid)}
