"""Mutant that claims an impossible zero distance for every distinct query."""

from __future__ import annotations


class ImpossibleRouter:
    def query(self, source: int, target: int) -> tuple[int, list[int]]:
        return 0, [source, target]


def build_router(
    vertex_count: int,
    edges: tuple[tuple[int, int, int], ...],
    deadline: float | None = None,
) -> ImpossibleRouter:
    del vertex_count, edges, deadline
    return ImpossibleRouter()
