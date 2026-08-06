"""Mutant that computes exact distances but fabricates direct-edge paths."""

from __future__ import annotations

import heapq


class BrokenPathRouter:
    def __init__(
        self,
        vertex_count: int,
        edges: tuple[tuple[int, int, int], ...],
    ) -> None:
        self.adjacency: list[list[tuple[int, int]]] = [
            [] for _ in range(vertex_count)
        ]
        for source, target, weight in edges:
            self.adjacency[source].append((target, weight))

    def query(self, source: int, target: int) -> tuple[int, list[int]]:
        distances: list[int | None] = [None] * len(self.adjacency)
        distances[source] = 0
        heap: list[tuple[int, int]] = [(0, source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distances[node] != distance:
                continue
            if node == target:
                return distance, [source, target]
            for neighbor, weight in self.adjacency[node]:
                candidate = distance + weight
                known = distances[neighbor]
                if known is None or candidate < known:
                    distances[neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))
        raise ValueError("target is unreachable")


def build_router(
    vertex_count: int,
    edges: tuple[tuple[int, int, int], ...],
    deadline: float | None = None,
) -> BrokenPathRouter:
    del deadline
    return BrokenPathRouter(vertex_count, edges)
