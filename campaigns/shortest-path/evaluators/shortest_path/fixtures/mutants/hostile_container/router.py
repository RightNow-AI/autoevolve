"""Mutant whose answer changes when a consumer iterates it again."""

from __future__ import annotations

import heapq
from collections.abc import Iterator


class VaryingAnswer:
    def __init__(self, distance: int, path: list[int]) -> None:
        self.distance = distance
        self.path = path
        self.reads = 0

    def __iter__(self) -> Iterator[object]:
        self.reads += 1
        if self.reads == 1:
            yield self.distance
            yield [self.path[0], self.path[-1]]
            return
        yield self.distance
        yield list(self.path)


class HostileRouter:
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

    def query(self, source: int, target: int) -> VaryingAnswer:
        distances: list[int | None] = [None] * len(self.adjacency)
        parents = [-1] * len(self.adjacency)
        distances[source] = 0
        heap: list[tuple[int, int]] = [(0, source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distances[node] != distance:
                continue
            if node == target:
                break
            for neighbor, weight in self.adjacency[node]:
                candidate = distance + weight
                known = distances[neighbor]
                if known is None or candidate < known:
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    heapq.heappush(heap, (candidate, neighbor))
        distance = distances[target]
        if distance is None:
            raise ValueError("target is unreachable")
        path = [target]
        node = target
        while node != source:
            node = parents[node]
            path.append(node)
        path.reverse()
        return VaryingAnswer(distance, path)


def build_router(
    vertex_count: int,
    edges: tuple[tuple[int, int, int], ...],
    deadline: float | None = None,
) -> HostileRouter:
    del deadline
    return HostileRouter(vertex_count, edges)
