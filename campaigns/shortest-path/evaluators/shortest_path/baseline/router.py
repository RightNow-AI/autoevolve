"""Textbook binary-heap Dijkstra seed for exact directed shortest paths."""

from __future__ import annotations


# EVOLVE-BLOCK-START
import heapq


class DijkstraRouter:
    """Store one adjacency list and run a fresh search for every query."""

    def __init__(
        self,
        vertex_count: int,
        edges: tuple[tuple[int, int, int], ...],
    ) -> None:
        adjacency: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
        for source, target, weight in edges:
            adjacency[source].append((target, weight))
        self.adjacency = adjacency

    def query(self, source: int, target: int) -> tuple[int, list[int]]:
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
            raise ValueError(f"no path from {source} to {target}")
        path = [target]
        node = target
        while node != source:
            node = parents[node]
            if node < 0:
                raise ValueError("path reconstruction failed")
            path.append(node)
        path.reverse()
        return distance, path


def build_router(
    vertex_count: int,
    edges: tuple[tuple[int, int, int], ...],
    deadline: float | None = None,
) -> DijkstraRouter:
    """Build the unremarkable seed adjacency list."""

    del deadline
    return DijkstraRouter(vertex_count, edges)
# EVOLVE-BLOCK-END
