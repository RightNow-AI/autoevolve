"""Exact brute-force seed for the ANN search campaign."""

from __future__ import annotations

import numpy as np


# EVOLVE-BLOCK-START
def build(vectors: np.ndarray, deadline: float | None = None) -> np.ndarray:
    """Retain a private database copy without constructing an approximate index."""

    del deadline
    return np.asarray(vectors, dtype=np.float32).copy()


def search(
    index: np.ndarray,
    queries: np.ndarray,
    k: int,
    deadline: float | None = None,
) -> list[list[int]]:
    """Return exact neighbours by scanning and sorting the full database."""

    del deadline
    database = np.asarray(index, dtype=np.float64)
    query_matrix = np.asarray(queries, dtype=np.float64)
    database_indices = np.arange(database.shape[0], dtype=np.int64)
    neighbours: list[list[int]] = []
    for query in query_matrix:
        delta = database - query
        squared_distances = np.einsum("ij,ij->i", delta, delta)
        order = np.lexsort((database_indices, squared_distances))
        neighbours.append([int(value) for value in order[:k]])
    return neighbours
# EVOLVE-BLOCK-END
