"""Seeded random-restart search for point spread in a unit square."""

from __future__ import annotations

import math
import time

import numpy as np

# EVOLVE-BLOCK-START


def _minimum_squared_distance(points: np.ndarray) -> float:
    deltas = points[:, None, :] - points[None, :, :]
    squared = np.einsum("ijk,ijk->ij", deltas, deltas)
    np.fill_diagonal(squared, np.inf)
    return float(np.min(squared))


def _repulsion_step(
    points: np.ndarray,
    rng: np.random.Generator,
    step_size: float,
) -> np.ndarray:
    deltas = points[:, None, :] - points[None, :, :]
    squared = np.einsum("ijk,ijk->ij", deltas, deltas)
    np.fill_diagonal(squared, np.inf)

    minimum = max(float(np.min(squared)), 1.0e-24)
    influence_radius = max(math.sqrt(minimum) * 1.75, 0.02)
    distances = np.sqrt(np.maximum(squared, 1.0e-24))
    weights = np.maximum(0.0, influence_radius - distances) / influence_radius
    np.fill_diagonal(weights, 0.0)

    directions = deltas / distances[:, :, None]
    forces = np.sum(directions * weights[:, :, None], axis=1)
    norms = np.linalg.norm(forces, axis=1)
    stalled = norms < 1.0e-12
    if np.any(stalled):
        forces[stalled] = rng.normal(size=(int(np.count_nonzero(stalled)), 2))
        norms = np.linalg.norm(forces, axis=1)
    forces /= np.maximum(norms[:, None], 1.0e-12)

    noise = rng.normal(scale=step_size * 0.05, size=points.shape)
    return np.clip(points + step_size * forces + noise, 0.0, 1.0)


def _improve_restart(
    start: np.ndarray,
    rng: np.random.Generator,
    deadline: float,
) -> tuple[np.ndarray, float]:
    current = start.copy()
    current_score = _minimum_squared_distance(current)
    incumbent = current.copy()
    incumbent_score = current_score
    step_size = max(0.015, 0.35 / math.sqrt(len(current)))
    stale_steps = 0
    steps = 0

    while time.monotonic() < deadline and stale_steps < 320 and steps < 2048:
        steps += 1
        proposal = _repulsion_step(current, rng, step_size)
        proposal_score = _minimum_squared_distance(proposal)
        if proposal_score > current_score:
            current = proposal
            current_score = proposal_score
            stale_steps = 0
            step_size = min(step_size * 1.02, 0.2)
            if current_score > incumbent_score:
                incumbent = current.copy()
                incumbent_score = current_score
        else:
            stale_steps += 1
            step_size = max(step_size * 0.992, 1.0e-4)
            if stale_steps % 64 == 0:
                current = np.clip(
                    incumbent + rng.normal(scale=step_size * 2.0, size=incumbent.shape),
                    0.0,
                    1.0,
                )
                current_score = _minimum_squared_distance(current)

    return incumbent, incumbent_score


def solve(
    n: int,
    deadline: float | None = None,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Search until the deadline and return the strongest measured incumbent."""

    if n < 2:
        raise ValueError("n must be at least 2")
    stop = deadline if deadline is not None else time.monotonic() + 1.0
    rng = np.random.default_rng(seed)
    incumbent = rng.random((n, 2))
    incumbent_score = _minimum_squared_distance(incumbent)

    while time.monotonic() < stop:
        restart = rng.random((n, 2))
        candidate, score = _improve_restart(restart, rng, stop)
        if score > incumbent_score:
            incumbent = candidate
            incumbent_score = score

    return [(float(x), float(y)) for x, y in incumbent]
# EVOLVE-BLOCK-END
