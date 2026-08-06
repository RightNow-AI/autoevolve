"""Deterministic local-search seed for binary sequences."""

from __future__ import annotations

import random
import time


# EVOLVE-BLOCK-START
def _correlations(sequence: list[int]) -> list[int]:
    n = len(sequence)
    return [
        sum(sequence[index] * sequence[index + lag] for index in range(n - lag))
        for lag in range(1, n)
    ]


def _energy(correlations: list[int]) -> int:
    return sum(value * value for value in correlations)


def _expand_skew_symmetric(free_spins: list[int]) -> list[int]:
    """Expand the left half and centre into an odd skew-symmetric sequence."""

    if not free_spins:
        raise ValueError("free_spins must not be empty")
    centre = len(free_spins) - 1
    sequence = free_spins[:]
    for offset in range(1, len(free_spins)):
        sign = -1 if offset % 2 else 1
        sequence.append(sign * free_spins[centre - offset])
    return sequence


def _after_flip(
    sequence: list[int],
    correlations: list[int],
    position: int,
) -> list[int]:
    n = len(sequence)
    sign = sequence[position]
    updated: list[int] = []
    for lag, old_value in enumerate(correlations, start=1):
        neighbours = 0
        if position + lag < n:
            neighbours += sequence[position + lag]
        if position - lag >= 0:
            neighbours += sequence[position - lag]
        updated.append(old_value - 2 * sign * neighbours)
    return updated


def _after_flips(
    sequence: list[int],
    correlations: list[int],
    positions: tuple[int, ...],
) -> tuple[list[int], list[int]]:
    updated_sequence = sequence[:]
    updated_correlations = correlations
    for position in positions:
        updated_correlations = _after_flip(
            updated_sequence,
            updated_correlations,
            position,
        )
        updated_sequence[position] = -updated_sequence[position]
    return updated_sequence, updated_correlations


def _steepest_descent(
    initial: list[int],
    deadline: float | None,
) -> tuple[list[int], int]:
    sequence = initial[:]
    correlations = _correlations(sequence)
    energy = _energy(correlations)
    step_limit = max(16, 4 * len(sequence))

    for _ in range(step_limit):
        best_position = None
        best_correlations = correlations
        best_energy = energy
        for position in range(len(sequence)):
            if deadline is not None and time.monotonic() >= deadline:
                return sequence, energy
            trial_correlations = _after_flip(sequence, correlations, position)
            trial_energy = _energy(trial_correlations)
            if trial_energy < best_energy:
                best_position = position
                best_correlations = trial_correlations
                best_energy = trial_energy
        if best_position is None:
            break
        sequence[best_position] = -sequence[best_position]
        correlations = best_correlations
        energy = best_energy
    return sequence, energy


def _steepest_skew_descent(
    initial_free_spins: list[int],
    deadline: float | None,
) -> tuple[list[int], int]:
    free_spins = initial_free_spins[:]
    sequence = _expand_skew_symmetric(free_spins)
    correlations = _correlations(sequence)
    energy = _energy(correlations)
    centre = len(free_spins) - 1
    step_limit = max(16, 4 * len(free_spins))

    for _ in range(step_limit):
        best_free_position = None
        best_sequence = sequence
        best_correlations = correlations
        best_energy = energy
        for free_position in range(len(free_spins)):
            if deadline is not None and time.monotonic() >= deadline:
                return sequence, energy
            if free_position == centre:
                positions = (free_position,)
            else:
                positions = (free_position, len(sequence) - 1 - free_position)
            trial_sequence, trial_correlations = _after_flips(
                sequence,
                correlations,
                positions,
            )
            trial_energy = _energy(trial_correlations)
            if trial_energy < best_energy:
                best_free_position = free_position
                best_sequence = trial_sequence
                best_correlations = trial_correlations
                best_energy = trial_energy
        if best_free_position is None:
            break
        free_spins[best_free_position] = -free_spins[best_free_position]
        sequence = best_sequence
        correlations = best_correlations
        energy = best_energy
    return sequence, energy


def solve(
    n: int,
    deadline: float | None = None,
    seed: int | None = None,
) -> list[int]:
    """Search random restarts and return the best exact-energy incumbent."""

    if n < 2:
        raise ValueError("n must be at least 2")
    rng = random.Random(n if seed is None else seed)
    restart_limit = max(8, min(64, 4096 // n))
    free_count = (n + 1) // 2
    incumbent = [rng.choice((-1, 1)) for _ in range(n)]
    incumbent_energy = _energy(_correlations(incumbent))

    for restart in range(restart_limit):
        if deadline is not None and time.monotonic() >= deadline:
            break
        if n % 2 == 1 and restart % 2 == 0:
            initial_free_spins = [rng.choice((-1, 1)) for _ in range(free_count)]
            candidate, candidate_energy = _steepest_skew_descent(
                initial_free_spins,
                deadline,
            )
        else:
            initial = [rng.choice((-1, 1)) for _ in range(n)]
            candidate, candidate_energy = _steepest_descent(initial, deadline)
        if candidate_energy < incumbent_energy:
            incumbent = candidate
            incumbent_energy = candidate_energy
    return incumbent
# EVOLVE-BLOCK-END
