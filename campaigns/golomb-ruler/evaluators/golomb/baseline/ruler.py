"""Seed Golomb ruler searcher.

build(order, deadline) returns marks whose pairwise differences are all
distinct. It is a SEARCH, not a formula: it may spend real compute until the
deadline and returns the best ruler it found.

This shape matters. Asking a language model to emit an optimal ruler makes it
recall a published constant on tabulated orders and guess badly on open ones.
Asking it to design a search puts the model where it is strong, writing the
strategy, and the CPU where it is strong, executing it.

The mutable region is the whole search. Improve the strategy: better
construction order, smarter restarts, local moves that shift or swap marks,
annealing, tabu, anything that returns a valid shorter ruler before the
deadline. The only rules are that the marks form a genuine Golomb ruler of the
requested order and that you return before the deadline, because overrunning
it scores zero rather than partial credit.
"""

from __future__ import annotations

import random
import time


def is_golomb(marks: list[int]) -> bool:
    """Return whether all pairwise differences are distinct. Frozen helper."""

    seen: set[int] = set()
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            diff = marks[j] - marks[i]
            if diff in seen:
                return False
            seen.add(diff)
    return True


# EVOLVE-BLOCK-START
def build(order: int, deadline: float | None = None) -> list[int]:
    """Search for a short Golomb ruler of the given order until the deadline."""

    if deadline is None:
        deadline = time.monotonic() + 5.0
    rng = random.Random(20260803)

    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        factor = 3
        while factor * factor <= n:
            if n % factor == 0:
                return False
            factor += 2
        return True

    def singer() -> list[int]:
        """Erdos-Turan Sidon set, valid at any order and a strong start."""

        prime = max(order, 2)
        while not is_prime(prime):
            prime += 1
        marks = sorted(2 * prime * k + (k * k) % prime for k in range(prime))[:order]
        return [value - marks[0] for value in marks]

    def under_cap(cap: int) -> list[int] | None:
        """Randomized greedy that refuses to place any mark beyond cap."""

        marks = [0]
        used: set[int] = set()
        candidate = 1
        while len(marks) < order:
            if candidate > cap or time.monotonic() > deadline:
                return None
            diffs = [candidate - m for m in marks]
            if len(set(diffs)) == len(diffs) and not used.intersection(diffs):
                if rng.random() < 0.12 and cap - candidate > order:
                    candidate += 1
                    continue
                used.update(diffs)
                marks.append(candidate)
            candidate += 1
        return marks

    best = singer()
    # Drive the length down: each success tightens the cap, so the search is
    # always trying to beat its own incumbent rather than wandering.
    while time.monotonic() < deadline:
        attempt = under_cap(best[-1] - 1)
        if attempt is not None:
            best = attempt

    return best
# EVOLVE-BLOCK-END
