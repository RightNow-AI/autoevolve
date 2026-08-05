"""Search for a K5-free colouring with few monochromatic K4s.

Adding one vertex to a K5-free colouring is possible exactly when the existing
vertices split into a red side containing no red K4 and a blue side containing
no blue K4. Each monochromatic K4 is therefore one exact clause that the new
vertex must satisfy. The sparse objective minimizes that clause count while
keeping every retained state K5-free.

The search starts from a verified certificate when one is supplied. It samples
several single-edge moves and prefers the best K5-preserving K4 delta. Equal K4
moves cross plateaus, uphill moves are annealed, recent flips are tabu, and a
stalled epoch restarts from the retained best certificate. Occasional two-edge
bridge proposals may pass through an intermediate K5 violation, but the pair is
accepted only when its final state has exactly zero monochromatic K5s.

Usage: search_sparse42.py <n> <seed> <seconds> [output.json] [start.json]
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_n import (  # noqa: E402
    Edge,
    brute_cost,
    brute_k4_cost,
    flip_edge,
    k4s_through,
    k5s_through,
    random_colouring,
    random_edge,
    remember_tabu,
    select_k5_edge,
    write_verified_certificate,
)

SparseCallback = Callable[[int, list[int]], None]


@dataclass(frozen=True)
class FlipMove:
    """One measured edge flip relative to the current state."""

    edge: Edge
    delta_k5: int
    delta_k4: int


@dataclass(frozen=True)
class PairMove:
    """Two sequential flips whose final state is K5-free."""

    first: FlipMove
    second: FlipMove
    delta_k4: int


def load_start(path: Path) -> tuple[int, list[int]]:
    """Load and validate an adjacency-form starting colouring."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("form") != "adjacency":
        raise ValueError(f"start file must use adjacency form: {path}")
    n = int(data["n"])
    if n < 5:
        raise ValueError("a Ramsey K5 certificate needs at least five vertices")
    red = [0] * n
    for raw_edge in data["red_edges"]:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError(f"invalid red edge in {path}: {raw_edge!r}")
        a, b = int(raw_edge[0]), int(raw_edge[1])
        if not 0 <= a < b < n:
            raise ValueError(f"red edge is outside 0..{n - 1}: {(a, b)!r}")
        red[a] |= 1 << b
        red[b] |= 1 << a
    return n, red


def measure_move(
    n: int,
    red: list[int],
    blue: list[int],
    edge: Edge,
) -> FlipMove:
    """Measure exact K5 and K4 deltas, restoring the original state."""

    u, v = edge
    was_red = bool(red[u] >> v & 1)
    before5_red, before5_blue = k5s_through(u, v, red, blue, n)
    before4_red, before4_blue = k4s_through(u, v, red, blue, n)
    before5 = before5_red if was_red else before5_blue
    before4 = before4_red if was_red else before4_blue

    flip_edge(red, blue, u, v)
    try:
        after5_red, after5_blue = k5s_through(u, v, red, blue, n)
        after4_red, after4_blue = k4s_through(u, v, red, blue, n)
        after5 = after5_blue if was_red else after5_red
        after4 = after4_blue if was_red else after4_red
    finally:
        flip_edge(red, blue, u, v)
    return FlipMove(edge=edge, delta_k5=after5 - before5, delta_k4=after4 - before4)


def sample_moves(
    n: int,
    red: list[int],
    blue: list[int],
    rng: random.Random,
    tabu: set[Edge],
    count: int,
) -> list[FlipMove]:
    """Measure distinct moves, including one violation-biased proposal."""

    edges: list[Edge] = []
    seen: set[Edge] = set()
    if count > 0:
        u, v, _, _ = select_k5_edge(
            n,
            red,
            blue,
            rng,
            tabu,
            tournament=min(8, count),
            exploration=0.0,
        )
        edges.append((u, v))
        seen.add((u, v))

    attempts = 0
    while len(edges) < count and attempts < count * 24:
        attempts += 1
        edge = random_edge(n, rng, tabu)
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return [measure_move(n, red, blue, edge) for edge in edges]


def choose_single_move(
    n: int,
    red: list[int],
    blue: list[int],
    current_k5: int,
    rng: random.Random,
    tabu: set[Edge],
    *,
    locked_k5_free: bool,
    samples: int = 16,
) -> FlipMove | None:
    """Choose the best sampled move under the current exact objective."""

    moves = sample_moves(n, red, blue, rng, tabu, samples)
    rng.shuffle(moves)
    if locked_k5_free:
        feasible = [move for move in moves if current_k5 + move.delta_k5 == 0]
        return min(feasible, key=lambda move: move.delta_k4, default=None)

    # One K5 dominates the largest K4 change one edge can cause. This makes
    # repair lexicographic without relying on a hand-tuned constant.
    repair_weight = math.comb(n - 2, 2) + 1
    return min(
        moves,
        key=lambda move: repair_weight * move.delta_k5 + move.delta_k4,
        default=None,
    )


def choose_pair_move(
    n: int,
    red: list[int],
    blue: list[int],
    rng: random.Random,
    tabu: set[Edge],
    *,
    first_samples: int = 12,
    second_samples: int = 20,
    max_intermediate_k5: int = 4,
) -> PairMove | None:
    """Find a two-flip bridge that returns exactly to the K5-free manifold."""

    first_moves = sample_moves(n, red, blue, rng, tabu, first_samples)
    bridges = [
        move for move in first_moves if 0 < move.delta_k5 <= max_intermediate_k5
    ]
    rng.shuffle(bridges)
    bridges.sort(key=lambda move: (move.delta_k5, move.delta_k4))
    best: PairMove | None = None

    for first in bridges[:4]:
        flip_edge(red, blue, *first.edge)
        try:
            blocked = set(tabu)
            blocked.add(first.edge)
            second_moves = sample_moves(
                n,
                red,
                blue,
                rng,
                blocked,
                second_samples,
            )
            rng.shuffle(second_moves)
            for second in second_moves:
                if first.delta_k5 + second.delta_k5 != 0:
                    continue
                candidate = PairMove(
                    first=first,
                    second=second,
                    delta_k4=first.delta_k4 + second.delta_k4,
                )
                if best is None or candidate.delta_k4 < best.delta_k4:
                    best = candidate
        finally:
            flip_edge(red, blue, *first.edge)
    return best


def search(
    n: int,
    seed: int,
    seconds: float,
    start: Path | None = None,
    on_certificate: SparseCallback | None = None,
) -> tuple[int | None, list[int] | None]:
    """Minimize exact K4 clauses and retain only K5-free best states."""

    if n < 5:
        raise ValueError("K5 search needs at least five vertices")
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    rng = random.Random(seed)
    if start is not None:
        start_n, red = load_start(start)
        if start_n != n:
            raise ValueError(f"requested n={n}, but {start} contains n={start_n}")
    else:
        red = random_colouring(n, rng)

    full = (1 << n) - 1
    blue = [(full ^ red[vertex]) & ~(1 << vertex) for vertex in range(n)]
    k5 = brute_cost(n, red)
    k4 = brute_k4_cost(n, red)
    if start is not None and k5 != 0:
        raise ValueError(f"start file is not K5-free: full recount found {k5}")

    locked_k5_free = k5 == 0
    best_k4 = k4 if locked_k5_free else None
    best_red = list(red) if locked_k5_free else None
    started = time.monotonic()
    deadline = started + seconds
    epoch_started = started
    last_improvement = started
    stagnation_window = max(10.0, min(120.0, seconds / 5.0))
    next_report = started + 60.0
    steps = 0
    restarts = 0
    accepted_pairs = 0
    dead_ends = 0
    tabu_limit = max(8, n // 2)
    tabu_queue: deque[Edge] = deque()
    tabu: set[Edge] = set()

    while (now := time.monotonic()) < deadline:
        if now - last_improvement >= stagnation_window:
            restarts += 1
            if best_red is not None and best_k4 is not None:
                red = list(best_red)
                blue = [(full ^ red[vertex]) & ~(1 << vertex) for vertex in range(n)]
                k5 = 0
                k4 = best_k4
                locked_k5_free = True
            else:
                red = random_colouring(n, rng)
                blue = [(full ^ red[vertex]) & ~(1 << vertex) for vertex in range(n)]
                k5 = brute_cost(n, red)
                k4 = brute_k4_cost(n, red)
            epoch_started = now
            last_improvement = now
            dead_ends = 0
            tabu_queue.clear()
            tabu.clear()
            continue

        steps += 1
        if now >= next_report:
            rate = steps / max(now - started, 1e-9)
            print(
                f"  n={n} seed={seed} t={now - started:.0f}s k5={k5} k4={k4} "
                f"best_k4={best_k4} restarts={restarts} pairs={accepted_pairs} "
                f"moves/s={rate:.0f}",
                flush=True,
            )
            next_report = now + 60.0

        epoch_fraction = min(1.0, (now - epoch_started) / stagnation_window)
        k4_temperature = max(0.25, 24.0 * math.exp(-4.0 * epoch_fraction))
        pair: PairMove | None = None
        if locked_k5_free and (dead_ends > 0 or rng.random() < 0.08):
            pair = choose_pair_move(n, red, blue, rng, tabu)

        if pair is not None:
            accepted = pair.delta_k4 <= 0 or rng.random() < math.exp(
                -pair.delta_k4 / k4_temperature
            )
            if accepted:
                flip_edge(red, blue, *pair.first.edge)
                flip_edge(red, blue, *pair.second.edge)
                k5 += pair.first.delta_k5 + pair.second.delta_k5
                k4 += pair.delta_k4
                if k5 != 0:
                    raise RuntimeError("accepted pair did not return to exact K5 cost zero")
                remember_tabu(pair.first.edge, tabu_queue, tabu, tabu_limit)
                remember_tabu(pair.second.edge, tabu_queue, tabu, tabu_limit)
                accepted_pairs += 1
                dead_ends = 0
            else:
                dead_ends += 1
        else:
            move = choose_single_move(
                n,
                red,
                blue,
                k5,
                rng,
                tabu,
                locked_k5_free=locked_k5_free,
            )
            if move is None:
                dead_ends += 1
                if dead_ends % 4 == 0:
                    tabu_queue.clear()
                    tabu.clear()
                continue

            if locked_k5_free:
                objective_delta = move.delta_k4
                temperature = k4_temperature
            else:
                repair_weight = math.comb(n - 2, 2) + 1
                objective_delta = repair_weight * move.delta_k5 + move.delta_k4
                temperature = max(0.5, 8.0 * math.exp(-4.0 * epoch_fraction))
            accepted = objective_delta <= 0 or rng.random() < math.exp(
                -objective_delta / temperature
            )
            if accepted:
                flip_edge(red, blue, *move.edge)
                k5 += move.delta_k5
                k4 += move.delta_k4
                if locked_k5_free and k5 != 0:
                    raise RuntimeError("K5-free move selection accepted an infeasible state")
                remember_tabu(move.edge, tabu_queue, tabu, tabu_limit)
                dead_ends = 0
            else:
                dead_ends += 1

        if k5 == 0 and (best_k4 is None or k4 < best_k4):
            locked_k5_free = True
            best_k4 = k4
            best_red = list(red)
            last_improvement = now
            if on_certificate is not None:
                on_certificate(best_k4, list(best_red))

    return best_k4, best_red


if __name__ == "__main__":
    n_arg = int(sys.argv[1])
    seed_arg = int(sys.argv[2])
    seconds_arg = float(sys.argv[3])
    output = (
        Path(sys.argv[4])
        if len(sys.argv) > 4
        else Path(f"sparse-n{n_arg}-seed{seed_arg}.json")
    )
    start_path = Path(sys.argv[5]) if len(sys.argv) > 5 else None

    best_count, best_adjacency = search(
        n_arg,
        seed_arg,
        seconds_arg,
        start_path,
    )
    if best_adjacency is None or best_count is None:
        print(f"n={n_arg} seed={seed_arg}: never reached a K5-free colouring", flush=True)
        raise SystemExit(0)

    verified_k5, verified_k4 = write_verified_certificate(
        output,
        n_arg,
        best_adjacency,
        expected_k5=0,
        expected_k4=best_count,
    )
    print(
        f"n={n_arg} seed={seed_arg}: best K5-free colouring has {verified_k4} "
        f"monochromatic K4s (recounted K5s = {verified_k5})",
        flush=True,
    )
    print(f"  wrote {output}", flush=True)
