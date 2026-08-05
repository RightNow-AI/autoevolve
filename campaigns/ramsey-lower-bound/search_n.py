"""Direct search for a K5-free two-colouring on n vertices.

A colouring of the complete graph on n vertices is K5-free when no five
vertices are joined entirely by red edges or entirely by blue edges. Such a
colouring on n vertices witnesses R(5,5) > n.

The search uses simulated annealing over single edge flips. It samples several
candidate edges and usually chooses one that currently participates in many
monochromatic K5s. Equal-cost flips cross plateaus, a short tabu memory avoids
immediate cycles, and a stalled epoch restarts near the best state found so far.

Why a flip is cheap. Flipping edge uv only changes the status of five-sets
that contain both u and v. When uv is red, the red K5s through it are exactly
the red triangles inside the common red neighbourhood of u and v, and there
are no blue K5s through it. After the flip those roles swap. One flip therefore
needs two small triangle counts rather than a pass over all C(n, 5) subsets.

Usage: search_n.py <n> <seed> <seconds> [output.json]
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import deque
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

CertificateCallback = Callable[[int, list[int]], None]
Edge = tuple[int, int]


def brute_cost(n: int, red: list[int]) -> int:
    """Count monochromatic K5s directly."""

    total = 0
    for group in combinations(range(n), 5):
        reds = sum(1 for a, b in combinations(group, 2) if red[a] >> b & 1)
        if reds in (0, 10):
            total += 1
    return total


def triangles_in(mask: int, adjacency: list[int]) -> int:
    """Count triangles inside the vertex set ``mask`` under ``adjacency``."""

    verts: list[int] = []
    remaining = mask
    while remaining:
        low = remaining & -remaining
        verts.append(low.bit_length() - 1)
        remaining ^= low
    count = 0
    for index, u in enumerate(verts):
        neighbours = adjacency[u] & mask
        for v in verts[index + 1 :]:
            if neighbours >> v & 1:
                count += (neighbours & adjacency[v] & mask).bit_count()
    return count // 3


def k5s_through(
    u: int,
    v: int,
    red: list[int],
    blue: list[int],
    n: int,
) -> tuple[int, int]:
    """Return monochromatic K5s through edge uv as red and blue counts."""

    full = (1 << n) - 1
    others = full ^ (1 << u) ^ (1 << v)
    red_common = red[u] & red[v] & others
    blue_common = blue[u] & blue[v] & others
    return triangles_in(red_common, red), triangles_in(blue_common, blue)


def edges_in(mask: int, adjacency: list[int]) -> int:
    """Count edges inside the vertex set ``mask`` under ``adjacency``."""

    verts: list[int] = []
    remaining = mask
    while remaining:
        low = remaining & -remaining
        verts.append(low.bit_length() - 1)
        remaining ^= low
    return sum((adjacency[u] & mask).bit_count() for u in verts) // 2


def k4s_through(
    u: int,
    v: int,
    red: list[int],
    blue: list[int],
    n: int,
) -> tuple[int, int]:
    """Return monochromatic K4s through edge uv as red and blue counts."""

    full = (1 << n) - 1
    others = full ^ (1 << u) ^ (1 << v)
    return (
        edges_in(red[u] & red[v] & others, red),
        edges_in(blue[u] & blue[v] & others, blue),
    )


def brute_k4_cost(n: int, red: list[int]) -> int:
    """Count monochromatic K4s directly."""

    total = 0
    for group in combinations(range(n), 4):
        reds = sum(1 for a, b in combinations(group, 2) if red[a] >> b & 1)
        if reds in (0, 6):
            total += 1
    return total


def flip_edge(red: list[int], blue: list[int], u: int, v: int) -> None:
    """Flip one undirected edge in both adjacency representations."""

    red[u] ^= 1 << v
    red[v] ^= 1 << u
    blue[u] ^= 1 << v
    blue[v] ^= 1 << u


def random_edge(n: int, rng: random.Random, tabu: set[Edge] | None = None) -> Edge:
    """Choose a canonical edge, avoiding tabu edges when another edge exists."""

    if n < 2:
        raise ValueError("an edge needs at least two vertices")
    blocked = tabu or set()
    for _ in range(64):
        u = rng.randrange(n)
        v = rng.randrange(n - 1)
        if v >= u:
            v += 1
        edge = (u, v) if u < v else (v, u)
        if edge not in blocked:
            return edge
    available = [edge for edge in combinations(range(n), 2) if edge not in blocked]
    if available:
        return rng.choice(available)
    return rng.choice(list(combinations(range(n), 2)))


def select_k5_edge(
    n: int,
    red: list[int],
    blue: list[int],
    rng: random.Random,
    tabu: set[Edge],
    *,
    tournament: int = 6,
    exploration: float = 0.15,
) -> tuple[int, int, bool, int]:
    """Choose an edge biased toward current monochromatic K5 participation."""

    candidates: list[tuple[int, int, bool, int]] = []
    seen: set[Edge] = set()
    attempts = 0
    while len(candidates) < tournament and attempts < tournament * 20:
        attempts += 1
        u, v = random_edge(n, rng, tabu)
        if (u, v) in seen:
            continue
        seen.add((u, v))
        was_red = bool(red[u] >> v & 1)
        red_count, blue_count = k5s_through(u, v, red, blue, n)
        badness = red_count if was_red else blue_count
        candidates.append((u, v, was_red, badness))
    if not candidates:
        u, v = random_edge(n, rng)
        was_red = bool(red[u] >> v & 1)
        red_count, blue_count = k5s_through(u, v, red, blue, n)
        return u, v, was_red, red_count if was_red else blue_count
    if rng.random() < exploration:
        return rng.choice(candidates)
    highest = max(candidate[3] for candidate in candidates)
    strongest = [candidate for candidate in candidates if candidate[3] == highest]
    return rng.choice(strongest)


def remember_tabu(
    edge: Edge,
    queue: deque[Edge],
    members: set[Edge],
    limit: int,
) -> None:
    """Remember an accepted flip without allowing the set and queue to drift."""

    if edge in members:
        return
    queue.append(edge)
    members.add(edge)
    while len(queue) > limit:
        members.remove(queue.popleft())


def random_colouring(n: int, rng: random.Random) -> list[int]:
    """Create a symmetric random red adjacency bitset."""

    red = [0] * n
    for a, b in combinations(range(n), 2):
        if rng.random() < 0.5:
            red[a] |= 1 << b
            red[b] |= 1 << a
    return red


def restart_colouring(
    n: int,
    rng: random.Random,
    best_red: list[int],
    restart_index: int,
) -> list[int]:
    """Restart near the retained best, with periodic fresh random diversity."""

    if restart_index % 4 == 0:
        return random_colouring(n, rng)
    red = list(best_red)
    blue = [0] * n
    full = (1 << n) - 1
    for vertex in range(n):
        blue[vertex] = (full ^ red[vertex]) & ~(1 << vertex)
    flipped: set[Edge] = set()
    while len(flipped) < max(4, n):
        edge = random_edge(n, rng, flipped)
        flipped.add(edge)
        flip_edge(red, blue, *edge)
    return red


def write_verified_certificate(
    path: Path,
    n: int,
    red: list[int],
    *,
    expected_k5: int,
    expected_k4: int | None = None,
) -> tuple[int, int | None]:
    """Recount from scratch and atomically write only a genuine certificate."""

    verified_k5 = brute_cost(n, red)
    if verified_k5 != expected_k5:
        raise ValueError(
            "incremental K5 cost disagreed with the full recount: "
            f"expected {expected_k5}, recounted {verified_k5}"
        )
    if verified_k5 != 0:
        raise ValueError(f"refusing to write a colouring with {verified_k5} monochromatic K5s")

    verified_k4: int | None = None
    if expected_k4 is not None:
        verified_k4 = brute_k4_cost(n, red)
        if verified_k4 != expected_k4:
            raise ValueError(
                "incremental K4 cost disagreed with the full recount: "
                f"expected {expected_k4}, recounted {verified_k4}"
            )

    edges = [[a, b] for a, b in combinations(range(n), 2) if red[a] >> b & 1]
    payload = {"form": "adjacency", "n": n, "red_edges": edges}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return verified_k5, verified_k4


def search(
    n: int,
    seed: int,
    seconds: float,
    on_certificate: CertificateCallback | None = None,
) -> tuple[int, list[int]]:
    """Search until the deadline and return the retained best exact state."""

    if n < 5:
        raise ValueError("K5 search needs at least five vertices")
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    rng = random.Random(seed)
    full = (1 << n) - 1
    red = random_colouring(n, rng)
    blue = [(full ^ red[vertex]) & ~(1 << vertex) for vertex in range(n)]
    cost = brute_cost(n, red)
    best = cost
    best_red = list(red)
    if best == 0:
        if on_certificate is not None:
            on_certificate(best, list(best_red))
        return best, best_red

    started = time.monotonic()
    deadline = started + seconds
    epoch_started = started
    last_improvement = started
    stagnation_window = max(5.0, min(60.0, seconds / 6.0))
    next_report = started + 60.0
    steps = 0
    restarts = 0
    tabu_limit = max(6, n // 2)
    tabu_queue: deque[Edge] = deque()
    tabu: set[Edge] = set()

    while (now := time.monotonic()) < deadline:
        if now - last_improvement >= stagnation_window:
            restarts += 1
            red = restart_colouring(n, rng, best_red, restarts)
            blue = [(full ^ red[vertex]) & ~(1 << vertex) for vertex in range(n)]
            cost = brute_cost(n, red)
            if cost < best:
                best = cost
                best_red = list(red)
                last_improvement = now
                if best == 0:
                    if on_certificate is not None:
                        on_certificate(best, list(best_red))
                    return best, best_red
            else:
                last_improvement = now
            epoch_started = now
            tabu_queue.clear()
            tabu.clear()
            continue

        steps += 1
        if now >= next_report:
            rate = steps / max(now - started, 1e-9)
            print(
                f"  n={n} seed={seed} t={now - started:.0f}s cost={cost} best={best} "
                f"restarts={restarts} flips/s={rate:.0f}",
                flush=True,
            )
            next_report = now + 60.0

        u, v, was_red, before = select_k5_edge(n, red, blue, rng, tabu)
        flip_edge(red, blue, u, v)
        after_red, after_blue = k5s_through(u, v, red, blue, n)
        after = after_blue if was_red else after_red
        delta = after - before
        epoch_fraction = min(1.0, (now - epoch_started) / stagnation_window)
        temperature = max(0.05, 3.0 * math.exp(-4.0 * epoch_fraction))

        # Equal-cost moves are always accepted so flat basins remain traversable.
        accepted = delta <= 0 or rng.random() < math.exp(-delta / temperature)
        if accepted:
            cost += delta
            remember_tabu((u, v), tabu_queue, tabu, tabu_limit)
            if cost < best:
                best = cost
                best_red = list(red)
                last_improvement = now
                if best == 0:
                    if on_certificate is not None:
                        on_certificate(best, list(best_red))
                    return best, best_red
        else:
            flip_edge(red, blue, u, v)

    return best, best_red


if __name__ == "__main__":
    n_arg = int(sys.argv[1])
    seed_arg = int(sys.argv[2])
    seconds_arg = float(sys.argv[3])
    output = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    best_cost, best_adjacency = search(n_arg, seed_arg, seconds_arg)
    print(
        f"n={n_arg} seed={seed_arg}: best monochromatic K5 count = {best_cost}",
        flush=True,
    )
    if best_cost == 0:
        target = output or Path(f"found-n{n_arg}-seed{seed_arg}.json")
        verified, _ = write_verified_certificate(
            target,
            n_arg,
            best_adjacency,
            expected_k5=best_cost,
        )
        print(f"  re-counted from scratch: {verified}", flush=True)
        print(f"  CERTIFICATE WRITTEN: {target}", flush=True)
