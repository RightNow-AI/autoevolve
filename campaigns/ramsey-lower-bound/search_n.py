"""Direct local search for a K5-free two-colouring on n vertices.

A colouring of the complete graph on n vertices is K5-free when no five
vertices are joined entirely by red edges or entirely by blue edges. Such a
colouring on n vertices witnesses R(5,5) > n. The published bound is
R(5,5) >= 43, which comes from n = 42, so n = 43 would improve a result that
has stood since 1989.

The search is simulated annealing over single edge flips. The cost is the
number of monochromatic K5s, and a cost of zero is a certificate.

Why a flip is cheap. Flipping edge uv only changes the status of five-sets
that contain both u and v. When uv is red, the red K5s through it are exactly
the red triangles inside the common red neighbourhood of u and v, and there
are no blue K5s through it. After the flip those roles swap. So one flip costs
two small triangle counts over sets of about twenty vertices, not a pass over
all C(n,5) subsets.

Usage: search_n.py <n> <seed> <seconds> [output.json]
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path


def brute_cost(n: int, red: list[int]) -> int:
    """Count monochromatic K5s directly. Used once, and to audit the delta."""

    total = 0
    for group in combinations(range(n), 5):
        pairs = list(combinations(group, 2))
        reds = sum(1 for a, b in pairs if red[a] >> b & 1)
        if reds in (0, 10):
            total += 1
    return total


def triangles_in(mask: int, adjacency: list[int]) -> int:
    """Count triangles inside the vertex set `mask` under `adjacency`.

    Every triangle is found once per edge, so the running total is divided by
    three at the end.
    """

    verts = []
    m = mask
    while m:
        low = m & -m
        verts.append(low.bit_length() - 1)
        m ^= low
    count = 0
    for i, u in enumerate(verts):
        nu = adjacency[u] & mask
        for v in verts[i + 1 :]:
            if not (nu >> v & 1):
                continue
            count += bin(nu & adjacency[v] & mask).count("1")
    return count // 3


def k5s_through(u: int, v: int, red: list[int], blue: list[int], n: int) -> tuple[int, int]:
    """Monochromatic K5s through edge uv, as (red count, blue count)."""

    full = (1 << n) - 1
    others = full ^ (1 << u) ^ (1 << v)
    red_common = red[u] & red[v] & others
    blue_common = blue[u] & blue[v] & others
    return triangles_in(red_common, red), triangles_in(blue_common, blue)


def search(n: int, seed: int, seconds: float) -> tuple[int, list[int]] | None:
    rng = random.Random(seed)
    full = (1 << n) - 1
    red = [0] * n
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.5:
                red[a] |= 1 << b
                red[b] |= 1 << a
    blue = [(full ^ red[i]) & ~(1 << i) for i in range(n)]

    cost = brute_cost(n, red)
    best = cost
    started = time.time()
    temperature = 3.0
    steps = 0
    next_report = started + 60.0
    while time.time() - started < seconds:
        steps += 1
        now = time.time()
        if now >= next_report:
            # Report while running. A search that only speaks at the end gives
            # no way to tell a descending cost from a stuck one.
            rate = steps / max(now - started, 1e-9)
            print(
                f"  n={n} seed={seed} t={now - started:.0f}s cost={cost} best={best} "
                f"temp={temperature:.2f} flips/s={rate:.0f}",
                flush=True,
            )
            next_report = now + 60.0
        u = rng.randrange(n)
        v = rng.randrange(n)
        while v == u:
            v = rng.randrange(n)
        was_red = bool(red[u] >> v & 1)
        before_red, before_blue = k5s_through(u, v, red, blue, n)
        before = before_red if was_red else before_blue

        red[u] ^= 1 << v
        red[v] ^= 1 << u
        blue[u] ^= 1 << v
        blue[v] ^= 1 << u

        after_red, after_blue = k5s_through(u, v, red, blue, n)
        after = after_blue if was_red else after_red
        delta = after - before

        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-6)):
            cost += delta
            if cost < best:
                best = cost
                if cost == 0:
                    return 0, red
        else:
            red[u] ^= 1 << v
            red[v] ^= 1 << u
            blue[u] ^= 1 << v
            blue[v] ^= 1 << u
        temperature = max(0.05, 3.0 * math.exp(-3.0 * (time.time() - started) / seconds))
    return best, red


if __name__ == "__main__":
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    seconds = float(sys.argv[3])
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    result = search(n, seed, seconds)
    assert result is not None
    best, red = result
    print(f"n={n} seed={seed}: best monochromatic K5 count = {best}", flush=True)
    if best == 0:
        verified = brute_cost(n, red)
        print(f"  re-counted from scratch: {verified}", flush=True)
        if verified != 0:
            print("  REJECTED: incremental cost disagreed with a full recount", flush=True)
            sys.exit(1)
        edges = [[a, b] for a in range(n) for b in range(a + 1, n) if red[a] >> b & 1]
        payload = {"form": "adjacency", "n": n, "red_edges": edges}
        target = out or Path(f"found-n{n}-seed{seed}.json")
        target.write_text(json.dumps(payload), encoding="utf-8")
        print(f"  CERTIFICATE WRITTEN: {target}", flush=True)
