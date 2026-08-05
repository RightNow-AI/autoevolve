"""Search for a K5-free colouring on n vertices with few monochromatic K4s.

Why this objective. Adding a 43rd vertex to a K5-free 42 vertex colouring is
satisfiable exactly when the 42 vertices split into R with no red K4 and B with
no blue K4. That is one clause per monochromatic K4 over 42 variables. Every
certificate produced so far carries about 1160 red and 1150 blue such K4s,
which is roughly 2300 clauses on 42 variables, a ratio near 55. Random 4-SAT
stops being satisfiable somewhere near a ratio of 10, and all three
certificates tested came back UNSAT.

The clauses here are structured rather than random, so that threshold is a
guide and not a proof. What it does say is that the useful thing to search for
is not another 42 vertex certificate, it is a 42 vertex certificate with far
fewer monochromatic K4s. This anneals on exactly that: K5s are forbidden with a
large weight, K4s are merely penalised.

Usage: search_sparse42.py <n> <seed> <seconds> [output.json]
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_n import (  # noqa: E402
    brute_cost,
    brute_k4_cost,
    k4s_through,
    k5s_through,
)

K5_WEIGHT = 10_000


def load_start(path: Path) -> tuple[int, list[int]]:
    """Start from an existing certificate instead of a random colouring.

    Reaching K5-free at 42 vertices from random is itself hard: a ten minute
    anneal from random stalls around 104 monochromatic K5s. Certificates that
    are already K5-free exist in this repository, so the search begins at cost
    zero and spends its whole budget on the objective that matters, which is
    reducing the monochromatic K4 count.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    n = int(data["n"])
    red = [0] * n
    for a, b in data["red_edges"]:
        red[int(a)] |= 1 << int(b)
        red[int(b)] |= 1 << int(a)
    return n, red


def search(n: int, seed: int, seconds: float, start: Path | None = None):
    rng = random.Random(seed)
    if start is not None:
        n, red = load_start(start)
    else:
        red = [0] * n
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < 0.5:
                    red[a] |= 1 << b
                    red[b] |= 1 << a
    full = (1 << n) - 1
    blue = [(full ^ red[i]) & ~(1 << i) for i in range(n)]

    k5 = brute_cost(n, red)
    k4 = brute_k4_cost(n, red)
    best_k4 = k4 if k5 == 0 else None
    best_red = list(red) if k5 == 0 else None
    started = time.time()
    steps = 0
    next_report = started + 60.0

    while time.time() - started < seconds:
        steps += 1
        now = time.time()
        if now >= next_report:
            print(
                f"  n={n} seed={seed} t={now - started:.0f}s k5={k5} k4={k4} "
                f"best_k4={best_k4} flips/s={steps / max(now - started, 1e-9):.0f}",
                flush=True,
            )
            next_report = now + 60.0

        u = rng.randrange(n)
        v = rng.randrange(n)
        while v == u:
            v = rng.randrange(n)
        was_red = bool(red[u] >> v & 1)

        b5r, b5b = k5s_through(u, v, red, blue, n)
        b4r, b4b = k4s_through(u, v, red, blue, n)
        before5 = b5r if was_red else b5b
        before4 = b4r if was_red else b4b

        red[u] ^= 1 << v
        red[v] ^= 1 << u
        blue[u] ^= 1 << v
        blue[v] ^= 1 << u

        a5r, a5b = k5s_through(u, v, red, blue, n)
        a4r, a4b = k4s_through(u, v, red, blue, n)
        after5 = a5b if was_red else a5r
        after4 = a4b if was_red else a4r

        d5 = after5 - before5
        d4 = after4 - before4
        delta = K5_WEIGHT * d5 + d4
        temperature = max(0.5, 60.0 * math.exp(-3.0 * (now - started) / seconds))

        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            k5 += d5
            k4 += d4
            if k5 == 0 and (best_k4 is None or k4 < best_k4):
                best_k4 = k4
                best_red = list(red)
        else:
            red[u] ^= 1 << v
            red[v] ^= 1 << u
            blue[u] ^= 1 << v
            blue[v] ^= 1 << u

    return best_k4, best_red


if __name__ == "__main__":
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    seconds = float(sys.argv[3])
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(f"sparse-n{n}-seed{seed}.json")
    start = Path(sys.argv[5]) if len(sys.argv) > 5 else None

    best_k4, best_red = search(n, seed, seconds, start)
    if best_red is None:
        print(f"n={n} seed={seed}: never reached a K5-free colouring", flush=True)
        raise SystemExit(0)

    verified_k5 = brute_cost(n, best_red)
    verified_k4 = brute_k4_cost(n, best_red)
    print(
        f"n={n} seed={seed}: best K5-free colouring has {verified_k4} monochromatic K4s "
        f"(recounted K5s = {verified_k5})",
        flush=True,
    )
    if verified_k5 != 0:
        print("  REJECTED: recount disagreed, not K5-free", flush=True)
        raise SystemExit(1)
    edges = [[a, b] for a in range(n) for b in range(a + 1, n) if best_red[a] >> b & 1]
    out.write_text(
        json.dumps({"form": "adjacency", "n": n, "red_edges": edges}), encoding="utf-8"
    )
    print(f"  wrote {out}", flush=True)
