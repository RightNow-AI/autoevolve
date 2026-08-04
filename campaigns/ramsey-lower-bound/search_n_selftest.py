"""Validate the incremental cost used by search_n.py against brute force.

The search decides it has found a certificate when its running cost reaches
zero. If the incremental update is wrong, it will announce a certificate that
does not exist. So the delta is checked against a full recount over many
random flips before the search is trusted, and the search itself re-counts
from scratch before writing anything out.
"""

from __future__ import annotations

import random
from itertools import combinations

from search_n import brute_cost, k5s_through, triangles_in


def brute_triangles(mask: int, adjacency: list[int]) -> int:
    verts = [i for i in range(64) if mask >> i & 1]
    return sum(
        1
        for a, b, c in combinations(verts, 3)
        if (adjacency[a] >> b & 1) and (adjacency[a] >> c & 1) and (adjacency[b] >> c & 1)
    )


rng = random.Random(20260804)

print("triangle counter against brute force:")
bad = 0
for _ in range(200):
    n = rng.randint(3, 14)
    adj = [0] * n
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.5:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
    mask = rng.randrange(1 << n)
    if triangles_in(mask, adj) != brute_triangles(mask, adj):
        bad += 1
print(f"  {200 - bad}/200 matched")

print("incremental cost against a full recount, over random flips:")
mismatches = 0
trials = 0
for _ in range(12):
    n = rng.randint(8, 12)
    full = (1 << n) - 1
    red = [0] * n
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.5:
                red[a] |= 1 << b
                red[b] |= 1 << a
    blue = [(full ^ red[i]) & ~(1 << i) for i in range(n)]
    cost = brute_cost(n, red)
    for _ in range(40):
        trials += 1
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
        cost += after - before
        if cost != brute_cost(n, red):
            mismatches += 1
            break
print(f"  {trials - mismatches}/{trials} flips kept the incremental cost exact")

print()
print("SEARCH COST MODEL TRUSTWORTHY" if bad == 0 and mismatches == 0 else "COST MODEL IS WRONG")
