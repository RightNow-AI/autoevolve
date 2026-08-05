"""Cross-check every exact cost primitive used by the Ramsey searches.

The searches decide that a certificate exists from incremental edge-flip
costs. A bad delta could therefore create a false mathematical claim. This
self-test compares K5 and K4 updates against full enumeration, checks the tabu
selector, checks sequential two-edge updates, and confirms that the shared
writer rejects an incremental/full-recount disagreement.
"""

from __future__ import annotations

import random
import tempfile
from itertools import combinations
from pathlib import Path

from search_n import (
    brute_cost,
    brute_k4_cost,
    edges_in,
    flip_edge,
    k4s_through,
    k5s_through,
    random_colouring,
    random_edge,
    select_k5_edge,
    triangles_in,
    write_verified_certificate,
)
from search_sparse42 import measure_move


def brute_triangles(mask: int, adjacency: list[int]) -> int:
    """Count triangles by direct triples."""

    verts = [vertex for vertex in range(len(adjacency)) if mask >> vertex & 1]
    return sum(
        1
        for a, b, c in combinations(verts, 3)
        if (adjacency[a] >> b & 1)
        and (adjacency[a] >> c & 1)
        and (adjacency[b] >> c & 1)
    )


def brute_edges(mask: int, adjacency: list[int]) -> int:
    """Count edges by direct pairs."""

    verts = [vertex for vertex in range(len(adjacency)) if mask >> vertex & 1]
    return sum(1 for a, b in combinations(verts, 2) if adjacency[a] >> b & 1)


def fresh_state(n: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """Create matching red and blue adjacency arrays."""

    red = random_colouring(n, rng)
    full = (1 << n) - 1
    blue = [(full ^ red[vertex]) & ~(1 << vertex) for vertex in range(n)]
    return red, blue


rng = random.Random(20260804)
failures: list[str] = []

print("triangle and edge counters against brute force:")
counter_matches = 0
for _ in range(200):
    size = rng.randint(3, 14)
    adjacency, _ = fresh_state(size, rng)
    selected = rng.randrange(1 << size)
    triangles_match = triangles_in(selected, adjacency) == brute_triangles(selected, adjacency)
    edges_match = edges_in(selected, adjacency) == brute_edges(selected, adjacency)
    if triangles_match and edges_match:
        counter_matches += 1
print(f"  {counter_matches}/200 matched")
if counter_matches != 200:
    failures.append("triangle or edge counter mismatch")

print("incremental K5 and K4 costs against full recounts:")
incremental_matches = 0
incremental_trials = 0
for _ in range(12):
    size = rng.randint(8, 12)
    red, blue = fresh_state(size, rng)
    k5_cost = brute_cost(size, red)
    k4_cost = brute_k4_cost(size, red)
    for _ in range(40):
        incremental_trials += 1
        u, v = random_edge(size, rng)
        was_red = bool(red[u] >> v & 1)
        before5_red, before5_blue = k5s_through(u, v, red, blue, size)
        before4_red, before4_blue = k4s_through(u, v, red, blue, size)
        before5 = before5_red if was_red else before5_blue
        before4 = before4_red if was_red else before4_blue
        flip_edge(red, blue, u, v)
        after5_red, after5_blue = k5s_through(u, v, red, blue, size)
        after4_red, after4_blue = k4s_through(u, v, red, blue, size)
        after5 = after5_blue if was_red else after5_red
        after4 = after4_blue if was_red else after4_red
        k5_cost += after5 - before5
        k4_cost += after4 - before4
        if k5_cost == brute_cost(size, red) and k4_cost == brute_k4_cost(size, red):
            incremental_matches += 1
        else:
            break
print(f"  {incremental_matches}/{incremental_trials} flips stayed exact")
if incremental_matches != incremental_trials:
    failures.append("incremental K5 or K4 mismatch")

print("measured sequential two-edge deltas against full recounts:")
pair_matches = 0
for _ in range(100):
    size = rng.randint(8, 12)
    red, blue = fresh_state(size, rng)
    before5 = brute_cost(size, red)
    before4 = brute_k4_cost(size, red)
    first_edge = random_edge(size, rng)
    first = measure_move(size, red, blue, first_edge)
    flip_edge(red, blue, *first.edge)
    second_edge = random_edge(size, rng, {first.edge})
    second = measure_move(size, red, blue, second_edge)
    flip_edge(red, blue, *second.edge)
    expected5 = before5 + first.delta_k5 + second.delta_k5
    expected4 = before4 + first.delta_k4 + second.delta_k4
    if expected5 == brute_cost(size, red) and expected4 == brute_k4_cost(size, red):
        pair_matches += 1
print(f"  {pair_matches}/100 pairs stayed exact")
if pair_matches != 100:
    failures.append("sequential pair delta mismatch")

print("tabu-aware edge selection:")
size = 8
red, blue = fresh_state(size, rng)
allowed = (size - 2, size - 1)
tabu = set(combinations(range(size), 2))
tabu.remove(allowed)
random_pick = random_edge(size, rng, tabu)
selected_u, selected_v, _, _ = select_k5_edge(size, red, blue, rng, tabu)
tabu_ok = random_pick == allowed and (selected_u, selected_v) == allowed
print("  avoided every tabu edge" if tabu_ok else "  selected a tabu edge")
if not tabu_ok:
    failures.append("tabu-aware selector mismatch")

print("verified certificate writer:")
writer_ok = False
with tempfile.TemporaryDirectory() as temporary_dir:
    directory = Path(temporary_dir)
    size = 5
    red = [0] * size
    red[0] |= 1 << 1
    red[1] |= 1 << 0
    good_path = directory / "good.json"
    write_verified_certificate(good_path, size, red, expected_k5=0)
    bad_path = directory / "bad.json"
    try:
        write_verified_certificate(bad_path, size, red, expected_k5=1)
    except ValueError:
        writer_ok = good_path.is_file() and not bad_path.exists()
print("  wrote only the fully recounted certificate" if writer_ok else "  writer check failed")
if not writer_ok:
    failures.append("verified writer did not fail closed")

print()
if failures:
    print("COST MODEL IS WRONG: " + "; ".join(failures))
    raise SystemExit(1)
print("SEARCH COST MODEL TRUSTWORTHY")
