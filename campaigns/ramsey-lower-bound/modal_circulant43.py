"""Exhaustively decide the circulant family for a K5-free colouring on n vertices.

For prime n a circulant colouring is determined by a subset S of the (n-1)/2
difference classes: the edge between i and j is red exactly when the circular
distance between them lies in S. That is 2**21 = 2097152 colourings at n = 43,
which is small enough to settle completely.

Two facts make each check cheap. A circulant is vertex transitive, so a
monochromatic K5 exists if and only if one exists through vertex 0. And a red
K5 through vertex 0 is vertex 0 together with a red K4 inside the red
neighbourhood of 0. So each candidate reduces to finding a K4 in an induced
subgraph on about (n-1)/2 vertices, in each of two colours.

The answer is decisive either way. A hit at n = 43 is a two colouring with no
monochromatic K5, which witnesses R(5,5) >= 44 and improves a bound that has
stood since 1989. An exhaustive miss proves no circulant colouring on 43
vertices works, which is a real negative result and redirects the search to
less symmetric families.

Every hit is re-verified from scratch, over all C(n,5) subsets, before it is
reported. A search that announces something it cannot re-check is worse than
one that finds nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"


def _head_sha() -> str:
    """Pin the image to the local commit, tolerating the container layout."""

    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:
        return "main"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    return result.stdout.strip() or "main"


COMMIT = _head_sha()

app = modal.App("autoevolve-circulant43")
store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO} {REPO_ROOT}",
        f"cd {REPO_ROOT} && git checkout {COMMIT}",
    )
)


@app.function(image=image, volumes={"/store": store}, timeout=60 * 60, cpu=2.0)
def sweep(n: int, shard: int, shard_count: int) -> dict:
    """Test every circulant colouring in one shard of the subset space."""

    from itertools import combinations

    half = (n - 1) // 2
    total = 1 << half

    def has_k4(members: list[int], adjacency: dict[int, int]) -> bool:
        """Whether the induced subgraph on `members` contains a K4."""

        count = len(members)
        for a in range(count):
            u = members[a]
            for b in range(a + 1, count):
                v = members[b]
                if not (adjacency[u] >> v & 1):
                    continue
                for c in range(b + 1, count):
                    w = members[c]
                    if not (adjacency[u] >> w & 1 and adjacency[v] >> w & 1):
                        continue
                    for d in range(c + 1, count):
                        x = members[d]
                        if (
                            adjacency[u] >> x & 1
                            and adjacency[v] >> x & 1
                            and adjacency[w] >> x & 1
                        ):
                            return True
        return False

    def distance(i: int, j: int) -> int:
        raw = (i - j) % n
        return min(raw, n - raw)

    hits: list[list[int]] = []
    tested = 0
    for mask in range(shard, total, shard_count):
        tested += 1
        red_classes = {index + 1 for index in range(half) if mask >> index & 1}
        if not red_classes or len(red_classes) == half:
            continue

        red_rows = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if distance(i, j) in red_classes:
                    red_rows[i] |= 1 << j
                    red_rows[j] |= 1 << i
        full = (1 << n) - 1
        blue_rows = [(full ^ red_rows[i]) & ~(1 << i) for i in range(n)]

        red_neighbours = [v for v in range(1, n) if red_rows[0] >> v & 1]
        if has_k4(red_neighbours, {v: red_rows[v] for v in red_neighbours}):
            continue
        blue_neighbours = [v for v in range(1, n) if blue_rows[0] >> v & 1]
        if has_k4(blue_neighbours, {v: blue_rows[v] for v in blue_neighbours}):
            continue

        # Survived the transitive check. Re-verify exhaustively before believing.
        monochromatic = 0
        for group in combinations(range(n), 5):
            reds = sum(1 for a, b in combinations(group, 2) if red_rows[a] >> b & 1)
            if reds in (0, 10):
                monochromatic += 1
                break
        if monochromatic == 0:
            hits.append(sorted(red_classes))

    return {"n": n, "shard": shard, "tested": tested, "hits": hits}


@app.local_entrypoint()
def main(n: int = 43, shard_count: int = 64) -> None:
    """Fan the whole subset space across containers and report the verdict."""

    import json

    tested = 0
    hits: list[list[int]] = []
    for result in sweep.starmap([(n, shard, shard_count) for shard in range(shard_count)]):
        tested += result["tested"]
        hits.extend(result["hits"])

    print(
        json.dumps(
            {
                "n": n,
                "subsets_tested": tested,
                "certificates_found": len(hits),
                "connection_sets": hits[:20],
                "verdict": (
                    f"FOUND a K5-free circulant colouring on {n} vertices, "
                    f"which witnesses R(5,5) >= {n + 1}"
                    if hits
                    else f"exhaustive: no circulant colouring on {n} vertices is K5-free"
                ),
            },
            indent=2,
        )
    )
