"""Independently verify a Ramsey certificate without touching the pack's code.

The point of this file is that it shares nothing with the evaluator that
produced the verdict. It reads the JSON, rebuilds the graph from scratch, and
brute forces every 5-subset in pure Python with exact integer and set logic.
If the pack's three verifiers were all wrong in the same way, this would
disagree with them.

A valid certificate on n vertices means no monochromatic K5, which witnesses
R(5,5) > n, that is R(5,5) >= n + 1.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

S = 5


def load(path: Path) -> tuple[int, set[tuple[int, int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["form"] != "adjacency":
        raise SystemExit(f"unsupported form {data['form']!r}")
    n = int(data["n"])
    red: set[tuple[int, int]] = set()
    for pair in data["red_edges"]:
        left, right = int(pair[0]), int(pair[1])
        if not 0 <= left < n or not 0 <= right < n:
            raise SystemExit(f"edge {pair} outside 0..{n - 1}")
        if left == right:
            raise SystemExit(f"self loop at {left}")
        red.add((min(left, right), max(left, right)))
    return n, red


def verify(n: int, red: set[tuple[int, int]]) -> tuple[bool, str]:
    """Brute force every 5-subset. Returns (valid, detail)."""

    checked = 0
    for group in combinations(range(n), S):
        pairs = list(combinations(group, 2))
        red_count = sum(1 for a, b in pairs if (a, b) in red)
        checked += 1
        if red_count == len(pairs):
            return False, f"RED K{S} on {group}"
        if red_count == 0:
            return False, f"BLUE K{S} on {group}"
    return True, f"checked {checked} subsets of size {S}, none monochromatic"


for argument in sys.argv[1:]:
    path = Path(argument)
    n, red = load(path)
    total_pairs = n * (n - 1) // 2
    valid, detail = verify(n, red)
    print(f"{path.name}")
    print(f"  vertices        {n}")
    print(f"  red edges       {len(red)} of {total_pairs} pairs")
    print(f"  blue edges      {total_pairs - len(red)}")
    print(f"  expected checks {len(list(combinations(range(n), S)))}")
    print(f"  verdict         {'VALID' if valid else 'INVALID'}: {detail}")
    if valid:
        print(f"  implies         R(5,5) >= {n + 1}")
    print()
