"""Check the extension solver against cases whose answers are known.

A buggy solver that returns UNSAT too eagerly would produce a confident and
wrong mathematical claim, so the solver is tested before its verdict is
believed. Small instances are cross-checked against exhaustive enumeration.
"""

from __future__ import annotations

import random
import sys
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extend_certificate import solve  # noqa: E402


def brute(n: int, clauses) -> bool:
    """Exhaustive truth table, for small n only."""

    for bits in product([False, True], repeat=n):
        ok = True
        for pos, neg in clauses:
            if any(bits[v] for v in pos) or any(not bits[v] for v in neg):
                continue
            ok = False
            break
        if ok:
            return True
    return False


def check(name: str, n: int, clauses, expected: bool) -> bool:
    got = solve(n, list(clauses)) is not None
    verdict = "ok" if got == expected else "WRONG"
    print(f"  {verdict}: {name} expected {'SAT' if expected else 'UNSAT'}, got "
          f"{'SAT' if got else 'UNSAT'}")
    return got == expected


print("fixed cases:")
passed = True
# No constraints at all is trivially satisfiable.
passed &= check("empty", 4, [], True)
# One red K4 forbids all four in R; putting any one in B satisfies it.
passed &= check("single red K4", 4, [(frozenset(), frozenset({0, 1, 2, 3}))], True)
# The same four vertices forbidden from being all-R and all-B at once is still
# satisfiable, because a mixed split violates neither.
passed &= check(
    "red and blue on the same quad",
    4,
    [(frozenset(), frozenset({0, 1, 2, 3})), (frozenset({0, 1, 2, 3}), frozenset())],
    True,
)
# One variable forced both ways is unsatisfiable.
passed &= check(
    "contradiction",
    1,
    [(frozenset({0}), frozenset()), (frozenset(), frozenset({0}))],
    False,
)
# Every 2-subset of 3 variables forbidden in both directions: with 3 variables
# some pair shares a value by pigeonhole, so this must be UNSAT.
pairs = list(combinations(range(3), 2))
clauses = [(frozenset(), frozenset(p)) for p in pairs]
clauses += [(frozenset(p), frozenset()) for p in pairs]
passed &= check("pigeonhole on 3", 3, clauses, False)

print("randomised cross-check against exhaustive enumeration:")
rng = random.Random(20260804)
mismatch = 0
for _trial in range(300):
    n = rng.randint(3, 9)
    clauses = []
    for _ in range(rng.randint(1, 14)):
        size = rng.randint(1, min(4, n))
        quad = frozenset(rng.sample(range(n), size))
        if rng.random() < 0.5:
            clauses.append((frozenset(), quad))
        else:
            clauses.append((quad, frozenset()))
    got = solve(n, list(clauses)) is not None
    want = brute(n, clauses)
    if got != want:
        mismatch += 1
        print(f"  MISMATCH n={n} clauses={clauses} solver={got} brute={want}")
print(f"  {300 - mismatch}/300 agreed with exhaustive enumeration")

print()
print("SOLVER TRUSTWORTHY" if passed and mismatch == 0 else "SOLVER IS WRONG, DO NOT USE")
