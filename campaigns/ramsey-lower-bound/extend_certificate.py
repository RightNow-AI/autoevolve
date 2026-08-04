"""Decide whether a verified 42 vertex certificate extends to 43 vertices.

The reduction. Add a new vertex v and join it red to a set R and blue to the
complement B. Any monochromatic K5 either avoids v, and then it already lives
in the 42 vertex graph which has none, or it contains v. A red K5 containing v
is v plus a red K4 all of whose vertices lie in R. A blue K5 containing v is v
plus a blue K4 inside B.

So the graph extends if and only if the 42 vertices split into R and B with no
red K4 inside R and no blue K4 inside B. That is a boolean problem in 42
variables: x[i] true means vertex i is in R. Each red K4 {a,b,c,d} forbids all
four being in R, giving the clause (~a or ~b or ~c or ~d). Each blue K4 forbids
all four being in B, giving (a or b or c or d).

Solved exactly by DPLL with unit propagation, so an UNSAT answer is a proof
that this particular certificate does not extend, not a failed search.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.setrecursionlimit(10000)


def load(path: Path) -> tuple[int, list[list[bool]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["form"] != "adjacency":
        raise SystemExit(f"unsupported form {data['form']!r}")
    n = int(data["n"])
    red = [[False] * n for _ in range(n)]
    for a, b in data["red_edges"]:
        red[int(a)][int(b)] = True
        red[int(b)][int(a)] = True
    return n, red


def monochromatic_k4s(n: int, red: list[list[bool]]) -> tuple[list, list]:
    reds, blues = [], []
    for quad in combinations(range(n), 4):
        pairs = list(combinations(quad, 2))
        if all(red[a][b] for a, b in pairs):
            reds.append(quad)
        elif not any(red[a][b] for a, b in pairs):
            blues.append(quad)
    return reds, blues


def solve(n: int, clauses: list[tuple[frozenset[int], frozenset[int]]]) -> list[bool] | None:
    """DPLL over 42 variables. clauses are (positives, negatives) literal sets."""

    assign: dict[int, bool] = {}

    def clause_state(clause):
        pos, neg = clause
        unassigned = []
        for v in pos:
            if v not in assign:
                unassigned.append((v, True))
            elif assign[v]:
                return "sat", None
        for v in neg:
            if v not in assign:
                unassigned.append((v, False))
            elif not assign[v]:
                return "sat", None
        if not unassigned:
            return "conflict", None
        if len(unassigned) == 1:
            return "unit", unassigned[0]
        return "open", None

    def dpll() -> bool:
        while True:
            unit = None
            for clause in clauses:
                state, lit = clause_state(clause)
                if state == "conflict":
                    return False
                if state == "unit" and unit is None:
                    unit = lit
            if unit is None:
                break
            assign[unit[0]] = unit[1]
        undecided = [v for v in range(n) if v not in assign]
        if not undecided:
            return True
        # Branch on the variable appearing in the most open clauses.
        counts = dict.fromkeys(undecided, 0)
        for clause in clauses:
            state, _ = clause_state(clause)
            if state != "open":
                continue
            for v in clause[0] | clause[1]:
                if v in counts:
                    counts[v] += 1
        pick = max(undecided, key=lambda v: counts[v])
        for value in (True, False):
            snapshot = dict(assign)
            assign[pick] = value
            if dpll():
                return True
            assign.clear()
            assign.update(snapshot)
        return False

    if dpll():
        return [assign.get(v, True) for v in range(n)]
    return None


for argument in sys.argv[1:]:
    path = Path(argument)
    n, red = load(path)
    reds, blues = monochromatic_k4s(n, red)
    print(f"{path.name}: n={n}, red K4s={len(reds)}, blue K4s={len(blues)}")
    clauses = [(frozenset(), frozenset(q)) for q in reds]
    clauses += [(frozenset(q), frozenset()) for q in blues]
    result = solve(n, clauses)
    if result is None:
        print("  UNSAT: this certificate provably does not extend to 43 vertices\n")
        continue
    in_r = [i for i in range(n) if result[i]]
    print(f"  SAT: |R|={len(in_r)}, |B|={n - len(in_r)}")
    edges = [[a, b] for a in range(n) for b in range(a + 1, n) if red[a][b]]
    edges += [[i, n] for i in in_r]
    out = Path(f"extended-{n + 1}-{path.stem}.json")
    out.write_text(
        json.dumps({"form": "adjacency", "n": n + 1, "red_edges": edges}),
        encoding="utf-8",
    )
    print(f"  wrote {out}\n")
