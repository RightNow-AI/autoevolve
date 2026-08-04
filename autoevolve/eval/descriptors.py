"""Program-structure behavior descriptors any evaluator can adopt.

MAP-elites needs each candidate to land somewhere in a behavior space. Without
that every candidate lands in one cell, the archive holds a single incumbent,
and the search degenerates into hill climbing. That is the largest defect this
project has found, and it was true of nine packs at once.

Some domains have obvious structural descriptors: a Golomb ruler has its gap
profile, a kernel has its launch shape. Many do not, and a pack author who has
to invent a behavior space before shipping tends to ship without one. These two
are the default for those packs. They describe the shape of the program rather
than how well it scored, so a fast solution and a slow one with the same
structure compete, while two different approaches at the same score both
survive.

Both are read from the source of the mutable region only. Frozen code is
identical across every candidate and would only add a constant.
"""

from __future__ import annotations

import ast
from pathlib import Path

START_MARKER = "EVOLVE-BLOCK-START"
END_MARKER = "EVOLVE-BLOCK-END"

#: Declare these in a pack alongside the metrics returned by source_metrics.
SOURCE_DESCRIPTORS = [
    {"name": "mutable_lines", "metric": "mutable_lines", "bins": 8, "lo": 1.0, "hi": 240.0},
    {"name": "call_diversity", "metric": "call_diversity", "bins": 8, "lo": 0.0, "hi": 48.0},
]


def mutable_source(text: str) -> str:
    """Return only the evolvable regions of a candidate file.

    A file with no markers is entirely mutable, which is how the simplest packs
    are written.
    """

    lines = text.splitlines()
    if not any(START_MARKER in line for line in lines):
        return text
    kept: list[str] = []
    inside = False
    for line in lines:
        if START_MARKER in line:
            inside = True
            continue
        if END_MARKER in line:
            inside = False
            continue
        if inside:
            kept.append(line)
    return "\n".join(kept)


def source_metrics(candidate_dir: Path, *entries: str) -> dict[str, float]:
    """Measure the structure of a candidate's mutable region.

    mutable_lines is how much code the candidate carries, which separates a
    terse closed form from an elaborate search. call_diversity is how many
    distinct things it calls, which separates a hand rolled loop from one
    leaning on a library. Neither says anything about quality.

    A file that cannot be parsed still gets a line count, because a candidate
    whose syntax is broken will fail its gate on its own and does not need to
    also crash the descriptor.
    """

    lines = 0
    names: set[str] = set()
    for entry in entries:
        path = candidate_dir / entry
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        region = mutable_source(text)
        lines += sum(1 for line in region.splitlines() if line.strip())
        try:
            tree = ast.parse(region)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return {
        "mutable_lines": float(lines),
        "call_diversity": float(len(names)),
    }
