"""Evaluator fixture that attempts networking from ceiling()."""

from __future__ import annotations

import socket
from pathlib import Path

from autoevolve.eval.contract import StageSpec

STAGES = [StageSpec(name="smoke", timeout_s=1.0)]
GATE = "correct"


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    del candidate_dir, stage
    return {GATE: 1.0}


def ceiling() -> dict[str, float]:
    socket.create_connection(("example.com", 80))
    return {"metric": "score", "value": 1.0}
