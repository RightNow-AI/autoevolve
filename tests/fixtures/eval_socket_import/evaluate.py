"""Evaluator fixture that attempts networking while its module loads."""

from __future__ import annotations

import socket
from pathlib import Path

from autoevolve.eval.contract import StageSpec

STAGES = [StageSpec(name="smoke", timeout_s=1.0)]
GATE = "correct"

socket.socket()


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    del candidate_dir, stage
    return {GATE: 1.0}
