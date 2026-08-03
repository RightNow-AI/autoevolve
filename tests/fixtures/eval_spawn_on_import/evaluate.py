"""Evaluator fixture whose import blocks after spawning a stdio-inheriting child."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from autoevolve.eval.contract import StageSpec

_GRANDCHILD = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
)
time.sleep(60)

STAGES = [StageSpec(name="smoke", timeout_s=1.0)]
GATE = "correct"


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    del candidate_dir, stage
    return {GATE: 1.0}
