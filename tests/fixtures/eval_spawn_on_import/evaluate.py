"""Evaluator fixture whose import blocks after spawning a stdio-inheriting child.

The grandchild records its process id where the test can find it, so the test
can assert the process actually died rather than inferring it from how quickly
the call returned. That inference failed on a loaded machine even when the
kill had worked.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from autoevolve.eval.contract import StageSpec

_GRANDCHILD = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
)
_PID_FILE = os.environ.get("AUTOEVOLVE_GRANDCHILD_PID_FILE")
if _PID_FILE:
    Path(_PID_FILE).write_text(str(_GRANDCHILD.pid), encoding="utf-8")
time.sleep(60)

STAGES = [StageSpec(name="smoke", timeout_s=1.0)]
GATE = "correct"


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    del candidate_dir, stage
    return {GATE: 1.0}
