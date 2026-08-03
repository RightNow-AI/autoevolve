"""Candidate fixture that leaves a stdio-inheriting grandchild sleeping."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

PID_FILE = "autoevolve-spawner-grandchild.pid"


def solve(values: list[int]) -> list[int]:
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    pid_path = Path(tempfile.gettempdir()) / PID_FILE
    pid_path.write_text(str(grandchild.pid), encoding="utf-8")
    time.sleep(30)
    return sorted(values)
