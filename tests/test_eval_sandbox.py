import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autoevolve.core.types import EvalError, StageSpec
from autoevolve.eval.sandbox import run_stage

FIXTURES = Path(__file__).parent / "fixtures"
TOY_EVALUATOR = FIXTURES / "eval_toy"
CANDIDATES = FIXTURES / "eval_toy_candidates"
SMOKE = StageSpec(name="smoke", timeout_s=20.0)
SPAWNER_PID_FILE = "autoevolve-spawner-grandchild.pid"


def _pid_exists(pid: int) -> bool:
    if sys.platform == "win32":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return f'"{pid}"' in completed.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_pid_exit(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)
    return not _pid_exists(pid)


def _force_kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_good_candidate_returns_metrics() -> None:
    metrics = run_stage(TOY_EVALUATOR, CANDIDATES / "good", 0, SMOKE)

    assert metrics["correct"] == 1.0
    assert metrics["score"] > 0.0


def test_broken_candidate_surfaces_failing_case() -> None:
    with pytest.raises(EvalError, match="case 0 failed"):
        run_stage(TOY_EVALUATOR, CANDIDATES / "broken", 0, SMOKE)


def test_slow_candidate_is_killed_at_wall_clock_timeout() -> None:
    started = time.monotonic()

    with pytest.raises(EvalError, match=r"timeout after 2\.0s at stage smoke"):
        run_stage(
            TOY_EVALUATOR,
            CANDIDATES / "slow",
            0,
            StageSpec(name="smoke", timeout_s=2.0),
        )

    assert time.monotonic() - started < 8.0


def test_timeout_kills_spawned_grandchild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    pid_path = tmp_path / SPAWNER_PID_FILE
    started = time.monotonic()

    with pytest.raises(EvalError, match=r"timeout after 2\.0s at stage smoke"):
        run_stage(
            TOY_EVALUATOR,
            CANDIDATES / "spawner",
            0,
            StageSpec(name="smoke", timeout_s=2.0),
        )

    assert time.monotonic() - started < 8.0
    assert pid_path.is_file()
    pid = int(pid_path.read_text(encoding="utf-8"))
    exited = _wait_for_pid_exit(pid, timeout_s=3.0)
    if not exited:
        _force_kill_pid(pid)
    assert exited


def test_noisy_candidate_does_not_corrupt_protocol() -> None:
    metrics = run_stage(TOY_EVALUATOR, CANDIDATES / "noisy", 0, SMOKE)

    assert metrics["correct"] == 1.0


def test_network_call_is_blocked() -> None:
    with pytest.raises(EvalError, match="network disabled in sandbox"):
        run_stage(TOY_EVALUATOR, CANDIDATES / "netcall", 0, SMOKE)


def test_environment_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOEVOLVE_CANARY", "must-not-leak")

    metrics = run_stage(TOY_EVALUATOR, CANDIDATES / "envspy", 0, SMOKE)

    assert metrics["correct"] == 1.0
