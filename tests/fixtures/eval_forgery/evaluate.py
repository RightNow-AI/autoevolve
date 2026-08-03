"""Evaluator whose gate rejects any answer other than 42.

Used to prove that candidate code cannot forge a passing verdict on the
runner's output channel. See tests/test_eval_sandbox.py.
"""

import importlib.util
import sys
from pathlib import Path

from autoevolve.eval.contract import EvalError, StageSpec

STAGES = [StageSpec(name="check", timeout_s=60.0)]
GATE = "correct"
METRIC = "score"
MAXIMIZE = True


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    spec = importlib.util.spec_from_file_location("forgery_cand", candidate_dir / "solution.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["forgery_cand"] = module
    spec.loader.exec_module(module)
    if module.answer() != 42:
        raise EvalError("wrong answer, gate failed")
    # A lazy import AFTER candidate code has run. If the candidate directory
    # is importable, this resolves to the candidate's shadow module.
    import statistics

    return {"correct": 1.0, "score": float(statistics.median([1.0, 1.0, 1.0]))}
