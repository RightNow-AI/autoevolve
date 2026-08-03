"""Sequential evaluator cascade with fail-closed gate handling."""

from __future__ import annotations

import math
from pathlib import Path

from autoevolve.core.types import EvalError, EvalOutcome
from autoevolve.eval.contract import Evaluator
from autoevolve.eval.sandbox import run_stage


def _coerce_metrics(metrics: object, stage: int) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise EvalError(f"evaluator metrics at stage {stage} are not a dict")
    coerced: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str):
            raise EvalError(f"metric name at stage {stage} is not a string")
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise EvalError(f"metric {name} is not numeric at stage {stage}") from exc
        if not math.isfinite(numeric):
            raise EvalError(f"metric {name} is not finite at stage {stage}")
        coerced[name] = numeric
    return coerced


def run_cascade(evaluator: Evaluator, candidate_dir: Path) -> EvalOutcome:
    """Run all stages in order, stopping at the first error or failed gate."""

    if not evaluator.stages:
        return EvalOutcome(False, {}, -1, "evaluator has no stages")

    deepest_scores: dict[str, float] = {}
    for index, spec in enumerate(evaluator.stages):
        try:
            raw_metrics = run_stage(evaluator.dir, candidate_dir, index, spec)
            metrics = _coerce_metrics(raw_metrics, index)
        except EvalError as exc:
            return EvalOutcome(False, {}, index, exc.reason)

        if evaluator.gate not in metrics:
            return EvalOutcome(False, {}, index, f"gate metric missing: {evaluator.gate}")
        if metrics[evaluator.gate] != 1.0:
            reason = f"gate metric {evaluator.gate} was {metrics[evaluator.gate]}, expected 1.0"
            return EvalOutcome(False, {}, index, reason)
        deepest_scores = metrics

    return EvalOutcome(True, deepest_scores, len(evaluator.stages) - 1)
