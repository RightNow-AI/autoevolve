from pathlib import Path

import pytest

from autoevolve.core.types import EvalError, StageSpec
from autoevolve.eval.contract import load_evaluator

FIXTURES = Path(__file__).parent / "fixtures"
TOY_EVALUATOR = FIXTURES / "eval_toy"


def test_load_evaluator_reads_child_reported_contract() -> None:
    evaluator = load_evaluator(TOY_EVALUATOR)

    assert evaluator.dir == TOY_EVALUATOR.resolve()
    assert evaluator.stages == [
        StageSpec(name="smoke", timeout_s=20.0),
        StageSpec(name="full", timeout_s=30.0),
    ]
    assert evaluator.gate == "correct"
    assert evaluator.has_ceiling is False
    assert "source compactness" in evaluator.spec_text
    assert evaluator.ceiling() is None


def test_load_evaluator_rejects_missing_evaluate_py(tmp_path: Path) -> None:
    evaluator_dir = tmp_path / "missing-entrypoint"
    (evaluator_dir / "baseline").mkdir(parents=True)

    with pytest.raises(EvalError, match="missing evaluate.py"):
        load_evaluator(evaluator_dir)


def test_load_evaluator_surfaces_import_failure(tmp_path: Path) -> None:
    evaluator_dir = tmp_path / "broken-import"
    (evaluator_dir / "baseline").mkdir(parents=True)
    (evaluator_dir / "evaluate.py").write_text(
        'raise RuntimeError("import exploded")\n',
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="RuntimeError: import exploded"):
        load_evaluator(evaluator_dir)


def test_loader_carries_metric_declaration(toy_evaluator_dir=None):
    from pathlib import Path

    from autoevolve.eval.contract import load_evaluator

    toy = Path(__file__).parent / "fixtures" / "eval_toy"
    evaluator = load_evaluator(toy)
    assert evaluator.metric == "score"
    assert evaluator.maximize is True
