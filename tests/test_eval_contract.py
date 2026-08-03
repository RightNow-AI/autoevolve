import time
from pathlib import Path

import pytest

import autoevolve.eval.contract as contract_module
from autoevolve.core.types import EvalError, StageSpec
from autoevolve.eval.contract import load_evaluator

FIXTURES = Path(__file__).parent / "fixtures"
TOY_EVALUATOR = FIXTURES / "eval_toy"
SOCKET_IMPORT_EVALUATOR = FIXTURES / "eval_socket_import"
SOCKET_CEILING_EVALUATOR = FIXTURES / "eval_socket_ceiling"
SPAWN_IMPORT_EVALUATOR = FIXTURES / "eval_spawn_on_import"


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


def test_load_evaluator_blocks_network_during_module_import() -> None:
    with pytest.raises(EvalError, match="network disabled in sandbox"):
        load_evaluator(SOCKET_IMPORT_EVALUATOR)


def test_evaluator_ceiling_blocks_network_call() -> None:
    evaluator = load_evaluator(SOCKET_CEILING_EVALUATOR)

    with pytest.raises(EvalError, match="network disabled in sandbox"):
        evaluator.ceiling()


def test_describe_timeout_kills_spawned_grandchild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract_module, "_RUNNER_TIMEOUT_S", 0.5)
    started = time.monotonic()

    with pytest.raises(EvalError, match="evaluator describe timed out after 30s"):
        load_evaluator(SPAWN_IMPORT_EVALUATOR)

    assert time.monotonic() - started < 5.0


def test_loader_carries_metric_declaration() -> None:
    evaluator = load_evaluator(TOY_EVALUATOR)
    assert evaluator.metric == "score"
    assert evaluator.maximize is True


def test_loader_carries_behavior_descriptors():
    """Without descriptors the archive is one cell and search is hill climbing."""

    from pathlib import Path

    from autoevolve.eval.contract import load_evaluator

    pack = Path(__file__).parents[1] / "campaigns" / "golomb-ruler" / "evaluators" / "golomb"
    evaluator = load_evaluator(pack)

    assert evaluator.descriptors
    names = {item["metric"] for item in evaluator.descriptors}
    assert names == {"max_gap", "gap_spread"}
    for item in evaluator.descriptors:
        assert item["bins"] >= 1
        assert item["hi"] > item["lo"]
