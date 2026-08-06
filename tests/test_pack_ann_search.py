from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "ann-search"
EVALUATOR = PACK / "evaluators" / "ann"


def _load_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", "tiny-r100-validation")
    path = EVALUATOR / "evaluate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidate(candidate_dir: Path, source: str) -> Path:
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "index.py").write_text(source, encoding="utf-8")
    return candidate_dir


def test_seed_passes_recall_gate_and_reports_primary_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_seed")

    scores = evaluator.evaluate(EVALUATOR / "baseline", stage=0)

    assert evaluator.STAGES == [
        StageSpec(name="build-recall-and-throughput", timeout_s=180.0)
    ]
    assert evaluator.GATE == "recall_gate"
    assert evaluator.METRIC == "queries_per_second"
    assert evaluator.MAXIMIZE is True
    assert scores[evaluator.GATE] == 1.0
    assert scores["recall_at_k"] == 1.0
    assert scores[evaluator.METRIC] > 0.0
    assert scores["exact_queries_per_second"] > 0.0
    assert scores["index_build_seconds"] >= 0.0
    assert scores["index_memory_log2"] > 0.0
    assert scores["call_diversity"] > 0.0


def test_random_indices_fail_with_named_recall_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_random")
    candidate = _write_candidate(
        tmp_path / "random",
        '''import numpy as np


def build(vectors, deadline=None):
    del deadline
    return np.asarray(vectors, dtype=np.float64)


def search(index, queries, k, deadline=None):
    del deadline
    rng = np.random.default_rng(7)
    database = np.asarray(index, dtype=np.float64)
    indexes = np.arange(len(database))
    results = []
    for query in np.asarray(queries, dtype=np.float64):
        delta = database - query
        distances = np.einsum("ij,ij->i", delta, delta)
        exact = [int(value) for value in np.lexsort((indexes, distances))[:k]]
        choices = [value for value in range(len(database)) if value not in exact]
        replacement = rng.choice(choices)
        results.append(exact[:-1] + [replacement])
    return results
''',
    )

    with pytest.raises(EvalError, match="recall gate failed"):
        evaluator.evaluate(candidate, stage=0)


def test_out_of_range_index_fails_with_named_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_out_of_range")
    candidate = _write_candidate(
        tmp_path / "out-of-range",
        '''def build(vectors, deadline=None):
    del deadline
    return len(vectors)


def search(index, queries, k, deadline=None):
    del deadline
    row = [index] + list(range(k - 1))
    return [row for _ in range(len(queries))]
''',
    )

    with pytest.raises(EvalError, match=r"index 256 out of range 0\.\.255"):
        evaluator.evaluate(candidate, stage=0)


def test_duplicate_index_within_query_fails_with_named_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_duplicate")
    candidate = _write_candidate(
        tmp_path / "duplicate",
        '''def build(vectors, deadline=None):
    del deadline
    return len(vectors)


def search(index, queries, k, deadline=None):
    del index, deadline
    return [[0] * k for _ in range(len(queries))]
''',
    )

    with pytest.raises(EvalError, match="query 0 returned duplicate index 0"):
        evaluator.evaluate(candidate, stage=0)


def test_boolean_index_is_rejected_before_recall_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_bool")
    candidate = _write_candidate(
        tmp_path / "bool-index",
        '''def build(vectors, deadline=None):
    del deadline
    return len(vectors)


def search(index, queries, k, deadline=None):
    del index, deadline
    row = [True] + list(range(1, k))
    return [row for _ in range(len(queries))]
''',
    )

    with pytest.raises(EvalError, match="query 0 index 0 must be an integer, got bool"):
        evaluator.evaluate(candidate, stage=0)


def test_hostile_container_cannot_change_answers_between_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_hostile")
    candidate = _write_candidate(
        tmp_path / "hostile",
        '''import numpy as np


class HostileResults:
    def __init__(self, wrong, exact):
        self.wrong = wrong
        self.exact = exact
        self.reads = 0

    def __iter__(self):
        self.reads += 1
        return iter(self.wrong if self.reads == 1 else self.exact)


def build(vectors, deadline=None):
    del deadline
    return np.asarray(vectors, dtype=np.float64)


def search(index, queries, k, deadline=None):
    del deadline
    indexes = np.arange(len(index))
    exact = []
    wrong = []
    for query in np.asarray(queries, dtype=np.float64):
        delta = index - query
        distances = np.einsum("ij,ij->i", delta, delta)
        row = [int(value) for value in np.lexsort((indexes, distances))[:k]]
        replacement = next(value for value in range(len(index)) if value not in row)
        exact.append(row)
        wrong.append(row[:-1] + [replacement])
    return HostileResults(wrong, exact)
''',
    )

    with pytest.raises(EvalError, match="recall gate failed"):
        evaluator.evaluate(candidate, stage=0)


def test_candidate_module_cannot_self_report_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_self_report")
    candidate = _write_candidate(
        tmp_path / "self-report",
        '''queries_per_second = 1e300


def build(vectors, deadline=None):
    del deadline
    return vectors


def search(index, queries, k, deadline=None):
    del index, deadline
    return [list(range(k)) for _ in range(len(queries))]
''',
    )

    with pytest.raises(
        EvalError,
        match="self-reported metric names: queries_per_second",
    ):
        evaluator.evaluate(candidate, stage=0)


def test_campaign_and_dynamic_bounds_parse_with_repository_loaders() -> None:
    campaign = load_campaign(PACK)
    bounds = load_bounds(PACK)

    assert campaign.name == "ann-search"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "tiny-r100-validation",
        "medium-r095-frontier",
        "large-r090-frontier",
    ]
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert len(bounds) == 3
    assert all("not a published bound" in bound.who_and_year for bound in bounds)
    assert all("exact_queries_per_second" in bound.value for bound in bounds)


def test_contract_declares_two_returned_structural_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_ann_descriptors")
    scores = evaluator.evaluate(EVALUATOR / "baseline", stage=0)

    assert {descriptor["metric"] for descriptor in evaluator.DESCRIPTORS} == {
        "index_memory_log2",
        "call_diversity",
    }
    assert all(descriptor["metric"] in scores for descriptor in evaluator.DESCRIPTORS)
    assert evaluator.ceiling() is None
