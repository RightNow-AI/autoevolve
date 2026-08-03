from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec, load_evaluator

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "superpermutation"
EVALUATOR = PACK / "evaluators" / "superperm"


def _load_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
    name: str,
) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    path = EVALUATOR / "evaluate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_builder(candidate_dir: Path, source: str) -> Path:
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "builder.py").write_text(source, encoding="utf-8")
    return candidate_dir


def test_seed_passes_exact_gate_and_reports_primary_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n5", "test_superperm_seed")
    scores = evaluator.evaluate(EVALUATOR / "baseline", stage=0)

    assert evaluator.STAGES == [
        StageSpec(name="build-and-verify", timeout_s=90.0),
        StageSpec(name="determinism-replay", timeout_s=240.0),
    ]
    assert evaluator.GATE == "complete"
    assert evaluator.METRIC == "length"
    assert evaluator.MAXIMIZE is False
    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] == 153.0
    assert scores["target_perms"] == 120.0
    assert scores["revisits"] == 0.0
    assert len(evaluator.DESCRIPTORS) == 2


def test_committed_missing_permutations_mutant_names_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n5", "test_superperm_mutant")
    mutant = EVALUATOR / "fixtures" / "mutants" / "missing_permutations"

    with pytest.raises(EvalError, match=r"missing 119 of 120 permutations"):
        evaluator.evaluate(mutant, stage=0)


def test_hostile_string_subclass_cannot_answer_differently_on_each_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n5", "test_superperm_hostile")
    candidate = _write_builder(
        tmp_path / "hostile",
        '''class Hostile(str):
    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.reads = 0
        return instance

    def __getitem__(self, index):
        self.reads += 1
        value = super().__getitem__(index)
        return value if self.reads == 1 else "9"


def build(n: int, deadline: float | None = None) -> str:
    del n, deadline
    return Hostile("123451234152341253412354123145231425314235142315423124531243512435"
                   "214352143521432514325143215432154321")
''',
    )

    with pytest.raises(EvalError, match="must return exact str or bytes"):
        evaluator.evaluate(candidate, stage=0)


def test_build_budget_exhaustion_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n5", "test_superperm_timeout")
    candidate = _write_builder(
        tmp_path / "timeout",
        '''def build(n: int, deadline: float | None = None) -> str:
    del n, deadline
    while True:
        pass
''',
    )

    with pytest.raises(EvalError, match="candidate build timed out"):
        evaluator._build_once(candidate, 5, 0.05)


def test_campaign_and_bounds_parse_through_product_loaders() -> None:
    campaign = load_campaign(PACK)
    bounds = load_bounds(PACK)
    loaded_evaluator = load_evaluator(EVALUATOR)

    assert campaign.name == "superpermutation"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == ["n5", "n6", "n7"]
    assert [cell.target for cell in campaign.cells] == [153.0, 871.0, 5905.0]
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert len(bounds) == 5
    assert all(bound.checked_on == "2026-08-04" for bound in bounds)
    assert all(bound.source_url.startswith("https://") for bound in bounds)
    assert loaded_evaluator.metric == "length"
    assert loaded_evaluator.maximize is False
    assert loaded_evaluator.descriptors == [
        {
            "name": "max_perm_gap",
            "metric": "max_perm_gap",
            "bins": 8,
            "lo": 1.0,
            "hi": 32.0,
        },
        {
            "name": "perm_gap_kinds",
            "metric": "perm_gap_kinds",
            "bins": 8,
            "lo": 1.0,
            "hi": 16.0,
        },
    ]
