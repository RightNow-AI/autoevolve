from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "frankl-union-closed"
PACK = CAMPAIGN / "evaluators" / "ucf"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _load_evaluator(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    monkeypatch.delenv("AUTOEVOLVE_UCF_NMAX", raising=False)
    monkeypatch.delenv("AUTOEVOLVE_UCF_MMAX", raising=False)
    return _load_module(PACK / "evaluate.py", name)


def test_seed_passes_every_stage_and_stage_zero_reports_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_frankl_seed")

    for stage in range(len(evaluator.STAGES)):
        scores = evaluator.evaluate(PACK / "baseline", stage=stage)
        assert set(scores) == {
            "union_closed_valid",
            "max_freq_ratio",
            "max_freq",
            "family_size",
            "ground_set_declared",
            "ground_set_used",
            "below_half",
            "half_margin",
            "nmax_in_force",
        }
        assert scores[evaluator.GATE] == 1.0
        assert scores[evaluator.METRIC] == 0.5
        assert scores["max_freq"] == 4.0
        assert scores["family_size"] == 8.0
        assert scores["ground_set_declared"] == 3.0
        assert scores["ground_set_used"] == 3.0
        assert scores["below_half"] == 0.0
        assert scores["half_margin"] == 0.0
        assert scores["nmax_in_force"] == 24.0

    stage_zero = evaluator.evaluate(PACK / "baseline", stage=0)
    assert evaluator.METRIC in stage_zero
    assert evaluator.METRIC == "max_freq_ratio"
    assert evaluator.MAXIMIZE is False


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("non_union_closed", r"0x1 \| 0x2 = 0x3 is not a member"),
        ("empty_family", "the empty family is excluded"),
        ("empty_only", r"\{empty set\} is excluded"),
    ],
)
def test_committed_mutants_fail_with_named_causes(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    reason: str,
) -> None:
    evaluator = _load_evaluator(monkeypatch, f"test_frankl_mutant_{name}")
    mutant = PACK / "fixtures" / "mutants" / name

    with pytest.raises(EvalError, match=reason):
        evaluator.evaluate(mutant, stage=0)


def test_hostile_mapping_getitem_cannot_change_the_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_frankl_hostile")
    candidate = tmp_path / "hostile"
    candidate.mkdir()
    (candidate / "model.py").write_text(
        """from collections.abc import Mapping


class Hostile(Mapping):
    def __init__(self):
        self.set_reads = 0

    def __iter__(self):
        return iter(("n", "sets"))

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == "n":
            return 3
        if key == "sets":
            self.set_reads += 1
            if self.set_reads == 1:
                return [0, 1, 2]
            return list(range(8))
        raise KeyError(key)


def build_family():
    return Hostile()
""",
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match=r"0x1 \| 0x2 = 0x3 is not a member"):
        evaluator.evaluate(candidate, stage=0)


def test_closure_operation_budget_exhaustion_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_frankl_budget")
    monkeypatch.setattr(evaluator, "_CLOSURE_PAIR_BUDGET", 35)

    with pytest.raises(EvalError, match="operation budget exhausted"):
        evaluator.evaluate(PACK / "baseline", stage=0)


def test_candidate_cannot_widen_import_time_universe_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_UCF_NMAX", "13")
    monkeypatch.delenv("AUTOEVOLVE_UCF_MMAX", raising=False)
    evaluator = _load_module(PACK / "evaluate.py", "test_frankl_import_time_cap")
    candidate = tmp_path / "cap_attack"
    candidate.mkdir()
    (candidate / "model.py").write_text(
        """import os


def build_family():
    os.environ["AUTOEVOLVE_UCF_NMAX"] = "24"
    return {"n": 14, "sets": [1]}
""",
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match=r"n must satisfy 1 <= n <= 13, got 14"):
        evaluator.evaluate(candidate, stage=0)


def test_campaign_and_bounds_parse_with_repository_loaders() -> None:
    campaign = load_campaign(CAMPAIGN)
    bounds = load_bounds(CAMPAIGN)

    assert campaign.name == "frankl-union-closed"
    assert campaign.evaluator_path == PACK.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "u12-validation",
        "u13-frontier",
        "u24-frontier",
    ]
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert len(bounds) == 3
    assert all(bound.checked_on == "2026-08-04" for bound in bounds)


def test_pack_contract_markers_descriptors_and_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "test_frankl_contract")

    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert len(evaluator.DESCRIPTORS) == 2
    assert evaluator.ceiling() is None

    baseline_source = (PACK / "baseline" / "model.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    spec_text = (PACK / "spec.md").read_text(encoding="utf-8")
    assert evaluator.GATE in spec_text
    assert evaluator.METRIC in spec_text
    assert "Fraction(f_max, m)" in spec_text
