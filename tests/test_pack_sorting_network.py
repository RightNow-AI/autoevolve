from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "sorting-network"
EVALUATOR = PACK / "evaluators" / "sortnet"


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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_passes_exact_stage_zero_gate_and_reports_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n11-validation",
        "test_sortnet_seed",
    )

    scores = evaluator.evaluate(EVALUATOR / "baseline", stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert evaluator.METRIC == "size"
    assert evaluator.METRIC in scores
    assert scores[evaluator.METRIC] > 35.0
    # The n11 cell verifies over 2**11 inputs. The zero-one principle makes
    # that exhaustive, so the count tracks the cell's channel width and is not
    # a constant borrowed from another cell.
    assert scores["binary_inputs"] == 2048.0
    assert scores["channels"] == 11.0
    assert scores["depth"] > 0.0
    assert scores["first_layer_channels"] > 0.0
    assert evaluator.MAXIMIZE is False


def test_committed_out_of_range_mutant_fails_with_named_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n11-validation",
        "test_sortnet_mutant",
    )
    mutant = EVALUATOR / "fixtures" / "mutants" / "out_of_range"

    with pytest.raises(EvalError, match=r"channel 11 is outside 0\.\.10"):
        evaluator.evaluate(mutant, stage=0)


def test_hostile_container_with_varying_reads_cannot_fool_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n11-validation",
        "test_sortnet_hostile",
    )
    candidate = tmp_path / "hostile"
    candidate.mkdir()
    (candidate / "network.py").write_text(
        '''from collections.abc import Iterator


class HostileNetwork:
    def __init__(self, valid: list[tuple[int, int]]) -> None:
        self.valid = valid
        self.reads = 0

    def __iter__(self) -> Iterator[tuple[int, int]]:
        self.reads += 1
        if self.reads == 1:
            return iter(())
        return iter(self.valid)


def build(channels: int, deadline: float | None = None) -> HostileNetwork:
    del deadline
    padded = 1
    while padded < channels:
        padded *= 2
    valid = []

    def add(left: int, right: int) -> None:
        if right < channels:
            valid.append((left, right))

    def merge(start: int, count: int, stride: int) -> None:
        next_stride = stride * 2
        if next_stride < count:
            merge(start, count, next_stride)
            merge(start + stride, count, next_stride)
            for left in range(start + stride, start + count - stride, next_stride):
                add(left, left + stride)
        else:
            add(start, start + stride)

    def sort_range(start: int, count: int) -> None:
        if count <= 1:
            return
        half = count // 2
        sort_range(start, half)
        sort_range(start + half, half)
        merge(start, count, 1)

    sort_range(0, padded)
    return HostileNetwork(valid)
''',
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="network fails on binary input"):
        evaluator.evaluate(candidate, stage=0)


def test_campaign_and_all_seven_field_bounds_parse_with_repo_loaders() -> None:
    campaign = load_campaign(PACK)
    bounds = load_bounds(PACK)

    assert campaign.name == "sorting-network"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "n11-validation",
        "n13-frontier",
        "n16-frontier",
        "n20-frontier",
    ]
    assert [cell.target for cell in campaign.cells] == [35.0, 44.0, 59.0, 90.0]
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert len(bounds) == 20
    for bound in bounds:
        assert bound.claim
        assert bound.value
        assert bound.direction
        assert bound.who_and_year
        assert bound.source_url.startswith("https://")
        assert bound.checked_on == "2026-08-05"
        assert bound.how_to_recheck


def test_contract_returns_both_structural_descriptor_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "n13-frontier",
        "test_sortnet_descriptors",
    )

    assert {descriptor["metric"] for descriptor in evaluator.DESCRIPTORS} == {
        "depth",
        "first_layer_channels",
    }
    assert evaluator.ceiling() is None
