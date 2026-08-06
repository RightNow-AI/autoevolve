from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "kidney-exchange"
PACK = CAMPAIGN / "evaluators" / "kidney"


def _load_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
    name: str,
) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    path = PACK / "evaluate.py"
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


def _write_candidate(root: Path, source: str) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "solver.py").write_text(source, encoding="utf-8")
    return candidate


def test_seed_passes_gate_and_reports_primary_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_seed",
    )

    scores = evaluator.evaluate(PACK / "baseline", stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert evaluator.METRIC == "transplants"
    assert evaluator.METRIC in scores
    assert scores[evaluator.METRIC] == scores["baseline_transplants"]
    assert scores["cycle_count"] == scores["baseline_cycle_count"]
    assert scores["chain_count"] == 0.0
    assert scores["chain_share"] == 0.0
    assert evaluator.MAXIMIZE is True


def test_reusing_one_vertex_in_two_cycles_fails_with_named_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_reused_vertex",
    )
    candidate = _write_candidate(
        tmp_path,
        """def solve(instance, deadline=None):
    del instance, deadline
    return {"cycles": [[0, 1], [0, 2]], "chains": []}
""",
    )

    with pytest.raises(EvalError, match="vertex 0 is used more than once"):
        evaluator.evaluate(candidate, stage=0)


def test_cycle_with_missing_directed_edge_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_missing_edge",
    )
    missing_pair = next(
        (left, right)
        for left in range(evaluator.INSTANCE.cell.pair_count)
        for right in range(left + 1, evaluator.INSTANCE.cell.pair_count)
        if right not in evaluator.INSTANCE.adjacency[left]
        or left not in evaluator.INSTANCE.adjacency[right]
    )
    candidate = _write_candidate(
        tmp_path,
        f"""def solve(instance, deadline=None):
    del instance, deadline
    return {{"cycles": [[{missing_pair[0]}, {missing_pair[1]}]], "chains": []}}
""",
    )

    with pytest.raises(EvalError, match=r"cycle 0 edge \d+->\d+ does not exist"):
        evaluator.evaluate(candidate, stage=0)


def test_cycle_exceeding_cell_cap_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_cycle_cap",
    )
    too_long = list(range(evaluator.INSTANCE.cell.cycle_cap + 1))
    candidate = _write_candidate(
        tmp_path,
        f"""def solve(instance, deadline=None):
    del instance, deadline
    return {{"cycles": [{too_long!r}], "chains": []}}
""",
    )

    with pytest.raises(EvalError, match=r"exceeds cycle cap 3"):
        evaluator.evaluate(candidate, stage=0)


def test_one_edge_altruist_chain_is_counted_as_one_transplant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_valid_chain",
    )
    altruist = evaluator.INSTANCE.altruists[0]
    patient = evaluator.INSTANCE.adjacency[altruist][0]
    candidate = _write_candidate(
        tmp_path,
        f"""def solve(instance, deadline=None):
    del instance, deadline
    return {{"cycles": [], "chains": [[{altruist}, {patient}]]}}
""",
    )

    scores = evaluator.evaluate(candidate, stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] == 1.0
    assert scores["chain_count"] == 1.0
    assert scores["chain_transplants"] == 1.0
    assert scores["chain_share"] == 1.0


def test_exact_solver_and_candidate_gate_agree_on_validation_optimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_exact",
    )
    exact_solution = evaluator.exact_validation_solution()
    exact_optimum = evaluator.exact_validation_optimum()
    candidate = _write_candidate(
        tmp_path,
        f"""def solve(instance, deadline=None):
    del instance, deadline
    return {exact_solution!r}
""",
    )

    scores = evaluator.evaluate(candidate, stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] == float(exact_optimum)


def test_hostile_container_cannot_change_after_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_hostile_container",
    )
    candidate = _write_candidate(
        tmp_path,
        """class HostileCycles(list):
    def __init__(self):
        super().__init__()
        self.reads = 0

    def __iter__(self):
        self.reads += 1
        if self.reads == 1:
            return iter(((0, 1), (0, 2)))
        return iter(())


def solve(instance, deadline=None):
    del instance, deadline
    return {"cycles": HostileCycles(), "chains": []}
""",
    )

    with pytest.raises(EvalError, match="vertex 0 is used more than once"):
        evaluator.evaluate(candidate, stage=0)


def test_candidate_global_matching_report_name_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_report_name",
    )
    candidate = _write_candidate(
        tmp_path,
        """transplants = 999


def solve(instance, deadline=None):
    del instance, deadline
    return {"cycles": [], "chains": []}
""",
    )

    with pytest.raises(EvalError, match="self-reported metric names: transplants"):
        evaluator.evaluate(candidate, stage=0)


def test_campaign_and_empty_bounds_registry_parse_with_repo_loaders() -> None:
    campaign = load_campaign(CAMPAIGN)
    bounds = load_bounds(CAMPAIGN)

    assert campaign.name == "kidney-exchange"
    assert campaign.evaluator_path == PACK.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "small-validation",
        "pairs-80-frontier",
        "pairs-160-frontier",
        "pairs-5000-frontier",
    ]
    assert all(cell.target is None for cell in campaign.cells)
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert bounds == ()


def test_pack_contract_descriptors_markers_and_honesty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_contract",
    )

    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert {descriptor["metric"] for descriptor in evaluator.DESCRIPTORS} == {
        "chain_share",
        "mean_cycle_length",
    }
    assert evaluator.ceiling()["value"] == 8.0

    baseline_source = (PACK / "baseline" / "solver.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    spec_text = (PACK / "spec.md").read_text(encoding="utf-8")
    assert "Saidman et al., 2006" in spec_text
    assert "70/20/10" in spec_text
    assert "correctness, not search capability" in spec_text
    assert "same process" in spec_text
