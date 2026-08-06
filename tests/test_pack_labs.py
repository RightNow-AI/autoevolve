from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "labs"
EVALUATOR = CAMPAIGN / "evaluators" / "labs"


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


def _load_baseline(name: str) -> ModuleType:
    path = EVALUATOR / "baseline" / "solver.py"
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


def _independent_energy(sequence: tuple[int, ...]) -> int:
    n = len(sequence)
    energy = 0
    for lag in range(1, n):
        correlation = 0
        for index in range(n - lag):
            correlation += sequence[index] * sequence[index + lag]
        energy += correlation * correlation
    return energy


def _brute_force_optimum(n: int) -> tuple[int, tuple[int, ...]]:
    best_energy = None
    best_sequence = None
    for mask in range(1 << n):
        sequence = tuple(1 if mask & (1 << index) else -1 for index in range(n))
        energy = _independent_energy(sequence)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_sequence = sequence
    assert best_energy is not None
    assert best_sequence is not None
    return best_energy, best_sequence


def test_exhaustive_validation_optimum_agrees_with_exact_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n13-validation", "test_labs_exact")
    true_energy, attaining_sequence = _brute_force_optimum(13)
    candidate = _write_candidate(
        tmp_path,
        f"""def solve(n):
    sequence = {attaining_sequence!r}
    assert len(sequence) == n
    return sequence
""",
    )

    scores = evaluator.evaluate(candidate, stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert scores["energy"] == float(true_energy)
    assert scores[evaluator.METRIC] == 13 * 13 / (2 * true_energy)


def test_non_skew_symmetric_sequence_is_still_accepted_and_scored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n13-validation", "test_labs_non_skew")
    sequence = (1,) * 13
    centre = len(sequence) // 2
    assert sequence[centre + 1] != -sequence[centre - 1]
    candidate = _write_candidate(
        tmp_path,
        f"""def solve(n):
    sequence = {sequence!r}
    assert len(sequence) == n
    return sequence
""",
    )

    scores = evaluator.evaluate(candidate, stage=0)

    assert scores[evaluator.GATE] == 1.0
    assert scores["energy"] == float(_independent_energy(sequence))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "def solve(n):\n    return [1] * (n - 1)\n",
            r"must contain exactly 13 entries; got 12",
        ),
        (
            "def solve(n):\n    return [1] * (n - 1) + [0]\n",
            r"sequence entry 12 must be -1 or \+1, got 0",
        ),
    ],
)
def test_invalid_candidate_output_fails_exact_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    message: str,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n13-validation", "test_labs_invalid")
    candidate = _write_candidate(tmp_path, source)

    with pytest.raises(EvalError, match=message):
        evaluator.evaluate(candidate, stage=0)


def test_deadline_seed_and_descriptors_are_evaluator_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n13-validation", "test_labs_contract")
    candidate = _write_candidate(
        tmp_path,
        f"""import time


def solve(n, deadline=None, seed=None):
    assert deadline is not None
    assert deadline > time.monotonic()
    assert seed == {evaluator.CELL_SEED!r}
    return [1] * n
""",
    )

    scores = evaluator.evaluate(candidate, stage=0)

    assert len(evaluator.DESCRIPTORS) == 2
    assert evaluator.DESCRIPTORS
    assert all(descriptor["name"] in scores for descriptor in evaluator.DESCRIPTORS)
    assert all(descriptor["metric"] in scores for descriptor in evaluator.DESCRIPTORS)
    assert evaluator.METRIC == "merit_factor"
    assert evaluator.MAXIMIZE is True


def test_candidate_builtin_rebinding_cannot_change_gate_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n13-validation", "test_labs_builtins")
    names = ("abs", "bool", "callable", "enumerate", "float", "getattr", "int", "iter")
    names += ("len", "max", "next", "range", "set", "str", "tuple", "type")
    originals = {name: getattr(builtins, name) for name in names}
    candidate = _write_candidate(
        tmp_path,
        f"""import builtins

SEQUENCE = {(1,) * 13!r}


def explode(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("candidate replaced a builtin")


for name in {names!r}:
    setattr(builtins, name, explode)


def solve(n):
    del n
    return SEQUENCE
""",
    )

    try:
        scores = evaluator.evaluate(candidate, stage=0)
    finally:
        for name, value in originals.items():
            setattr(builtins, name, value)

    assert scores[evaluator.GATE] == 1.0
    assert scores["length"] == 13.0


def test_baseline_is_fenced_search_and_passes_validation_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(monkeypatch, "n13-validation", "test_labs_baseline")
    source = (EVALUATOR / "baseline" / "solver.py").read_text(encoding="utf-8")

    assert "# EVOLVE-BLOCK-START" in source
    assert "# EVOLVE-BLOCK-END" in source
    assert "_steepest_descent" in source
    assert "_steepest_skew_descent" in source
    assert "_expand_skew_symmetric" in source
    scores = evaluator.evaluate(EVALUATOR / "baseline", stage=0)
    assert scores[evaluator.GATE] == 1.0
    assert scores["length"] == 13.0


def test_skew_expansion_matches_identity_and_zeroes_odd_lags() -> None:
    baseline = _load_baseline("test_labs_skew_helper")
    free_spins = [1, -1, -1, 1]

    sequence = baseline._expand_skew_symmetric(free_spins)
    m = len(free_spins)

    assert len(sequence) == 2 * m - 1
    for offset in range(1, m):
        expected = ((-1) ** offset) * sequence[m - 1 - offset]
        assert sequence[m - 1 + offset] == expected
    for lag in range(1, len(sequence), 2):
        correlation = sum(
            sequence[index] * sequence[index + lag]
            for index in range(len(sequence) - lag)
        )
        assert correlation == 0


@pytest.mark.parametrize("cell", ["n41-calibration", "n61-calibration"])
def test_calibration_cells_receive_full_stage_timeout(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
) -> None:
    evaluator = _load_evaluator(monkeypatch, cell, f"test_labs_{cell}")

    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert evaluator.STAGES[0].timeout_s == 300.0


@pytest.mark.parametrize(
    "cell",
    [
        "n71-frontier",
        "n81-frontier",
        "n91-frontier",
        "n101-frontier",
        "n121-frontier",
    ],
)
def test_frontier_cells_receive_full_stage_timeout(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
) -> None:
    evaluator = _load_evaluator(monkeypatch, cell, f"test_labs_{cell}")

    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert evaluator.STAGES[0].timeout_s == 300.0


def test_campaign_and_empty_bounds_registry_parse_with_repo_loaders() -> None:
    campaign = load_campaign(CAMPAIGN)
    bounds = load_bounds(CAMPAIGN)

    assert campaign.name == "labs"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "n13-validation",
        "n41-calibration",
        "n61-calibration",
        "n71-frontier",
        "n81-frontier",
        "n91-frontier",
        "n101-frontier",
        "n121-frontier",
    ]
    assert all(cell.target is None for cell in campaign.cells)
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert bounds == ()
    assert (CAMPAIGN / "bounds.json").read_text(encoding="utf-8").strip() == '{"bounds": []}'


def test_candidate_compute_is_advertised_without_stored_results() -> None:
    evaluator_spec = (EVALUATOR / "spec.md").read_text(encoding="utf-8")
    campaign_spec = (CAMPAIGN / "spec.md").read_text(encoding="utf-8")

    assert "may burn its whole budget searching" in evaluator_spec
    assert "return its best sequence" in evaluator_spec
    assert "no published best known energy or merit factor" in evaluator_spec.lower()
    assert "no published best known energy or merit factor" in campaign_spec.lower()
    assert "n <= 66 is solved" in campaign_spec
    assert "matched-known-optimum" in campaign_spec
    assert "skew symmetry is never a gate condition" in evaluator_spec.lower()
