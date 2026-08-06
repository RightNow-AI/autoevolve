from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "matmul-decomp"
EVALUATOR = PACK / "evaluators" / "matmul"


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


def _write_candidate(
    tmp_path: Path,
    name: str,
    decomposition: dict[str, list[list[complex | float | int]]],
) -> Path:
    candidate = tmp_path / name
    candidate.mkdir()
    (candidate / "solver.py").write_text(
        "DECOMPOSITION = "
        + repr(decomposition)
        + "\n\n"
        + "def solve(problem, deadline=None, seed=None):\n"
        + "    del problem, deadline, seed\n"
        + "    return DECOMPOSITION\n",
        encoding="utf-8",
    )
    return candidate


def _strassen_rank_seven() -> dict[str, list[list[int]]]:
    """Independent gate witness. Published coefficients belong only in this test."""

    return {
        "U": [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ],
        "V": [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        "W": [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
    }


def _schoolbook(m: int, k: int, n: int) -> dict[str, list[list[int]]]:
    rank = m * k * n
    u = [[0] * (m * k) for _ in range(rank)]
    v = [[0] * (k * n) for _ in range(rank)]
    w = [[0] * (m * n) for _ in range(rank)]
    row = 0
    for i in range(m):
        for j in range(k):
            for ell in range(n):
                u[row][i * k + j] = 1
                v[row][j * n + ell] = 1
                w[row][i * n + ell] = 1
                row += 1
    return {"U": u, "V": v, "W": w}


def test_independent_strassen_rank_seven_passes_exact_validation_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "2x2-real-r7-validation",
        "test_matmul_strassen",
    )
    candidate = _write_candidate(tmp_path, "strassen", _strassen_rank_seven())

    scores = evaluator.evaluate(candidate)

    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] == 7.0
    assert scores["numeric_tolerance"] == 0.0
    assert evaluator.MAXIMIZE is False
    assert len(evaluator.DESCRIPTORS) == 2
    for descriptor in evaluator.DESCRIPTORS:
        assert descriptor["name"] in scores
        assert descriptor["metric"] in scores


def test_schoolbook_rank_eight_passes_and_scores_eight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "2x2-real-r7-validation",
        "test_matmul_schoolbook",
    )
    candidate = _write_candidate(tmp_path, "schoolbook", _schoolbook(2, 2, 2))

    scores = evaluator.evaluate(candidate)

    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] == 8.0


def test_near_discrete_coefficient_is_rejected_without_rounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "2x2-real-r7-validation",
        "test_matmul_near_miss",
    )
    decomposition = _strassen_rank_seven()
    decomposition["U"][0][0] = 1.0 + 2.0**-40
    candidate = _write_candidate(tmp_path, "near-miss", decomposition)

    with pytest.raises(EvalError, match="outside the declared discrete set"):
        evaluator.evaluate(candidate)


def test_allowed_but_perturbed_coefficient_fails_exact_tensor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "2x2-real-r7-validation",
        "test_matmul_exact_perturbation",
    )
    decomposition = _strassen_rank_seven()
    decomposition["U"][0][0] = 0
    candidate = _write_candidate(tmp_path, "exact-perturbation", decomposition)

    with pytest.raises(EvalError, match="tensor identity failed"):
        evaluator.evaluate(candidate)


@pytest.mark.parametrize("matrix_name", ["U", "V", "W"])
def test_wrong_matrix_shape_raises_eval_error(
    matrix_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "2x2-real-r7-validation",
        f"test_matmul_bad_shape_{matrix_name}",
    )
    decomposition = _schoolbook(2, 2, 2)
    decomposition[matrix_name][0] = decomposition[matrix_name][0][:-1]
    candidate = _write_candidate(tmp_path, f"bad-{matrix_name}", decomposition)

    with pytest.raises(EvalError, match=rf"{matrix_name}\[0\] must contain exactly"):
        evaluator.evaluate(candidate)


def test_numeric_and_complex_cells_accept_generated_schoolbook_decompositions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numeric = _load_evaluator(
        monkeypatch,
        "3x3-real-r23-frontier",
        "test_matmul_numeric_cell",
    )
    numeric_candidate = _write_candidate(tmp_path, "numeric", _schoolbook(3, 3, 3))
    numeric_scores = numeric.evaluate(numeric_candidate)
    assert numeric_scores[numeric.GATE] == 1.0
    assert numeric_scores[numeric.METRIC] == 27.0
    assert numeric_scores["numeric_tolerance"] > 0.0

    exact_complex = _load_evaluator(
        monkeypatch,
        "4x4-complex-r48-frontier",
        "test_matmul_complex_cell",
    )
    complex_candidate = _write_candidate(tmp_path, "complex", _schoolbook(4, 4, 4))
    complex_scores = exact_complex.evaluate(complex_candidate)
    assert complex_scores[exact_complex.GATE] == 1.0
    assert complex_scores[exact_complex.METRIC] == 64.0
    assert complex_scores["numeric_tolerance"] == 0.0
    assert exact_complex.STAGES[0].timeout_s == 600.0


def test_campaign_and_empty_bounds_parse_with_repo_loaders() -> None:
    campaign = load_campaign(PACK)
    bounds = load_bounds(PACK)

    assert campaign.name == "matmul-decomp"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "2x2-real-r7-validation",
        "3x3-real-r23-frontier",
        "4x4-complex-r48-frontier",
    ]
    assert [cell.target for cell in campaign.cells] == [7.0, 23.0, 48.0]
    assert campaign.budget(full=False).is_bounded()
    assert campaign.budget(full=True).is_bounded()
    assert bounds == ()
