from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.cli.campaign import load_bounds, load_campaign
from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "campaigns" / "cublas-frontier"
EVALUATOR = PACK / "evaluators" / "cublas"


def _load_evaluator(name: str) -> ModuleType:
    path = EVALUATOR / "evaluate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "triton"
        or name.startswith("triton.")
    }


def test_mock_seed_passes_gate_and_returns_primary_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_CUBLAS_MOCK", "1")
    monkeypatch.setenv("AUTOEVOLVE_CELL", "odd-1000-validation")
    evaluator = _load_evaluator("test_cublas_mock_seed")
    before = _gpu_modules()

    scores = evaluator.evaluate(EVALUATOR / "baseline")

    assert evaluator.METRIC == "speedup"
    assert evaluator.MAXIMIZE is True
    assert scores[evaluator.GATE] == 1.0
    assert evaluator.METRIC in scores
    assert scores["mock_mode"] == 1.0
    assert scores["tile_area_log2"] > 0.0
    assert scores["kernel_launches"] == 2.0
    assert _gpu_modules() == before


def test_committed_wrong_result_mutant_fails_with_parity_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_CUBLAS_MOCK", "1")
    monkeypatch.setenv("AUTOEVOLVE_CELL", "skinny-4096x8-frontier")
    evaluator = _load_evaluator("test_cublas_wrong_result")
    mutant = EVALUATOR / "fixtures" / "mutants" / "wrong_result"

    with pytest.raises(EvalError, match="failed float64 parity"):
        evaluator.evaluate(mutant)


def test_real_mode_rejects_candidate_cpu_tensor() -> None:
    evaluator = _load_evaluator("test_cublas_cpu_tensor")

    class CpuDevice:
        type = "cpu"

        def __str__(self) -> str:
            return "cpu"

    class CpuTensor:
        device = CpuDevice()
        shape = (4, 4)
        dtype = "float32"

    with pytest.raises(EvalError, match="must return a CUDA tensor"):
        evaluator._require_cuda_tensor(
            CpuTensor(),
            expected_device="cuda:0",
            expected_shape=(4, 4),
            expected_dtype="float32",
            label="candidate",
        )


def test_self_reported_speedup_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_CUBLAS_MOCK", "1")
    monkeypatch.setenv("AUTOEVOLVE_CELL", "odd-1000-validation")
    evaluator = _load_evaluator("test_cublas_self_report")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = (EVALUATOR / "baseline" / "kernel.py").read_text(encoding="utf-8")
    honest_return = "return np.asarray(output, dtype=np.float32)"
    claimed_return = (
        'return {"output": np.asarray(output, dtype=np.float32), "speedup": 1e300}'
    )
    changed = source.replace(honest_return, claimed_return, 1)
    assert changed != source, "baseline mock return shape drifted"
    (candidate / "kernel.py").write_text(changed, encoding="utf-8")

    with pytest.raises(EvalError, match="self-reported metrics: speedup"):
        evaluator.evaluate(candidate)


def test_campaign_and_dynamic_bounds_parse_with_repository_loaders() -> None:
    campaign = load_campaign(PACK)
    bounds = load_bounds(PACK)

    assert campaign.name == "cublas-frontier"
    assert campaign.evaluator_path == EVALUATOR.resolve()
    assert [cell.key for cell in campaign.cells] == [
        "odd-1000-validation",
        "skinny-4096x8-frontier",
        "batched-1024x32-frontier",
        "fused-bias-relu-frontier",
    ]
    assert len(bounds) == 4
    assert all("not a published bound" in bound.who_and_year for bound in bounds)
    assert all("dynamic cublas_ms" in bound.value for bound in bounds)


def test_contract_constants_and_descriptors_are_well_formed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOEVOLVE_FORCE_CUBLAS_MOCK", "1")
    evaluator = _load_evaluator("test_cublas_contract")

    assert evaluator.GATE == "correctness"
    assert evaluator.METRIC == "speedup"
    assert evaluator.MAXIMIZE is True
    assert evaluator.STAGES
    assert all(isinstance(stage, StageSpec) for stage in evaluator.STAGES)
    assert all(stage.timeout_s > 0.0 for stage in evaluator.STAGES)
    assert len(evaluator.DESCRIPTORS) == 2
    assert {item["metric"] for item in evaluator.DESCRIPTORS} == {
        "tile_area_log2",
        "kernel_launches",
    }
    assert evaluator.ceiling() is None
