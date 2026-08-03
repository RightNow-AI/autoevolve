"""GPU-real and CPU-mocked evaluator for vector add-and-scale kernels."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [StageSpec(name="kernel-cases", timeout_s=60.0)]
GATE: str = "mock_parity"

PACK_DIR = Path(__file__).resolve().parent
BASELINE_DIR = PACK_DIR / "baseline"
FIXTURE_DIR = PACK_DIR / "fixtures"
Case = tuple[str, np.ndarray, np.ndarray, float]

_REFERENCE_MODULE: ModuleType | None = None


def _load_module(candidate_dir: Path) -> ModuleType:
    entry_path = candidate_dir / "kernel.py"
    if not entry_path.is_file():
        raise EvalError(f"candidate is missing {entry_path.name}")
    module_name = f"_autoevolve_triton_kernel_{abs(hash(entry_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {entry_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    return module


def _load_cases() -> list[Case]:
    raw = json.loads((FIXTURE_DIR / "cases.json").read_text(encoding="utf-8"))
    cases: list[Case] = []
    for item in raw["cases"]:
        cases.append(
            (
                str(item["name"]),
                np.asarray(item["x"], dtype=np.float32),
                np.asarray(item["y"], dtype=np.float32),
                float(item["alpha"]),
            )
        )
    return cases


def _reference_module() -> ModuleType:
    global _REFERENCE_MODULE
    if _REFERENCE_MODULE is None:
        _REFERENCE_MODULE = _load_module(BASELINE_DIR)
    return _REFERENCE_MODULE


def _real_mode_available() -> bool:
    if os.environ.get("AUTOEVOLVE_FORCE_TRITON_MOCK") == "1":
        return False
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None:
        return False
    torch = importlib.import_module("torch")
    return bool(torch.cuda.is_available())


def _run_candidate(
    candidate: ModuleType,
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    real: bool,
) -> np.ndarray:
    try:
        output = candidate.run(x.copy(), y.copy(), alpha, real=real)
    except Exception as exc:
        raise EvalError(f"candidate execution failed: {exc}") from exc
    array = np.asarray(output, dtype=np.float32)
    if array.shape != x.shape:
        raise EvalError(f"candidate returned shape {array.shape}, expected {x.shape}")
    return array


def _check_parity(candidate: ModuleType, cases: list[Case], *, real: bool) -> None:
    reference = _reference_module()
    for name, x, y, alpha in cases:
        expected = reference.ref(x, y, alpha)
        try:
            actual = _run_candidate(candidate, x, y, alpha, real=real)
        except EvalError as exc:
            raise EvalError(f"case {name} failed: {exc.reason}") from exc
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
            difference = float(np.max(np.abs(actual - expected)))
            raise EvalError(f"case {name} failed parity; max absolute error {difference}")


def _static_mock_score(entry_path: Path, sizes: list[int]) -> float:
    tree = ast.parse(entry_path.read_text(encoding="utf-8"), filename=str(entry_path))
    constants: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"BLOCK", "num_warps"}:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                constants[target.id] = node.value.value
    block = constants.get("BLOCK", 256)
    warps = constants.get("num_warps", 4)
    if block <= 0 or warps <= 0:
        raise EvalError("mock launch constants must be positive")
    utilization = [size / (math.ceil(size / block) * block) for size in sizes]
    warp_balance = min(warps, 4) / max(warps, 4)
    return float(sum(utilization) / len(utilization) * warp_balance)


def _mock_score(candidate: ModuleType, candidate_dir: Path, sizes: list[int]) -> float:
    hook = getattr(candidate, "mock_schedule", None)
    if hook is None:
        return _static_mock_score(candidate_dir / "kernel.py", sizes)
    scores: list[float] = []
    for size in sizes:
        try:
            result = hook(size)
            score = float(result["score"])
        except Exception as exc:
            raise EvalError(f"mock_schedule failed for n={size}: {exc}") from exc
        if not math.isfinite(score):
            raise EvalError(f"mock_schedule returned a non-finite score for n={size}")
        scores.append(score)
    return sum(scores) / len(scores)


def _measure_real(candidate: ModuleType, cases: list[Case]) -> tuple[float, float]:
    torch = importlib.import_module("torch")
    elapsed_values: list[float] = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _, x, y, alpha in cases:
            _run_candidate(candidate, x, y, alpha, real=True)
        end.record()
        torch.cuda.synchronize()
        elapsed_values.append(float(start.elapsed_time(end)))
    candidate_ms = max(min(elapsed_values), 1e-9)
    total_flops = float(sum(2 * x.size for _, x, _, _ in cases))
    tflops = total_flops / (candidate_ms / 1_000.0) / 1e12
    return tflops, candidate_ms


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Run parity first, then return real throughput or explicit mock metrics."""
    if stage != 0:
        raise EvalError(f"unknown stage {stage}")
    cases = _load_cases()
    candidate = _load_module(candidate_dir)
    real = _real_mode_available()
    _check_parity(candidate, cases, real=real)
    if not real:
        return {
            GATE: 1.0,
            "mock_score": _mock_score(candidate, candidate_dir, [x.size for _, x, _, _ in cases]),
        }
    tflops, candidate_ms = _measure_real(candidate, cases)
    return {GATE: 1.0, "tflops": tflops, "candidate_ms": candidate_ms}


def ceiling() -> dict[str, float | str] | None:
    """Return the memory-bandwidth roofline when a compatible CUDA GPU exists."""
    if not _real_mode_available():
        return None
    torch = importlib.import_module("torch")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    memory_clock_khz = getattr(properties, "memory_clock_rate", None)
    memory_bus_bits = getattr(properties, "memory_bus_width", None)
    if memory_clock_khz is None or memory_bus_bits is None:
        return None
    bandwidth_bytes = float(memory_clock_khz) * 1_000.0 * 2.0 * float(memory_bus_bits) / 8.0
    bytes_moved_per_element = 12.0
    flops_per_element = 2.0
    roofline_tflops = bandwidth_bytes / bytes_moved_per_element * flops_per_element / 1e12
    return {
        "metric": "tflops",
        "value": roofline_tflops,
        "method": "roofline: memory-bound",
    }
