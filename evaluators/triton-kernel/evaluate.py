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
_CELLS = frozenset({"add-1k", "add-8k", "scale-1k", "scale-8k"})

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


def _selected_cell() -> str | None:
    cell = os.environ.get("AUTOEVOLVE_CELL")
    if cell is not None and cell not in _CELLS:
        allowed = ", ".join(sorted(_CELLS))
        raise EvalError(f"AUTOEVOLVE_CELL must be one of: {allowed}")
    return cell


def _load_cases(cell: str | None = None) -> list[Case]:
    raw = json.loads((FIXTURE_DIR / "cases.json").read_text(encoding="utf-8"))
    cases: list[Case] = []
    for item in raw["cases"]:
        if cell is not None and item["cell"] != cell:
            continue
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
    """Real measurement needs a CUDA device, not one particular kernel language.

    The certificate is numerical parity against the reference plus elapsed
    time on this device. A candidate may reach that through Triton, plain
    torch, CuPy, or anything else it can import, so requiring Triton here
    would strand working hardware in mock mode. Triton ships no Windows
    wheel, which is exactly that situation.
    """

    if os.environ.get("AUTOEVOLVE_FORCE_TRITON_MOCK") == "1":
        return False
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def available_backends() -> tuple[str, ...]:
    """Kernel backends a candidate may import in real mode on this machine."""

    names = ("triton", "torch", "cupy", "numba")
    return tuple(name for name in names if importlib.util.find_spec(name) is not None)


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
    if real:
        _require_device_result(output)
        output = output.detach().to("cpu")
    array = np.asarray(output, dtype=np.float32)
    if array.shape != x.shape:
        raise EvalError(f"candidate returned shape {array.shape}, expected {x.shape}")
    return array


def _require_device_result(output: object) -> None:
    """Reject a real-mode result that never touched the GPU.

    A candidate that quietly falls back to CPU still produces correct
    numbers, so parity alone would pass it and the reported throughput
    would be a CPU measurement published as a GPU one. Real mode therefore
    requires a CUDA tensor, which is exact evidence of where the work ran.
    This check reads an attribute and never synchronizes, so it is safe to
    call inside the timing loop.
    """

    device = getattr(output, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        raise EvalError(
            "real mode requires the result to be a CUDA tensor so the "
            "measurement provably ran on the device; got "
            f"{type(output).__name__} on device {device}"
        )


def _launch_for_timing(
    candidate: ModuleType,
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> None:
    """Launch the candidate on the device with no host transfer in the loop."""

    try:
        output = candidate.run(x, y, alpha, real=True)
    except Exception as exc:
        raise EvalError(f"candidate execution failed: {exc}") from exc
    _require_device_result(output)


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


def _launch_constants(entry_path: Path) -> tuple[int, int]:
    """Read the tile size and warp count the candidate declared.

    These describe the launch shape, so they are read from the source rather
    than from anything the candidate reports at run time.
    """

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
        raise EvalError("launch constants must be positive")
    return block, warps


def _launch_descriptors(candidate_dir: Path) -> dict[str, float]:
    block, warps = _launch_constants(candidate_dir / "kernel.py")
    return {
        "block_log2": math.log2(block),
        "warp_log2": math.log2(warps),
    }


def _static_mock_score(entry_path: Path, sizes: list[int]) -> float:
    block, warps = _launch_constants(entry_path)
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
        # The mock score is a utilization product and is therefore bounded by
        # one. The candidate reports it, so an unbounded value is not a
        # measurement, it is a claim, and evolution will make that claim
        # arbitrarily large: an unclamped run reached 1e300 in 46 programs.
        if not 0.0 <= score <= 1.0:
            raise EvalError(
                f"mock_schedule returned {score} for n={size}; the utilization "
                "model is bounded to [0, 1] and a self-reported score outside "
                "it is rejected"
            )
        scores.append(score)
    return sum(scores) / len(scores)


def _benchmark_elements() -> int:
    """Element count for the throughput workload.

    The parity fixtures are deliberately tiny so correctness checks stay
    fast. Timing them would measure kernel launch overhead and the host
    transfer rather than throughput, which for a memory-bound operation can
    never approach the device roofline and would make the metric
    meaningless. Throughput therefore uses a separate, device-resident
    workload large enough that the kernel dominates.
    """

    raw = os.environ.get("AUTOEVOLVE_KERNEL_ELEMENTS")
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise EvalError("AUTOEVOLVE_KERNEL_ELEMENTS must be an integer") from exc
        if value < 1 << 16:
            raise EvalError("AUTOEVOLVE_KERNEL_ELEMENTS must be at least 65536")
        return value
    return 1 << 24


def _measure_real(candidate: ModuleType, cases: list[Case]) -> tuple[float, float]:
    """Time the candidate on device-resident data, excluding host transfer."""

    torch = importlib.import_module("torch")
    device = torch.device("cuda")
    elements = _benchmark_elements()
    generator = torch.Generator(device=device).manual_seed(20260803)
    x_tensor = torch.rand(elements, device=device, dtype=torch.float32, generator=generator)
    y_tensor = torch.rand(elements, device=device, dtype=torch.float32, generator=generator)
    alpha = float(cases[0][3]) if cases else 1.0

    for _ in range(3):
        _launch_for_timing(candidate, x_tensor, y_tensor, alpha)
    torch.cuda.synchronize()

    repeats = 20
    elapsed_values: list[float] = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            _launch_for_timing(candidate, x_tensor, y_tensor, alpha)
        end.record()
        torch.cuda.synchronize()
        elapsed_values.append(float(start.elapsed_time(end)) / repeats)

    candidate_ms = max(min(elapsed_values), 1e-9)
    total_flops = 2.0 * float(elements)
    tflops = total_flops / (candidate_ms / 1_000.0) / 1e12
    return tflops, candidate_ms


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Run parity first, then return real throughput or explicit mock metrics."""
    if stage != 0:
        raise EvalError(f"unknown stage {stage}")
    cases = _load_cases(_selected_cell())
    candidate = _load_module(candidate_dir)
    real = _real_mode_available()
    _check_parity(candidate, cases, real=real)
    descriptors = _launch_descriptors(candidate_dir)
    if not real:
        return {
            GATE: 1.0,
            "mock_score": _mock_score(
                candidate,
                candidate_dir,
                [x.size for _, x, _, _ in cases],
            ),
            **descriptors,
        }
    tflops, candidate_ms = _measure_real(candidate, cases)
    return {GATE: 1.0, "tflops": tflops, "candidate_ms": candidate_ms, **descriptors}


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

# Primary metric declaration consumed by the engine when locking a contract.
# Mock mode exposes only mock_ metrics, so the declaration follows the mode.
METRIC = "tflops" if _real_mode_available() else "mock_score"
MAXIMIZE = True

# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent.
#
# Both describe the launch shape a candidate chose, not how fast it ran. A
# kernel is defined by its tile size and how much parallelism it asks the
# device for, and two kernels with the same throughput today can sit at
# opposite ends of that space with very different room left to improve. The
# archive should hold both rather than discarding one for being a tie.
DESCRIPTORS = [
    {"name": "block_log2", "metric": "block_log2", "bins": 8, "lo": 3.0, "hi": 14.0},
    {"name": "warp_log2", "metric": "warp_log2", "bins": 5, "lo": 0.0, "hi": 5.0},
]
