"""Strict same-run cuBLAS comparison with a CPU-only CI mock."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [StageSpec(name="cuda-parity-and-timing", timeout_s=900.0)]
GATE = "correctness"
METRIC = "speedup"
MAXIMIZE = True

PACK_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = PACK_DIR / "fixtures" / "cases.json"

# Rounding error in a dot product grows with the reduction length, so the
# budget is the standard backward error bound gamma_K = K * eps / (1 - K * eps)
# applied to the sum of absolute products rather than to the output. A flat
# tolerance failed cuBLAS itself at K = 4096, because cancellation makes the
# output small while the accumulated error does not shrink with it.
FLOAT32_EPS = 2.0**-24
# The sentinel deliberately fills a row and a column with the same value, so
# the entry at index zero sums thousands of identical same-signed products.
# That is the worst case for floating point accumulation, and implementations
# differ in summation order, so a factor of four proved 3 percent too tight
# against an honest ieee-precision Triton kernel. Sixteen leaves room for any
# reasonable accumulation order.
#
# This is still nowhere near admitting a precision downgrade. TF32 carries 11
# mantissa bits against 24, so its error budget is about 8192 times larger, and
# this factor spends well under one percent of that. A TF32 result fails by
# roughly five hundred times the allowance rather than marginally.
ERROR_BUDGET_SAFETY = 16.0
RTOL = 5e-5
ATOL = 2e-5


def _gamma(reduction_length: int) -> float:
    """Backward error factor for summing `reduction_length` float32 products."""

    product = reduction_length * FLOAT32_EPS
    if product >= 1.0:
        raise EvalError(f"reduction length {reduction_length} is too large for float32")
    return ERROR_BUDGET_SAFETY * product / (1.0 - product)


WARMUP_LAUNCHES = 3
TIMED_ROUNDS = 5
REPEATS_PER_ROUND = 3
DEADLINE_HEADROOM_S = 90.0

_PROTECTED_REPORT_NAMES = frozenset(
    {
        "CANDIDATE_MS",
        "CUBLAS_MS",
        "METRICS",
        "SPEEDUP",
        "candidate_ms",
        "cublas_ms",
        "metrics",
        "speedup",
    }
)


class CellSpec(NamedTuple):
    """One fixed GEMM workload and its reduced CPU-mock analogue."""

    key: str
    role: str
    batch: int
    m: int
    n: int
    k: int
    bias: bool
    activation: str
    seed: int
    mock_batch: int
    mock_m: int
    mock_n: int
    mock_k: int


class Workload(NamedTuple):
    """Inputs and the trusted output computed before candidate import."""

    a: Any
    b: Any
    bias: Any | None
    expected: Any
    # |A| @ |B|, which bounds accumulated rounding error even where the signed
    # sum cancels. A tolerance keyed to the output misjudges exactly that case.
    # Required, not defaulted: one parity call site was missed and silently fell
    # back to the weak check, so two GPU launches failed on a budget that was
    # never applied. A missing value must be a loud error, not a quiet one.
    magnitude: Any


def _load_candidate(candidate_dir: Path) -> ModuleType:
    entry_path = candidate_dir / "kernel.py"
    if not entry_path.is_file():
        raise EvalError("candidate is missing kernel.py")
    module_name = f"_autoevolve_cublas_{abs(hash(entry_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {entry_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    claimed = sorted(_PROTECTED_REPORT_NAMES.intersection(vars(module)))
    if claimed:
        names = ", ".join(claimed)
        raise EvalError(f"candidate declared self-reported metric names: {names}")
    return module


def _load_cells() -> tuple[str, dict[str, CellSpec]]:
    try:
        raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot load cell fixtures: {exc}") from exc
    default_cell = str(raw["default_cell"])
    cells: dict[str, CellSpec] = {}
    for item in raw["cells"]:
        cell = CellSpec(
            key=str(item["key"]),
            role=str(item["role"]),
            batch=int(item["batch"]),
            m=int(item["m"]),
            n=int(item["n"]),
            k=int(item["k"]),
            bias=bool(item["bias"]),
            activation=str(item["activation"]),
            seed=int(item["seed"]),
            mock_batch=int(item["mock_batch"]),
            mock_m=int(item["mock_m"]),
            mock_n=int(item["mock_n"]),
            mock_k=int(item["mock_k"]),
        )
        if min(
            cell.batch,
            cell.m,
            cell.n,
            cell.k,
            cell.mock_batch,
            cell.mock_m,
            cell.mock_n,
            cell.mock_k,
        ) <= 0:
            raise EvalError(f"cell {cell.key} has a non-positive dimension")
        if cell.activation not in {"none", "relu"}:
            raise EvalError(f"cell {cell.key} has an unsupported activation")
        cells[cell.key] = cell
    if default_cell not in cells:
        raise EvalError("fixture default_cell does not name a configured cell")
    return default_cell, cells


def _selected_cell() -> CellSpec:
    default_cell, cells = _load_cells()
    key = os.environ.get("AUTOEVOLVE_CELL", default_cell)
    if key not in cells:
        choices = ", ".join(sorted(cells))
        raise EvalError(f"AUTOEVOLVE_CELL must be one of: {choices}")
    return cells[key]


def _real_mode_available() -> bool:
    if os.environ.get("AUTOEVOLVE_FORCE_CUBLAS_MOCK") == "1":
        return False
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _shape(batch: int, m: int, width: int) -> tuple[int, ...]:
    if batch == 1:
        return (m, width)
    return (batch, m, width)


def _set_precision_sentinel(a: Any, b: Any, batch: int) -> None:
    """Make TF32 input rounding fail deterministically on output zero, zero."""

    if batch == 1:
        a[0, :] = 1.0003
        b[:, 0] = 1.0003
    else:
        a[:, 0, :] = 1.0003
        b[:, :, 0] = 1.0003


def _numpy_reference(
    a: np.ndarray,
    b: np.ndarray,
    bias: np.ndarray | None,
    activation: str,
) -> np.ndarray:
    result = np.matmul(a.astype(np.float64), b.astype(np.float64))
    if bias is not None:
        result = result + bias.astype(np.float64)
    if activation == "relu":
        result = np.maximum(result, 0.0)
    return result


def _make_numpy_workload(cell: CellSpec, seed_offset: int = 0) -> Workload:
    rng = np.random.default_rng(cell.seed + seed_offset)
    a = rng.uniform(
        -1.0,
        1.0,
        size=_shape(cell.mock_batch, cell.mock_m, cell.mock_k),
    ).astype(np.float32)
    b = rng.uniform(
        -1.0,
        1.0,
        size=_shape(cell.mock_batch, cell.mock_k, cell.mock_n),
    ).astype(np.float32)
    _set_precision_sentinel(a, b, cell.mock_batch)
    bias = None
    if cell.bias:
        bias = rng.uniform(-0.25, 0.25, size=(cell.mock_n,)).astype(np.float32)
    expected = _numpy_reference(a, b, bias, cell.activation)
    # |A| @ |B| bounds the accumulated rounding error even when the signed sum
    # cancels to something small, which is exactly the case a tolerance keyed
    # to the output misjudges.
    magnitude = np.matmul(np.abs(a).astype(np.float64), np.abs(b).astype(np.float64))
    return Workload(a=a, b=b, bias=bias, expected=expected, magnitude=magnitude)


def _torch_reference(
    torch: Any,
    matmul: Callable[[Any, Any], Any],
    relu: Callable[[Any], Any],
    a: Any,
    b: Any,
    bias: Any | None,
    activation: str,
) -> Any:
    result = matmul(a.to(dtype=torch.float64), b.to(dtype=torch.float64))
    if bias is not None:
        result = result + bias.to(dtype=torch.float64)
    if activation == "relu":
        result = relu(result)
    return result


def _make_torch_workload(
    torch: Any,
    matmul: Callable[[Any, Any], Any],
    relu: Callable[[Any], Any],
    device: Any,
    cell: CellSpec,
    seed_offset: int,
) -> Workload:
    generator = torch.Generator(device=device).manual_seed(cell.seed + seed_offset)
    a = torch.rand(
        _shape(cell.batch, cell.m, cell.k),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    b = torch.rand(
        _shape(cell.batch, cell.k, cell.n),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    a = a.mul(2.0).sub(1.0)
    b = b.mul(2.0).sub(1.0)
    _set_precision_sentinel(a, b, cell.batch)
    bias = None
    if cell.bias:
        bias = torch.rand(
            (cell.n,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        bias = bias.mul(0.5).sub(0.25)
    expected = _torch_reference(torch, matmul, relu, a, b, bias, cell.activation)
    # Computed in float64 without the bias or activation, because it bounds the
    # error of the reduction itself. A tolerance keyed to the output would
    # misjudge every entry where the signed sum cancels.
    magnitude = matmul(
        a.abs().to(dtype=torch.float64),
        b.abs().to(dtype=torch.float64),
    )
    return Workload(a=a, b=b, bias=bias, expected=expected, magnitude=magnitude)


def _reject_reported_output(output: object) -> None:
    if isinstance(output, Mapping):
        names = sorted(
            str(name) for name in output if str(name) in _PROTECTED_REPORT_NAMES
        )
        if names:
            raise EvalError(f"candidate returned self-reported metrics: {', '.join(names)}")
        raise EvalError("candidate must return only its output, not a mapping")
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, Mapping):
                raise EvalError("candidate returned self-reported metadata beside its output")


def _call_candidate(
    candidate: ModuleType,
    workload: Workload,
    activation: str,
    *,
    real: bool,
    deadline: float,
) -> object:
    run = getattr(candidate, "run", None)
    if not callable(run):
        raise EvalError("candidate must define callable run")
    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError) as exc:
        raise EvalError(f"cannot inspect candidate run signature: {exc}") from exc
    args = [workload.a, workload.b, workload.bias, activation]
    kwargs: dict[str, object] = {"real": real}
    deadline_parameter = signature.parameters.get("deadline")
    arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if deadline_parameter is not None:
        if deadline_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(deadline)
        else:
            kwargs["deadline"] = deadline
    elif arbitrary_keywords:
        kwargs["deadline"] = deadline
    try:
        output = run(*args, **kwargs)
    except Exception as exc:
        raise EvalError(f"candidate execution failed: {exc}") from exc
    _reject_reported_output(output)
    return output


def _assert_numpy_close(
    output: object,
    expected: np.ndarray,
    label: str,
    magnitude: np.ndarray,
    reduction_length: int,
) -> None:
    """Compare against the float64 reference with a K-aware error budget.

    A flat tolerance is wrong here. Rounding error in a dot product grows with
    the reduction length, so a constant that suits K=64 rejects a correct
    result at K=4096. This gate did exactly that: it failed cuBLAS itself, the
    reference implementation, by 3.01e-05 against a 2.88e-05 allowance.

    The budget is the standard backward error bound for floating point
    summation, gamma_K times the sum of absolute products, which is the
    quantity that actually bounds the error. `magnitude` carries |A| @ |B|
    computed in float64 when the caller knows it.

    This stays far from admitting a precision downgrade. TF32 carries 11 bits
    of mantissa against 24, so its bound is roughly 8000 times larger, and the
    sentinel row planted in the inputs pushes a TF32 result orders of magnitude
    outside this allowance rather than marginally past it.
    """

    try:
        actual = np.asarray(output)
    except Exception as exc:
        raise EvalError(f"{label} returned a non-array output: {exc}") from exc
    if actual.shape != expected.shape:
        raise EvalError(f"{label} returned shape {actual.shape}, expected {expected.shape}")
    if actual.dtype != np.float32:
        raise EvalError(f"{label} returned dtype {actual.dtype}, expected float32")
    difference = np.abs(actual.astype(np.float64) - expected)
    allowance = ATOL + _gamma(reduction_length) * np.abs(magnitude)
    if not bool(np.all(difference <= allowance)):
        index = np.unravel_index(int(np.argmax(difference - allowance)), difference.shape)
        # Report the ratio and the reduction length too. A bare pair of numbers
        # cannot distinguish a marginally tight budget from a real precision
        # downgrade, and that distinction cost two launches to work out.
        ratio = float(difference[index]) / max(float(allowance[index]), 1e-30)
        raise EvalError(
            f"{label} failed float64 parity at {index}; absolute error "
            f"{float(difference[index])} exceeds {float(allowance[index])} "
            f"by {ratio:.2f}x with reduction length {reduction_length}"
        )


def _require_cuda_tensor(
    output: object,
    expected_device: object,
    expected_shape: Sequence[int],
    expected_dtype: object,
    label: str,
) -> None:
    device = getattr(output, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        raise EvalError(
            f"{label} must return a CUDA tensor on the input device; got device {device}"
        )
    if device != expected_device:
        raise EvalError(f"{label} returned device {device}, expected {expected_device}")
    shape = tuple(getattr(output, "shape", ()))
    if shape != tuple(expected_shape):
        raise EvalError(f"{label} returned shape {shape}, expected {tuple(expected_shape)}")
    dtype = getattr(output, "dtype", None)
    if dtype != expected_dtype:
        raise EvalError(f"{label} returned dtype {dtype}, expected {expected_dtype}")


def _assert_torch_close(
    torch: Any,
    output: Any,
    expected: Any,
    label: str,
    magnitude: Any,
    reduction_length: int,
) -> None:
    difference = (output.to(dtype=torch.float64) - expected).abs()
    allowance = ATOL + _gamma(reduction_length) * magnitude.abs()
    failed = difference > allowance
    if bool(failed.any().item()):
        excess = difference - allowance
        flat_index = int(excess.reshape(-1).argmax().item())
        absolute_error = float(difference.reshape(-1)[flat_index].item())
        allowed_error = float(allowance.reshape(-1)[flat_index].item())
        raise EvalError(
            f"{label} failed float64 parity at flat index {flat_index}; absolute error "
            f"{absolute_error} exceeds {allowed_error}"
        )


def _integer_assignment(tree: ast.Module, name: str) -> int | None:
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            return int(value.value)
    return None


def _source_descriptors(candidate_dir: Path) -> tuple[dict[str, float], int]:
    entry_path = candidate_dir / "kernel.py"
    try:
        source = entry_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(entry_path))
    except (OSError, SyntaxError) as exc:
        raise EvalError(f"cannot read candidate launch declarations: {exc}") from exc
    block_m = _integer_assignment(tree, "BLOCK_M")
    block_n = _integer_assignment(tree, "BLOCK_N")
    launches = _integer_assignment(tree, "KERNEL_LAUNCHES")
    if block_m is None or block_n is None or launches is None:
        raise EvalError("candidate must declare integer BLOCK_M, BLOCK_N, and KERNEL_LAUNCHES")
    if block_m <= 0 or block_n <= 0 or not 1 <= launches <= 64:
        raise EvalError("launch declarations must be positive and launches must not exceed 64")
    return {
        "tile_area_log2": math.log2(block_m * block_n),
        "kernel_launches": float(launches),
    }, launches


def _set_ieee_matmul(torch: Any) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    matmul_backend = torch.backends.cuda.matmul
    if hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = False


def _baseline_output(
    workload: Workload,
    matmul: Callable[[Any, Any], Any],
    relu: Callable[[Any], Any],
    activation: str,
) -> Any:
    output = matmul(workload.a, workload.b)
    if workload.bias is not None:
        output = output + workload.bias
    if activation == "relu":
        output = relu(output)
    return output


def _warm_up(
    launch: Callable[[Workload], Any],
    workload: Workload,
    select_device: Callable[[], None],
    synchronize: Callable[[], None],
) -> None:
    for _ in range(WARMUP_LAUNCHES):
        select_device()
        launch(workload)
    select_device()
    synchronize()


def _measure_cuda(
    launch: Callable[[Workload], Any],
    workloads: Sequence[Workload],
    event_factory: Callable[..., Any],
    select_device: Callable[[], None],
    synchronize: Callable[[], None],
) -> tuple[float, list[Any]]:
    expected_count = TIMED_ROUNDS * REPEATS_PER_ROUND
    if len(workloads) != expected_count:
        raise EvalError(f"timing requires exactly {expected_count} one-use workloads")
    elapsed: list[float] = []
    outputs: list[Any] = []
    offset = 0
    for _ in range(TIMED_ROUNDS):
        select_device()
        start = event_factory(enable_timing=True)
        end = event_factory(enable_timing=True)
        synchronize()
        start.record()
        start.synchronize()
        for workload in workloads[offset : offset + REPEATS_PER_ROUND]:
            select_device()
            outputs.append(launch(workload))
        # The device-wide synchronization makes launches on any CUDA stream
        # finish before the ending timestamp. Otherwise a side stream could
        # return early and make asynchronous work disappear from the score.
        select_device()
        synchronize()
        end.record()
        end.synchronize()
        elapsed.append(float(start.elapsed_time(end)) / REPEATS_PER_ROUND)
        offset += REPEATS_PER_ROUND
    return max(min(elapsed), 1e-9), outputs


def _profile_launch_count(
    launch: Callable[[Workload], Any],
    workload: Workload,
    profile_factory: Callable[..., Any],
    profiler_activity: Any,
    select_device: Callable[[], None],
    synchronize: Callable[[], None],
) -> tuple[int, Any]:
    select_device()
    with profile_factory(
        activities=[profiler_activity.CPU, profiler_activity.CUDA]
    ) as profile:
        select_device()
        output = launch(workload)
        select_device()
        synchronize()
    count = sum(
        1
        for event in profile.events()
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    )
    return count, output


def _stage_deadline(started: float, stage: int) -> float:
    budget = STAGES[stage].timeout_s - DEADLINE_HEADROOM_S
    if budget <= 0.0:
        raise EvalError("stage timeout leaves no candidate deadline headroom")
    return started + budget


def _evaluate_mock(
    candidate_dir: Path,
    cell: CellSpec,
    deadline: float,
) -> dict[str, float]:
    candidate = _load_candidate(candidate_dir)
    workload = _make_numpy_workload(cell)
    output = _call_candidate(
        candidate,
        workload,
        cell.activation,
        real=False,
        deadline=deadline,
    )
    _assert_numpy_close(
        output,
        workload.expected,
        f"cell {cell.key}",
        magnitude=workload.magnitude,
        reduction_length=cell.mock_k,
    )
    descriptors, _ = _source_descriptors(candidate_dir)
    return {
        GATE: 1.0,
        METRIC: 0.0,
        "mock_mode": 1.0,
        **descriptors,
    }


def _evaluate_real(candidate_dir: Path, cell: CellSpec, deadline: float) -> dict[str, float]:
    torch = importlib.import_module("torch")
    _set_ieee_matmul(torch)
    device = torch.device("cuda", torch.cuda.current_device())

    # Capture trusted callables before candidate import. The baseline and all
    # references are also complete before import, so candidate module side
    # effects cannot change the comparison it is trying to beat.
    matmul = torch.matmul
    relu = torch.relu
    set_device = torch.cuda.set_device
    synchronize_device = torch.cuda.synchronize
    event_factory = torch.cuda.Event
    profile_factory = torch.profiler.profile
    profiler_activity = torch.profiler.ProfilerActivity

    def select_device() -> None:
        set_device(device)

    def synchronize() -> None:
        synchronize_device(device)

    select_device()

    setup = _make_torch_workload(torch, matmul, relu, device, cell, 0)
    profile_workload = _make_torch_workload(torch, matmul, relu, device, cell, 1)
    timed = [
        _make_torch_workload(torch, matmul, relu, device, cell, 10 + index)
        for index in range(TIMED_ROUNDS * REPEATS_PER_ROUND)
    ]

    def baseline_launch(workload: Workload) -> Any:
        return _baseline_output(workload, matmul, relu, cell.activation)

    baseline_setup = baseline_launch(setup)
    _require_cuda_tensor(
        baseline_setup,
        device,
        setup.expected.shape,
        torch.float32,
        "cuBLAS baseline",
    )
    _assert_torch_close(
        torch,
        baseline_setup,
        setup.expected,
        "cuBLAS baseline",
        magnitude=setup.magnitude,
        reduction_length=cell.k,
    )
    _warm_up(baseline_launch, setup, select_device, synchronize)
    cublas_ms, baseline_outputs = _measure_cuda(
        baseline_launch,
        timed,
        event_factory,
        select_device,
        synchronize,
    )
    for index, (output, workload) in enumerate(
        zip(baseline_outputs, timed, strict=True)
    ):
        _assert_torch_close(
            torch,
            output,
            workload.expected,
            f"cuBLAS timed output {index}",
            magnitude=workload.magnitude,
            reduction_length=cell.k,
        )

    candidate = _load_candidate(candidate_dir)
    descriptors, declared_launches = _source_descriptors(candidate_dir)

    def candidate_launch(workload: Workload) -> Any:
        output = _call_candidate(
            candidate,
            workload,
            cell.activation,
            real=True,
            deadline=deadline,
        )
        _require_cuda_tensor(
            output,
            device,
            workload.expected.shape,
            torch.float32,
            "candidate",
        )
        return output

    candidate_setup = candidate_launch(setup)
    synchronize()
    _assert_torch_close(
        torch,
        candidate_setup,
        setup.expected,
        "candidate setup",
        magnitude=setup.magnitude,
        reduction_length=cell.k,
    )
    _warm_up(candidate_launch, setup, select_device, synchronize)

    measured_launches, profile_output = _profile_launch_count(
        candidate_launch,
        profile_workload,
        profile_factory,
        profiler_activity,
        select_device,
        synchronize,
    )
    _assert_torch_close(
        torch,
        profile_output,
        profile_workload.expected,
        "candidate profile",
        magnitude=profile_workload.magnitude,
        reduction_length=cell.k,
    )
    if measured_launches == 0:
        raise EvalError("candidate performed no CUDA kernel launch on a fresh input")
    if measured_launches != declared_launches:
        raise EvalError(
            f"candidate declared {declared_launches} CUDA kernel launches but profiler "
            f"observed {measured_launches}"
        )
    descriptors["kernel_launches"] = float(measured_launches)

    candidate_ms, candidate_outputs = _measure_cuda(
        candidate_launch,
        timed,
        event_factory,
        select_device,
        synchronize,
    )
    for index, (output, workload) in enumerate(
        zip(candidate_outputs, timed, strict=True)
    ):
        _assert_torch_close(
            torch,
            output,
            workload.expected,
            f"candidate timed output {index}",
            magnitude=workload.magnitude,
            reduction_length=cell.k,
        )

    # A faster candidate has a smaller denominator, so this direction makes
    # larger fitness mean faster code. Reversing it would reward regressions.
    speedup = cublas_ms / candidate_ms
    return {
        GATE: 1.0,
        METRIC: speedup,
        "cublas_ms": cublas_ms,
        "candidate_ms": candidate_ms,
        **descriptors,
    }


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Gate numerical parity, then measure the same-run cuBLAS ratio."""

    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")
    started = time.monotonic()
    deadline = _stage_deadline(started, stage)
    cell = _selected_cell()
    if not _real_mode_available():
        return _evaluate_mock(candidate_dir, cell, deadline)
    return _evaluate_real(candidate_dir, cell, deadline)


def ceiling() -> None:
    """No static ceiling exists for a same-device latency ratio."""

    return None


DESCRIPTORS = [
    {
        "name": "tile_area_log2",
        "metric": "tile_area_log2",
        "bins": 10,
        "lo": 4.0,
        "hi": 18.0,
    },
    {
        "name": "kernel_launches",
        "metric": "kernel_launches",
        "bins": 8,
        "lo": 1.0,
        "hi": 65.0,
    },
]
