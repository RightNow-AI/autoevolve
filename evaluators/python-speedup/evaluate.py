"""Evaluator for the bundled pure-Python image pipeline speed task."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from types import ModuleType

from autoevolve.eval.contract import EvalError, StageSpec
from autoevolve.eval.descriptors import SOURCE_DESCRIPTORS, source_metrics

STAGES: list[StageSpec] = [
    StageSpec(name="small-images", timeout_s=20.0),
    StageSpec(name="all-images", timeout_s=60.0),
]
GATE: str = "correct"

PACK_DIR = Path(__file__).resolve().parent
BASELINE_DIR = PACK_DIR / "baseline"
FIXTURE_DIR = PACK_DIR / "fixtures"
Image = list[list[float]]
ImageCase = tuple[str, Image]

_BASELINE_MODULE: ModuleType | None = None
_BASELINE_TIMES: dict[int, float] = {}


def _load_module(candidate_dir: Path) -> ModuleType:
    entry_path = candidate_dir / "pipeline.py"
    if not entry_path.is_file():
        raise EvalError(f"candidate is missing {entry_path.name}")
    module_name = f"_autoevolve_python_speedup_{abs(hash(entry_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {entry_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(f"candidate import failed: {exc}") from exc
    return module


def _load_cases() -> list[ImageCase]:
    raw = json.loads((FIXTURE_DIR / "images.json").read_text(encoding="utf-8"))
    cases: list[ImageCase] = []
    for item in raw["images"]:
        pixels = [[float(value) for value in row] for row in item["pixels"]]
        cases.append((str(item["name"]), pixels))
    return cases


def _load_expected() -> dict[str, float]:
    raw = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))
    return {str(item["name"]): float(item["value"]) for item in raw["outputs"]}


def _stage_cases(stage: int) -> list[ImageCase]:
    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")
    cases = _load_cases()
    return cases[:2] if stage == 0 else cases


def _copy_image(image: Image) -> Image:
    return [row[:] for row in image]


def _run_pipeline(module: ModuleType, image: Image) -> float:
    blurred = module.box_blur(image)
    magnitudes = module.sobel_magnitude(blurred)
    output = module.threshold_count(magnitudes)
    try:
        value = float(output)
    except (TypeError, ValueError) as exc:
        raise EvalError(f"candidate returned a non-numeric output: {output!r}") from exc
    if not math.isfinite(value):
        raise EvalError(f"candidate returned a non-finite output: {value!r}")
    return value


def _check_gate(module: ModuleType, cases: list[ImageCase]) -> None:
    expected = _load_expected()
    for name, image in cases:
        try:
            actual = _run_pipeline(module, _copy_image(image))
        except EvalError as exc:
            raise EvalError(f"case {name} failed: {exc.reason}") from exc
        except Exception as exc:
            raise EvalError(f"case {name} raised {type(exc).__name__}: {exc}") from exc
        target = expected[name]
        if not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-9):
            raise EvalError(f"case {name} expected {target} but candidate returned {actual}")


def _measure(module: ModuleType, cases: list[ImageCase]) -> float:
    durations: list[float] = []
    for _ in range(3):
        prepared = [(name, _copy_image(image)) for name, image in cases]
        started = time.perf_counter()
        try:
            for _, image in prepared:
                _run_pipeline(module, image)
        except Exception as exc:
            raise EvalError(f"candidate timing run failed: {exc}") from exc
        durations.append(time.perf_counter() - started)
    return max(min(durations), 1e-12)


def _baseline_module() -> ModuleType:
    global _BASELINE_MODULE
    if _BASELINE_MODULE is None:
        _BASELINE_MODULE = _load_module(BASELINE_DIR)
    return _BASELINE_MODULE


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Gate exact outputs, then measure speed against the bundled baseline."""
    cases = _stage_cases(stage)
    candidate = _load_module(candidate_dir)
    _check_gate(candidate, cases)
    if stage not in _BASELINE_TIMES:
        _BASELINE_TIMES[stage] = _measure(_baseline_module(), cases)
    candidate_seconds = _measure(candidate, cases)
    return {
        GATE: 1.0,
        "speedup": _BASELINE_TIMES[stage] / candidate_seconds,
        "candidate_ms": candidate_seconds * 1_000.0,
        **source_metrics(candidate_dir, "pipeline.py"),
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no fixed ceiling because the plateau policy governs this task."""
    return None

# Primary metric declaration consumed by the engine when locking a contract.
METRIC = "speedup"
MAXIMIZE = True


# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent. These describe the shape of the program rather than how well it
# scored, so two different approaches at the same score both survive.
DESCRIPTORS = SOURCE_DESCRIPTORS
