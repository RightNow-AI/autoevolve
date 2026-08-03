from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import EvalError

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evaluators" / "python-speedup"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_gate_passes_and_speedup_is_near_one() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_python_speedup_evaluator")
    scores = evaluator.evaluate(PACK / "baseline", stage=0)
    assert scores["correct"] == 1.0
    assert 0.5 <= scores["speedup"] <= 2.0
    assert scores["candidate_ms"] > 0.0
    full_scores = evaluator.evaluate(PACK / "baseline", stage=1)
    assert full_scores["correct"] == 1.0


def test_wrong_threshold_mutant_names_the_failed_case() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_python_speedup_wrong")
    mutant = PACK / "fixtures" / "mutants" / "wrong_threshold"
    with pytest.raises(EvalError, match="synthetic-24"):
        evaluator.evaluate(mutant, stage=0)


def test_expected_outputs_match_fresh_baseline_recomputation() -> None:
    baseline = _load_module(PACK / "baseline" / "pipeline.py", "test_speedup_baseline")
    images = json.loads((PACK / "fixtures" / "images.json").read_text(encoding="utf-8"))
    expected_data = json.loads(
        (PACK / "fixtures" / "expected.json").read_text(encoding="utf-8")
    )
    expected = {item["name"]: item["value"] for item in expected_data["outputs"]}
    actual: dict[str, int] = {}
    for case in images["images"]:
        blurred = baseline.box_blur(case["pixels"])
        magnitudes = baseline.sobel_magnitude(blurred)
        actual[case["name"]] = baseline.threshold_count(magnitudes)
    assert actual == expected


def test_fixture_regeneration_is_byte_identical(tmp_path: Path) -> None:
    generator = _load_module(
        PACK / "fixtures" / "make_fixtures.py",
        "test_python_speedup_fixture_generator",
    )
    generator.write_fixtures(tmp_path)
    for name in ("images.json", "expected.json"):
        assert (tmp_path / name).read_bytes() == (PACK / "fixtures" / name).read_bytes()
