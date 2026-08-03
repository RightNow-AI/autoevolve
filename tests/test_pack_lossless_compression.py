from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from autoevolve.eval.contract import EvalError, StageSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "evaluators" / "lossless-compression"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_candidate(candidate_dir: Path, source: str) -> Path:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "codec.py").write_text(source, encoding="utf-8")
    return candidate_dir


@pytest.mark.parametrize("stage", [0, 1])
def test_baseline_passes_lossless_gate_at_every_stage(stage: int) -> None:
    evaluator = _load_module(
        PACK / "evaluate.py",
        f"test_lossless_compression_baseline_{stage}",
    )
    scores = evaluator.evaluate(PACK / "baseline", stage=stage)
    assert scores[evaluator.GATE] == 1.0
    assert scores[evaluator.METRIC] > 0.0
    assert scores["compressed_bytes"] > 0.0
    assert scores["original_bytes"] > 0.0


def test_stage_zero_contains_primary_metric() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lossless_stage_zero")
    scores = evaluator.evaluate(PACK / "baseline", stage=0)
    assert evaluator.METRIC == "compression_ratio"
    assert evaluator.MAXIMIZE is True
    assert evaluator.METRIC in scores


def test_lossy_mutant_names_sample_and_first_differing_offset() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lossless_mutant")
    mutant = PACK / "fixtures" / "mutants" / "lossy"
    with pytest.raises(
        EvalError,
        match=r"sample near-random\.txt failed lossless gate at byte offset \d+",
    ):
        evaluator.evaluate(mutant, stage=0)


def test_fixture_regeneration_is_byte_identical(tmp_path: Path) -> None:
    generator = _load_module(
        PACK / "fixtures" / "make_fixtures.py",
        "test_lossless_fixture_generator",
    )
    generator.write_fixtures(tmp_path)
    for name in generator.FIXTURE_NAMES:
        assert (tmp_path / name).read_bytes() == (
            PACK / "fixtures" / "corpus" / name
        ).read_bytes()


def test_pack_metadata_markers_and_spec() -> None:
    evaluator = _load_module(PACK / "evaluate.py", "test_lossless_metadata")
    assert evaluator.STAGES
    assert all(type(stage) is StageSpec for stage in evaluator.STAGES)
    assert all(stage.timeout_s > 0.0 for stage in evaluator.STAGES)

    baseline_source = (PACK / "baseline" / "codec.py").read_text(encoding="utf-8")
    assert "# EVOLVE-BLOCK-START" in baseline_source
    assert "# EVOLVE-BLOCK-END" in baseline_source

    spec_text = (PACK / "spec.md").read_text(encoding="utf-8")
    assert evaluator.GATE in spec_text
    assert evaluator.METRIC in spec_text
    assert evaluator.ceiling() is None


def test_hostile_list_subclass_cannot_change_gate_answers(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path / "hostile",
        """\
class HostileList(list):
    def __init__(self, values):
        super().__init__(values)
        self.reads = 0

    def __getitem__(self, index):
        value = super().__getitem__(index)
        self.reads += 1
        return value if self.reads % 2 else value ^ 1


def compress(data: bytes) -> bytes:
    return bytes(data)


def decompress(blob: bytes) -> bytes:
    return HostileList(blob)
""",
    )
    hostile_module = _load_module(candidate / "codec.py", "test_hostile_container")
    hostile = hostile_module.HostileList([65])
    assert hostile[0] != hostile[0]

    evaluator = _load_module(PACK / "evaluate.py", "test_lossless_hostile")
    with pytest.raises(EvalError, match="decompress must return exact bytes"):
        evaluator.evaluate(candidate, stage=0)


def test_forbidden_compression_import_names_module(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path / "zlib_candidate",
        """\
import zlib


def compress(data: bytes) -> bytes:
    return zlib.compress(data)


def decompress(blob: bytes) -> bytes:
    return zlib.decompress(blob)
""",
    )
    evaluator = _load_module(PACK / "evaluate.py", "test_lossless_zlib")
    with pytest.raises(EvalError, match="forbidden compression module zlib"):
        evaluator.evaluate(candidate, stage=0)


def test_forbidden_file_access_is_rejected(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path / "file_candidate",
        """\
def compress(data: bytes) -> bytes:
    with open("corpus.txt", "rb") as corpus:
        return corpus.read()


def decompress(blob: bytes) -> bytes:
    return blob
""",
    )
    evaluator = _load_module(PACK / "evaluate.py", "test_lossless_file_access")
    with pytest.raises(EvalError, match="forbidden file access open"):
        evaluator.evaluate(candidate, stage=0)
