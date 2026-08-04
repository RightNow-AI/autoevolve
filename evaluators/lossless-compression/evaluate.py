"""Evaluator for the bundled lossless compression task."""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from autoevolve.eval.contract import EvalError, StageSpec

STAGES: list[StageSpec] = [
    StageSpec(name="two-smaller-samples", timeout_s=15.0),
    StageSpec(name="all-samples", timeout_s=30.0),
]
GATE: str = "lossless"
METRIC: str = "compression_ratio"
MAXIMIZE: bool = True

PACK_DIR = Path(__file__).resolve().parent
CORPUS_DIR = PACK_DIR / "fixtures" / "corpus"
CORPUS_NAMES = (
    "natural-language.txt",
    "repetitive.txt",
    "near-random.txt",
)
_DENIED_COMPRESSION_MODULES = frozenset(
    {
        "_bz2",
        "_compression",
        "_lzma",
        "_zlib",
        "brotli",
        "brotlicffi",
        "bz2",
        "compression",
        "gzip",
        "lzma",
        "tarfile",
        "zipfile",
        "zlib",
    }
)
_DENIED_FILE_MODULES = frozenset({"os", "pathlib"})

CorpusSample = tuple[str, bytes]
CodecFunction = Callable[[bytes], object]


def _module_root(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


def _check_import(name: str, relative_path: str) -> None:
    root = _module_root(name)
    if root in _DENIED_COMPRESSION_MODULES:
        raise EvalError(
            f"candidate imports forbidden compression module {root} in {relative_path}"
        )
    if root in _DENIED_FILE_MODULES:
        raise EvalError(
            f"candidate imports forbidden file access module {root} in {relative_path}"
        )


def _literal_import_target(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, ast.Constant) or type(first.value) is not str:
        return None
    function = node.func
    if isinstance(function, ast.Name) and function.id == "__import__":
        return first.value
    if isinstance(function, ast.Attribute) and function.attr == "import_module":
        return first.value
    return None


def _scan_candidate(candidate_dir: Path) -> None:
    source_paths = sorted(path for path in candidate_dir.rglob("*.py") if path.is_file())
    for source_path in source_paths:
        relative_path = source_path.relative_to(candidate_dir).as_posix()
        try:
            source = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvalError(f"cannot read candidate source {relative_path}") from exc
        try:
            tree = ast.parse(source, filename=relative_path)
        except SyntaxError as exc:
            raise EvalError(
                f"candidate source {relative_path} is invalid Python: {exc.msg}"
            ) from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _check_import(alias.name, relative_path)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                _check_import(node.module, relative_path)
            elif isinstance(node, ast.Call):
                imported = _literal_import_target(node)
                if imported is not None:
                    _check_import(imported, relative_path)
                function = node.func
                if isinstance(function, ast.Name) and function.id == "open":
                    raise EvalError(
                        f"candidate uses forbidden file access open in {relative_path}"
                    )
                if isinstance(function, ast.Attribute) and function.attr in {
                    "open",
                    "read_bytes",
                    "read_text",
                }:
                    raise EvalError(
                        f"candidate uses forbidden file access {function.attr} "
                        f"in {relative_path}"
                    )


def _load_module(candidate_dir: Path, purpose: str) -> ModuleType:
    entry_path = candidate_dir / "codec.py"
    if not entry_path.is_file():
        raise EvalError("candidate is missing codec.py")

    module_name = "_autoevolve_lossless_candidate"
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load candidate entry file {entry_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvalError(
            f"candidate {purpose} namespace import raised {type(exc).__name__}"
        ) from exc
    return module


def _discard_new_imports(known_modules: frozenset[str]) -> None:
    for name in tuple(sys.modules):
        if name not in known_modules:
            sys.modules.pop(name, None)


def _load_samples(stage: int) -> list[CorpusSample]:
    if stage < 0 or stage >= len(STAGES):
        raise EvalError(f"unknown stage {stage}")

    samples: list[CorpusSample] = []
    for name in CORPUS_NAMES:
        path = CORPUS_DIR / name
        try:
            samples.append((name, path.read_bytes()))
        except OSError as exc:
            raise EvalError(f"cannot read corpus sample {name}") from exc
    samples.sort(key=lambda sample: (len(sample[1]), sample[0]))
    return samples[:2] if stage == 0 else samples


def _required_function(module: ModuleType, name: str) -> CodecFunction:
    function = getattr(module, name, None)
    if not callable(function):
        raise EvalError(f"candidate does not define callable {name}()")
    return function


def _snapshot_bytes(value: object, sample_name: str, operation: str) -> bytes:
    if type(value) is not bytes:
        raise EvalError(
            f"sample {sample_name} {operation} must return exact bytes"
        )
    return bytes(value)


def _first_differing_offset(expected: bytes, actual: bytes) -> int:
    for offset, (expected_byte, actual_byte) in enumerate(
        zip(expected, actual, strict=False)
    ):
        if expected_byte != actual_byte:
            return offset
    return min(len(expected), len(actual))


def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Gate exact roundtrips, then measure the bytes produced by the candidate."""
    samples = _load_samples(stage)
    _scan_candidate(candidate_dir)

    known_modules = frozenset(sys.modules)
    compressor_module = _load_module(candidate_dir, "compress")
    compress = _required_function(compressor_module, "compress")
    original_bytes = 0
    compressed_bytes = 0
    sample_ratios: list[float] = []
    alphabet: set[int] = set()

    for sample_index, (sample_name, sample) in enumerate(samples):
        try:
            raw_blob = compress(sample)
        except Exception as exc:
            raise EvalError(
                f"sample {sample_name} compress raised {type(exc).__name__}"
            ) from exc
        blob = _snapshot_bytes(raw_blob, sample_name, "compress")
        _discard_new_imports(known_modules)
        if sample and not blob:
            raise EvalError(
                f"sample {sample_name} compress returned an empty blob for non-empty input"
            )

        decoder_module = _load_module(candidate_dir, f"decompress_{sample_index}")
        decompress = _required_function(decoder_module, "decompress")
        try:
            raw_roundtrip = decompress(blob)
        except Exception as exc:
            raise EvalError(
                f"sample {sample_name} decompress raised {type(exc).__name__}"
            ) from exc
        roundtrip = _snapshot_bytes(raw_roundtrip, sample_name, "decompress")
        _discard_new_imports(known_modules)
        if roundtrip != sample:
            offset = _first_differing_offset(sample, roundtrip)
            raise EvalError(
                f"sample {sample_name} failed lossless gate at byte offset {offset}"
            )

        original_bytes += len(sample)
        compressed_bytes += len(blob)
        if sample and blob:
            sample_ratios.append(len(sample) / len(blob))
        alphabet.update(blob)

    if compressed_bytes <= 0:
        raise EvalError("candidate produced no compressed bytes")
    return {
        GATE: 1.0,
        METRIC: original_bytes / compressed_bytes,
        "compressed_bytes": float(compressed_bytes),
        "original_bytes": float(original_bytes),
        "ratio_spread": (max(sample_ratios) - min(sample_ratios)) if sample_ratios else 0.0,
        "blob_alphabet": len(alphabet) / 256.0,
    }


def ceiling() -> dict[str, float | str] | None:
    """Return no certified ceiling for a general lossless compressor."""
    return None


# MAP-elites behavior descriptors. Without these every candidate lands in one
# archive cell and the search degenerates into hill climbing on a single
# incumbent, which is the largest defect this project has found.
#
# Both are structural rather than quality measures. ratio_spread is the gap
# between the candidate's best and worst sample, which separates a generalist
# from a coder tuned to one kind of input; neither is better, and the archive
# should hold both. blob_alphabet is the fraction of the 256 byte values the
# compressed output uses, which separates coder families: bit-packed
# arithmetic output touches nearly every value, while a dictionary coder that
# emits literals stays inside a narrow band.
DESCRIPTORS = [
    {"name": "ratio_spread", "metric": "ratio_spread", "bins": 6, "lo": 0.0, "hi": 6.0},
    {"name": "blob_alphabet", "metric": "blob_alphabet", "bins": 8, "lo": 0.0, "hi": 1.0},
]
