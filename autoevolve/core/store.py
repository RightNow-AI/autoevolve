"""Content-addressed storage for candidate source trees."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

from autoevolve.core.db import resolve_home


def _normalize_path(raw: str) -> str:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(raw)
    if (
        not normalized
        or not path.parts
        or path.is_absolute()
        or windows_path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"candidate path must be a clean relative path: {raw!r}")
    return path.as_posix()


def _validate_ref(ref: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", ref) is None:
        raise ValueError(f"invalid content reference: {ref!r}")


def _normalize_files(files: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_path, content in files.items():
        if not isinstance(content, str):
            raise TypeError(f"candidate file {raw_path!r} must contain text")
        path = _normalize_path(raw_path)
        if path in normalized:
            raise ValueError(f"duplicate normalized candidate path: {path}")
        normalized[path] = content
    return normalized


def content_ref(files: dict[str, str]) -> str:
    """Hash sorted normalized path and content pairs into one stable reference."""

    normalized = _normalize_files(files)
    payload = json.dumps(
        sorted(normalized.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ContentStore:
    """Persist and materialize immutable candidate file mappings."""

    def __init__(self, home: Path | None = None):
        self.home = resolve_home(home)
        self.root = self.home / "store"
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, files: dict[str, str]) -> str:
        """Store files idempotently and return their SHA-256 reference."""

        normalized = _normalize_files(files)
        ref = content_ref(normalized)
        target = self.root / ref
        if target.is_dir():
            return ref

        temporary = self.root / f".{ref}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=False)
        try:
            for relative, content in sorted(normalized.items()):
                destination = temporary.joinpath(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8", newline="")
            try:
                temporary.replace(target)
            except OSError:
                if not target.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return ref

    def get(self, ref: str) -> dict[str, str]:
        """Load all files for a reference as normalized relative paths."""

        _validate_ref(ref)
        source = self.root / ref
        if not source.is_dir():
            raise KeyError(f"unknown content reference: {ref}")
        files: dict[str, str] = {}
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            files[relative] = path.read_text(encoding="utf-8")
        return files

    def materialize(self, ref: str, dest: Path) -> None:
        """Write a stored candidate beneath an existing or new destination."""

        destination = Path(dest)
        destination.mkdir(parents=True, exist_ok=True)
        for relative, content in self.get(ref).items():
            path = destination.joinpath(*PurePosixPath(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")


def put(files: dict[str, str], home: Path | None = None) -> str:
    """Store files in the configured home."""

    return ContentStore(home).put(files)


def get(ref: str, home: Path | None = None) -> dict[str, str]:
    """Load files from the configured home."""

    return ContentStore(home).get(ref)


def materialize(ref: str, dest: Path, home: Path | None = None) -> None:
    """Materialize files from the configured home."""

    ContentStore(home).materialize(ref, dest)
