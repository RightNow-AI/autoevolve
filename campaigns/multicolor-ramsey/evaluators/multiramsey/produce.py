"""Trusted producer wrapper for one multicolor Ramsey certificate."""

from __future__ import annotations

import importlib.util
import inspect
import json
import operator
import os
import socket
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import NoReturn

_OS_CLOSE = os.close
_OS_DUP2 = os.dup2
_OS_EXIT = os._exit
_OS_FSYNC = os.fsync
_OS_OPEN = os.open
_OS_REPLACE = os.replace
_OS_WRITE = os.write


class _BlockedSocket:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("network disabled in multicolor Ramsey producer")


def _blocked_create_connection(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise RuntimeError("network disabled in multicolor Ramsey producer")


def _install_network_block() -> None:
    socket.socket = _BlockedSocket  # type: ignore[assignment]
    socket.create_connection = _blocked_create_connection  # type: ignore[assignment]


def _snapshot_value(raw: object, path: str) -> object:
    """Read one candidate value once and retain only plain immutable data."""

    if raw is None or type(raw) in {str, bool}:
        return raw
    if not isinstance(raw, bool):
        try:
            return int(operator.index(raw))
        except TypeError:
            pass
    if isinstance(raw, Mapping):
        try:
            items = tuple(raw.items())
        except Exception as exc:
            raise TypeError(f"{path} could not be read as a mapping: {exc}") from exc
        snapshot: dict[str, object] = {}
        for index, item in enumerate(items):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(f"{path}.items()[{index}] is not a key-value pair")
            key, value = item
            if type(key) is not str:
                raise TypeError(f"{path} key {index} must be a plain string")
            if key in snapshot:
                raise TypeError(f"{path} contains duplicate key {key!r}")
            snapshot[key] = _snapshot_value(value, f"{path}.{key}")
        return snapshot
    if isinstance(raw, list | tuple):
        try:
            items = tuple(raw)
        except Exception as exc:
            raise TypeError(f"{path} could not be read as a sequence: {exc}") from exc
        return [_snapshot_value(item, f"{path}[{index}]") for index, item in enumerate(items)]
    raise TypeError(f"{path} contains unsupported value type {type(raw).__name__}")


def snapshot_certificate(raw: object) -> dict[str, object]:
    """Snapshot a candidate return before any gate clause can inspect it."""

    snapshot = _snapshot_value(raw, "certificate")
    if not isinstance(snapshot, dict):
        raise TypeError("construct() must return a mapping")
    return snapshot


def canonical_bytes(snapshot: dict[str, object]) -> bytes:
    """Serialize the one trusted snapshot in the accepted wire format."""

    text = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )
    return text.encode("utf-8")


def _load_candidate(candidate_dir: Path) -> ModuleType:
    path = candidate_dir / "construct.py"
    if not path.is_file():
        raise FileNotFoundError("candidate is missing construct.py")
    name = f"_multiramsey_candidate_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load construct.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _call_construct(module: ModuleType, n_cap: int, deadline: float) -> object:
    construct = getattr(module, "construct", None)
    if not callable(construct):
        raise TypeError("construct.py must define callable construct()")
    signature = inspect.signature(construct)
    try:
        signature.bind(n_cap, deadline)
    except TypeError:
        try:
            signature.bind(n_cap, deadline=deadline)
        except TypeError:
            signature.bind(n_cap)
            return construct(n_cap)
        return construct(n_cap, deadline=deadline)
    return construct(n_cap, deadline)


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = Path(f"{path}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = _OS_OPEN(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += _OS_WRITE(descriptor, view[written:])
        _OS_FSYNC(descriptor)
    finally:
        _OS_CLOSE(descriptor)
    _OS_REPLACE(temporary, path)


def _run(argv: list[str]) -> int:
    if len(argv) != 5:
        raise ValueError("usage: produce.py CANDIDATE_DIR OUTPUT_PATH N_CAP SEARCH_BUDGET_S")
    candidate_dir = Path(argv[1]).resolve()
    output_path = Path(argv[2]).resolve()
    n_cap = int(argv[3])
    search_budget_s = float(argv[4])
    if search_budget_s <= 0.0:
        raise ValueError("SEARCH_BUDGET_S must be positive")
    _install_network_block()
    module = _load_candidate(candidate_dir)
    deadline = time.monotonic() + search_budget_s
    snapshot = snapshot_certificate(_call_construct(module, n_cap, deadline))
    _write_atomic(output_path, canonical_bytes(snapshot))
    return 0


def main(argv: list[str] | None = None) -> NoReturn:
    """Run once with stdout disconnected from every verdict channel."""

    null_descriptor = _OS_OPEN(os.devnull, os.O_WRONLY)
    _OS_DUP2(null_descriptor, 1)
    _OS_CLOSE(null_descriptor)
    try:
        code = _run(list(sys.argv if argv is None else argv))
    except BaseException as exc:  # noqa: BLE001 - producer must report every exit
        message = f"{type(exc).__name__}: {exc}\n".encode("utf-8", errors="replace")[-2000:]
        try:
            _OS_WRITE(2, message)
        except OSError:
            pass
        code = 1
    _OS_EXIT(code)


if __name__ == "__main__":
    main()
