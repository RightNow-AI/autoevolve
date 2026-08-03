"""Isolated candidate builder used by the superpermutation evaluator."""

from __future__ import annotations

import importlib.util
import inspect
import os
import socket
import sys

_bytes = bytes
_callable = callable
_float = float
_len = len
_memoryview = memoryview
_os_close = os.close
_os_dup = os.dup
_os_dup2 = os.dup2
_os_exit = os._exit
_os_open = os.open
_os_write = os.write
_str = str
_type = type

_OUTPUT_BYTES = 1_000_001
_NETWORK_DISABLED = "network disabled in superpermutation builder"


class _BlockedSocket:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(_NETWORK_DISABLED)


def _blocked_create_connection(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise RuntimeError(_NETWORK_DISABLED)


def _load_builder(candidate_dir: str) -> object:
    path = os.path.join(candidate_dir, "builder.py")
    spec = importlib.util.spec_from_file_location("_superperm_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load builder.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.append(candidate_dir)
    spec.loader.exec_module(module)
    return module


def _emit_bytes(descriptor: int, payload: bytes) -> None:
    remaining = _memoryview(payload)
    while remaining:
        written = _os_write(descriptor, remaining)
        remaining = remaining[written:]


def _failure(saved_stderr: int, exc: BaseException) -> None:
    try:
        _os_dup2(saved_stderr, 2)
        _os_close(saved_stderr)
        message = f"build failed: {_type(exc).__name__}: {_str(exc)}\n"
        _emit_bytes(2, message.encode("utf-8", "replace")[:4096])
    finally:
        _os_exit(1)


def main() -> None:
    saved_stdout = _os_dup(1)
    saved_stderr = _os_dup(2)
    null_descriptor = _os_open(os.devnull, os.O_WRONLY)
    _os_dup2(null_descriptor, 1)
    _os_dup2(null_descriptor, 2)
    _os_close(null_descriptor)

    try:
        if _len(sys.argv) != 3:
            raise ValueError("expected candidate directory and n")
        candidate_dir = os.path.abspath(sys.argv[1])
        n = int(sys.argv[2])
        deadline = _float(os.environ["AUTOEVOLVE_BUILD_DEADLINE"])

        socket.socket = _BlockedSocket  # type: ignore[assignment]
        socket.create_connection = _blocked_create_connection  # type: ignore[assignment]

        module = _load_builder(candidate_dir)
        build = getattr(module, "build", None)
        if not _callable(build):
            raise TypeError("builder.py must define callable build()")
        if _len(inspect.signature(build).parameters) >= 2:
            raw = build(n, deadline)
        else:
            raw = build(n)

        if _type(raw) is str:
            certificate = raw.encode("ascii", "strict")
        elif _type(raw) is bytes:
            certificate = raw
        else:
            raise TypeError(
                f"build() must return exact str or bytes, got {_type(raw).__name__}"
            )
        certificate = certificate[:_OUTPUT_BYTES]
    except BaseException as exc:
        _failure(saved_stderr, exc)

    _os_dup2(saved_stdout, 1)
    _os_close(saved_stdout)
    _os_close(saved_stderr)
    _emit_bytes(1, certificate)
    _os_exit(0)


if __name__ == "__main__":
    main()
