"""Deliberately lossy codec used to prove the exact roundtrip gate."""

from __future__ import annotations


def compress(data: bytes) -> bytes:
    """Return the input without compression."""
    return bytes(data)


def decompress(blob: bytes) -> bytes:
    """Corrupt one byte so every non-empty roundtrip is invalid."""
    output = bytearray(blob)
    if output:
        offset = len(output) // 2
        output[offset] ^= 1
    return bytes(output)
