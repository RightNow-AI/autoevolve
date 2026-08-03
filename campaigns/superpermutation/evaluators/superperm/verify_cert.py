"""Exact, total verification for a superpermutation certificate."""

from __future__ import annotations

import hashlib
import itertools
import math

from autoevolve.eval.contract import EvalError

MAX_CERT_BYTES = 1_000_000

_bytes = bytes
_float = float
_frozenset = frozenset
_int = int
_len = len
_range = range
_set = set
_type = type


def verify_certificate(certificate: bytes, n: int) -> dict[str, float]:
    """Gate one immutable byte snapshot and return metrics derived from it."""

    if _type(certificate) is not bytes:
        raise EvalError("internal error: certificate snapshot is not exact bytes")
    length = _len(certificate)
    if length < n:
        raise EvalError(f"certificate length {length} is shorter than n = {n}")
    if length > MAX_CERT_BYTES:
        raise EvalError(
            f"certificate length exceeds the {MAX_CERT_BYTES}-byte evaluator limit"
        )

    alphabet = b"123456789"[:n]
    allowed = _frozenset(alphabet)
    for index, value in enumerate(certificate):
        if value not in allowed:
            raise EvalError(
                f"certificate byte {index} is {value}; expected one of {alphabet!r}"
            )

    expected = _frozenset(
        _bytes(permutation) for permutation in itertools.permutations(alphabet)
    )
    seen: set[bytes] = _set()
    permutation_starts: list[int] = []
    permutation_windows = 0
    for start in _range(length - n + 1):
        window = certificate[start : start + n]
        if window in expected:
            permutation_windows += 1
            permutation_starts.append(start)
            seen.add(window)

    if seen != expected:
        missing = expected.difference(seen)
        example = min(missing).decode("ascii")
        raise EvalError(
            f"certificate is missing {_len(missing)} of {_len(expected)} permutations; "
            f"first missing permutation is {example}"
        )

    gaps = [
        permutation_starts[index] - permutation_starts[index - 1]
        for index in _range(1, _len(permutation_starts))
    ]
    target_permutations = math.factorial(n)
    fingerprint = _int(hashlib.sha256(certificate).hexdigest()[:11], 16)
    return {
        "complete": 1.0,
        "length": _float(length),
        "n": _float(n),
        "target_perms": _float(target_permutations),
        "perm_windows": _float(permutation_windows),
        "revisits": _float(permutation_windows - target_permutations),
        "cert_fp": _float(fingerprint),
        "max_perm_gap": _float(max(gaps)),
        "perm_gap_kinds": _float(_len(_set(gaps))),
    }
