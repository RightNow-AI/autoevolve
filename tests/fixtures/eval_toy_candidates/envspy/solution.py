import os

from autoevolve.eval.contract import EvalError


def solve(xs: list[int]) -> list[int]:
    canary = os.environ.get("AUTOEVOLVE_CANARY")
    if canary is not None:
        raise EvalError(f"AUTOEVOLVE_CANARY leaked into sandbox: {canary}")
    return sorted(xs)
