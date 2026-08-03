import os

from autoevolve.eval.contract import EvalError


def solve(xs: list[int]) -> list[int]:
    canary = os.environ.get("SECRET_CANARY")
    if canary is not None:
        raise EvalError(f"SECRET_CANARY leaked into sandbox: {canary}")
    return sorted(xs)
