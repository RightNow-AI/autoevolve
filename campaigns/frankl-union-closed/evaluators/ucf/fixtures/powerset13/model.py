"""Scale fixture for the Frankl union-closed evaluator."""


def build_family() -> dict[str, object]:
    """Return the powerset of a 13-element ground set."""

    n = 13
    return {"n": n, "sets": list(range(1 << n))}
