"""Seed certificate producer for the ramsey-lower-bound pack."""

from __future__ import annotations


# EVOLVE-BLOCK-START
def construct(s: int, n_cap: int, deadline: float | None = None) -> dict[str, object]:
    """Return a deterministic circulant Ramsey coloring certificate."""

    del deadline
    if s == 3:
        if n_cap < 4:
            raise ValueError("n_cap is too small for the K3 seed")
        return {"form": "circulant", "n": 4, "red_diffs": [1]}
    if s == 4:
        q = 17
    elif s == 5:
        q = 37
    else:
        raise ValueError(f"unsupported clique size: {s}")
    if n_cap < q:
        raise ValueError(f"n_cap is too small for the Paley seed of order {q}")
    residues = {(value * value) % q for value in range(1, q)}
    differences = sorted({min(value, q - value) for value in residues})
    return {"form": "circulant", "n": q, "red_diffs": differences}
# EVOLVE-BLOCK-END
