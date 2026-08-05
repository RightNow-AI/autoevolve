"""First-principles seed for the multicolor Ramsey campaign."""

from __future__ import annotations

from itertools import combinations


# EVOLVE-BLOCK-START
def construct(n_cap: int, deadline: float | None = None) -> dict[str, object]:
    """Color a 5-cycle red and its complementary 5-cycle blue."""

    del deadline
    n = 5
    if n_cap < n:
        raise ValueError("n_cap is too small for the five-vertex seed")
    red_edges = {
        (0, 1),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
    }
    edge_colors = [0 if edge in red_edges else 1 for edge in combinations(range(n), 2)]
    return {"n": n, "edge_colors": edge_colors}
# EVOLVE-BLOCK-END
