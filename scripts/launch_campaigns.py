"""Start research campaigns without holding the connection open.

`modal run --detach` still needs the local client to survive the whole startup
handshake, and on a flaky link that is enough to lose the run: three campaigns
died here to WinError 10054 after the app had already been created, leaving no
container behind. Spawning is the fix. Deploy the researcher once, then each
launch is a single short call that returns a call id immediately, so a dropped
connection afterwards costs nothing.

Usage:
    modal deploy scripts/modal_researcher.py
    uv run python scripts/launch_campaigns.py            # every campaign below
    uv run python scripts/launch_campaigns.py labs71     # one of them
"""

from __future__ import annotations

import sys

import modal

APP_NAME = "autoevolve-researcher"
FUNCTION_NAME = "research"

_NO_RECALL = (
    "You are searching, not recalling. Do not write an answer you believe you "
    "remember from a table or a paper. Every gate here recomputes the score from "
    "your output, so a remembered answer that is even slightly wrong scores worse "
    "than an honest search, and a remembered answer that is right teaches this "
    "project nothing. Report your best result honestly, including and especially "
    "when it falls short. An accurate shortfall is worth more here than an "
    "optimistic number."
)

CAMPAIGNS: dict[str, dict[str, object]] = {
    "labs71": {
        "evaluator": "campaigns/labs/evaluators/labs",
        "cell": "n71-frontier",
        "store_name": "research-labs71",
        "mission": (
            "Maximise the merit factor of a low autocorrelation binary sequence at "
            "this length. Energy is the sum over lags of the squared aperiodic "
            "autocorrelation and the merit factor is n*n/(2E), so lower energy is "
            "better. For odd n = 2m-1, skew symmetry sets s[m-1+l] = ((-1)**l) * "
            "s[m-1-l], which forces every odd lag correlation to zero and cuts the "
            "space from 2**n to 2**m. The baseline already searches that subspace, "
            "so beating it needs better search: self avoiding walks, tabu tenure "
            "tuned to the length, memetic restarts, and above all an incremental "
            "flip evaluation that updates the correlation table in O(n) rather than "
            "recomputing it in O(n squared), because that multiplies how many flips "
            "fit in the budget. You may search the full space too, and on a long run "
            "that is where a genuinely new record would come from. " + _NO_RECALL
        ),
    },
    # The 4x4 target, named explicitly. An earlier launch omitted the cell and
    # silently ran the 2x2 validation cell for its whole budget.
    "matmul4x4": {
        "evaluator": "campaigns/matmul-decomp/evaluators/matmul",
        "cell": "4x4-complex-r48-frontier",
        "store_name": "research-matmul4x4",
        "mission": (
            "Find a bilinear decomposition of the 4x4 by 4x4 matrix multiplication "
            "tensor in as few scalar multiplications as possible. The target is 48, "
            "which is known to be achievable over the complex numbers, so this is a "
            "replication rather than a lottery. Strassen recursion gives 49 and has "
            "stood since 1969. The gate reconstructs the whole tensor from your U, V "
            "and W and demands exact equality, so an almost correct decomposition "
            "scores zero. Worth your budget: alternating least squares over the three "
            "factor matrices with many restarts, then discrete rounding onto the "
            "allowed coefficient set followed by an exact repair pass; nonlinear least "
            "squares with a regulariser pushing coefficients toward that set; and "
            "exploiting symmetry, because good decompositions tend to be structured "
            "rather than random. A valid rank 49 that is not simply Strassen recursion "
            "is already a real result worth recording before you push for 48. "
            + _NO_RECALL
        ),
    },
    # Diagnostic. n=31 matched the published packing exactly, and this asks
    # whether that came from searching or from remembering.
    "circle43": {
        "evaluator": "campaigns/circle-packing/evaluators/circlepack",
        "cell": "n43-frontier",
        "store_name": "research-circle43",
        "mission": (
            "Place n points in the unit square maximising the minimum pairwise "
            "distance. This is continuous global optimisation with an enormous number "
            "of local optima, so method matters more than effort. What works: "
            "billiards or perturbation plus repulsion until contacts stabilise, then a "
            "local polish that raises the true minimum distance; formulating the local "
            "improvement as maximise d subject to pairwise separation and solving a "
            "sequence of linear programs around the current contact graph; and "
            "restarting from structured configurations such as hexagonal and square "
            "lattice fragments rather than only from random points, because good "
            "packings tend to be lattice fragments with defects. Watch the contact "
            "graph: in a locally optimal packing most points are pinned, and a point "
            "with no contacts is a rattler that moves for free, which is often where "
            "an improvement hides. " + _NO_RECALL
        ),
    },
}


def main() -> None:
    requested = sys.argv[1:] or list(CAMPAIGNS)
    unknown = [name for name in requested if name not in CAMPAIGNS]
    if unknown:
        raise SystemExit(f"unknown campaign(s): {', '.join(unknown)}")

    research = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    for name in requested:
        spec = CAMPAIGNS[name]
        call = research.spawn(
            evaluator=spec["evaluator"],
            mission=spec["mission"],
            cell=spec["cell"],
            store_name=spec["store_name"],
            hours=5.0,
            rounds=10,
        )
        print(f"{name}: spawned {call.object_id} -> store {spec['store_name']}")


if __name__ == "__main__":
    main()
