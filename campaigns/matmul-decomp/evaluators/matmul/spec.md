# Matrix multiplication tensor evaluator

This evaluator implements the contract in `../../spec.md` for one cell selected
by `AUTOEVOLVE_CELL`. It is CPU-only, uses NumPy, and requires no network or
external data.

`solver.py` defines `solve(problem, deadline=None, seed=None)`. The evaluator
uses `inspect.signature`, passes the problem positionally, and requires the
signature to accept both the absolute monotonic deadline and deterministic cell
seed. Candidate compute is allowed and expected because producing `U`, `V`, and
`W` is the search task. Five seconds of each stage are retained for trusted gate
work. Frontier stages have a 600 second total timeout.

The return value has exactly the keys `U`, `V`, and `W`. The evaluator consumes
candidate-controlled containers once, copies coefficients into evaluator-owned
arrays, and validates the full shapes before scoring. Exact-mode coefficients
must belong to the declared real or Gaussian half-integer grid. Values outside
the grid fail with `EvalError`; they are never rounded.

The gate `tensor_identity` reconstructs every tensor entry with `numpy.einsum`.
Exact cells use scaled integer arithmetic and equality. The numeric cell uses a
per-entry forward-error bound derived from machine epsilon, rank, and the sum of
absolute reconstruction term magnitudes. `rank` is minimized.

The two returned MAP-elites descriptors are `coefficient_sparsity` and
`distinct_coefficient_values`. Only code inside the baseline solver's
EVOLVE-BLOCK markers may be mutated.
