# Matrix multiplication decomposition campaign

## Goal and claim boundary

This campaign searches for low-rank bilinear decompositions of small matrix
multiplication tensors. For matrices of shapes `m x k` and `k x n`, flatten the
left input, right input, and output in row-major order. The trusted tensor has

```text
T[(i, j), (j2, l), (i2, l2)] = 1
```

exactly when `j == j2`, `l == l2`, and `i == i2`. Every other entry is zero. A
candidate returns coefficient matrices `U`, `V`, and `W` with one row per scalar
multiplication. The gate reconstructs the complete tensor and accepts only when
it matches the selected cell's tensor contract.

The primary metric is `rank`, minimized. A result belongs to its exact cell,
coefficient mode, seed, run id, and evaluator revision. It is a candidate until
it passes the campaign promotion ladder. `bounds.json` is intentionally empty.
The orchestrator adds literature bounds only with cited sources.

## Cells

| cell | field | coefficient mode | target rank | stage timeout |
| --- | --- | --- | ---: | ---: |
| `2x2-real-r7-validation` | real | exact half-integers | 7 | 120 s |
| `3x3-real-r23-frontier` | real | numeric float64 | 23 | 600 s |
| `4x4-complex-r48-frontier` | complex | exact half-Gaussian integers | 48 | 600 s |

The exact cells allow real and imaginary numerators in `{-2, -1, 0, 1, 2}`
over denominator 2. Real cells additionally require a zero imaginary part. The
2x2 cell is validation: the test suite supplies an independent rank 7 witness
and also checks a generated schoolbook rank 8 witness. Published decomposition
coefficients do not appear in the campaign, evaluator, or baseline.

## Candidate compute contract

The candidate entry file is `solver.py` and defines:

```python
def solve(problem, deadline=None, seed=None):
    ...
    return {"U": U, "V": V, "W": W}
```

The evaluator inspects the signature. It passes the problem mapping
positionally and supplies both an absolute `time.monotonic()` deadline and the
cell seed. The deadline is derived from the selected stage timeout with five
seconds reserved for normalization, reconstruction, and reporting. Candidate
compute is allowed and expected. Useful approaches include alternating least
squares followed by discrete projection, direct combinatorial search over the
declared coefficient grid, and hybrid local search. Candidate code has no
network access under the evaluator sandbox.

The problem mapping contains the cell key, `m`, `k`, `n`, field, coefficient
mode, target rank, seed, and the exact grid numerators and denominator when the
cell uses exact coefficients. `U`, `V`, and `W` must have shapes
`(R, m*k)`, `(R, k*n)`, and `(R, m*n)` for one common positive `R`.

The baseline does real search. It runs bounded seeded CP-ALS attempts at the
target rank, projects to the declared grid for exact cells, and checks the full
tensor before returning an incumbent. If no searched target-rank incumbent
passes, it returns a schoolbook decomposition generated directly from the
matrix multiplication index loops. It contains no stored decomposition table.

## Gate

For exact cells, every coefficient is normalized without rounding. A value
outside the declared discrete set raises `EvalError`. The evaluator scales
accepted coefficients to Gaussian integer pairs, reconstructs the full tensor
with `numpy.einsum`, and compares integer tensors with exact equality. There is
no tolerance in an exact cell.

For the numeric cell, reconstruction uses float64. Its per-entry tolerance is a
forward-error bound. The evaluator computes the sum of absolute rank-one term
magnitudes with a second full `einsum`, derives the standard gamma factor from
machine epsilon and the number of floating-point operations, and scales the
bound by the reconstructed magnitude. No fixed decimal tolerance decides the
gate.

## MAP-elites descriptors

Every passing result reports exactly two structural descriptors:

- `coefficient_sparsity`: the fraction of all coefficient entries that are
  exactly zero
- `distinct_coefficient_values`: the number of distinct normalized coefficient
  values used across `U`, `V`, and `W`

These descriptors separate sparse formulas from coefficient-diverse formulas
without duplicating the minimized rank metric.

## Honesty

A gate pass proves only the tensor identity under the selected coefficient
contract. It does not prove novelty. Rank, target status, and any comparison
must come from a recorded run. A timeout, gate failure, plateau, or exhausted
budget is reported as such. Validation success is not a frontier claim.
