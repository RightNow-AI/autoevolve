# Circle-packing evaluator

## What is measured

For the selected cell, place exactly `n` points in the closed unit square `[0, 1] x [0, 1]`.

- `GATE` is `coordinates_valid`.
- `METRIC` is `min_pairwise_distance`.
- `MAXIMIZE` is `True`.
- `circle_radius` is the secondary metric `d / (2 * (1 + d))`.

The evaluator contains no published coordinate table, stored record distance, or comparison
against a known solution. It derives every reported geometric metric from the returned points.

## Candidate compute contract

The candidate entry file is `solver.py` and exports a callable compatible with:

```python
def solve(n: int, deadline: float | None = None, seed: int = 0):
    ...
```

Return an iterable containing exactly `n` points. Each point must contain exactly two finite
numeric coordinates.

Candidate compute is allowed and expected. This is a continuous global optimization problem,
and a candidate may burn its whole candidate budget searching. The evaluator passes a fixed seed
and an absolute `time.monotonic()` deadline derived from the selected stage timeout while keeping
three seconds for normalization, exact verification, and reporting. It inspects the entrypoint
signature, so `solve(n)`, `solve(n, deadline)`, keyword-only deadline or seed parameters, and
variadic forms remain usable. The outer sandbox timeout still applies if a candidate ignores the
deadline.

The baseline performs seeded random restarts followed by repulsion-based local improvement. It
returns the strongest incumbent it measured when the deadline arrives. It does not contain a
cell-specific coordinate set or distance.

## Gate and arithmetic

`CONTAINMENT_TOLERANCE` is a tiny explicit allowance for roundoff introduced when coordinates
move through NumPy and Python scalar conversions. A coordinate below
`-CONTAINMENT_TOLERANCE` or above `1 + CONTAINMENT_TOLERANCE` raises `EvalError`. Coordinates
inside only that tolerance band are clamped to the exact closed square before scoring. This keeps
the tolerance from buying geometric spread.

After structural and finite-value validation, the evaluator independently checks both axes of
every point. It then builds the full squared-distance matrix with NumPy, selects the strict upper
triangle, and takes the minimum over exactly `n * (n - 1) / 2` unique pairs. It takes one square
root only after the minimum squared distance is known. Coincident points are valid geometry with
a measured minimum distance of exactly zero.

## Cells

| cell | role | point count | timeout |
| --- | --- | ---: | ---: |
| `n2-validation` | validation | 2 | 15 seconds |
| `n10-calibration` | calibration | 10 | 300 seconds |
| `n20-calibration` | calibration | 20 | 300 seconds |
| `n30-calibration` | calibration | 30 | 300 seconds |
| `n31-frontier` | frontier | 31 | 300 seconds |
| `n37-frontier` | frontier | 37 | 300 seconds |
| `n43-frontier` | frontier | 43 | 300 seconds |
| `n51-frontier` | frontier | 51 | 300 seconds |
| `n62-frontier` | frontier | 62 | 300 seconds |

## Behavior descriptors

Exactly two descriptors are returned for every gate-passing candidate:

- `boundary_point_count` counts points within `DESCRIPTOR_TOLERANCE` of at least one edge.
- `contact_pair_fraction` divides the number of pairs within `DESCRIPTOR_TOLERANCE` of the
  measured minimum distance by the total unique pair count.

Both descriptors describe the geometry of a packing rather than replacing the primary score.
They keep structurally different boundary and contact patterns in separate MAP-elites cells.

## Hardware and honesty

The evaluator and baseline require CPU Python and NumPy only. Candidate execution has no network
access. The validation cell demonstrates the gate arithmetic. Calibration cells measure whether
the search can match a proven optimum and can never support a record claim. Frontier scores remain
measured candidates until repeated and independently checked against cited bounds.
