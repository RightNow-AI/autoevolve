# Equal-circle packing campaign

## Goal and claim boundary

This campaign evolves algorithms that place `n` points in the closed unit square and maximize
their minimum pairwise Euclidean distance. The primary metric is
`min_pairwise_distance`, with `MAXIMIZE = True`. For interpretation as equal circles in a unit
square, the evaluator also reports the derived radius
`r = d / (2 * (1 + d))` as `circle_radius`.

The point spread `d` is the quantity used for evolution and gating. No published coordinate set,
record distance, or frontier target is stored in this pack. `bounds.json` is intentionally empty
until the orchestrator supplies independently cited bounds.

## Cells

| cell | role | point count | stage timeout |
| --- | --- | ---: | ---: |
| `n2-validation` | elementary gate validation | 2 | 15 seconds |
| `n10-calibration` | proven-range calibration | 10 | 300 seconds |
| `n20-calibration` | proven-range calibration | 20 | 300 seconds |
| `n30-calibration` | proven-range calibration | 30 | 300 seconds |
| `n31-frontier` | frontier search | 31 | 300 seconds |
| `n37-frontier` | frontier search | 37 | 300 seconds |
| `n43-frontier` | frontier search | 43 | 300 seconds |
| `n51-frontier` | frontier search | 51 | 300 seconds |
| `n62-frontier` | frontier search | 62 | 300 seconds |

Packing is proven optimal for every `n <= 30`. The `n10-calibration`, `n20-calibration`, and
`n30-calibration` cells exist only to test whether the searcher can reach a known optimum without
storing that optimum in this pack. Any success on those cells is a matched-known-optimum result
and must never be reported as a record. Only cells above 30 are frontier searches where a genuine
improvement could exist.

`AUTOEVOLVE_CELL` selects the cell before candidate code loads. An unknown value fails closed.
All campaign targets are null so the run is governed by its explicit evaluation budget rather
than a recalled distance.

## Exact gate

The gate is `coordinates_valid`. A candidate must return exactly `n` two-dimensional points with
finite coordinates. The evaluator independently checks both axes of every point against the
closed unit square. A tiny named containment tolerance admits only floating-point representation
drift. Accepted drift is clamped back to the square before any distance is scored, so it cannot
improve the metric.

The evaluator recomputes all `n * (n - 1) / 2` pair distances from the normalized coordinates.
It computes squared distances first, takes the minimum over the unique pairs, and performs one
square root after the minimum is known. A candidate never supplies its own score.

## Candidate compute and diversity

This is a continuous global optimization problem. Candidates are expected to search and may use
their entire supplied deadline. The evaluator seeds the entrypoint and derives an absolute
monotonic deadline from the selected stage timeout with verification headroom. A legacy
entrypoint accepting only `n` remains valid.

MAP-elites receives exactly two evaluator-computed descriptors:

- `boundary_point_count`, the number of points within descriptor tolerance of any square edge.
- `contact_pair_fraction`, the fraction of unique pairs whose distance is within descriptor
  tolerance of the measured minimum.

The validation cell checks the arithmetic and gate only. A calibration score can establish search
credibility but cannot establish a record. A frontier score is a candidate, not a literature
result. A claim requires the recorded run id, repeated measured seeds, an independent rerun, and
comparison against bounds populated from cited sources.
