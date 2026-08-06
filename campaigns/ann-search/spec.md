# Approximate nearest-neighbour search campaign

## Goal and claim boundary

This campaign evolves CPU indexes for Euclidean nearest-neighbour search. A candidate
must return almost all exact neighbours before its query throughput counts. A result
applies only to its cell, CPU, software image, and run id. No stored benchmark or
published throughput enters the score.

The task cannot be solved by recalling a published record. The indexed vectors and
queries are generated for this pack, and the fastest index for one generated workload
on one measured CPU is not a tabulated answer. The tiny cell is validation only. It
proves the gate and timing harness, not frontier search ability.

## Data generator

Every cell uses `numpy.random.default_rng` with the fixed seed in this table. Cluster
centres are independent normal samples with standard deviation 6.0. Database vectors
select a centre uniformly and add independent normal noise with standard deviation
0.65. Query vectors independently select centres and add noise with the same standard
deviation. Arrays are stored as float32. Distances are recomputed in float64.

| cell | role | N | D | queries | k | clusters | seed | recall gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny-r100-validation` | validation | 256 | 16 | 24 | 5 | 8 | 1101 | 1/1 |
| `medium-r095-frontier` | frontier | 4096 | 32 | 96 | 10 | 48 | 2202 | 19/20 |
| `large-r090-frontier` | frontier | 12000 | 48 | 160 | 10 | 96 | 3303 | 9/10 |

Uniform high-dimensional noise is deliberately absent. It makes neighbourhoods
indistinguishable and turns approximate indexing into a degenerate task.

## Ground truth and recall gate

The evaluator generates the selected workload and computes exact top-k neighbours once
by full NumPy brute force before importing candidate code. That timed computation is
both the immutable ground-truth snapshot and the same-run exact baseline. Ground truth
is never passed to candidate code.

Candidate results are consumed once into tuples of plain integers before scoring.
Boolean values are rejected. NumPy integer scalars are accepted through
`operator.index()`. Every row must contain exactly k distinct, in-range indices. Recall
is the total exact-neighbour intersection divided by `queries * k`. The gate comparison
uses integer cross multiplication, so no float decides whether a candidate passes.

The boolean gate is `recall_gate`. The evaluator also reports the measured
`recall_at_k`. A candidate below the selected threshold raises `EvalError` and receives
no throughput score.

## Timing and metrics

`METRIC = "queries_per_second"` and `MAXIMIZE = True`. Candidate query timing includes
the complete `search()` call and one-pass materialisation of a lazy result. Input-copy
setup, recall scoring, and descriptor inspection are outside the timed interval.

The evaluator also returns:

- `index_build_seconds`, including candidate module import and `build()`.
- `candidate_search_seconds`, the measured query interval.
- `exact_queries_per_second`, from the one ground-truth brute-force pass.
- `exact_search_seconds`, the corresponding same-run baseline interval.

Index build time is not folded into query throughput, but it is always recorded so a
candidate cannot make setup cost invisible. Performance claims must cite the candidate
and exact metrics from the same run.

## Structural descriptors

MAP-elites receives two non-quality descriptors that the evaluator computes:

- `index_memory_log2` estimates bytes reachable from the object returned by `build()`
  and reports their base-2 logarithm. NumPy array payloads and ordinary container
  contents are included. Module-global caches are outside this descriptor.
- `call_diversity` is the number of distinct call targets in the mutable candidate
  source, computed by `autoevolve.eval.descriptors.source_metrics`.

These descriptors separate compact and elaborate index designs without rewarding
recall or speed directly.

## Candidate compute contract

The candidate entry file is `index.py` and defines:

```python
def build(vectors, deadline=None):
    ...


def search(index, queries, k, deadline=None):
    ...
```

`vectors` and `queries` are independent NumPy copies owned by the candidate call. A
candidate may build an index, compile helpers, and autotune during evaluation. The
evaluator derives one absolute monotonic deadline from the stage timeout with reporting
headroom. It inspects both signatures and passes the deadline positionally, by keyword,
or not at all according to the declared signature. The outer stage timeout still
applies when candidate code ignores the deadline.

Candidate modules may not define names matching any reported quality, timing, baseline,
or descriptor metric. Search must return only neighbour indices, never a claimed recall
or throughput value.

## Seed and promotion

The seed copies the database in `build()` and performs exact float64 brute force in
`search()`. It has recall 1.0 and is intentionally slow. It is not an approximate index
and contains no recalled neighbour table.

A single passing frontier score is a candidate. A performance claim requires three
completed improving seeds on the same CPU and image, followed by an independent rerun.
Budget exhaustion, a failed recall gate, or a plateau remains a negative result.
