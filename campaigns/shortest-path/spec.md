# Exact shortest-path speed campaign

## Goal and claim boundary

This campaign evolves exact query implementations for one fixed weighted directed road graph
per cell. The score is measured on the machine that runs the evaluator. There is no published
constant to recall. A performance result applies only to its cell, hardware, software image,
and run id.

The graphs are synthetic. Evaluation-time downloads would add network availability and mutable
upstream data to the contract. Instead, the evaluator uses a documented deterministic model:
an integer-weighted rectangular street grid, asymmetric travel times in each direction,
occasional diagonal connectors, and lower-cost arterial skip edges. Graph and query seeds are
fixed in `evaluate.py`.

| cell | role | workload |
| --- | --- | --- |
| `small-validation` | validation | 8 by 8 road grid and every ordered distinct vertex pair |
| `large-frontier` | frontier | 72 by 72 road grid and 512 fixed seeded long-distance queries |

`AUTOEVOLVE_CELL` selects the cell before candidate code loads. An unknown value fails closed.

## Exact gate

The gate is `exact_shortest_paths`. The evaluator runs its own plain binary-heap Dijkstra for
every query before candidate import. Candidate distance values must equal those reference
integers exactly. Candidate paths must start and end at the query endpoints, contain no repeated
vertex, use only directed edges in the graph, and have an exact integer edge sum equal to the
returned distance.

Every candidate result is consumed once into immutable integer and tuple primitives. Every gate
clause reads only that snapshot. Python and NumPy booleans are rejected where integers are
required. Other integer-like values are normalized with `operator.index()`, which accepts NumPy
integer scalars without accepting floats.

The candidate module may not define any name that matches a reported metric. The evaluator
computes every metric itself.

## Metric and timing

`METRIC = "queries_per_second"` and `MAXIMIZE = True`. The metric is the fixed query count
divided by the total wall-clock time for all candidate query calls and one-pass result
normalization. Gate verification runs after that immutable snapshot is complete.

The evaluator times its plain Dijkstra reference in the same interpreter, on the same graph and
query set, during the same run. It reports `reference_query_seconds`,
`reference_queries_per_second`, `query_seconds`, and `query_speedup`. No stored or remembered
benchmark enters the score.

Candidate preprocessing is timed separately as `preprocessing_seconds`. Lazy preprocessing done
inside the first query remains inside `query_seconds`, so it cannot disappear from both metrics.

## Candidate compute contract

The candidate entrypoint is `router.py` and exports:

```python
def build_router(
    vertex_count: int,
    edges: tuple[tuple[int, int, int], ...],
    deadline: float | None = None,
) -> object:
    ...
```

The returned object must provide `query(source: int, target: int)`. Each query returns exactly
two values: an integer distance and an iterable of vertex integers describing the path.

Candidates may preprocess and autotune. The evaluator derives an absolute `time.monotonic()`
deadline from the stage timeout while reserving 30 seconds for query timing, exact checks, and
reporting. It inspects `build_router` and omits the deadline for a candidate that does not accept
it. The outer stage timeout still applies to a candidate that ignores the deadline.

## Seed and descriptors

The seed is textbook Dijkstra with a binary heap, written from first principles. It builds an
adjacency list once and performs an ordinary query-time search. It contains no tuned library,
landmark heuristic, bidirectional search, contraction hierarchy, or arc flag.

MAP-elites uses `mutable_lines` and `call_diversity` from
`autoevolve.eval.descriptors`. They describe candidate source structure, not quality, and both
are returned by every gate-passing evaluation.

## Promotion and honesty

The small cell validates the entire harness with an exhaustive all-pairs query set. It is not
evidence about frontier speed. A large-cell score is a candidate. A performance claim requires
three improving completed seeds on the same hardware and software image, followed by an
independent rerun. Budget exhaustion and plateau are negative results, not target claims.
