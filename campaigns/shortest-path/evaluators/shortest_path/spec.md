# Exact shortest-path evaluator contract

## Workload

This CPU evaluator generates a synthetic directed road graph deterministically. The model is an
integer-weighted rectangular street grid with asymmetric direction costs, occasional diagonal
connectors, and arterial skip edges. There is no evaluation-time download. Graph and query seeds
are fixed in `evaluate.py`.

`small-validation` uses an 8 by 8 graph and every ordered pair of distinct vertices. The all-pairs
query set validates the harness exhaustively. `large-frontier` uses a 72 by 72 graph and 512 fixed
seeded long-distance queries. `AUTOEVOLVE_CELL` is read before candidate code loads.

## Gate and metrics

The gate is `exact_shortest_paths`. For every query, the evaluator computes an integer reference
distance with its own plain binary-heap Dijkstra. A candidate passes only when its distance equals
that reference and its path has the requested endpoints, has no repeated vertex, uses real
directed edges, and has an exact edge-weight sum equal to the distance.

The primary metric is `queries_per_second`, measured in queries per wall-clock second and
maximized. `preprocessing_seconds` reports build time separately. `query_seconds`,
`reference_query_seconds`, `reference_queries_per_second`, and `query_speedup` expose the same-run
timing evidence. There is no static target or ceiling.

No float decides the gate. Candidate values are consumed once into immutable integers and tuples.
Python and NumPy booleans are rejected. `operator.index()` accepts NumPy integer scalars.

## Candidate API

`router.py` defines `build_router(vertex_count, edges, deadline=None)`. The returned object defines
`query(source, target)` and returns exactly `(distance, path)`. The evaluator reserves 30 seconds
of its stage timeout and passes the remaining absolute `time.monotonic()` deadline when the
inspected signature accepts it. Candidates that omit the deadline still run under the outer stage
timeout.

The seed is textbook query-time Dijkstra with a binary heap. Only its EVOLVE-BLOCK region is
mutable. The two structural, non-quality descriptors are `mutable_lines` and `call_diversity`,
and both are returned by every passing evaluation.
