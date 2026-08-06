# Capacitated vehicle routing with time windows campaign

## Goal and objective

This campaign evolves solvers for the capacitated vehicle routing problem with
time windows. Every route starts at depot 0, visits customers, and returns to
depot 0. Each customer has a demand, service time, and closed time window.
Vehicles are identical and each has the instance capacity.

The objective follows the standard Solomon convention exactly and is
lexicographic:

1. Minimize `vehicle_count`.
2. Among solutions with the same vehicle count, minimize `total_distance`.

`METRIC = "total_distance"` and `MAXIMIZE = False` because the evaluator
contract has one scalar primary metric. The evaluator also returns
`vehicle_count`, and the first MAP-elites descriptor bins it in one-vehicle
increments. The archive therefore keeps a distance frontier for distinct fleet
sizes. Any campaign summary or winner decision must compare the recorded tuple
`(vehicle_count, total_distance)` in that order. A smaller distance from a
larger fleet is not a better Solomon result.

The second descriptor is `mean_route_customers`, the customer count divided by
the number of nonempty routes. Both descriptor names are present in every
gate-passing metric dictionary.

## Distance convention

`DISTANCE_CONVENTION` is
`euclidean_double_sum_round_half_up_2dp`. The evaluator computes every edge as
double-precision Euclidean distance, sums all route edges without per-edge
rounding, then rounds the complete solution distance to two decimal places
with round-half-up behavior. Time propagation uses the unrounded Euclidean edge
distance. The gate, returned metric, log template, and any campaign comparison
use this same convention.

Changing this convention creates a different benchmark contract. A result from
another convention must not be compared as if it were from this pack.

## Cells and data

Two generated fixtures are committed so evaluation and CI need no network:

| cell | role | fixture |
| --- | --- | --- |
| `tiny-12-validation` | hand-auditable validation | 12 customers generated from seed 730121 |
| `generated-100` | generated search cell | 100 customers generated from seed 730209 |

The generator is deterministic and the evaluator fails closed if either file
does not match its committed seed. The generated instances are unpublished and
carry no remembered answer.

Public frontier cells cover the C1, C2, R1, R2, RC1, and RC2 families from both
the Solomon and Gehring-Homberger sets. They read standard text files from the
same `fixtures/` directory. Those files are intentionally fetched outside the
candidate sandbox with:

```text
modal run campaigns/vrp/fetch_instances.py
```

`evaluate.py` performs no network access. Selecting a public cell before its
fixture exists fails with a fetch instruction. Every public frontier stage has
`timeout_s = 300.0`.

The source index pages are:

- https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/
- https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/

The fetch entrypoint copies only instance definitions from a pinned public
mirror. It does not fetch best-known tables or solution routes.

## Candidate compute

Candidates define `solve(instance, deadline=None, seed=0)` in `solver.py` and
return exactly:

```python
{"routes": [[0, customer_id, ..., 0], ...]}
```

The evaluator uses `inspect.signature`. Older candidates that accept only the
instance still run. A candidate that names `deadline` or `seed`, or accepts
variadic arguments, receives them. `deadline` is an absolute
`time.monotonic()` value derived from the stage timeout with three seconds of
headroom. Candidate compute, preprocessing, randomized search, and local search
are allowed. The expected behavior is to return the best incumbent before the
deadline.

The seed solver uses Clarke-Wright-style savings and nearest-neighbor
constructions, followed by route merge, relocate, and within-route reversal
moves. Every incumbent comparison uses `(vehicle_count, total_distance)`.

## Exact gate

The gate is `routes_feasible`. Candidate-controlled containers are consumed
once into immutable tuples of plain integer ids. Python and NumPy booleans are
rejected where integer ids are required. The evaluator then recomputes:

1. Every route starts and ends at depot 0.
2. Depot 0 never appears in the customer positions of a route.
3. Every customer id is valid and appears exactly once across all routes.
4. No customer is missing.
5. The route count does not exceed the instance fleet limit.
6. Each route's total demand does not exceed capacity.
7. Arrival time is propagated from the depot with unrounded Euclidean travel.
8. Early arrival waits until the window opens.
9. Service time is added before the next leg.
10. Every customer service start and the return to the depot meet their latest
    times.
11. `vehicle_count`, `total_distance`, and both descriptors are computed by the
    evaluator from the accepted routes.

Any failed clause raises `EvalError` with a route, customer, capacity, coverage,
or time-window cause. There is no partial credit.

## Promotion and honesty

The validation cell proves parser and gate behavior. The generated 100-customer
cell proves the offline search path. Public instances become benchmark evidence
only when the exact cell, run id, program id, distance convention, vehicle
count, distance, stop reason, and artifacts are retained.

`bounds.json` remains empty until the orchestrator adds cited sources. This pack
contains no published best-known distance, best-known vehicle count, or known
solution route. A model-written constant is not evidence. Budget exhaustion,
plateau, or infeasibility remains that recorded outcome and is never promoted
to a target claim.
