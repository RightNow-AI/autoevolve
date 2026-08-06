# Solomon-format CVRPTW evaluator

## Contract

The selected fixture describes depot 0, a bounded identical fleet, vehicle
capacity, customer coordinates and demands, and one service window per stop.
The candidate receives plain data and returns explicit depot-to-depot routes.

```python
def solve(instance, deadline=None, seed=0):
    return {"routes": [[0, ..., 0], ...]}
```

The result mapping has exactly the key `routes`. Routes contain integer-like
ids only. The evaluator snapshots the mapping, outer route collection, and each
route exactly once before the gate reads them.

The preferred signature receives an absolute monotonic deadline and a stable
cell seed. Signature inspection preserves compatibility with an older
`solve(instance)` implementation. The outer sandbox timeout remains final.

## Objective and metrics

Solomon ordering is lexicographic: minimize vehicle count first, then minimize
distance for equal vehicle counts. The canonical comparison key is
`(vehicle_count, total_distance)`.

The evaluator declarations are:

```text
GATE = routes_feasible
METRIC = total_distance
MAXIMIZE = False
DISTANCE_CONVENTION = euclidean_double_sum_round_half_up_2dp
```

The distance metric is the sum of double-precision Euclidean route edges,
rounded once at the end to two decimal places with round-half-up behavior.
Travel time uses the same Euclidean edges before rounding. `vehicle_count` is a
separate returned metric and the first MAP-elites descriptor. The second
descriptor, `mean_route_customers`, is the number of customers divided by the
vehicle count.

Because the scalar evaluator contract names distance, campaign reporting must
read both recorded metrics and apply the lexicographic key. The vehicle-count
descriptor preserves separate distance leaders instead of collapsing all fleet
sizes into one archive cell.

## Parser and fixtures

`parse_solomon_text` reads the standard Solomon text layout with a vehicle
number and capacity row followed by seven customer columns:

```text
CUST NO. XCOORD. YCOORD. DEMAND READY TIME DUE DATE SERVICE TIME
```

Ids must be unique and contiguous from depot 0. Numeric values must be finite,
time windows cannot be reversed, service and demand cannot be negative, depot
demand is zero, and no customer demand may exceed capacity.

`generated-tiny-12.txt` and `generated-100.txt` are created by the committed
deterministic generator from fixed seeds. The evaluator compares their full
text with regenerated text before parsing. Public Solomon and
Gehring-Homberger fixtures are read from subdirectories created by the Modal
fetch entrypoint. There is no network code in this module.

## Exact feasibility gate

The trusted gate validates the complete route structure and rejects empty
routes, missing depot endpoints, a depot in customer position, unknown ids,
duplicates, missing customers, excess fleet use, capacity overflow, a late
customer, or a late depot return.

For each route, time starts at the depot opening. Each leg adds unrounded
Euclidean travel. Service starts at the later of arrival and the stop opening,
which implements waiting. Starting service after the closing time is rejected.
Service duration is then added before the next leg.

The evaluator computes every reported metric after the gate. Candidate modules
may not declare names that collide with the gate, score, or descriptor metrics.

## Baseline

The mutable region in `baseline/solver.py` is fenced with EVOLVE-BLOCK markers.
It builds multiple deterministic and seeded incumbents with savings and nearest
neighbor construction. It then searches with route merge, relocate, and
within-route reversal moves, accepts only lexicographic improvements, checks the
deadline, and returns its incumbent.

## Claim boundary

The generated fixtures have no published answer. Public instance results are
valid only for this parser, gate, distance convention, cell, run id, program id,
and recorded stop reason. This evaluator contains no best-known objective value
and no known solution route.
