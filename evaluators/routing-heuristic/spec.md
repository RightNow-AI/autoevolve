# Routing heuristic evaluator

## Task

The candidate constructs one Euclidean cycle for each point set. The bundled baseline
starts at point zero and repeatedly selects the nearest unvisited point.

## Metrics and gate

The `valid` gate requires the returned tour to be a list and an exact permutation of
`range(n)`. Every selected instance must pass before any score is returned. A failure
raises `EvalError` and names the instance.

The headline metric is `tour_cost`. It is the sum of Euclidean cycle lengths across
the selected instances in fixture coordinate units. `mean_cost` is the arithmetic
mean cycle length in the same units. The exact scorer includes the closing edge from
the final point to the first point.

The target semantics for `tour_cost` are minimize. The engine records direction in
the locked contract and handles minimization. Stage 0 uses the three smallest
instances and has a 20 second timeout. Stage 1 uses all eight instances and has a
60 second timeout.

`ceiling()` returns `None`. This evaluator does not compute optimal tours and does not
publish a fake optimum.

## Hardware and dependencies

This evaluator needs only a CPU and the Python standard library. Costs are computed
in process on the current machine.

## Fixture provenance

`instances.json` contains eight seeded uniform point sets with sizes 30, 40, 50, 60,
75, 90, 105, and 120. Coordinates are uniform on `[0, 1000]` and rounded to nine
decimal places. Python `random.Random` seed `314159` generates the data.

Regenerate the fixture with:

```text
python evaluators/routing-heuristic/fixtures/make_fixtures.py
```

The script is deterministic and rewrites byte-identical JSON.

## Candidate guidance

Agents may change only code between `# EVOLVE-BLOCK-START` and
`# EVOLVE-BLOCK-END` in `heuristic.py`. They must preserve
`build_tour(points: list[tuple[float, float]]) -> list[int]`.
