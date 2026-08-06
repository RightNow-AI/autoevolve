# Generated kidney exchange matching evaluator

## Why this problem matters

A patient may have a willing living donor whose kidney is medically
incompatible with them. Kidney exchange redirects donors through cycles and
chains so each participating patient receives a compatible kidney. Every edge
accepted by this evaluator represents one donor-to-patient transplant.

Cycles must be performed simultaneously, so this pack caps them at three
transplants. Chains begin with a nondirected altruistic donor and may be longer
because their transplants can proceed sequentially. Finding a maximum packing
of vertex-disjoint cycles and chains is NP-hard. Operational solvers therefore
run under a time limit and return their best incumbent. This evaluator measures
that time-limited regime on generated instances whose answers are not published.

## Generated data and sources

The graph generator follows the core Saidman model: independently draw patient
and donor ABO types, retain only pairs whose intended donation is incompatible,
and make every potential donor-to-patient edge depend on ABO compatibility and
the patient's sensitization class.

The source is Susan L. Saidman, Alvin E. Roth, Tayfun Sonmez, M. Utku Unver, and
Francis L. Delmonico, "Increasing the Opportunity of Live Kidney Donation by
Matching for Two- and Three-Way Exchanges," *Transplantation* 81(5), 2006,
773-782:

https://web.stanford.edu/~alroth/papers/SaidmanRothSonmezUnverDelmonico.Transplantation.2006.pdf

The pack uses these source-backed integer distributions:

- ABO type weights O/A/B/AB = 45/40/11/4 percent
  [lit: Saidman et al., 2006].
- Positive crossmatch probabilities for low/medium/high PRA = 5/45/90 percent
  [lit: Saidman et al., 2006].

The later methodological review by Delorme et al. describes the Saidman
generator as the standard synthetic generator and documents why its output
should not be mistaken for a calibrated modern registry:

https://doi.org/10.1016/j.cor.2022.105707

I could not verify a primary-source population share for the three PRA tiers
from the accessible Saidman paper text. This pack therefore declares its
70/20/10 low/medium/high PRA mix as an explicit modeling choice [no-claim]. It
is not attributed to Saidman et al. The choice keeps most generated patients in
the low-sensitization class while retaining a smaller hard-to-match tail.

The number of altruists and the chain caps are also pack choices, not Saidman
parameters. They are chosen to exercise chain logic without allowing an
unbounded route. The small validation cell is deterministically regenerated,
with a bounded attempt cap, until it contains at least one cycle and at least
one altruist-to-pair edge, plus at least one absent pair edge. Exhausting that
cap fails closed.

All random decisions use a cell seed and exact integer draws. No floating-point
value decides graph membership or gate validity.

## Candidate contract and deadline

The candidate defines `solve(instance, deadline=None)` in `solver.py`. The
trusted evaluator detects the callable signature with `inspect.signature`.
It always passes the generated instance positionally. It supplies `deadline`
when the function names that parameter or accepts variadic positional or
keyword arguments.

`deadline` is an absolute `time.monotonic()` value derived from the selected
stage timeout with three seconds of evaluator headroom. A candidate should
watch its clock and return its incumbent before the deadline. The outer stage
timeout is a total loss, so no timed-out partial result is scored.

The instance mapping contains:

```text
{
  "cell": <str>,
  "seed": <int>,
  "pair_count": <int>,
  "altruists": (<vertex id>, ...),
  "cycle_cap": <int>,
  "chain_cap": <int>,
  "edges": ((<patient vertex id>, ...), ...)
}
```

Rows in `edges` are indexed by donor vertex. Pair vertices are numbered from
zero through `pair_count - 1`. Altruistic donor vertices follow them. Edge
targets are always pair vertices because altruists have no patient.

The result has exactly two keys:

```text
{
  "cycles": [[pair_vertex, ...], ...],
  "chains": [[altruist_vertex, pair_vertex, ...], ...]
}
```

A cycle of `k` vertices uses `k` transplant edges, including the closing edge.
A chain of `k + 1` vertices uses `k` transplant edges. Candidate code runs in
the same interpreter as the evaluator. The evaluator therefore consumes the
mapping, each outer route collection, and every route exactly once. It retains
only immutable tuples of plain integer vertex ids and reads only that snapshot
afterward. Python booleans and numpy booleans are rejected. Integer-like numpy
scalars are accepted through `operator.index()`.

The candidate module may not define a global whose name matches any returned
metric. The evaluator rejects those names before and after `solve()` so a
candidate cannot present its own transplant count, baseline count, or
descriptor as trusted output.

## Exact gate

`GATE` is `matching_valid`. The gate checks every returned edge and vertex:

1. Every cycle contains only pair vertices, has no repeated vertex, respects
   the cell cycle cap, follows directed compatibility edges, and closes back to
   its start.
2. Every chain starts at an altruistic donor, continues only through pair
   vertices, has no repeated vertex, respects the cell chain edge cap, and
   follows directed compatibility edges.
3. All cycles and chains are vertex disjoint across the complete solution.
4. Every integer field is normalized exactly. No float decides a gate clause.

Any failure raises `EvalError` with the specific cycle, chain, edge, or reused
vertex. There is no partial credit.

## Metric, descriptors, and baseline

`METRIC` is `transplants`, and `MAXIMIZE` is `True`. It is the exact number of
donor-to-patient edges in the accepted cycles and chains. Reporting metrics
include cycle count, chain count, cycle transplants, and chain transplants.

The two MAP-elites descriptors are structural and are returned by every passing
evaluation:

- `chain_share`: chain transplant edges divided by all transplant edges
- `mean_cycle_length`: cycle transplant edges divided by the number of cycles

The evaluator reruns a straightforward greedy matcher in the same process, on
the same immutable instance, during every evaluation. It repeatedly chooses
the shortest lexicographically first available valid cycle and never builds a
chain. The returned metrics include its transplant count, cycle count, chain
count, and measured runtime. The honest comparison is only candidate versus
that in-run greedy baseline under this evaluator's fixed stage budget. It is
not a claim about a registry, an integer-program implementation, or a world
record.

The seed candidate is the same simple cycle-only greedy rule. It is deliberately
untuned and contains no recalled construction.

## Cells and ground truth

- `small-validation` has 8 incompatible pairs, 1 altruist, cycle cap 3, chain
  cap 4, and a 15 second stage [no-claim]. The pack enumerates every feasible
  cycle and chain, then solves the resulting finite set-packing problem exactly
  by dynamic programming over vertex masks.
- `pairs-80-frontier` has 80 incompatible pairs, 2 altruists, cycle cap 3,
  chain cap 6, and a 45 second stage [no-claim].
- `pairs-160-frontier` has 160 incompatible pairs, 4 altruists, cycle cap 3,
  chain cap 8, and a 60 second stage [no-claim].

The tests feed the exact small-cell solution back through the ordinary candidate
gate and require the gate's transplant count to equal the exact optimum.
Matching this small generated optimum proves that the exact solver and gate
agree. It proves correctness, not search capability. The frontier cells are
large generated instances with no published answer. Their evidence comes only
from the fixed time budget and the in-run baseline.

The theoretical ceiling is the pair count because vertex disjointness permits
at most one incoming transplant for each paired patient.

## Hardware and Modal search

The evaluator is CPU only and uses Python 3.11 standard library plus numpy.
`modal_search.py` builds a repository-HEAD-pinned image with Node 22 and Codex,
selects the requested cell through `AUTOEVOLVE_CELL`, sets
`AUTOEVOLVE_AGENT_RUNTIME=codex`, mounts the shared `autoevolve-store` volume,
uses a problem-specific store directory, checkpoints while running, and commits
the volume in a `finally` block. The founder laptop is not campaign compute.

## Honesty

Every frontier graph is generated from a committed seed, so there is no
published certificate for a model to recall. A passing result is still only a
candidate tied to its exact run id and cell. It becomes stronger evidence only
after replication and independent rechecking. Budget exhaustion, timeout, or a
failure to beat the in-run greedy baseline is reported plainly.
