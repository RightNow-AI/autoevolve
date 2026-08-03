# Ramsey lower-bound evaluator

## Headline problem

The diagonal Ramsey number `R(s,s)` is the smallest `N` such that every red-blue
coloring of the edges of `K_N` contains a monochromatic `K_s`. The exact value of
`R(5,5)` has resisted determination for roughly 75 years. The current published box
is `43 <= R(5,5) <= 46` [lit: Exoo, 1989; Angeltveit and McKay, 2024 and 2026].

A certificate on `n` vertices with no monochromatic `K_s` proves `R(s,s) > n`, or
equivalently `R(s,s) >= n + 1`. This pack maximizes the largest certified `n`.

## Certificate format

The candidate defines `construct(s, n_cap, deadline)` in `construct.py`. The deadline
argument is optional. The trusted producer detects the signature, so a closed-form
candidate that defines only `construct(s, n_cap)` still works. The judge supplies
`s`, `n_cap`, and the deadline from frozen cell constants. The candidate cannot choose
them and does not receive `AUTOEVOLVE_CELL`.

The returned object has exactly one of these forms. Extra or missing keys fail.

### Form A: circulant

```text
{"form": "circulant", "n": <int>, "red_diffs": [<int>, ...]}
```

`S <= n <= CAP`. `red_diffs` is strictly increasing and every entry is in
`1..n // 2`. The edge `{i,j}` is red exactly when
`min((i-j) mod n, (j-i) mod n)` is listed. Every other edge is blue. If `n` is even,
the self-inverse difference `n / 2` contributes `n / 2` undirected edges, one
incident edge per vertex.

This representation is compact and makes mutation and verification cheap by cyclic
symmetry. It also restricts the search space. A valid coloring that is not circulant
cannot be found while a candidate stays in this form. A run that only searches Form A
must say so and cannot claim to have exhausted all colorings.

### Form B: adjacency

```text
{"form": "adjacency", "n": <int>, "red_edges": [[i, j], ...]}
```

Every edge is a two-integer array with `0 <= i < j < n`. The list is strictly
increasing in lexicographic order. Listed edges are red and every unlisted edge is
blue. This general form keeps the certificate space open to non-circulant frontier
colorings.

The trusted producer snapshots the returned object into plain immutable values in
one pass, rejects `bool` where an integer is required, accepts exact numpy integers
through `operator.index()`, and writes canonical JSON atomically. The file must be
valid UTF-8 and at most 1,000,000 bytes. Candidate flags, claimed scores, comments,
precomputed clique counts, and `verified: true` are not trusted. They make the schema
wrong and fail.

## Exact gate

`GATE` is `mono_clique_free`. The judge decodes every pair into exactly one color and
checks disjointness, symmetry, self loops, and total coverage before scoring.

Gate A is an exact bitset clique search in both colors. Its limit is 5,000,000
recursive states per color. At the largest cap, the complete search tree through
depth 5 has at most `sum(C(50,k), k=0..5) = 2,369,936` states, so the production limit
cannot truncate a cell in this pack. It still fails closed if a future configuration
exhausts the limit. Gate B is independently coded and enumerates every `S`-vertex
subset from the normalized certificate without using Gate A's adjacency masks. It
accepts only when no subset is all red and no subset is all blue. There is no floating
point arithmetic and no sampling in either decision.

The exact subset counts are:

| cell | seed subsets | cap subsets |
|---|---:|---:|
| `k3-smoke` | `C(4,3) = 4` | `C(8,3) = 56` |
| `k4-climb` | `C(17,4) = 2,380` | `C(24,4) = 10,626` |
| `k5-frontier` | `C(37,5) = 435,897` | `C(50,5) = 2,118,760` |

Even the largest cell is designed as a seconds-scale exhaustive check over a little
more than two million subsets. Full enumeration is used, so no sampling or symmetry
reduction is needed at the cap. This lane did not execute the gate under its explicit
no-run instruction. The orchestrator must confirm the wall times before merge.

## Metric, descriptors, and gaming resistance

- `METRIC` is `n_vertices`, maximized. It is exactly the normalized certificate's
  vertex count.
- `mono_clique_free = 1.0` means both colors passed the exact total gate.
- Reporting metrics are `red_edge_count`, `diff_class_count`, `target_clique`,
  `certificate_persisted`, `replay_identical`, and `stage_reached`.
- The two MAP-elites behavior descriptors are `red_density` and `is_circulant`.
  They describe structure, not quality.

The main gaming risk is shrinking `n` until the clique condition becomes trivial.
The schema rejects `n < S`, so a vacuous certificate never reaches the gate. More
importantly, `n_vertices = float(n)` and `MAXIMIZE = True`. A smaller valid certificate
always scores strictly worse than a larger valid certificate. It cannot displace the
seed on fitness. Padding does not help because every added pair is colored, and an
unlisted adjacency edge is blue rather than uncovered.

## Stages and candidate compute

Stage 0, `produce-and-gate`, has a 75 second wall clock. One fresh producer process
gets at most 45 seconds. A candidate that accepts the deadline gets 33.75 seconds,
which is 75 percent of the producer timeout, and must return its incumbent before it.
The judge normalizes once and runs Gate A in both colors.

Stage 1, `replay-and-total-recheck`, has a 180 second wall clock. Two fresh producer
processes run independently. Their normalized immutable snapshots must be identical.
Gate A runs again and Gate B exhausts all subsets. A disagreement fails closed. A
qualifying canonical certificate is then written under `certificates/<cell>/` with a
content-derived file name. The persisted file is pure canonical certificate JSON and
contains no provenance or claimed score.

A timeout is a total loss, not partial credit. Search code must watch the deadline and
return its best deterministic incumbent. numpy is available. Network access is
disabled in the producer before candidate code loads.

## Cells

- `k3-smoke`: `S=3`, `CAP=8`, current ceiling certificate size 5. This is a harness
  smoke cell.
- `k4-climb`: `S=4`, `CAP=24`, current ceiling certificate size 17. This is a
  validation cell seeded by the Paley graph of order 17. Matching it reproduces
  `R(4,4) = 18` [lit: Greenwood and Gleason, 1955] and proves nothing about search.
- `k5-frontier`: `S=5`, `CAP=50`, current ceiling certificate size 45. This is the
  frontier cell. The seed is the Paley graph of order 37. A certificate above 42
  vertices would beat the published lower bound and begins only at the
  improvement-candidate tier.

Each cap is above the cited ceiling certificate size. The cap blocks denial of
service claims such as a million-vertex certificate. It does not silently reject a
small certificate that would improve or contradict the current literature box.

## Hardware and fixtures

CPU only. Python 3.11 standard library plus numpy are allowed. The problem is self
contained and has no data fixture. The committed `monochromatic` mutant returns an
all-blue graph and must fail with a named blue-clique cause. The `hostile_container`
mutant changes its `__getitem__` answer across reads and must still fail from the one
immutable snapshot.
