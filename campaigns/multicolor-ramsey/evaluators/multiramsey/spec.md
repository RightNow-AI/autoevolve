# Multicolor Ramsey lower-bound evaluator

## Headline problem

`R(K3,K4,C4,C4)` is the smallest `N` such that every four-coloring of the edges
of `K_N` contains at least one of these color-specific forbidden subgraphs:

- a red triangle `K3`
- a blue complete graph `K4`
- a green cycle `C4`
- a yellow cycle `C4`

A valid coloring on `n` vertices proves `R(K3,K4,C4,C4) > n`, equivalently
`R(K3,K4,C4,C4) >= n + 1`. The published bounds are
`49 <= R(K3,K4,C4,C4) <= 75` [lit: Wesley, 2025, arXiv 2509.03784; Boza and
Radziszowski, 2025, as cited there]. The paper improves the previous lower bound
of 43 and obtains its 48-vertex coloring with Cayley constraints over
`SmallGroup(48,4)`, which is isomorphic to `Z8 x S3` [lit: Wesley, 2025, arXiv
2509.03784].

The measured metric is `n_vertices`, maximized. It is the exact vertex count of
the candidate certificate that passes the total gate.

## Why the C4 gate uses common neighbors

A graph contains a `C4` exactly when some pair of vertices has at least two
distinct common neighbors. If `u` and `v` share distinct neighbors `x` and `y`,
then `u-x-v-y-u` is a 4-cycle. The converse follows by taking the opposite
vertices of any 4-cycle. Extra diagonal edges do not matter because the forbidden
cycle need not be induced. Green and yellow are therefore `C4`-free exactly when
every vertex pair has at most one common neighbor in that color.

## Certificate and candidate compute contract

The candidate defines `construct(n_cap, deadline)` in `construct.py`. The deadline
argument is optional. The trusted producer supplies the frozen cell cap and a
monotonic deadline. A candidate must return its deterministic incumbent before
that deadline. Timeouts fail the gate with no partial credit.

The return value has exactly this form:

```text
{"n": <int>, "edge_colors": [<int>, ...]}
```

`5 <= n <= n_cap`. The color list has exactly `C(n,2)` entries in lexicographic
pair order:

```text
(0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
```

Colors are `0=red`, `1=blue`, `2=green`, and `3=yellow`. Every entry must be an
integer in `0..3`. Boolean values are rejected even though Python makes `bool` a
subclass of `int`. Exact numpy integer scalars are accepted through
`operator.index()`.

Candidate code runs in the same interpreter as the trusted producer. The producer
therefore snapshots the complete return value once into plain JSON-compatible
primitives. Every later schema check, gate clause, metric, and persistence step
reads only the immutable normalized certificate. A container whose answers change
between reads cannot influence two different clauses.

The canonical certificate is valid UTF-8 JSON and is limited to 1,000,000 bytes.
Extra keys, missing pairs, extra pairs, invalid colors, claimed scores, and claimed
verification flags all fail.

## Exact gate and stages

`GATE` is `forbidden_subgraph_free`.

Stage 0 is `produce-and-gate`. It launches one fresh producer process, normalizes
the certificate once, reconstructs four symmetric adjacency bitsets, and verifies
that every pair has exactly one color. It then runs these exact checks:

- red triangle search by bitset neighborhood intersections
- blue `K4` search by nested bitset intersections
- green `C4` search by common-neighbor counts
- yellow `C4` search by common-neighbor counts

Stage 1 is `replay-and-exhaustive-recheck`. It launches the producer again in a
fresh interpreter and requires the two immutable certificates to be identical. It
reruns the fast gate and then uses a separately coded exhaustive verifier over the
certificate colors. That verifier enumerates every triple for red, every quadruple
for blue, and all three possible 4-cycles on every quadruple for green and yellow.
The fast and exhaustive verdicts must agree or the gate fails closed. No float and
no sampling decides validity.

Only the last stage persists a canonical certificate. The validation cell persists
at five vertices. The frontier cell persists at 48 vertices for an honest matched
result and at 49 vertices for an improvement candidate
[lit: Wesley, 2025, arXiv 2509.03784].

## Descriptors and reporting metrics

The two MAP-elites descriptors are structural and are returned by `evaluate()`:

- `red_density`, the fraction of all pairs colored red
- `distinct_color_class_sizes`, the number of distinct values among the four edge
  counts

Neither descriptor measures certificate quality. Other reporting metrics include
the four color class sizes, `certificate_persisted`, `replay_identical`, and
`stage_reached`.

## Cells

- `n5-validation` caps the certificate at five vertices. The seed colors a 5-cycle
  red and its complement blue. Both graphs are 5-cycles, so red has no triangle,
  blue has no `K4`, and the empty green and yellow graphs have no `C4`. This is a
  validation cell derived from first principles.
- `n49-frontier` caps the certificate at 49 vertices. A passing 49-vertex coloring
  would exceed the published 48-vertex certificate and begins only at the
  improvement-candidate tier.

## Search and hardware

`search.py` provides direct simulated annealing over one-edge recolorings. Its
tracked cost is exactly the sum of red triangles, blue `K4` subgraphs, green `C4`
cycles, and yellow `C4` cycles. Every recoloring delta counts only forbidden
subgraphs containing the changed edge. Restarts, accepted zero-delta moves, and an
explicit plateau move keep the search mobile. Progress is printed at startup and at
least once per minute.

The same file also provides a circulant Cayley mode over `Z_n`. That mode colors an
edge by its cyclic difference class and recolors one complete difference class per
move. It explores a much smaller structured space and does not claim to exhaust all
colorings.

Before writing any certificate, the search independently recounts all four
violation totals from scratch. A mismatch with the incremental totals rejects the
write. `modal_search.py` fans independent seeds across Modal CPU containers and
writes only fully recounted certificates to the shared `autoevolve-store` volume.

CPU only. Python 3.11 standard library plus numpy are allowed. No founder-laptop
compute is part of this pack's evidence.

## Honesty

The published lower bound 49 came from a coloring on 48 vertices. Reaching 48
vertices only matches that bound. Only a valid coloring on 49 or more vertices is
an improvement candidate. It is not born verified. It still requires a fresh bounds
check, a committed canonical certificate, the exact run id, replication, and an
independent implementation or third-party recheck before any stronger claim.
