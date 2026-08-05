# ramsey-lower-bound campaign log

Append-only. Negative results get the same block format as wins.

## 2026-08-04, extension result: both 42 vertex certificates are dead ends

Negative result, proved rather than searched.

Adding a 43rd vertex to a K5-free 42 vertex colouring reduces exactly. Join the
new vertex red to a set R and blue to the complement B. A monochromatic K5 that
avoids the new vertex already lives in the 42 vertex graph, which has none. A
red K5 through it is the new vertex plus a red K4 lying inside R, and a blue K5
through it is a blue K4 inside B. So an extension exists if and only if the 42
vertices split into R with no red K4 and B with no blue K4.

That is a boolean problem in 42 variables, one clause per monochromatic K4.

| certificate | red K4s | blue K4s | extends to 43 |
|---|---|---|---|
| n42-79e15fd6c25d1729 | 1169 | 1145 | no, UNSAT |
| n42-ffb3b30e31b519e8 | 1164 | 1145 | no, UNSAT |

`extend_certificate.py` decides this with DPLL and unit propagation, so UNSAT
is a proof for that certificate and not a failed search.
`extend_certificate_selftest.py` checks the solver against five cases with
known answers and against exhaustive enumeration on 300 random instances, all
of which agree.

This says nothing about `R(5,5)` itself. It says these two graphs do not
extend. Other 42 vertex K5-free colourings may. The search therefore needs
structurally different 42 vertex certificates rather than more copies of the
ones already found, and fewer monochromatic K4s means fewer constraints on the
split. That redirection is now in the run's discovery ledger.

## 2026-08-04, k5-frontier, run r6ae4654984, local agentic workers

Best: 42 vertices. Seed: 37. Operator: agentic. Matches literature.

Tier: matched. A 42 vertex two-colouring with no monochromatic K5 witnesses
`R(5,5) >= 43`, which is exactly the published lower bound [lit: Exoo, 1989].
This equals that bound. It does not improve it. Improving it needs 43
vertices.

Two distinct certificates were produced by two different programs and both are
in the repo:

- `certificates/k5-frontier/n42-79e15fd6c25d1729.json`, program p3f5515a557,
  436 red edges and 425 blue of 861 pairs
- `certificates/k5-frontier/n42-ffb3b30e31b519e8.json`, program pc5c7e51728,
  426 red edges and 435 blue of 861 pairs

Both passed the pack's stage 1: replayed in a fresh interpreter with identical
snapshots required, then an exhaustive check over every 5-subset that had to
agree with two independent fast verifiers. Both were then re-checked by a
verifier written separately from this pack, sharing none of its code, which
brute forced all 850,668 subsets of size 5 in each and found no monochromatic
K5 in either colour.

Whether either graph is isomorphic to Exoo's is not established here and is
not claimed. What is established is that two valid 42 vertex certificates
exist in this repo and can be rechecked by anyone in about a minute.

The method matters more than the number. The agent did not recall a
construction: it wrote a parallel search over the factorisations of 42, ran
`search42.py 7 6`, `14 3`, `21 2`, `6 7` and `3 14` concurrently alongside
`run42.py` and `run_sym.py`, and produced two structurally different answers.
A recalled graph would have produced one. This was only possible because the
agentic operator was given the ability to execute code the same day; before
that it had never produced a single accepted child.

## 2026-08-04, k5-frontier, run r2dcea50679

Best: 41 vertices. Seed: 37. Operators: diff and rewrite. Below literature.

Tier: below literature. 41 vertices establishes R(5,5) >= 42. The published
lower bound is R(5,5) >= 43 [lit: Exoo, 1989], which needs 42 vertices to
match and 43 to improve. This result is not a match and not an improvement.
It is recorded because a campaign log that only holds wins is not evidence of
anything.

The number is measured at the last stage, not screened at stage 0. That means
the certificate was re-derived in a fresh interpreter and required to come
back identical, then checked exhaustively across every 5-subset of the 41
vertices by a verifier that had to agree with two independent fast ones or the
result would have been rejected closed. 41 programs passed that full gate.

Why this run existed at all is worth recording. The pack had previously sat at
zero programs for a day, not because the search was weak but because the
describe probe withheld AUTOEVOLVE_CELL, so a pack that selects its instance
at import time could not be loaded. See docs/reviews/2026-08-04-live-run-defects.md.

`k4-climb` is a validation cell. Matching 17 vertices only reproduces the published
`R(4,4) = 18` result [lit: Greenwood and Gleason, 1955]. It is not evidence that the
search can improve a frontier.

`k5-frontier` is the open cell. Any result above 42 vertices must remain an
improvement-candidate until its canonical certificate is independently rechecked and
the literature bounds are re-read after the run.

## 2026-08-05 exhaustive: the circulant family at n=43 is empty

Negative result, decided rather than searched, and validated by a control.

For prime n a circulant colouring is determined by a subset of the (n-1)/2
difference classes, and a circulant is vertex transitive, so a monochromatic K5
exists exactly when one exists through vertex 0. That reduces each candidate to
finding a K4 in an induced subgraph on about (n-1)/2 vertices, which makes the
whole family decidable.

| n | subsets tested | certificates found |
|---|---|---|
| 37 (control) | 262,144 | 110 |
| 43 (target) | 2,097,152 | 0 |

The control is what makes the target believable. At n = 37 the sweep
rediscovers 110 K5-free circulant colourings, which is expected because the
Paley graph of order 37 is circulant and K5-free and is this pack's own seed. A
sweep that could not find those would say nothing when it returned zero.

So no circulant colouring on 43 vertices avoids a monochromatic K5. This does
not improve `R(5,5) >= 43` and it is not claimed as novel: circulant graphs are
the first family anyone searches for these bounds, so the absence is very
likely already known to the people who set the record. What it does here is
close the family off with an exhaustive check in this repository, and redirect
the search: any 43 vertex certificate must be less symmetric than a circulant.

That redirection is consistent with what the general search is seeing. The
batched GPU annealer, which searches the unrestricted space, reached 2
monochromatic K5s at n = 43 across 16384 chains and plateaued there.

Reproduce with `modal run campaigns/ramsey-lower-bound/modal_circulant43.py
--n 43 --shard-count 64`, and check the tool first with `--n 37`.
