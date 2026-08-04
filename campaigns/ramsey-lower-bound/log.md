# ramsey-lower-bound campaign log

Append-only. Negative results get the same block format as wins.

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
