# ramsey-lower-bound campaign log

Append-only. Negative results get the same block format as wins.

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
