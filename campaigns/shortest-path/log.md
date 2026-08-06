# shortest-path campaign log

Append-only. Negative results get the same block format as wins.

## 2026-08-05 large cell, agent-controlled campaign, store research-path-large

Reported: query speedup 154.46 against the in-run reference Dijkstra, on a
generated graph of 5,184 vertices and 23,050 edges.

```
query_speedup                154.46
reference_queries_per_second 344.05
query_seconds (512 queries)   0.0096
preprocessing_seconds         30.90
validation_all_pairs           0.0
```

Tier: improvement over the in-run baseline, with a confound that has to be
stated before the number is quoted anywhere.

The agent's own note for the winning round reads: "direct two-argument native
FASTCALL installed on instance, eliminating Python frame and capsule argument."
The reference is a Python Dijkstra running at 344 queries per second. So a
large and unseparated share of this 154x is native compiled code beating
interpreted code, not a better routing algorithm beating a worse one. The
honest sentence is that queries got 154x faster, not that we found a faster
shortest path algorithm.

The algorithmic content that is present is also not novel. Preprocessing for
30.9 seconds to make queries near instant is what contraction hierarchies
(Geisberger, Sanders, Schultes, Delling 2008) and ALT landmark A* (Goldberg and
Harrelson 2005) have done for years, and both are standard in production
routing.

Correctness scope: `validation_all_pairs` is 0.0, meaning the exhaustive
all-pairs cross-check did not run on this cell, which is expected because it is
quadratic and this cell is large. Every one of the 512 measured queries was
still gate-checked for exact distance agreement with the reference and for path
validity, so the result is verified on what was measured and is not verified
beyond it.

To separate language from algorithm, the next run needs a native reference
Dijkstra rather than a Python one. Until that exists this pack cannot support
an algorithmic claim.
