# ann-search campaign log

Append-only. Negative results get the same block format as wins.

## 2026-08-05 default cell, agent-controlled campaign, store research-ann2

Reported: 73,908 queries per second against an exact reference at 19,302, at
recall 1.0. That ratio is 3.83 [no-claim], marked because it compares against a
reference measured in the same run rather than against any published result.

```
recall_at_k                     1.0
queries_per_second           73,908
exact_queries_per_second     19,302
index_build_seconds          0.0028
index_memory_log2             14.01
```

Tier: improvement over the in-run baseline. Not a literature claim, and not
comparable to any published ANN benchmark, because the instance is generated
here and no external library was run in this environment.

The notable part is the recall. This pack gates on approximate recall and would
have accepted a much larger speedup bought by returning worse neighbours. The
winning candidate did not take that trade. Its note says "same algorithm and
exact top-k", and recall_at_k is exactly 1.0, so it returns identical answers
to the exact reference and simply computes them faster with a transposed AVX2
distance kernel plus Ofast, LTO, and removal of semantic interposition.

That makes this the cleanest of the three agent-controlled results, because
there is no accuracy quality being silently spent to buy the speed. It is also
the least novel: a SIMD blocked exact top-k kernel is what FAISS and every
serious vector search library already ship, and that ratio [no-claim] is an
ordinary payoff for vectorising an inner loop.

An earlier run of this pack (store research-ann, superseded) reported roughly
172x [no-claim] at recall 0.9025. It is withdrawn and must not be quoted. It was
produced under an evaluator path typo of mine that pointed the researcher at
`evaluators/annsearch` instead of `evaluators/ann`, so the describe step failed,
the metric silently defaulted, and no `best_candidate/` was ever written. There
is no source to read for it, and with a seeded query set memorisation would
have produced exactly that shape of number. The typo now raises instead of
defaulting.
