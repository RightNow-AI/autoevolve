# ANN search evaluator contract

The evaluator implements the campaign contract in `../../spec.md` for one selected
cell. It needs a CPU with NumPy and no network, GPU, public dataset, or external file.

The correctness gate is `recall_gate`. It compares an immutable one-pass snapshot of
candidate indices with exact brute-force neighbours generated and timed before candidate
import. The measured recall is reported as `recall_at_k`. The gate threshold is selected
by `AUTOEVOLVE_CELL` and is checked with integer arithmetic.

The primary metric is `queries_per_second`, maximized over the cell's fixed query set.
`index_build_seconds`, `candidate_search_seconds`, `exact_queries_per_second`, and
`exact_search_seconds` are reported separately. The two MAP-elites descriptors are
`index_memory_log2` and `call_diversity`.

The candidate entry file is `index.py`. It defines `build(vectors, deadline=None)` and
`search(index, queries, k, deadline=None)`. The evaluator inspects each signature and
passes an absolute monotonic deadline only when accepted. Only content inside the seed's
EVOLVE-BLOCK markers may change.
