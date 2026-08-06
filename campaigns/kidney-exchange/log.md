# Kidney exchange campaign log

Generated frontier cells have no published optimum. Results are comparisons
against the greedy baseline measured in the same evaluator process and run.

## 2026-08-05 pairs-160-frontier, agent-controlled campaign, store research-kidney

Reported: 87 transplants against a baseline of 64, on the generated cell with
160 incompatible pairs, 4 altruistic donors, cycle cap 3 and chain cap 8.

```
cycle_transplants     79      baseline_transplants  64
chain_transplants      8      baseline_cycle_count  32
total                 87      baseline_chain_count   0
mean_cycle_length  2.724      baseline_time_ms   6.559
```

Tier: improvement over the in-run baseline. NOT a literature claim. There is no
published optimum for a self-generated instance, so this figure is meaningful
only against the greedy baseline timed beside it in the same process.

The baseline packs 32 disjoint two-cycles and uses no chains at all, which is
why 64 divides exactly by 32. The candidate reaches 87 by mixing three-cycles
in (mean cycle length 2.724 across about 29 cycles) and by using the altruists,
which the baseline never touches.

The candidate is a real solver and this is worth stating precisely, because
four earlier results in this project turned out to be recall of a published
constant or an artifact of the harness. Reading `best_candidate/solver.py`
shows large neighbourhood search: it holds an incumbent packing, destroys a
random subset of 2 to 10 selected routes, greedily repacks every freed patient
over all routes rather than only the destroyed ones, and keeps the result on
improvement plus a small sideways acceptance. Disjointness is a bitmask test.
Construction is a rarity-weighted randomized greedy run repeatedly to establish
a strong incumbent first. The whole thing is deadline-aware and returns its
incumbent when the clock runs out.

None of that is novel. Large neighbourhood search on the cycle and chain
packing formulation is the standard approach to this problem and predates this
project by roughly two decades. What the run demonstrates is that the system
wrote a working solver from the problem statement, not that it found a new
method.

Open question, and the reason this entry does not say the number is good: the
true optimum of this cell is unknown. The pack's exact solver is bitmask
dynamic programming over the full option set and `_exact_result` deliberately
refuses any cell without `require_validation_shape`, so it answers only the
8-pair validation cell. 160 pairs at cycle cap 3 is well inside the range a
position-indexed ILP handles, so the optimum is computable and simply has not
been computed. Until it is, 87 is known to beat greedy and is not known to be
near optimal.
