# Golomb ruler evaluator

## What is measured

A Golomb ruler of order n is a set of n integer marks whose pairwise
differences are all distinct. This pack searches for short ones.

- `GATE` is `golomb`. It is exact and total: the order must match the cell,
  the first mark must be 0, marks must strictly increase, and all C(n,2)
  pairwise differences must be distinct. Integer arithmetic only, no
  tolerance anywhere.
- `METRIC` is `length`, the largest mark, minimized.
- Also reported: `order`, `distinct_differences`, and the two behavior
  descriptors `max_gap` and `gap_spread`, which describe a ruler's structural
  shape rather than its quality so the archive keeps diverse footholds.

## The candidate compute contract

**You may compute.** `build(order, deadline)` receives a `time.monotonic()`
deadline and may spend real CPU searching until it. Return the best valid
ruler found. A candidate that ignores the second argument still works.

The budget is 75 percent of the stage wall clock, currently 45 of 60 seconds.
Overrunning the stage timeout scores zero rather than partial credit, so a
search must watch its own deadline and return its incumbent.

numpy is available. Any integer type is accepted on return, including numpy
integers; `bool` is refused.

This is stated explicitly because it was previously true but undocumented,
and no candidate ever used a millisecond of it. On tabulated orders that made
recall the fastest route to a passing gate.

## Measured results on this pack

Order 29, all measured on this machine:

| approach | length |
|---|---|
| Erdos-Turan closed form, original seed | 1624 |
| construction plus capped randomized greedy, current seed | 986 |
| Singer perfect difference set, found by evolution in run ra1ac573da6 | 623 |
| best known in the literature | 553 [lit: OEIS A003022 and the distributed.net OGR project] |

The ordering is the honest lesson of this pack. A hand-written metaheuristic
lost to a mathematical construction that evolution discovered by itself, so
"let the candidate search" is not automatically better than "let the candidate
think". The strongest candidates will likely do both: start from a good
construction, then spend the deadline improving it.

## Cells

Orders 8, 11, and 13 are **validation cells**. Their optima are proven and
tabulated, so a result matching them is a rediscovery and says nothing about
search capability. See docs/FRONTIER.md section 4a.

Order 29 and above are **frontier cells**: optimality is unproven past order
28, so recall cannot substitute for search there.

## Fixture provenance

None. The problem is self contained and needs no data.
