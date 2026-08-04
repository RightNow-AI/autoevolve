# golomb-ruler campaign log

Append-only. Negative results get the same block format as wins.

## 2026-08-04 order-29, run r3b7e1b68af, local agentic workers

Cell order-29, seed length 1007, human best 553. Operator: agentic only.

Result: length 623 from the first accepted child, all 406 pairwise differences
distinct, gate passed. Three children total, best unchanged at 623.

Tier: below literature. 623 is 70 above the best known 553, so this improves
nothing. It is recorded for the comparison it supports, which is the reason
the run existed.

The comparison: Modal run rcb5428d2ed spent 3001 programs of diff and rewrite
on this exact cell and finished budget_exhausted at 623. One agentic program
reached the same number. That is the clearest measurement yet of what the
agentic operator is worth per program, and it is also the ceiling of what it
achieved here, because three agentic children did not improve on the first.

Both operators stopping at exactly 623 is worth reading as a warning rather
than a coincidence. It suggests a recalled construction rather than a search,
which is the same pattern seen at order-11 where the answer was a tabulated
constant. On the Ramsey pack the same operator did search, writing and running
a parallel program over the factorisations of 42, so the capability is real
and the question is why it was not used here.

## 2026-08-03 order-11, run r48c09efb6d

Cell order-11, seed length 221 from the Erdos-Turan construction, proven
optimum 72. Budget 60 evaluations, four parallel workers, diff operator only,
seed 101.

Result: length 72 after 1 accepted child, status target_hit, zero gate
failures. The certificate is genuine: the gate verified all 55 pairwise
differences distinct in exact integer arithmetic, and 55 is C(11,2), so the
ruler is a real optimal order-11 Golomb ruler.

Tier: matched. This is a rediscovery of a published, tabulated result, and
the mechanism was recall rather than search. The winning program returns a
literal mark list guarded by `if order == 11` and leaves the general
construction untouched. See docs/FRONTIER.md section 4a.

What this run does establish: the pack loads, its bounds register as fresh,
the exact gate accepts a true certificate, the target ends the run as
target_hit, and artifacts render. What it does not establish: any search
capability whatsoever.

Next: a frontier cell at an order with no published optimum, where recall
cannot substitute for search.
