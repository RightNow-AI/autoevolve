# golomb-ruler campaign log

Append-only. Negative results get the same block format as wins.

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
