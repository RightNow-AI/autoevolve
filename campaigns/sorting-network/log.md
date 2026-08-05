# sorting-network campaign log

Append-only. Validation rediscoveries, frontier candidates, gate failures, and negative
results use the same run block format.

## 2026-08-05 n13-frontier, run raed4a12e6d, Modal diff and rewrite

Best: 45 comparators. Seed: Batcher odd-even mergesort. Tier: recall, not a
result.

45 matches the best published size for 13 channels [lit: Juille, 1995]. The
network is valid: the gate verifies exhaustively over all 8192 binary inputs by
the zero-one principle, and 50 of 54 programs passed that gate.

It is still not a search result. Program p83bf07603d hardcodes the answer as a
literal list of comparator pairs behind a plain `if channels == 13` branch,
with the surrounding Batcher construction left intact as dead code for every
other width.

The lesson is worth more than the number. This pack was built deliberately to
avoid recall: the seed is Batcher built from first principles, well above the
record, so that any improvement would have to be found. That was not enough.
Forbidding a recalled seed does not prevent recall, because the mutation
operator can inject a constant at any later point, and no gate can tell a
recalled valid network from a discovered one. They are the same object.

What does hold is the target. Cell targets are set to the best published value
minus one, so `target_hit` cannot fire on a recalled constant: there is nothing
published at 44 comparators to recall. A result at or below 44 would have to be
found. That is the only structural defence that survives contact with a model
that has read the literature.

Fourth instance of this pattern in this project, after the order-11 Golomb
ruler, the first Ramsey attempts, and an 872 character superpermutation pasted
in as a string literal.
