# Sorting network evaluator

## What is measured

A sorting network on `n` channels is a fixed sequence of compare-exchange operations. A
comparator `(i, j)` replaces the value on the lower-numbered channel with the minimum and
the value on the higher-numbered channel with the maximum. `METRIC` is `size`, the number
of comparators, minimized.

`GATE` is `sorts_all_binary_inputs`. The zero-one principle states that a comparator
network sorts every input from any totally ordered domain if and only if it sorts all
`2^n` binary inputs. The gate therefore represents all binary inputs in exact integer
bitsets, applies every normalized comparator, and rejects if any output has a one before
a zero. There is no sampling, floating point decision, or search inside the gate.

The two MAP-elites descriptors are structural schedule properties:

- `depth` is the number of parallel layers in the earliest legal schedule that preserves
  each channel's comparator order.
- `first_layer_channels` is the number of channels touched by that first parallel layer.

Both descriptor metrics are returned by `evaluate`. They describe the network shape used
for archive diversity. Only `size` decides fitness.

## Candidate format and compute contract

The candidate defines `build(channels, deadline)` in `network.py` and returns an iterable
of two-item comparator iterables. The deadline is an absolute `time.monotonic()` value. A
candidate may search until it and should return its best valid incumbent before it. The
evaluator detects the entry point signature, so `build(channels)` also works for a
closed-form construction that ignores the deadline.

The stage wall clock is 120 seconds. The candidate deadline receives 75 percent of it,
currently 90 seconds, leaving headroom for import, exhaustive verification, and process
teardown. A stage timeout is a total loss rather than partial credit.

Every comparator index must be integral, distinct, and in `0..channels - 1`. Exact numpy
integer values are accepted through `operator.index()`. `bool` is rejected explicitly
because it subclasses `int`. Reversed pairs are canonicalized because a comparator is an
unordered connection between two channels.

Candidate code runs in the same interpreter as the evaluator. The returned outer
container and every comparator container are read exactly once into tuples of plain Python
integers before validation, verification, descriptor calculation, or scoring. Every later
clause reads only that immutable snapshot. A container subclass that changes its answer on
later reads cannot split the gate from the metric.

## Cells and targets

| cell | role | channels | target |
|---|---|---:|---:|
| `n11-validation` | validation | 11 | 35 [lit: Harder, 2020] |
| `n13-frontier` | frontier | 13 | 44, strictly below 45 [lit: Dobbelaere maintained list] |
| `n16-frontier` | frontier | 16 | 59, strictly below 60 [lit: Green, 1969] |
| `n20-frontier` | frontier | 20 | 90, strictly below 91 [lit: Dobbelaere maintained list] |

The validation cell has a proven optimum. It measures whether the harness can reproduce a
published result and proves nothing about search. Size optimality is open for every
frontier cell in this pack.

For reference, the live upper records checked on 2026-08-05 were:

| n | size | depth |
|---:|---:|---:|
| 11 | 35 | 8 [lit: Harder, 2020; Dobbelaere maintained list] |
| 12 | 39 | 8 [lit: Harder, 2020; Dobbelaere maintained list] |
| 13 | 45 | 9 [lit: Juille, 1995; Dobbelaere maintained list] |
| 14 | 51 | 9 [lit: Knuth; Dobbelaere maintained list] |
| 15 | 56 | 9 [lit: Knuth; Dobbelaere maintained list] |
| 16 | 60 | 9 [lit: Green, 1969; Van Voorhis, 1972] |
| 17 | 71 | 10 [lit: Al-Haj Baddar, 2009; Ehlers and Muller, 2014] |
| 18 | 77 | 11 [lit: Dobbelaere, 2020; Al-Haj Baddar and Batcher, 2009] |
| 19 | 85 | 11 [lit: Dobbelaere, 2017; Ehlers and Muller, 2014] |
| 20 | 91 | 11 [lit: Dobbelaere, 2017; Ehlers and Muller, 2014] |

The current Dobbelaere and Wikipedia tables agree on these values. Attribution is less
clean. The maintained n=17 row credits Al-Haj Baddar for the size upper bound while its
individual-network note says Valsalam and Miikkulainen also reached size 71. Wikipedia
names Schwiebert's 2001 genetic search for the n=11 depth-8 network while an earlier
sentence broadly attributes the n<=16 depth networks to Knuth's 1973 edition. These
historical conflicts do not change the values stored in `bounds.json`.

## Seed and fixture provenance

The seed generates Batcher odd-even mergesort for the next power of two and omits
comparators that touch padded high channels. Padding conceptually uses positive-infinity
sentinels, so omitted real-to-padding comparisons are no-ops. This is a construction stated
from first principles, not a recalled best known network.

The committed `out_of_range` mutant returns a comparator that names channel `n` and must
fail before exhaustive verification with the invalid channel named.

CPU only. Python 3.11 standard library plus numpy are available. No external data fixture
is needed.

## Honesty

Matching a published size is recall or rediscovery, even when the certificate is valid.
Only a strictly smaller network is a frontier result. A strict improvement starts at the
`improvement-candidate` tier. It still requires a fresh bounds check, the exact network
committed in-repo, and independent re-verification before it can become a verified
improvement. A budget or timeout stop is reported as such and never converted into an
optimality claim.
