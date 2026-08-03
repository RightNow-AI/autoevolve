# Superpermutation frontier evaluator

## Problem and cells

A superpermutation on `n` symbols is a string that contains every permutation of those symbols as
a contiguous substring. This pack minimizes certificate length.

| cell | role | classical seed length | campaign target | cited interval or optimum |
|---|---|---:|---:|---:|
| `n5` | validation | 153 | 153 | 153 [lit: Chaffin 2014, summarized by Engen and Vatter 2021] |
| `n6` | frontier | 873 | 871 | 867 to 872 [lit: Houston, Pantone, and Vatter 2018; Houston 2014] |
| `n7` | frontier | 5913 | 5905 | 5884 to 5906 [lit: Houston, Pantone, and Vatter 2018; Egan 2019] |

`n5` is a validation cell. Its optimum is published, so matching 153 may measure recall. It does
not establish search capability. `n6` and `n7` are frontier cells in this pack. Their targets beat
the cited constructions by one character. A recalled published certificate cannot hit either
target.

The bounds registry was re-read on 2026-08-04. The registry keeps lower and upper bounds as
separate literature entries. Any result that beats a cited construction still follows
`docs/FRONTIER.md`: it begins as an `improvement-candidate`, needs a fresh bounds recheck, and needs
the exact certificate committed in-repo.

## Certificate format

The candidate directory contains `builder.py` defining `build(n)` or `build(n, deadline)`. The
returned certificate has these invariants:

1. The return value has exact type `str` or `bytes`. Subclasses are rejected.
2. A `str` is encoded with strict ASCII. The resulting immutable `bytes` snapshot has length from
   `n` through 1,000,000 inclusive. A larger result is truncated once to 1,000,001 bytes by the
   build runner and then rejected by the gate.
3. Every byte is exactly one of `b"123456789"[:n]`.
4. Every one of the `n!` permutations occurs at least once as a contiguous length-`n` window.

The baseline layout is a module docstring, an `EVOLVE-BLOCK-START` line, the complete `build`
function, and an `EVOLVE-BLOCK-END` line. The Engine freezes text outside those markers in parent
files that already contain markers. It does not confine the whole candidate tree to that fence.
Evolution may add new unfenced Python helper modules. Module-level code in those files runs when a
candidate imports them. Evolution may also add data files, but the content store reads every
candidate file as UTF-8 text. Binary blobs do not survive the store round trip.

The build runner loads `builder.py` by explicit path. It appends the private candidate directory to
`sys.path`, never prepends it. Helper modules can import, but candidate files cannot shadow the
standard library. The builder process has no `PYTHONPATH`, `HOME`, or `USERPROFILE` and receives a
private `TEMP` and `TMP`. Python starts with `-P`, `-s`, and `-B`. Python-level sockets are blocked
before candidate code loads.

## Candidate compute contract

Candidates may search at evaluation time. The build runner inspects the `build` signature. A
two-argument build receives an absolute `time.monotonic()` deadline. A one-argument build still
works. The stage timeout is a total loss, not partial credit, so a search must watch the deadline
and return its best valid incumbent before it expires.

The compute budget is deliberately generous. Stage 0 gives one fresh builder process 72 seconds
inside a 90-second stage. Stage 1 gives two fresh builder processes 108 seconds each inside a
240-second stage. This leaves 18 seconds at stage 0 and 24 seconds at stage 1 for process startup,
private copies, exact verification, and teardown. The campaign proxy budget is 30 evaluations or
4 hours. The opt-in full budget is 300 evaluations or 48 hours. At least one bound always closes a
run.

Python 3.11 or newer, the standard library, and numpy are available. Python-level sockets are
blocked at evaluation time. As with the repository sandbox, native extensions that issue raw
network syscalls are outside that guarantee, so this is not a boundary for hostile untrusted code.
The evaluator is CPU-only and needs no special hardware.

## Exact verifier

`AUTOEVOLVE_CELL` is read at evaluator module import, before any candidate process exists. The only
accepted values are `n2` through `n7`. There is no default. Module import remains valid without a
cell so evaluator description and ceiling discovery work, but `evaluate()` fails closed until a
valid cell is set.

The candidate never runs in the gate process. `evaluate.py` copies the candidate to a private
temporary directory and launches `build_runner.py` in a fresh interpreter. The runner normalizes
the exact return value once and emits only an immutable byte snapshot. Every later gate clause and
every metric reads that snapshot.

`verify_cert.py` constructs the exact set of `n!` permutation byte strings. It scans the
certificate from left to right once over all length-`n` windows. Each permutation window is counted
and inserted into a set. The gate passes only when the collected set equals the expected set.
There is no sampling, floating-point gate decision, model judgment, or bounded proof search.

Stage 0 already runs the complete gate. Stage 1 rebuilds twice. Each build gets a new interpreter,
a new private candidate copy, a new empty temporary directory, and the same hash seed. The two byte
snapshots must be identical before the first is gated. Nothing from stage 0 reaches stage 1 except
the engine's recorded float metrics.

this is a cheap FILTER, not a proof of reproducibility. A candidate that caches to a hardcoded absolute path outside the directories we control still replays identically, and two independent randomized builds can coincide by chance (observed once with a 1-in-6 random suffix before the test was strengthened to 16 random digits). The reproducibility that FRONTIER.md section 2 demands for `improvement-candidate` comes from the out-of-band re-derivation on a fresh checkout, recorded in log.md -- never from stage 1 alone.

## Metrics and descriptors

`GATE` is `complete`. It is `1.0` only after the exact total verifier passes.

`METRIC` is `length`, the byte length of the gated certificate, minimized. `MAXIMIZE` is `False`.
The only way to lower the score is to return a shorter byte string that still contains all `n!`
permutations. The metric is computed with `len()` on the same immutable bytes object that passed the
gate. A longer certificate scores worse. For the 873-byte classical `n6` seed, deleting byte 0
makes one permutation absent, so the edited certificate fails rather than receiving a lower score.

Secondary metrics are exact integers cast losslessly to float:

- `n`: the selected instance.
- `target_perms`: `n!`.
- `perm_windows`: permutation windows counted with multiplicity.
- `revisits`: `perm_windows - n!`.
- `cert_fp`: a 44-bit SHA-256 prefix tying a score row to a certificate.
- `max_perm_gap`: the largest distance between consecutive permutation-window starts.
- `perm_gap_kinds`: the number of distinct such distances.

The last two are structural behavior descriptors, not the fitness target. They preserve different
overlap patterns in the MAP-elites archive. `max_perm_gap` uses 8 bins from 1 through 32.
`perm_gap_kinds` uses 8 bins from 1 through 16. Values outside a descriptor range land in an edge
bin under the normal archive rules.

## Seed construction

The seed is the classical recursive construction. Start with `S_1 = "1"`. To lift
`S_(m-1)` to `S_m`, scan `S_(m-1)` left to right. Copy each character. Whenever the length-`m-1`
window ending at that character contains every old symbol exactly once, append symbol `m` followed
by that window.

For a permutation window `w`, the inserted block is `w + m + w`. Its length-`m` windows are the
rotations `suffix_k(w) + m + prefix_(m-1-k)(w)` for `k` from 0 through `m-1`. Every permutation on
`m` symbols decomposes uniquely in that form around symbol `m`, so induction proves completeness.
The construction adds exactly `m!` characters at lift `m`. Its length is therefore
`1! + 2! + ... + n!`, which gives 153, 873, and 5913 for the three campaign cells. Each seed visits
every target permutation exactly once, so `revisits` is 0.

## Fixture provenance

The problem is self-contained and needs no external data fixture. The committed
`fixtures/mutants/missing_permutations/builder.py` returns only one permutation. It deterministically
fails with the count of missing permutations in the error.
