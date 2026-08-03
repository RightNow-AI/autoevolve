# Frankl union-closed sets evaluator

## What is measured

Frankl's union-closed sets conjecture asks whether every finite union-closed
family other than the two vacuous exceptions has an element in at least half
of its member sets [lit: Frankl, 1979, as cataloged by Douglas West]. It has
been open since 1979. A valid family below one half would be a counterexample.
A false gate pass would therefore be a false claim about a 47-year-old problem.

The candidate supplies a finite family. Each set is an integer bitmask over
the ground set `{0, ..., n - 1}`. Union is bitwise OR.

- `GATE` is `union_closed_valid`. It checks the complete certificate contract
  and every unordered pair of family members, including diagonal pairs.
- `METRIC` is `max_freq_ratio`, minimized. If `m` is the family size and
  `f_max` is the largest element frequency, the metric is `f_max / m`.
- The evaluator holds the ratio as `Fraction(f_max, m)`. The counterexample
  metric `below_half` is 1 only when that exact fraction is less than
  `Fraction(1, 2)`. No float decides a gate or a counterexample claim.
- `half_margin` is the signed integer `2 * f_max - m`.
- Also reported are `max_freq`, `family_size`, `ground_set_declared`,
  `ground_set_used`, and `nmax_in_force`.

The float form of `max_freq_ratio` is only the engine's optimization signal
and display value. The family size cap makes the frontier cell target
`0.49999` a safe orchestration threshold for every exact ratio below one half,
but the target is not the mathematical verdict.

## Certificate format

The candidate directory may contain files, but the evaluator reads and copies
only `model.py`. The copied file must be self-contained and no larger than 1
MiB. It defines callable `build_family` and returns a mapping that normalizes
to a JSON object with exactly the keys `n` and `sets`.

- `n` is an exact integer with `1 <= n <= N_MAX`. Booleans are rejected.
  Numpy integer scalars are accepted through `operator.index()`.
- `sets` is an iterable of exact integer masks with
  `1 <= len(sets) <= M_MAX` and `0 <= mask < 2**n`.
- Masks are strictly ascending. This is the canonical encoding and makes
  duplicate-member inflation structurally impossible.
- At least one member is nonempty. The empty set itself may be present.
- The normalized JSON wire uses sorted keys, compact separators, UTF-8, and
  `allow_nan=False`. It is capped at 4 MiB.

`N_MAX` is `min(24, AUTOEVOLVE_UCF_NMAX)`. `M_MAX` is
`min(20000, AUTOEVOLVE_UCF_MMAX)`. Missing or blank values use the hard cap.
Invalid values fail the evaluator. A configured value can only tighten a cap.
Both values are read when `evaluate.py` imports, before candidate code loads.
Candidate code shares the evaluator interpreter, so every later clause reads
only the immutable snapshot and never re-reads the returned candidate objects.

The two excluded degenerate families are rejected explicitly before metric
arithmetic:

1. `F = {}` is rejected because there is no member and no frequency maximum.
2. `F = {empty set}` is rejected because there is no ground element whose
   frequency could witness the conjecture.

Padding the declared universe with unused elements cannot lower `f_max`.
Unused elements have frequency zero, while the required nonempty member makes
some used element have positive frequency.

## Candidate compute contract

Candidates may search at evaluation time. `build_family(deadline)` may accept
an absolute `time.monotonic()` deadline. A zero-argument `build_family()` also
works. Signature inspection selects the call shape.

The total candidate compute allowance is 75 percent of the current stage
timeout. Stage 1 divides that allowance equally between two fresh executions.
The remaining wall time is reserved for exact verification and teardown. A
candidate that ignores its deadline still runs, but crossing the stage timeout
is a total gate failure. Search code must return its best valid incumbent in
time. Numpy and the Python standard library are available.

## Verification stages

Stage 0, `exact-gate`, has a 90 second timeout. It performs one isolated copy
and explicit-path load of `model.py`, snapshots the returned mapping and set
iterable once into immutable Python integers, checks every shape clause, checks
all unions with a dense direct-address array, and derives frequencies itself.

Stage 1, `replay-and-cross-check`, has a 240 second timeout. It repeats stage
0, executes a second fresh copy of `model.py`, normalizes that result through a
separate implementation, and requires identical normalized JSON bytes. It
then repeats the primary verifier and independently checks all unions with a
stdlib `bytearray`, recounts frequencies by scanning each element, and checks
the used-ground-set invariant.

The pair checks have an exact operation budget equal to every unordered pair
allowed by `M_MAX`. If that budget is ever exhausted before all required pairs
are checked, the gate fails closed. A stage timeout also fails closed.

The replay is a reproducibility filter, not a proof that candidate code is
deterministic. A randomized candidate can return the same output twice by
chance. The accepted artifact is the normalized family, and stage 0 already
fully proves union closure for that artifact.

## Metric integrity

The candidate cannot provide a ratio, frequency, verdict, or metric key. Any
extra certificate key is rejected. The evaluator derives every value from the
immutable normalized masks.

A non-closed low-frequency family fails with a concrete union witness. A
duplicate mask fails the strict ascending rule. Neither empty degenerate can
reach arithmetic. A passed `below_half = 1` therefore means a finite, exact,
independently replayed union-closed family whose maximum element frequency is
strictly less than one half.

Current theory proves that some element occurs in at least
`(3 - sqrt(5)) / 2`, about 0.38, of the sets [lit: Alweiss, Huang, and Sellke,
2024]. The conjectured threshold remains one half. Reaching exactly one half
is expected and is not news. The powerset seed does exactly that. Only an
exact value below one half would disprove the conjecture.

## Cells

- `u12-validation` applies `N_MAX = 12`. The conjecture is published as
  verified through 12 ground elements [lit: Vuckovic and Zivkovic, 2017].
  Together with the powerset equality example, its minimum is one half. This
  cell validates the harness and proves nothing about frontier search.
- `u13-frontier` applies `N_MAX = 13`. No published answer is supplied to the
  search. Any below-half result must be treated first as an
  `improvement-candidate` and independently scrutinized.
- `u24-frontier` applies the hard `N_MAX = 24` cap and has the same claim
  discipline. Recall of a published family cannot solve an open cell.

The two MAP-elites descriptors are `ground_set_used` and `family_size`.
They describe family structure, not quality, and preserve diverse archive
footholds instead of collapsing the campaign to one hill-climbing cell.

## Seed and fixtures

The seed is the full powerset of a three-element ground set, encoded by masks
`0` through `7`. It is union-closed because every OR remains in that range.
Each of the three elements occurs in exactly four of the eight members, so its
exact ratio is one half. The seed is a valid equality example, not progress on
the conjecture.

`fixtures/powerset13/model.py` supplies the full 13-element powerset as a scale
fixture. The committed mutants cover a concrete missing union and both
excluded degenerate families. No external dataset is used.

## Hardware

CPU only. Python 3.11 or newer is required. The evaluator uses the standard
library and numpy. It performs no network access.
