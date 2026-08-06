# Low Autocorrelation Binary Sequences evaluator

## Problem and metric

For a sign sequence `s` of length `n`, every entry is exactly `-1` or `+1`. Its aperiodic
autocorrelation at lag `k` is:

```text
C_k = sum(s[i] * s[i + k] for i in 0 .. n - 1 - k)
```

The energy is the exact non-negative integer:

```text
E = sum(C_k * C_k for k in 1 .. n - 1)
```

`METRIC` is `merit_factor`, computed as `n * n / (2 * E)`, and `MAXIMIZE` is `True`.
Every `C_k` and `E` are computed with Python integer arithmetic. Floating point is used only
to expose the final scalar metric and reporting values after the exact energy is known.

## Candidate compute contract

The candidate file is `solver.py` and defines:

```python
def solve(n, deadline=None, seed=None):
    ...
```

The evaluator inspects the callable signature. It always passes `n` positionally. It passes
`deadline` and `seed` by their declared names, through variadic arguments, or omits them when
the callable accepts only `(n)`.

`deadline` is an absolute `time.monotonic()` value derived from the selected stage timeout
with three seconds reserved for import, result normalization, the exact gate, and process
teardown. The candidate may burn its whole budget searching and should return its best
sequence before the deadline. A stage timeout is a total loss and no partial result is
scored. `seed` is fixed per cell so search behavior is reproducible.

The baseline is a real search procedure. It runs deterministic random restarts, applies
single-flip steepest descent using exact incremental autocorrelation updates, and returns its
best incumbent when its own restart cap or the deadline is reached.

## Skew-symmetric search tactic

For odd length `n = 2m - 1`, a sequence is skew symmetric when, for every valid positive
offset `l`:

```text
s[m - 1 + l] = ((-1) ** l) * s[m - 1 - l]
```

The first `m` spins determine the complete sequence, so this structure reduces the search
space from `2 ** n` sequences to `2 ** m`. Every odd-lag autocorrelation is then exactly zero.
This is a search method, not a stored answer. The baseline expands `m` free spins with this
identity and performs steepest descent in that reduced space, while alternating with
unrestricted full-space restarts.

Skew symmetry is never a gate condition. The evaluator accepts every valid `-1` and `+1`
sequence of the selected length, including non-skew-symmetric sequences, and recomputes every
autocorrelation and the energy from scratch.

## Exact gate

`GATE` is `valid_binary_sequence`. The candidate result is consumed exactly once into an
immutable tuple of plain Python integers. The gate requires exactly `n` entries and rejects
booleans, non-integral entries, and every integer other than `-1` and `+1`.

After normalization, the evaluator recomputes every autocorrelation and the energy from the
immutable sequence. It never accepts a candidate-reported energy, merit factor, descriptor,
or bound. No stored answer and no network access participate in the gate or score.

Candidate code imports into the evaluator process. Trusted references to builtins,
`operator.index`, signature primitives, timing, and import helpers are therefore bound before
candidate import. Later gate work uses those references so candidate code cannot change a
dependency by rebinding a builtin.

## MAP-elites descriptors

Exactly two evaluator-computed descriptors spread the archive:

- `positive_fraction`: the count of `+1` entries divided by `n`, with range 0 through 1.
- `normalized_max_abs_autocorrelation`: the maximum `abs(C_k)` divided by `n`, with range
  0 through 1.

Both descriptor names are returned in the metrics mapping for every passing candidate. They
describe sequence structure and do not replace `merit_factor` as fitness.

## Cells and resources

| cell | role | n | timeout |
| --- | --- | ---: | ---: |
| `n13-validation` | validation | 13 | 30.0 seconds |
| `n41-calibration` | calibration | 41 | 300.0 seconds |
| `n61-calibration` | calibration | 61 | 300.0 seconds |
| `n71-frontier` | frontier | 71 | 300.0 seconds |
| `n81-frontier` | frontier | 81 | 300.0 seconds |
| `n91-frontier` | frontier | 91 | 300.0 seconds |
| `n101-frontier` | frontier | 101 | 300.0 seconds |
| `n121-frontier` | frontier | 121 | 300.0 seconds |

CPU only. Python 3.11 standard library is sufficient.

## Honesty

The pack contains no published best known energy or merit factor and has no numeric target.
The validation optimum is computed inside the test by exhaustive enumeration and is never
written into the repository. A frontier score is a measured candidate result, not a record
claim. A calibration score can only match or miss an already solved optimum and is never a
record claim. Budget exhaustion and timeout remain their recorded stop reasons.
