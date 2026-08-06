# Low Autocorrelation Binary Sequences campaign

## Goal and claim boundary

This campaign evolves search procedures for Low Autocorrelation Binary Sequences. A
candidate returns a sign sequence for one selected length. The trusted evaluator derives
every autocorrelation, the energy, the merit factor, and both archive descriptors from that
returned sequence.

The pack contains no published best known energy or merit factor. Every campaign target is
null and `bounds.json` is empty. The orchestrator may populate the bounds registry later from
cited sources, outside the candidate-facing pack. A passing candidate is evidence only for
its exact run id, cell, and measured output.

## Cells

| cell | role | length | stage timeout |
| --- | --- | ---: | ---: |
| `n13-validation` | exhaustive validation | 13 | 30 seconds |
| `n41-calibration` | calibration | 41 | 300 seconds |
| `n61-calibration` | calibration | 61 | 300 seconds |
| `n71-frontier` | frontier | 71 | 300 seconds |
| `n81-frontier` | frontier | 81 | 300 seconds |
| `n91-frontier` | frontier | 91 | 300 seconds |
| `n101-frontier` | frontier | 101 | 300 seconds |
| `n121-frontier` | frontier | 121 | 300 seconds |

`AUTOEVOLVE_CELL` selects the cell before candidate code loads. Unknown cell names fail
closed. The validation test enumerates all sign sequences of length 13, computes the true
minimum energy directly, and feeds an attaining sequence through the ordinary evaluator.
That validates the exact gate. It is not evidence that the baseline search can rediscover
the optimum.

Packebusch and Mertens, *Low autocorrelation binary sequences*, Journal of Physics A 49
165001 (2016), computed all optimal LABS sequences for `n <= 66` by exhaustive branch and
bound. In this pack, n <= 66 is solved. The length 41 and 61 cells are therefore labelled
calibration, not frontier. A calibration result that reaches the cited optimum is a
matched-known-optimum result and must never be reported as a record or improvement.

The same work exhaustively solved the skew-symmetric subset through length 119. That does not
solve the unrestricted LABS problem above length 66. For frontier lengths through 119, a
skew-only result may match a known skew-subset optimum while an unrestricted general
improvement can still exist. Length 121 lies beyond both exhaustive ranges stated here.

## Search and promotion

Candidate compute is part of the contract. Search, restarts, incremental scoring, and other
self-contained methods are allowed inside the stage budget. The evaluator supplies a fixed
cell seed and an absolute monotonic deadline when the candidate signature accepts them.

The seed solver uses deterministic single-flip steepest descent with random restarts. On odd
lengths it alternates between the full search space and the reduced skew-symmetric search
space. It is a working search baseline, not a stored certificate. Frontier outputs remain
candidates in the unrestricted problem because every frontier length is above 66. They require
three completed improving seeds and independent rechecking. A timeout, exhausted budget, or
failure to improve remains a negative result.
