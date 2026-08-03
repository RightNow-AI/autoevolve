# Python speedup evaluator

## Task

The candidate implements a pure-Python grayscale image pipeline. It applies a 3 by 3
box blur, a Sobel magnitude transform, and a threshold count. The bundled baseline is
deliberately direct and uses nested Python loops.

## Metrics and gate

The `correct` gate compares the threshold-count output for every selected image with
`expected.json`. The absolute tolerance is `1e-9` and the relative tolerance is zero.
Every selected case must pass. A failure raises `EvalError` and names the case.

The headline metric is `speedup`. It is the baseline elapsed time divided by the
candidate elapsed time. Each elapsed time is the minimum of three in-process repeats.
The bundled baseline is timed once per stage per evaluator process. `candidate_ms` is
the candidate elapsed time in milliseconds for one pass over the selected images.

The target semantics for `speedup` are maximize. `ceiling()` returns `None`. There is
no honest fixed upper bound, so plateau detection governs the run.

Stage 0 uses the two smallest images and has a 20 second timeout. Stage 1 uses all six
images and has a 60 second timeout.

## Hardware and dependencies

This evaluator needs only a CPU. NumPy is installed and candidates may use it. Timing
is measured on the current machine during the current evaluation.

## Fixture provenance

The six synthetic images use sizes 24 by 24 through 96 by 96. They are generated with
Python `random.Random` seed `104729`. Values are rounded to nine decimal places.
`expected.json` contains the bundled baseline outputs. It is regenerated from the
baseline, not copied from an external claim.

Regenerate both files with:

```text
python evaluators/python-speedup/fixtures/make_fixtures.py
```

The script is deterministic and rewrites byte-identical JSON for unchanged code.

## Candidate guidance

Agents may change only code between `# EVOLVE-BLOCK-START` and
`# EVOLVE-BLOCK-END` in `pipeline.py`. They must preserve the documented signatures
of `box_blur`, `sobel_magnitude`, and `threshold_count`.
