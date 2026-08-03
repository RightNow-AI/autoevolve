# Symbolic regression evaluator

## Task

The candidate predicts the published Nguyen-7 benchmark target
`f(x) = ln(x + 1) + ln(x^2 + 1)`. The bundled model is the deliberately weak linear
seed `0.5 + 0.9 * x`.

This is a science rediscovery demo. Recovering Nguyen-7 is reported as rediscovery of
a known public result. It is not a novelty claim. A discovery claim still requires a
reproducible artifact, held-out validation, and an exact run id under
`docs/HONESTY.md`.

## Metrics and gate

The `finite` gate requires a Python float prediction that is finite for every training
point. Every training point must pass before any score is returned.

`r2_heldout` is the coefficient of determination on the 40 held-out points. It is
unitless. `complexity` is the AST node count across the body of `predict`. It is a
node count. The headline metric is `fitness`, defined as
`r2_heldout - 0.001 * complexity`. The target semantics for `fitness` are maximize.

Stage 0 checks the 60 training points and returns the gate plus complexity. It has a
15 second timeout. Stage 1 repeats the train gate and then computes held-out metrics.
It has a 30 second timeout. `ceiling()` returns `None` because the complexity penalty
depends on the candidate program.

## Hardware and dependencies

This evaluator needs only a CPU and the Python standard library. All predictions and
metrics are computed in process on the current machine.

## Fixture provenance

The data follow the Nguyen benchmark suite by name and exact Nguyen-7 formula.
Python `random.Random` seed `271828` samples `x` uniformly on `[0, 2]`. The first 60
samples form the training split and the next 40 form the held-out split. Inputs are
rounded to nine decimal places. Targets are computed from those rounded inputs and
rounded to twelve decimal places.

Regenerate both files with:

```text
python evaluators/symbolic-regression/fixtures/make_fixtures.py
```

The script is deterministic and rewrites byte-identical JSON.

## Candidate guidance

Agents may change only code between `# EVOLVE-BLOCK-START` and
`# EVOLVE-BLOCK-END` in `model.py`. They must preserve
`predict(x: float) -> float`.
