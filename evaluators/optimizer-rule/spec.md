# Optimizer update rule evaluator

## Task and fixed harness

The candidate implements only an optimizer update rule. `baseline/rule.py` exposes
`init_state(shape)` and `update(param, grad, state, step)`. The evaluator owns the model,
data, initialization, gradients, loss, validation, and training budget. A candidate cannot
change those parts of the experiment through its public interface.

The fixed model is a NumPy MLP with two inputs, one hidden layer of 16 `tanh` units, and one
binary output logit. It trains on a committed noisy two-moons classification fixture. Every
replicate uses exactly 300 full-batch update steps. The training and validation splits are
disjoint. The candidate receives one parameter tensor, its training gradient, its own state,
and the step number. It never receives validation features or labels.

The seed rule is plain SGD with learning rate `0.05`. At the small fixed initialization used
here, the full-batch binary cross-entropy gradient is nonzero. A step of `0.05` moves each
parameter against that gradient and is deliberately small relative to the bounded inputs and
bounded `tanh` derivative. Repeating that descent step has real training headroom while avoiding
the immediate instability demonstrated by the committed diverging mutant.

## Gate and metrics

The `trained` gate requires the final training loss to be finite and strictly lower than the
initial training loss in every replicate in the selected stage. A non-finite returned parameter
or state, a changed parameter shape or dtype, or an invalid state fails immediately with an
`EvalError` that names the step and parameter. A gate failure returns no score.

The headline metric is `val_loss`. It is the mean validation binary cross-entropy after exactly
300 full-batch steps, and lower is better, so `MAXIMIZE` is `False`. The evaluator also returns
`train_loss`, mean `val_accuracy`, and `steps`. `steps` is `300.0` in both stages because it is
the fixed per-replicate budget.

Stage 0, `single-seed-proxy`, runs initialization seed `20260803`. Stage 1,
`three-seed-replication`, runs seeds `20260803`, `20260809`, and `20260821` and reports the mean
metrics. This is a mini replication ladder. Every candidate sees the same fixed seed or seed set.
All random construction uses explicit NumPy `Generator` objects. The evaluator never reads or
changes NumPy's global random state, so the same deterministic candidate produces the exact same
`val_loss` on repeated evaluations.

`ceiling()` returns `None`. The Bayes error of this synthetic noisy task is not computed, so this
pack makes no theoretical validation-loss ceiling claim.

## Certificate and proxy limits

This pack answers the three domain questions in `docs/DOMAINS.md`. The certificate is executable:
the evaluator checks finite, decreasing training loss under the locked harness. The check is cheap
because the model and data are small. The primary metric is graded validation loss rather than a
binary success flag. The validation split is held out from the candidate interface, and stage 1
adds replicated initialization evidence.

This is a PROXY task. A rule winning here is only a proxy win under `docs/HONESTY.md`. It says
nothing about large-scale training, different architectures, different datasets, or distributed
optimization until the rule is replicated at that scale.

## Hardware, dependencies, and fixtures

This evaluator is deterministic, offline, and CPU-only. It uses Python 3.11 and NumPy. It does not
use Torch, SciPy, scikit-learn, a network, or wall-clock measurements.

Fixture seed `314159` creates 144 noisy two-moons rows with an explicit
`numpy.random.default_rng` generator. The committed split contains 96 training rows and 48
validation rows with non-overlapping sample ids. Values are rounded to nine decimal places.

Regenerate the fixture with:

```text
python evaluators/optimizer-rule/fixtures/make_fixtures.py
```

The generator rewrites byte-identical JSON for unchanged code.

## Candidate guidance

Agents may change only code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` in
`rule.py`. Preserve both public signatures. Return an exact two-item tuple containing a finite
NumPy array with the same shape and dtype as `param`, plus an exact plain dict whose values are
finite NumPy arrays or exact Python floats.
