"""Embedded evaluator contract rules for synthesis prompts."""

from __future__ import annotations

CONTRACT_RULES = """The evaluator directory must contain spec.md, evaluate.py, baseline/,
and fixtures/.
spec.md states metric names and units, the correctness gate, target semantics, hardware needs,
and fixture provenance. evaluate.py imports StageSpec and EvalError from
autoevolve.eval.contract. It defines STAGES from cheap to expensive, GATE as the name of a
boolean 1.0 or 0.0 metric, and evaluate(candidate_dir: Path, stage: int = 0) returning a
non-empty dict[str, float] that always includes GATE. Correctness is checked before score.
Gate failures raise EvalError with a useful reason. Metrics are measured on this machine and
run, never inherited, estimated, or looked up. The gate must be deterministic for identical
input. Fixtures are versioned in the evaluator folder. Stage 0 must be cheap. An optional
ceiling() may return a metric, value, and method. Candidate code must never be exec'd or eval'd
inside the engine process."""

FILE_BLOCK_RULES = """Return only complete files in path-labeled blocks:
### FILE: relative/path.py
```python
complete content
```
Use only spec.md, evaluate.py, baseline/<files>, and fixtures/<files>."""


def synthesis_prompt(goal_text: str, domain: str) -> str:
    """Build the first-pass evaluator synthesis prompt."""

    return f"""Create a production evaluator for this English goal:
{goal_text}

Classified domain: {domain}

Contract rules:
{CONTRACT_RULES}

The evaluator must have a measured scalar optimization metric, a deterministic correctness
gate, explicit STAGES and GATE, realistic fixtures, and a runnable baseline implementation.
Do not claim measurements that have not been performed. Use placeholders in spec.md for
baseline and target values that the engine must measure and lock later.

{FILE_BLOCK_RULES}"""


def repair_prompt(goal_text: str, domain: str, error: str, previous_output: str) -> str:
    """Build the single correction prompt after validation fails."""

    return f"""Correct the evaluator response for this goal:
{goal_text}

Classified domain: {domain}
Validation error: {error}

Previous response:
````text
{previous_output}
````

Return a complete corrected evaluator, not a patch.

Contract rules:
{CONTRACT_RULES}

{FILE_BLOCK_RULES}"""
