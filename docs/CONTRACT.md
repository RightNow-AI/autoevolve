# CONTRACT.md

Normative evaluator contract. CLAUDE.md section 6 is the constitution; this
file is the spec evaluator authors and the eval package implement. Keep in
sync in the same commit as any change.

## 1. Evaluator folder

```
<name>/
  spec.md          # what is measured, metrics, target semantics, hardware needs
  evaluate.py      # the contract entrypoint
  baseline/        # the seed program (a directory of files)
  fixtures/        # correctness data (parity sets, test vectors)
```

spec.md MUST state: the metric names and units, what the gate checks, target
semantics (maximize or a value), hardware needs, and how fixtures were made.
GPU evaluators MUST declare hardware in spec.md and ship a CPU mock path so CI
stays green.

## 2. evaluate.py module API (exact)

```python
from autoevolve.eval.contract import StageSpec, EvalError

STAGES: list[StageSpec]   # cascade, cheap to expensive, each with timeout_s
GATE: str                 # name of the boolean correctness metric (1.0 or 0.0)
METRIC: str               # OPTIONAL: the primary metric the contract locks to
MAXIMIZE: bool            # OPTIONAL: the primary metric's direction

def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    """Measure the candidate. Raise EvalError(reason) on gate failure.
    Never return {}. The GATE metric MUST appear in the returned dict."""

def ceiling() -> dict | None:      # OPTIONAL
    """Theoretical max, e.g. roofline. {"metric": ..., "value": ...,
    "method": "..."} or None."""
```

Declare METRIC and MAXIMIZE. Without them the engine falls back to the
ceiling's metric, then to the alphabetically first non-gate metric with
maximize true, and a guessed direction can reward the wrong thing. Every
bundled pack declares both. METRIC may be computed at import time when the
measured metric depends on the environment (the triton pack declares tflops
on GPU and mock_score in mock mode).

## 3. Rules (non-negotiable)

- Correctness gate BEFORE any score counts. Gate fail means fitness 0. No
  exceptions, no partial credit. Evolution will delete the work to make the
  number go up; the gate is what makes results real.
- Metrics are scalars measured on this machine, this run. No inherited
  numbers, no estimates, no lookups.
- evaluate() must be deterministic enough that the gate never flips on
  identical input. Timing metrics may vary; gates may not.
- Fixtures live in the evaluator folder and are versioned with it.

## 4a. Verdict integrity (the property everything else rests on)

The verdict is decided by the evaluator and reported by the runner. Candidate
code must never be able to influence what the engine records.

- The runner emits its verdict on file descriptor 1. While evaluator and
  candidate code runs, fd 1 is the null device, so nothing that code writes
  can reach the verdict channel, including writes through `sys.__stdout__`
  or a raw `os.write`.
- The real descriptor is restored only after the payload is decided, written
  with a raw `os.write` that ignores tampering with `sys.stdout`, and the
  process then leaves through `os._exit`, so no `atexit` handler registered
  by candidate code can append a second verdict.
- The parent reads the FIRST verdict line, never a later one, and treats a
  nonzero exit code as a failure.

This is not theoretical. Before these rules, a candidate returning a wrong
answer could register an `atexit` handler printing a passing payload, and
the parent, which took the last line of stdout, believed it. Evolution
optimizes whatever is measured, so a forgeable verdict channel means it
learns to forge rather than to solve. Regression tests live in
tests/test_eval_sandbox.py.

Candidate code still runs in the same interpreter as the evaluator that
judges it, so an evaluator that trusts values a candidate returns can still
be misled about its own computation. Evaluator authors must recompute every
quantity that reaches a metric rather than accepting a candidate's report of
it. Frontier packs carry stricter rules in docs/FRONTIER.md section 5.

## 4b. Environment visibility

The child environment is scrubbed to an allowlist, plus `AUTOEVOLVE_`
prefixed workload configuration such as `AUTOEVOLVE_CELL`, which campaign
cells use to select their workload. Engine and model configuration is
excluded even under that prefix: `AUTOEVOLVE_HOME` would hand a candidate
the path to the run database and therefore the ability to edit its own
scores, and endpoint and model settings are no business of a candidate.
Credential shaped names never pass under any prefix.

## 4. Sandbox guarantees and limits (honest)

Candidates run in a subprocess spawned by the engine, never in-process:

- fresh temp working directory containing a copy of the candidate dir
- wall-clock kill at StageSpec.timeout_s, enforced on all platforms
- environment scrubbed to an allowlist (PATH, SYSTEMROOT, TEMP, HOME,
  PYTHONHASHSEED); no secrets can leak into candidate code
- network disabled: the runner replaces socket.socket before the evaluator or
  candidate module loads. This blocks Python-level networking. Native
  extensions doing raw syscalls are NOT blocked; do not run adversarial
  untrusted code and rely on this alone.
- memory (RLIMIT_AS) and cpu (RLIMIT_CPU) limits applied on POSIX when
  StageSpec sets them. On Windows these are best-effort: wall-clock timeout
  is the enforced bound. This asymmetry is accepted and documented, not
  hidden.
- `exec`/`eval` of candidate source in the engine process is forbidden,
  including in tests.

## 5. Cascade semantics

Stages run in order 0..N-1. The gate is checked at every stage; the first
EvalError stops the cascade and the candidate scores 0. A candidate's recorded
scores are those of the deepest stage reached. StageSpec timeouts are per
stage. Stage 0 must be cheap enough to run hundreds of times locally.

## 6. Feasibility

If the evaluator defines ceiling() and the contract target exceeds it, the run
is declared infeasible BEFORE any evolution compute burns. The infeasibility
report (ceiling value, method, maximum plausible target) is a successful
outcome and is rendered in report.md.
