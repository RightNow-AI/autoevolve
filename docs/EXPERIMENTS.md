# Experimental record

Every number here comes from a run database in this repository, identified by
run id. Nothing is estimated, remembered, or carried over from another
environment. Where a result is below the published literature it says so, and
the negative results are recorded in the same format as the positive ones,
because a record that only holds wins measures nothing.

All runs are dated 2026-08-04 unless stated. Modal runs used CPU containers
except E9, which used an NVIDIA A10.

---

## 1. Verification, before any result

No result below means anything without this section.

A candidate never reports its own score. It runs in a subprocess whose stdout
is the null device while evaluator and candidate code execute; the runner
restores a saved descriptor only to write one verdict line, writes it with a
raw `os.write`, and exits through `os._exit` so no `atexit` handler can append
a second verdict. The parent reads the first line and rejects a nonzero exit.
The runner launches with `-P` and `PYTHONSAFEPATH=1`, so the candidate
directory is never on `sys.path` and cannot shadow a module the judge imports.

Both of those defenses exist because both attacks were demonstrated end to end
against earlier versions of this system, not because they were anticipated:

| attack | result before the fix | status |
|---|---|---|
| `atexit` handler writes a second verdict to stdout | a candidate that failed its gate scored 999999 | closed, regression test |
| candidate ships a module that shadows one the judge imports | scored 777777 | closed, regression test |

Operators are on the same footing. An agentic operator returns files, and the
engine re-runs the whole cascade on those files in the sandbox, so an agent's
own measurement is a note in the record and never a score.

---

## 2. E1: a matched lower bound on R(5,5)

**Run r6ae4654984**, cell `k5-frontier`, operator `agentic` only, 6 children
from a 37-vertex seed.

A two-colouring of the complete graph on 42 vertices with no monochromatic K5
witnesses `R(5,5) >= 43`. That is exactly the published lower bound
[lit: Exoo, 1989]. This run produced two of them.

| certificate | program | red edges | blue edges |
|---|---|---|---|
| `n42-79e15fd6c25d1729.json` | p3f5515a557 | 436 | 425 |
| `n42-ffb3b30e31b519e8.json` | pc5c7e51728 | 426 | 435 |

Both are in `campaigns/ramsey-lower-bound/evaluators/ramsey/certificates/k5-frontier/`.

Three independent checks agree on both:

1. The pack's stage 1 re-derives the certificate in a fresh interpreter and
   requires a byte-identical snapshot, then runs an exhaustive check over
   every 5-subset that must agree with two independent fast verifiers or the
   result fails closed.
2. `campaigns/ramsey-lower-bound/verify_certificate.py`, written separately
   and sharing no code with the pack, brute forces all **850,668** subsets of
   size 5 in each graph. Neither contains a monochromatic K5 in either colour.
3. The two certificates differ from each other in edge count and structure, so
   they are not one answer counted twice.

**Claim tier: matched.** This equals the published bound. It does not improve
it. Improving it requires 43 vertices.

**Not claimed:** that either graph is new, or that either is non-isomorphic to
Exoo's. Neither was tested.

### Evidence this was search and not recall

The distinction matters because an earlier result in this project (Golomb
order-11, run r48c09efb6d) reached a proven optimum purely by reproducing a
tabulated constant, and that is not evidence of search.

During E1 the agent wrote and ran its own parallel search over the
factorisations of 42, executing `search42.py 7 6`, `14 3`, `21 2`, `6 7` and
`3 14` concurrently alongside `run42.py`, `run41.py`, `search42b.py` and
`run_sym.py semi42`. It returned two structurally different valid answers.
Recall of a single known construction produces one.

This was only possible because the agentic operator gained the ability to
execute code on the same day. See section 6.

---

## 3. E2: operator efficiency on the same problem

The central quantitative comparison. Same pack, same cell, same gate.

| run | operators | programs | best | stage |
|---|---|---|---|---|
| r2dcea50679 (Modal) | diff, rewrite, crossover | 401 | 41 vertices | 1 |
| r6ae4654984 (local) | agentic | 6 | **42 vertices** | 1 |

The agentic operator reached a strictly better result in **6 programs than
401 programs** of diff and rewrite reached, and the better result is the one
that matches the literature.

**Threat to validity, stated plainly:** these programs are not equal in cost.
A diff program is one model call taking seconds. An agentic program is a full
coding session with a budget of up to 25 minutes that spawns its own parallel
subprocesses. This comparison establishes efficiency *per program*, which is
the unit the archive and the bandit operate on. It does **not** establish
efficiency per wall-clock second, per token, or per dollar, and no such claim
is made. The two runs also used different hardware.

## 4. E3: the same comparison where it does not flatter the operator

**Golomb ruler, order 29.** Best known length 553.

| run | operators | programs | best | end |
|---|---|---|---|---|
| rcb5428d2ed (Modal) | diff, rewrite, crossover | 3001 | 623 | budget exhausted |
| r3b7e1b68af (local) | agentic | 3 | 623 | stopped |

One agentic program reached the length that 3001 diff and rewrite programs
reached. Three agentic children never improved on the first.

Both operator families terminating on **exactly** 623 is recorded here as a
warning rather than a success. It is the signature of a recalled construction,
the same pattern as the order-11 result. On E1 the same operator demonstrably
searched, so the capability is real and the open question is why it was not
used here. Neither result approaches 553.

**Claim tier: below literature**, by 70.

## 5. E4: a domain with no literature to recall

**Run re34f50e5c3**, `python-speedup`, operator `agentic`, 2 children.

| program | speedup vs baseline |
|---|---|
| seed | 0.998 |
| first agentic child | 2.006 |
| second agentic child | **2.169** |

Both children passed the correctness gate on output equality against fixtures.
The improvements were ordinary optimisation work with nothing to recall:
zip-based sliding windows replacing index arithmetic, a walrus operator reusing
Sobel gradient terms, and a locally bound `append`.

---

## 6. Platform findings, which gate every result above

These are reported because each one silently invalidated or prevented
experiments, and because a green test suite detected none of them. Each is now
covered by a regression test that fails against the old code.

| finding | measured effect |
|---|---|
| Archive descriptors missing on 9 of 14 packs | `cell_key` constant `"0"`, so MAP-elites held one cell and one elite; inspirations returned empty on all 153 cycles of a run; crossover and migration could never fire |
| Gate failures counted toward the plateau | runs closed at 153 of 3000 evaluations; after the fix, run rcb5428d2ed ran 3001 and exhausted its budget, a **20x** increase in search performed |
| Bandit allowlist laundering | the selector hinted `agentic`, the CLI substituted `diff`, and the pull was recorded against `diff`; three of four operators showed zero pulls ever |
| Describe probe withheld workload configuration | a pack that reads its cell at import time, which the frontier rules require, could not be loaded at all; the Ramsey campaign sat at **zero programs** for a day behind a store that looked idle |
| Agentic operator judged by process exit code | the agent completed its edits and exited nonzero because of a host teardown hook; seven consecutive cycles discarded finished work and the operator had **never** produced an accepted child in the project's history |
| Agentic operator denied execution | on Golomb order 29 a probe recorded 7 permission denials, all Bash or PowerShell, hit the turn limit, and made no edit; the operator's entire premise is that it can run and measure things |
| Timeout discarded landed edits | two full cycles lost work that was already on disk |

### E10: reward hacking, observed live

**Run r5a42774d14**, `triton-kernel` in mock mode. The mock scorer accepted a
candidate-reported `result["score"]`. Evolution drove that self-reported value
to **1e+300 within 46 programs**. The metric is a utilization product and is
bounded by 1 by construction, so any value outside `[0, 1]` is a claim rather
than a measurement, and is now rejected. This run is retained in the record as
a negative control and its number must never be quoted as a result.

---

## 7. Negative results

| run | problem | programs | best | published | outcome |
|---|---|---|---|---|---|
| r7aa64a8c93 | Frankl union-closed | 905 | 0.5 | counterexample needs < 0.5 | no counterexample; the search sat exactly on the conjecture's own boundary, which is what a true conjecture looks like from the inside |
| r6aba13595d | superpermutation n=6 | 50 | 873 | 872 | seed never improved |
| rde500192d7 | Golomb ruler order 30 | 747 | 680 | 585 | below literature by 95 |
| r14b0eac3ff | lossless compression | 2001 | 4.272 ratio | n/a | improved over its own baseline, no external comparison run in this environment |
| r0d334e58e7 | E9: GPU kernel, NVIDIA A10 | 401 | 0.0828 TFLOPS | roofline in-pack | real device-resident measurement; real mode requires the result to be a CUDA tensor, so a silent CPU fallback cannot be reported as GPU throughput |

---

## 8. E11: moving every workload off the laptop, and what that cost to prove

Measured 2026-08-05. None of this is a discovery. It is recorded because the
agentic operator was laptop-only until now, and it is the only operator that
has ever produced a frontier result here, so this is what unblocked scale.

| claim | evidence |
|---|---|
| The quality gate runs remotely | `scripts/modal_gate.py` returned `ruff_ok` true, `pytest_ok` true, 322 passed, 0 failed, in a container |
| A coding agent runs remotely | codex-cli 0.146.1 installs in the image and authenticates from the `OPENAI_API_KEY` already present in the `autoevolve-model` secret |
| It genuinely edits files | the agentic preflight returns `changed: true, ok: true` |

Two failures had to be found first, and both are the same shape as the older
defects in section 6: something reported success while doing nothing.

The preflight first returned **exit code 0 with `changed: false`**. A probe of
three documented invocations found why. With `-s workspace-write` codex prints
"the filesystem sandbox failed on all attempts", exits 0, and changes nothing.
`--full-auto` behaves identically. Only `--dangerously-bypass-approvals-and-sandbox`
edits the file. Codex's own Linux sandbox cannot initialise in that image. The
bypass is opt-in, off by default, and set only by the Modal entrypoint, where
a disposable container holding one repo clone already provides the isolation.

The remote gate then caught four defects before any search ran: three Modal
entrypoints crash-looped because they resolved the repository HEAD at import
time, which also happens inside a container that has no git, and one pack test
asserted 8192 binary inputs against the n=11 cell whose correct count is 2048.

### E12: batched GPU annealing, validated before it was believed

`campaigns/ramsey-lower-bound/modal_gpu.py --mode selftest` on an NVIDIA A10G:

- **5,376 delta comparisons** against a brute force reference that enumerates
  every clique containing the flipped edge, for both K5 and K4, over n = 8 to
  14, on evolving rather than fresh states. All exact.
- n = 30 reached zero monochromatic K5s in 54.3 seconds, host-verified from
  scratch, `stop_reason: zero_k5`.
- Throughput: **639,641 aggregate flips per second across 8,192 chains**,
  against a measured single-chain CPU reference of **58,000 per second**. Both
  numbers are raw measurements rather than a ratio, because the ratio is what
  the claims lint asks to be grounded and the two absolute figures are the
  actual evidence. Reproduce with `modal run
  campaigns/ramsey-lower-bound/modal_gpu.py --mode selftest --gpu A10G`.

The exactness check is the point. A batched delta that drifted would announce
certificates that do not exist, and this project has already been burned once
by a search reporting something it could not re-verify.

## 9. What this record does not support

- No claim that any published bound was improved. E1 matches one; nothing beats one.
- No claim of per-second, per-token, or per-dollar efficiency for any operator. Only per program, on the runs named.
- No comparison against any baseline that was not run in this environment.
- No claim about graph novelty or isomorphism in E1.
- Results marked below literature are reported as such and are not evidence that the method works, only that it ran.
