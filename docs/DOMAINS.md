# DOMAINS.md

How to point autoevolve at a new domain, and how to tell in advance whether
it will work. The engine is domain-general: core, eval, mutate, mcp, and cli
contain nothing about kernels or graphs or equations. The evaluator is the
entire domain. Adding a domain means writing one, not changing the system.

## 1. The three questions

Ask these in order. A domain that fails question one cannot be attacked at
all, no matter how much compute is applied.

**Q1. Is there a certificate?** Given a candidate answer, can a program
decide whether it is correct, without a human and without asking a model?
This is the correctness gate. Evolution deletes whatever is not checked, so
an unchecked property will be destroyed to make the number go up.

**Q2. Is the check cheaper than the search?** Verification must be fast
enough to run hundreds or thousands of times. If checking costs as much as
solving, evolution has no advantage over direct search.

**Q3. Is the metric graded, not binary?** Evolution needs partial credit. A
score that is zero for every candidate until it is suddenly perfect gives
search no gradient and is indistinguishable from random guessing. Most
apparently impossible problems are binary problems that have a graded
sibling, and finding that sibling is the actual skill.

## 2. Certificate taxonomy

| kind | check | examples | how autoevolve handles it |
|---|---|---|---|
| exact | deterministic, total, cheap | sorting networks, Golomb rulers, compression size, kernel parity, tour length | ordinary evaluator, gate is the check |
| simulated | a simulator is the ground truth | trajectories, control policies, circuits, aerodynamics | evaluator drives the simulator; the simulator's fidelity is the honest limit and belongs in spec.md |
| staged | cheap proxy now, expensive truth later | protein binders, catalysts, chip layouts, model architectures | `STAGES` cascade plus the campaign promotion ladder; a proxy win is labeled a proxy win, always |
| held-out | a labeled dataset scores it | detection accuracy, forecast error, grammar correction | evaluator scores against data the candidate never sees; guard against fitting the test set |
| absent | no program can decide it | is this face beautiful, is this essay good, is this theorem true without a proof term | not evolvable; find the graded sibling or formalize it |

## 3. Mapping real domains

These are worked answers, not aspirations. Each names what is evolvable and
what is not, in the same domain.

**AI algorithms.** The strongest fit. Optimizer variants, attention blocks,
routing functions, quantization schemes, and sampling strategies all score on
validation loss under a fixed compute budget, which is exact, cheap at proxy
scale, and graded. The `arch-search` campaign is this shape. What is not
evolvable: claiming a proxy win transfers to large scale without running it,
which is why the promotion ladder exists.

**Space.** Trajectory design, transfer optimization, station keeping, and
landing control all have exact objectives (delta-v, arrival error, fuel) with
constraints a program verifies by integrating the dynamics. Simulated
certificate, excellent fit. Not evolvable: anything whose truth depends on
hardware that has not flown.

**Cars.** Racing lines and control policies score exactly in a physics
simulator as lap time subject to staying on track. Route optimization is
already the `routing-heuristic` shape. Aerodynamics is a staged certificate,
because CFD is the gate and it is expensive. Not evolvable: ride feel.

**Vision and faces.** Detection, landmarking, recognition, and tracking score
exactly against held-out labeled data, so algorithm and pipeline design is
evolvable. Generating a face that looks good is not: there is no program that
decides beauty, and substituting a model's opinion invites reward hacking on
that model's blind spots rather than progress.

**English.** Compression is an exact certificate measured in bytes, which is
why the Hutter Prize is a real target. Grammar correction and structured
extraction score against labeled sets. Summarization and style do not have
certificates, and metrics like BLEU are proxies that optimize into nonsense
when pushed hard.

**Mathematics.** Certificates exist wherever the answer is an object rather
than an argument: a coloring, a ruler, a packing, a counterexample. Proofs
become certificates only when formalized, since a Lean or Coq proof term
either type-checks or does not. See docs/FRONTIER.md.

## 4. The anti-pattern

Do not use a model as the judge in core. CLAUDE.md section 16 forbids it, and
the reason is mechanical, not philosophical: evolution optimizes the judge,
not the goal. Given enough attempts it finds the inputs where the judge is
wrong, and the result is a candidate that scores brilliantly and is worthless.
A model may propose, explain, or summarize. It may not decide.

If a domain has no certificate, the honest options are to find the graded
sibling, formalize the goal until a program can check it, or accept that this
is not the right tool. Inventing a scorer that merely feels right produces
confident numbers about nothing, which is worse than no result.

## 5. Adding a domain

1. Answer the three questions in writing. If Q1 fails, stop.
2. `uv run autoevolve init <name>` scaffolds the folder.
3. Write the gate first, before the metric. Make it exact and total, and make
   any bounded search inside it fail closed.
4. Declare `METRIC` and `MAXIMIZE` so the engine never guesses the direction.
5. Normalize whatever the candidate returns into immutable primitives in one
   pass before checking anything. Candidate code shares your interpreter, so
   a container it returns may answer differently on each read. See
   docs/CONTRACT.md section 4a.
6. Ship a seed that is a real, valid, known-good candidate, and fixtures with
   a deterministic generator whose seed is stated in spec.md.
7. Write spec.md so a stranger can tell what was measured, on what hardware,
   and what the result would and would not prove.
