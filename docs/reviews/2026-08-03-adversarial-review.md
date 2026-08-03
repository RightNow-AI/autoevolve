# Adversarial review, 2026-08-03

Twenty-six agents reviewed the merged v0 across five dimensions (core math,
sandbox security, integration seams, honesty system, test quality), with one
adversarial verifier per finding instructed to refute against the real
source. 21 findings raised, 20 confirmed, 1 refuted.

## Confirmed and fixed

1. critical, consent: pre-approval synthesis executed generated evaluate.py
   through the describe loader. synthesize() gained load=False on the opened
   path; a behavioral test bans any subprocess there.
2. major, core: crossover hinted with no possible partner, deadlocking
   workers into the skip cap. Hint now degrades when no partner exists.
3. major, core: submissions correlated to samples by parent id only,
   corrupting island and edge records under concurrency. ParentBundle
   carries parent_sample_seq end to end; legacy submissions match newest
   first.
4. major, sandbox: socket block only covered --evaluate. It now installs
   before any mode loads evaluator code.
5. major, sandbox: describe and ceiling runner lacked process-tree kill and
   could hang the engine on Windows. They use the stage sandbox's isolation
   and kill path.
6. major, engine: goal-only open_run passed endpoint None into synthesis.
   It resolves strong then cheap or raises naming the env vars.
7. major, gh: issue config target never reached the contract. It does.
8. major, campaigns: the campaign loader shadowed pack METRIC and MAXIMIZE
   declarations. It forwards them.
9. major, honesty: the claims lint missed promised claim shapes. Patterns
   extended; the widened lint immediately caught and grounded one README
   line.
10. major, packs: kernel-frontier cells measured identical fixtures. The
    triton pack selects fixture groups by AUTOEVOLVE_CELL.
11. major, tests: sandbox tree-kill and consent re-verification had no
    failing-path coverage. Regression tests added for both.
12. minor, gh: the terminal PR winner query could select a gate-failed
    candidate. It requires the gate metric at 1.0 in the scored stage.
13. minor, docs: three agent-facing doc drifts (plateau shape, rejection
    shape, discoveries fields) corrected to the real return shapes.
14. minor, honesty: the dual-worker run's numbers had no in-repo artifact.
    Its generated report is in docs/gallery.

## Accepted as known-minor

- tests/fixtures/viz/make_fixture.py emits synthetic event payloads that do
  not byte-match engine-produced payloads. The renderer's load-bearing proof
  is the two real-run artifact sets in docs/gallery, both rendered from real
  stores. Aligning the fixture is tracked as future hardening, not a v0
  defect.

## Refuted

- One core-math finding did not survive verification against the source.
