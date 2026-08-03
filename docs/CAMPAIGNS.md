# CAMPAIGNS.md

Normative campaign pack format. CLAUDE.md section 11 is the constitution; this
file is the spec the campaigns/ packs and the campaign runner implement. Keep
in sync in the same commit as any change.

## 1. What a campaign is

A campaign is a long-running, resumable set of evolution runs plus a report
pipeline, aimed at producing genuinely new artifacts. A campaign is made of
cells. Each cell is one evolution run target (one op and shape for kernels,
one dataset for equations, one instance family for algorithms, one component
slot for architectures). Cells share the campaign's evaluator and differ by
configuration.

## 2. Pack layout

```
campaigns/<name>/
  spec.md          # goal, method, promotion ladder, honesty section (required)
  campaign.json    # machine config, schema below (required)
  evaluators/      # one or more evaluator folders obeying docs/CONTRACT.md
  log.md           # append-only campaign log, includes negative results
```

## 3. campaign.json schema

```json
{
  "name": "kernel-frontier",
  "domain": "triton-kernel",
  "evaluator": "evaluators/vector-op",
  "cells": [
    {"key": "add-1k", "env": {"AUTOEVOLVE_CELL": "add-1k"}, "target": null}
  ],
  "proxy_budget": {"max_evals": 30},
  "full_budget": {"max_evals": 300},
  "ladder": ["proxy", "replicate-3", "scaled"],
  "replicate_seeds": 3
}
```

- evaluator paths are relative to the pack directory.
- Each cell's env is applied to the evaluator sandbox for that cell's run so
  one evaluator can serve many cells.
- proxy_budget is what CI and local demo runs use. full_budget is opt-in.
- ladder names the promotion stages in order. A result's ladder position is
  computed from the db, never asserted by hand.

## 4. Runner semantics (autoevolve campaign, built with the packs)

- `autoevolve campaign list` discovers packs and prints name, domain, cells,
  ladder.
- `autoevolve campaign run <name> [--cell KEY] [--proxy|--full] [--seed N]`
  opens one run per selected cell with goal_text "campaign:<name>:<cell>",
  drives the local worker loop, and appends a dated result block to log.md
  with the run id, cell, budget, best fitness, and end cause.
- `autoevolve campaign report <name>` reconstructs the campaign state from the
  db (all runs whose goal_text carries the campaign tag): a table of cells,
  their best results with run ids, and their ladder position. Results below
  replication requirements are labeled candidate. Rediscoveries are labeled
  rediscovery.
- Resume is free: rerunning a cell opens a new run and the report shows the
  best across runs, each with its id.

## 5. Honesty enforcement (in code, not prose)

- The claims lint runs as part of the test suite. It scans README.md, docs/,
  and campaigns/**/spec.md and log.md: any line making a measured claim (a
  multiplier like 10x [no-claim], a percentage gain, TFLOPS, tok/s, or the words faster
  or speedup next to a number) must contain a run id (pattern r[0-9a-f]{10})
  or the explicit marker [no-claim] for illustrative text. A violating line
  fails the suite.
- The campaign report generator labels every result candidate, rediscovery,
  or discovery per docs/HONESTY.md. discovery requires held-out or replicated
  validation recorded in the db plus the exact run id, enforced in the report
  code path.
- log.md is append-only. Negative results get the same block format as wins.
