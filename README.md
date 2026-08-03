# autoevolve

Agent-native evolutionary optimization. You state a goal in english. The system
synthesizes a scoring contract, measures the baseline, checks feasibility, then
evolves code toward the target with a parallel population of coding-agent
workers, and ships the result as a report or a PR.

The promise on every run: hit the target, or deliver best-found plus an
evidence-backed explanation of the ceiling. Both are successful outcomes.

## Status

v0 under construction. Every number that ever appears in this README will come
from `autoevolve report` over a real run artifact with a run id. No number
appears before that. See docs/HONESTY.md.

## How it works

1. English in, contract out. Before any compute burns, autoevolve synthesizes
   `evaluate.py`, measures the baseline, computes a feasibility ceiling where
   possible, and locks the contract.
2. The agent is the mutation operator. Claude Code and Codex sessions mutate
   candidates. They can profile, read compiler output, and debug the evaluator.
3. The population outlives sessions. All evolution state lives in one SQLite
   store owned by the autoevolve server. Workers are stateless and disposable.
   Any MCP-speaking agent can join a run mid-flight.
4. It gets smarter every run. Top lineage diffs are distilled into a discovery
   ledger that future runs sample into mutation context.
5. It evolves itself. Operator selection is a persistent per-domain bandit over
   measured child-improvement rates.

## Surfaces

| surface | shape |
|---|---|
| CLI | `autoevolve init\|run\|watch\|join\|report\|campaign\|render\|serve` |
| MCP server + skill | any MCP client joins a run and works the loop |
| GitHub issue mode | issue in, contract comment, approval label, evolution, PR out |
| Dashboard | self-contained HTML plus evolution.gif plus lineage poster |

## Docs

- docs/ARCHITECTURE.md is the normative module and interface spec.
- docs/CONTRACT.md is the normative evaluator contract.
- docs/HONESTY.md is the claims policy. It is enforced, not aspirational.

## License

Apache-2.0
