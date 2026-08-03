# HONESTY.md

The claims policy. Mirrors CLAUDE.md sections 11 and 13. Enforced by CI lint
(U9), not aspirational.

## Rules

- Measured-or-null. Every number in README, docs, dashboards, and reports
  comes from `autoevolve report` over a real run artifact, or it does not
  appear.
- A benchmark claim without a run id is a defect. Fix by running or deleting.
- Failure is a first-class result. Ceiling, budget, and plateau causes are
  reported plainly in one paragraph. Silent death is a defect.
- Never present a proxy-task result as an at-scale result. Proxy wins are
  always labeled proxy wins.
- Never compare against a baseline you did not run in the same environment.

Campaigns that attack published open problems carry extra rules, because
their baseline is somebody else's record rather than a number we measured.
See docs/FRONTIER.md for the claim tiers, the cited bounds registry, the
staleness rule, and the `[lit: source]` citation marker.

## Campaign claims ladder

- "candidate": any interesting artifact from a run. Needs nothing but the
  run id.
- "discovery": requires ALL of (1) reproducible artifact in-repo, (2)
  held-out or replicated validation, (3) the exact run id. Anything less
  stays "candidate".
- arch-search promotions: proxy win, then 3-seed proxy replication, then
  scaled validation run, and only then a claim.
- Rediscovery of known results is reported as rediscovery. It is the
  credibility demo, not the novelty claim.
- Negative results are reported in the campaign log. The ledger of what
  failed is part of the compounding asset.
