# FRONTIER.md

Normative rules for campaigns that attack published open problems. CLAUDE.md
section 11 and docs/HONESTY.md are the constitution; this file is the extra
discipline that applies when a campaign's baseline is somebody else's
published record rather than a number we measured ourselves.

## 1. Why this file exists

Ordinary campaigns compare against a baseline we measured on this machine.
Frontier campaigns compare against the literature. That introduces two
failure modes ordinary honesty rules do not cover:

1. Citing a bound as if we measured it.
2. Announcing an improvement over a record that was already beaten while we
   were running.

Both are answered structurally below, not by good intentions.

## 2. Claim tiers

Every frontier result carries exactly one tier. The campaign report assigns
the tier from database facts and the bounds registry, never by hand.

| tier | means | requires |
|---|---|---|
| `literature` | a published bound we did not measure | a citation with who and year in bounds.json |
| `matched` | our run reproduced a known bound | our run id, and a measured value equal to the cited bound |
| `improvement-candidate` | our run beat the cited bound | our run id, the exact certificate in-repo, and a bounds re-check no older than the run |
| `verified-improvement` | the candidate survived scrutiny | all of the above, plus an independent re-verification of the certificate by a second implementation or a third party, recorded with its date |

Nothing is ever born a `verified-improvement`. The path is always
`improvement-candidate` first. A result that beats a bound whose registry
entry is stale is reported as `improvement-candidate (stale bound)` and is
not announced anywhere until the bound is re-checked.

## 3. The bounds registry

Every frontier pack ships `bounds.json` beside its `campaign.json`:

```json
{
  "bounds": [
    {
      "claim": "smallest known 5-chromatic unit-distance graph",
      "value": 509,
      "direction": "lower_is_better",
      "who_and_year": "Jaan Parts, 2020",
      "source_url": "https://...",
      "checked_on": "2026-08-03",
      "how_to_recheck": "search for 5-chromatic unit-distance graph vertex count; the community tracks reductions publicly"
    }
  ]
}
```

Rules:

- `checked_on` is the date a human or an agent actually re-read the source,
  not the date the entry was written.
- A bound is **stale** when `checked_on` is more than 30 days before the run
  that cites it. `autoevolve campaign bounds <name>` prints staleness, and
  `autoevolve campaign run` warns loudly on a stale bound before it starts.
  This encodes the operational rule that a campaign re-checks its bound the
  day it starts.
- Beating a bound never edits that bound's entry. Add a new entry citing our
  own run id, so the literature value and our measurement stay separable
  forever.

## 4. Citing literature inside prose

The claims lint requires a run id on every measured claim. Literature bounds
are not measurements, so they carry an explicit citation marker instead:

```
The best known length at order 29 is 553 [lit: Distributed.net OGR-29, 2014].
```

`[lit: ...]` satisfies the lint only when it names a source. It is a promise
that the number came from outside and is not ours. Using `[lit: ...]` on a
number produced by one of our runs is a defect of the same severity as a
benchmark claim without a run id.

`[no-claim]` remains available for purely illustrative numbers that assert
nothing about the world.

## 5. Gate discipline for frontier packs

A frontier gate is stricter than an ordinary evaluator gate, because a false
pass here does not just waste compute, it produces a false mathematical
claim.

- The gate must be exact. No floating point comparison decides whether a
  certificate is valid. Algebraic quantities use exact arithmetic over
  rationals or integers.
- The gate must be total. Every case is checked, never sampled. A gate that
  verifies a random subset of the constraints is a defect.
- Any bounded search inside a gate fails closed. If a colorability search or
  an exhaustiveness check hits its operation budget without a conclusion, the
  gate fails. It never reports a bound it did not prove.
- The candidate produces the certificate; the evaluator judges it. Evolution
  can only edit inside EVOLVE-BLOCK markers, and no gate logic lives there.

## 6. What gets announced

Nothing leaves this repository as a claim about mathematics unless it is at
least `improvement-candidate` with a fresh bound, and the certificate is
committed so anyone can re-verify it independently. Matching a known bound is
reported plainly as a rediscovery and is a successful outcome, because it is
what validates the harness. Failing to reach the known bound is reported too,
with the ceiling or budget or plateau cause, exactly like any other run.
