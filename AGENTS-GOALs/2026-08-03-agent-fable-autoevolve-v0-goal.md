# Goal: AutoEvolve v0 end to end (2026-08-03)

Owner: Fable orchestrator session. Founder directive: build the entire CLAUDE.md
roadmap U0 through U9 to production quality, fully autonomous, no pickers,
unlimited Codex lane credits, merge when ready, no Claude co-author lines
anywhere.

## State when this file was written

- U0 merged on main: scaffold, seam types, normative docs, CI. Gate green
  (ruff clean, 4 tests).
- Seven Codex sol lanes running in standalone clones (job ids in the session
  scratchpad lanes.md and in RunPipe/ultramerge/LEDGER.md): u1-core, u2-eval,
  u3-mutate, u4-mcp, u5-cli, u6-gh, u7-packs. Briefs embed the full spec;
  lanes write files only, orchestrator gates and commits.
- Live-docs recon done: mcp SDK installed is v2.0.0 (MCPServer, no FastMCP);
  codex exec flags verified from codex-cli 0.146.0; claude -p flags and skill
  frontmatter from live docs. These facts are baked into the U3/U4 briefs.

## Final state (end of day 2026-08-03)

v0 DONE per CLAUDE.md section 15: all U0 through U9 merged, all U8 proofs
green (proof 1 r59ff8810dd, proof 2 r8d0a8d799d at 10.90x target_hit,
proof 3 rda6528a177 dual runtime), 233 tests green with ruff clean, README
gallery from real runs, adversarial review of 26 agents with 20 findings
fixed same day (docs/reviews/2026-08-03-adversarial-review.md). No remote
configured; nothing was pushed anywhere. Lane clones AutoEvolve-* may
linger until the codex companion releases directory handles; safe to
delete afterward.

## Merge discipline

Per unit: uv sync --dev in lane clone, ruff + pytest, read the diff, commit in
lane, fetch into main, merge, re-gate on main, journal entry, release claim,
delete clone. Unit order U1, U2 first (wave 1), then U3, U4, U5, then U6, U7,
then U9. CLAUDE.md section 15 order holds for merges even though lanes built
in parallel.
