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

## Remaining after lane merges

- U9 campaigns lane (launch after wave merges; brief not yet written).
- U8 proofs run by the orchestrator with real executions.
- Final ultracode review workflow, fixes, README gallery from real runs,
  journal entries per unit.

## Merge discipline

Per unit: uv sync --dev in lane clone, ruff + pytest, read the diff, commit in
lane, fetch into main, merge, re-gate on main, journal entry, release claim,
delete clone. Unit order U1, U2 first (wave 1), then U3, U4, U5, then U6, U7,
then U9. CLAUDE.md section 15 order holds for merges even though lanes built
in parallel.
