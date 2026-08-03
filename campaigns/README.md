# campaigns

Research campaign packs: long-running, resumable runs plus a report pipeline
aimed at producing genuinely new artifacts. Each pack has spec.md, evaluators,
a promotion ladder, and an honesty section per docs/HONESTY.md.

v0 ships four packs: kernel-frontier, arch-search, algorithm-frontier, and
equation-discovery.

List the packs with `autoevolve campaign list`. Run proxy cells with
`autoevolve campaign run <name> --proxy`. Add `--cell <key>` to select one cell.
Use `--full` only for the opt-in full budget. Reconstruct ladder state from the
database with `autoevolve campaign report <name>`.
