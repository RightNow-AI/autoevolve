## What this changes

<!-- One paragraph. What behavior is different after this merges. -->

## Evidence

<!-- Required. Paste the output you actually ran, not a description of it. -->

```
uv run ruff check .
uv run pytest -q
```

- [ ] Every fix ships a test that fails against the broken code.
- [ ] Any number in this PR or in the changed code carries its run id.
- [ ] Normative docs updated in this same commit if behavior they specify
      changed (docs/ARCHITECTURE.md, docs/CONTRACT.md, docs/CAMPAIGNS.md).

## Risk

<!-- What could this break, and what did you check to rule that out. Say
     "none known" only if you actually looked. -->
