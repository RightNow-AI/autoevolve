"""Dense product prompts for model-backed mutation operators."""

from __future__ import annotations

from autoevolve.core.types import Contract, ParentBundle


def build_diff_prompt(bundle: ParentBundle, contract: Contract) -> str:
    """Build the SEARCH/REPLACE mutation prompt."""

    return _common_prompt(bundle, contract) + "\n\n" + _diff_output_contract()


def build_rewrite_prompt(bundle: ParentBundle, contract: Contract) -> str:
    """Build the full-file rewrite mutation prompt."""

    return _common_prompt(bundle, contract) + "\n\n" + _rewrite_output_contract()


def _common_prompt(bundle: ParentBundle, contract: Contract) -> str:
    direction = "maximize" if contract.maximize else "minimize"
    target = direction if contract.target is None else str(contract.target)
    sections = [
        "Improve the parent against this locked contract.",
        "\n".join(
            (
                f"Goal: {contract.goal}",
                f"Metric: {contract.metric} ({direction})",
                f"Target: {target}",
                f"Correctness gate: {contract.gate}",
            )
        ),
        (
            "Mutation law: change ONLY content between lines containing "
            "EVOLVE-BLOCK-START and EVOLVE-BLOCK-END. Marker lines and all content "
            "outside them are immutable."
        ),
        _render_parent_files(bundle.parent_files),
        _render_inspirations(bundle),
        _render_discoveries(bundle.discoveries),
    ]
    return "\n\n".join(sections)


def _render_parent_files(files: dict[str, str]) -> str:
    rendered = ["Parent files. Mutable regions are explicitly fenced in each file:"]
    for path, content in sorted(files.items()):
        mode = "contains mutable regions" if _has_markers(content) else "read-only context"
        rendered.append(f"\n### PARENT FILE: {path} [{mode}]\n````text\n{content}\n````")
    return "\n".join(rendered)


def _render_inspirations(bundle: ParentBundle) -> str:
    if not bundle.inspirations:
        return "Inspirations: none supplied."
    lines = ["Inspirations. Source text is not carried by ParentBundle; use these references:"]
    for program, scores in bundle.inspirations[:3]:
        score_text = ", ".join(f"{key}={value}" for key, value in sorted(scores.items()))
        lines.append(
            f"- {program.id} path={program.code_ref} operator={program.operator} "
            f"scores=[{score_text}]"
        )
    return "\n".join(lines)


def _render_discoveries(discoveries: list[str]) -> str:
    if not discoveries:
        return "Prior knowledge: none supplied."
    return "Prior knowledge from earlier runs:\n" + "\n".join(
        f"- {discovery}" for discovery in discoveries
    )


def _has_markers(content: str) -> bool:
    return "EVOLVE-BLOCK-START" in content and "EVOLVE-BLOCK-END" in content


def _diff_output_contract() -> str:
    return """Return only one or more exact SEARCH/REPLACE blocks:
<<<<<<< SEARCH relative/path.py
exact original text including indentation
=======
replacement text including indentation
>>>>>>> REPLACE
Each search must match the parent exactly. Keep every edit inside an EVOLVE-BLOCK region."""


def _rewrite_output_contract() -> str:
    return """Return only complete files that contain EVOLVE-BLOCK markers, using this format:
### FILE: relative/path.py
```python
complete file content
```
Do not return read-only files. Preserve marker lines and all content outside marked regions
exactly."""
