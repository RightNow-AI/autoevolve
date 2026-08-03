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
    target = (
        "no fixed target, push as far as possible"
        if contract.target is None
        else str(contract.target)
    )
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
        _render_standing(bundle, contract),
        (
            "Mutation law: change ONLY content between lines containing "
            "EVOLVE-BLOCK-START and EVOLVE-BLOCK-END. Marker lines and all content "
            "outside them are immutable."
        ),
        _render_parent_files(bundle.parent_files),
        _render_inspirations(bundle),
        _render_failures(bundle.recent_failures),
        _render_discoveries(bundle.discoveries),
    ]
    return "\n\n".join(sections)


def _render_standing(bundle: ParentBundle, contract: Contract) -> str:
    """Tell the model where it stands. Without this it optimizes blind."""

    metric = contract.metric
    lines = ["Where this parent stands:"]
    parent_value = bundle.parent_scores.get(metric)
    best_value = bundle.best_scores.get(metric)
    if parent_value is None:
        lines.append("- The parent's score was not recorded.")
    else:
        lines.append(f"- Parent {metric}: {parent_value}")
    if best_value is not None:
        lines.append(f"- Best {metric} anywhere in this run: {best_value}")
    if contract.baseline is not None:
        lines.append(f"- Measured baseline {metric}: {contract.baseline}")
    if parent_value is not None and best_value is not None and parent_value != best_value:
        lines.append(
            "- This parent is NOT the current best. Beating the best above is what counts."
        )
    other = {
        key: value for key, value in sorted(bundle.parent_scores.items()) if key != metric
    }
    if other:
        detail = ", ".join(f"{key}={value}" for key, value in other.items())
        lines.append(f"- Parent's other measurements: {detail}")
    return "\n".join(lines)


def _render_failures(failures: list[str]) -> str:
    if not failures:
        return "Recent gate failures: none recorded."
    lines = [
        "Recent gate failures in this run. Do not repeat these mistakes:",
    ]
    lines.extend(f"- {reason}" for reason in failures)
    return "\n".join(lines)


def _render_parent_files(files: dict[str, str]) -> str:
    rendered = ["Parent files. Mutable regions are explicitly fenced in each file:"]
    for path, content in sorted(files.items()):
        mode = "contains mutable regions" if _has_markers(content) else "read-only context"
        rendered.append(f"\n### PARENT FILE: {path} [{mode}]\n````text\n{content}\n````")
    return "\n".join(rendered)


def _render_inspirations(bundle: ParentBundle) -> str:
    """Show other elites WITH their source, so they can actually inspire.

    A hash and a score teach nothing. The mutable region of each inspiration
    is what carries a transferable idea, so it is included verbatim.
    """

    if not bundle.inspirations:
        return "Other elites: none yet, this parent is the only one in the archive."
    lines = [
        "Other elites in the population. These passed the gate with the scores shown. "
        "Borrow what works and combine it with the parent:"
    ]
    for index, (program, scores) in enumerate(bundle.inspirations[:3]):
        score_text = ", ".join(f"{key}={value}" for key, value in sorted(scores.items()))
        lines.append(f"\n### ELITE {program.id} operator={program.operator} scores=[{score_text}]")
        files = (
            bundle.inspiration_files[index]
            if index < len(bundle.inspiration_files)
            else {}
        )
        excerpt = _mutable_excerpt(files)
        lines.append(f"````text\n{excerpt}\n````" if excerpt else "(source unavailable)")
    return "\n".join(lines)


def _mutable_excerpt(files: dict[str, str], limit: int = 4000) -> str:
    """Return the mutable regions of an inspiration, which is where its idea lives."""

    parts: list[str] = []
    for path, content in sorted(files.items()):
        if not _has_markers(content):
            continue
        for segment in content.split("EVOLVE-BLOCK-START")[1:]:
            body = segment.split("EVOLVE-BLOCK-END")[0]
            parts.append(f"# from {path}\n{body.strip()}")
    if not parts:
        return ""
    joined = "\n\n".join(parts)
    return joined if len(joined) <= limit else joined[:limit] + "\n# ... truncated"


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
