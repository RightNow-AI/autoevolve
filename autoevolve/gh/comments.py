"""Pure Markdown renderers for GitHub issue mode."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def contract_proposal_comment(
    goal: str,
    config: dict[str, Any],
    evaluator_source: str | None,
    synthesized: bool,
) -> str:
    """Render the reviewable contract proposal posted before approval."""

    if synthesized and evaluator_source is None:
        raise ValueError("Synthesized evaluator source is required for review.")

    metric = str(config.get("metric", "defined by evaluator"))
    target = _number(config["target"]) if "target" in config else "maximize"
    budget = _budget_text(config)
    lines = [
        "Proposed contract. No evaluator or candidate code has run.",
        "",
        "```text",
        "CONTRACT",
        f"goal: {_single_line(goal)}",
        f"metric: {metric}  baseline: measured after approval  target: {target}",
        f"gate: evaluator correctness gate   budget: {budget}",
        "feasibility: measured after approval",
        "```",
        "",
    ]

    evaluator = config.get("evaluator")
    if synthesized:
        opening_fence, closing_fence = _code_fence(evaluator_source or "")
        lines.extend(
            [
                "Synthesized evaluator source for maintainer review:",
                "",
                opening_fence,
                evaluator_source or "",
                closing_fence,
                "",
            ]
        )
    elif evaluator:
        lines.extend([f"Evaluator: `{evaluator}`", ""])

    lines.append(
        "Applying the `evolve:approved` label records maintainer consent and starts execution."
    )
    return _clean("\n".join(lines))


def approval_declined_comment(actor: str, reason: str) -> str:
    """Render a polite approval-integrity rejection."""

    return _clean(
        f"Approval from @{actor} was not accepted. {_as_sentence(reason)} "
        "No evaluator or candidate code was executed."
    )


def milestone_comment(
    run_id: str,
    evals_used: int,
    budget: int,
    best_fitness: float | None,
    baseline: float | None,
    curve_rows: list[tuple[int, float]],
    artifacts_note: str,
) -> str:
    """Render one measured progress update for a running approved contract."""

    rows = [
        "| Evaluation | Best fitness |",
        "|---:|---:|",
    ]
    rows.extend(f"| {index} | {_number(fitness)} |" for index, fitness in curve_rows)
    if not curve_rows:
        rows.append("| none recorded | none recorded |")

    text = "\n".join(
        [
            "### Run milestone",
            "",
            f"Run id: `{run_id}`",
            f"Evaluations for run `{run_id}`: {evals_used} of {budget}.",
            f"Baseline for run `{run_id}`: {_number_or_pending(baseline)}.",
            f"Best fitness for run `{run_id}`: {_number_or_pending(best_fitness)}.",
            "",
            *rows,
            "",
            _as_sentence(artifacts_note),
        ]
    )
    return _clean(text)


def terminal_comment(run_id: str, status: str, summary_paragraph: str) -> str:
    """Render the final issue comment for a completed or failed run."""

    summary = " ".join(summary_paragraph.split())
    return _clean(
        f"### Run complete\n\nRun id: `{run_id}`\n\nStatus: `{status}`\n\n{_as_sentence(summary)}"
    )


def ceiling_analysis_comment(run_id: str, analysis: str) -> str:
    """Render the successful infeasibility outcome."""

    paragraph = " ".join(analysis.split())
    return _clean(
        f"### Contract feasibility result\n\nRun id: `{run_id}`\n\n"
        f"{_as_sentence(paragraph)} No evolution was started for run `{run_id}`."
    )


def configuration_error_comment(reason: str) -> str:
    """Render a parse or proposal-generation error without exposing a traceback."""

    return _clean(
        f"I could not prepare the contract proposal. {_as_sentence(reason)} "
        "No evaluator or candidate code was executed."
    )


def _budget_text(config: dict[str, Any]) -> str:
    bounds = [f"{config.get('budget_evals', 150)} evaluations"]
    if "wall_clock_s" in config:
        bounds.append(f"{_number(config['wall_clock_s'])} seconds")
    return ", ".join(bounds)


def _number_or_pending(value: float | None) -> str:
    return "not available" if value is None else _number(value)


def _number(value: object) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _as_sentence(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "No additional detail was provided."
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def _clean(value: str) -> str:
    return value.replace("\N{EM DASH}", "-")


def _code_fence(source: str) -> tuple[str, str]:
    fence = "```"
    while fence in source:
        fence += "`"
    return f"{fence}python", fence


def contains_em_dash(comments: Iterable[str]) -> bool:
    """Return whether rendered comments contain the forbidden punctuation."""

    return any("\N{EM DASH}" in comment for comment in comments)
