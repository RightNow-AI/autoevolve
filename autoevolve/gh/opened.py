"""Safe proposal handling for opened and edited issues.

This module deliberately has no import path to the engine, evaluator sandbox,
or process execution. It parses issue text, optionally accepts generated source
from an injected source-only callback, and posts exactly one review comment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from autoevolve.gh.comments import configuration_error_comment, contract_proposal_comment
from autoevolve.gh.parse import extract_config, extract_goal


class CommentClient(Protocol):
    """The only GitHub capability available to the proposal handler."""

    def post_comment(self, issue_number: int, body: str) -> dict[str, Any]: ...


def handle_opened(
    client: CommentClient,
    issue: dict[str, Any],
    synthesize_source: Callable[[str], str],
) -> None:
    """Post one proposal comment without measuring or executing issue code."""

    issue_number = int(issue["number"])
    try:
        goal = extract_goal(str(issue.get("title", "")), issue.get("body"))
        config = extract_config(issue.get("body"))
        synthesized = "evaluator" not in config
        evaluator_source = synthesize_source(goal) if synthesized else None
        comment = contract_proposal_comment(goal, config, evaluator_source, synthesized)
    except (KeyError, TypeError, ValueError) as exc:
        comment = configuration_error_comment(str(exc))
    except Exception as exc:
        comment = configuration_error_comment(
            f"Evaluator source generation failed before execution. Error type: "
            f"{type(exc).__name__}"
        )

    client.post_comment(issue_number, comment)
