from __future__ import annotations

import pytest

from autoevolve.gh.parse import extract_config, extract_goal


def test_extract_goal_strips_evolve_prefix_and_config() -> None:
    body = "Keep syntax unchanged.\n\n```autoevolve\nbudget_evals: 20\n```\nignored"

    assert extract_goal("evolve: make parsing faster", body) == (
        "make parsing faster\n\nKeep syntax unchanged."
    )


def test_extract_goal_without_prefix_uses_full_title_and_body() -> None:
    assert extract_goal("Make parsing faster", "Keep syntax unchanged.") == (
        "Make parsing faster\n\nKeep syntax unchanged."
    )


def test_extract_config_happy_path_and_unknown_keys() -> None:
    body = """Goal.

```autoevolve
budget_evals: 40
wall_clock_s: 12.5
workers: 3
evaluator: evaluators/parser
target: 9.25
metric: cases_per_second
target_path: src/parser
ignored_key: ignored
```
"""

    assert extract_config(body) == {
        "budget_evals": 40,
        "wall_clock_s": 12.5,
        "workers": 3,
        "evaluator": "evaluators/parser",
        "target": 9.25,
        "metric": "cases_per_second",
        "target_path": "src/parser",
    }


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("budget_evals: many", "budget_evals"),
        ("workers: 0", "workers"),
        ("wall_clock_s: never", "wall_clock_s"),
        ("target: inf", "target"),
        ("evaluator: ../outside", "evaluator"),
    ],
)
def test_extract_config_rejects_malformed_known_values(line: str, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        extract_config(f"```autoevolve\n{line}\n```")


def test_extract_config_defaults() -> None:
    assert extract_config(None) == {"budget_evals": 150, "workers": 2}
