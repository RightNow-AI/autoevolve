from __future__ import annotations

from autoevolve.gh.comments import (
    approval_declined_comment,
    ceiling_analysis_comment,
    configuration_error_comment,
    contract_proposal_comment,
    milestone_comment,
    terminal_comment,
)


def test_contract_proposal_has_contract_shape_consent_and_unmeasured_baseline() -> None:
    comment = contract_proposal_comment(
        "Make parsing faster",
        {
            "budget_evals": 25,
            "workers": 2,
            "metric": "cases_per_second",
            "target": 5000.0,
        },
        "def evaluate(candidate_dir, stage=0):\n    return {'gate': 1.0}",
        True,
    )

    assert "CONTRACT\ngoal: Make parsing faster" in comment
    assert (
        "metric: cases_per_second  baseline: measured after approval  target: 5000" in comment
    )
    assert "gate: evaluator correctness gate   budget: 25 evaluations" in comment
    assert "feasibility: measured after approval" in comment
    assert "```python" in comment
    assert "Applying the `evolve:approved` label records maintainer consent" in comment
    assert "baseline: 0" not in comment
    assert "baseline: 1" not in comment


def test_milestone_comment_renders_curve_table() -> None:
    comment = milestone_comment(
        "r123",
        20,
        100,
        4.5,
        1.25,
        [(0, 1.25), (10, 3.0), (20, 4.5)],
        "Artifacts will be attached at close",
    )

    assert "Run id: `r123`" in comment
    assert "| Evaluation | Best fitness |" in comment
    assert "| 20 | 4.5 |" in comment


def test_declined_comment_names_actor() -> None:
    comment = approval_declined_comment("visitor", "Write permission is required")

    assert "@visitor" in comment
    assert "No evaluator or candidate code was executed" in comment


def test_no_comment_renderer_emits_an_em_dash() -> None:
    comments = [
        contract_proposal_comment(
            "Goal",
            {"budget_evals": 10, "workers": 1, "evaluator": "eval/demo"},
            None,
            False,
        ),
        approval_declined_comment("actor", "reason"),
        milestone_comment("r1", 10, 20, 2.0, 1.0, [(10, 2.0)], "artifact note"),
        terminal_comment("r1", "target_hit", "Measured result"),
        ceiling_analysis_comment("r1", "Target exceeds ceiling"),
        configuration_error_comment("Bad value"),
    ]

    assert all("\N{EM DASH}" not in comment for comment in comments)
