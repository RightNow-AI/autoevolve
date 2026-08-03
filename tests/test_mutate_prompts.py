from autoevolve.core.types import Budget, Contract, ParentBundle, Program
from autoevolve.mutate.prompts import build_diff_prompt, build_rewrite_prompt


def _program(program_id: str) -> Program:
    return Program(program_id, "r1", None, "diff", f"store/{program_id}", 0, None, "now")


def test_operator_prompts_include_contract_context_and_output_law():
    bundle = ParentBundle(
        _program("parent"),
        {
            "main.py": "# EVOLVE-BLOCK-START\nvalue = 1\n# EVOLVE-BLOCK-END\n",
            "context.txt": "read only\n",
        },
        inspirations=[
            (_program("i1"), {"score": 3.0}),
            (_program("i2"), {"score": 2.0}),
            (_program("i3"), {"score": 1.0}),
            (_program("i4"), {"score": 0.0}),
        ],
        discoveries=["Unrolling the inner loop improved the measured score."],
    )
    contract = Contract(
        "make the loop faster",
        "python-speedup",
        "milliseconds",
        False,
        10.0,
        5.0,
        "correct",
        Budget(max_evals=10),
    )

    diff_prompt = build_diff_prompt(bundle, contract)
    rewrite_prompt = build_rewrite_prompt(bundle, contract)

    for prompt in (diff_prompt, rewrite_prompt):
        assert "Goal: make the loop faster" in prompt
        assert "Metric: milliseconds (minimize)" in prompt
        assert "Target: 5.0" in prompt
        assert "ONLY content" in prompt
        assert "main.py [contains mutable regions]" in prompt
        assert "context.txt [read-only context]" in prompt
        # Inspirations must carry SOURCE, not an unusable content hash.
        assert "### ELITE i1" in prompt
        assert "Where this parent stands:" in prompt
        assert "scores=[score=3.0]" in prompt
        assert "### ELITE i3" in prompt
        assert "i4" not in prompt
        assert "- Unrolling the inner loop improved the measured score." in prompt
    assert "<<<<<<< SEARCH relative/path.py" in diff_prompt
    assert "### FILE: relative/path.py" in rewrite_prompt


def test_prompt_shows_standing_failures_and_inspiration_source():
    """The model must see its score, the best score, and why attempts failed."""

    from autoevolve.core.types import Budget, Contract, ParentBundle, Program
    from autoevolve.mutate.prompts import build_diff_prompt

    contract = Contract(
        goal="shorten the ruler",
        domain="golomb",
        metric="length",
        maximize=False,
        baseline=221.0,
        target=None,
        gate="golomb",
        budget=Budget(max_evals=10),
    )
    parent = Program("p1", "r1", None, "diff", "ref1", 0, "0,0", "now")
    elite = Program("i1", "r1", None, "diff", "ref2", 1, "1,1", "now")
    bundle = ParentBundle(
        parent=parent,
        parent_files={"ruler.py": "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"},
        inspirations=[(elite, {"length": 640.0})],
        inspiration_files=[
            {"ruler.py": "# EVOLVE-BLOCK-START\nsmart = 'idea'\n# EVOLVE-BLOCK-END\n"}
        ],
        parent_scores={"length": 700.0},
        best_scores={"length": 623.0},
        recent_failures=["difference 12 repeats: marks[2],marks[5]"],
    )

    prompt = build_diff_prompt(bundle, contract)

    assert "Parent length: 700.0" in prompt
    assert "Best length anywhere in this run: 623.0" in prompt
    assert "Measured baseline length: 221.0" in prompt
    assert "This parent is NOT the current best" in prompt
    assert "difference 12 repeats" in prompt
    assert "smart = 'idea'" in prompt
    assert "no fixed target" in prompt
