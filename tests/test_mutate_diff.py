import random

import pytest

from autoevolve.core.types import Budget, Contract, EvalOutcome, ParentBundle, Program
from autoevolve.mutate.base import OperatorContext, OperatorError
from autoevolve.mutate.diff import DiffOperator
from tests.fakes import FakeEndpoint


def _bundle() -> ParentBundle:
    program = Program("p1", "r1", None, "seed", "ref", 0, None, "now")
    return ParentBundle(
        program,
        {
            "main.py": (
                "frozen = 1\n"
                "# EVOLVE-BLOCK-START\n"
                "value = 1\n"
                "# EVOLVE-BLOCK-END\n"
                "tail = 2\n"
            )
        },
    )


def _context(endpoint, tmp_path) -> OperatorContext:
    return OperatorContext(
        contract=Contract(
            "increase value",
            "general",
            "score",
            True,
            1.0,
            2.0,
            "correct",
            Budget(max_evals=10),
        ),
        rng=random.Random(1),
        endpoint_cheap=endpoint,
        endpoint_strong=None,
        evaluate_locally=lambda files: EvalOutcome(True, {"score": 2.0}, 0),
        workdir=tmp_path,
    )


def test_diff_mutates_marker_content_and_reports_counts(tmp_path):
    endpoint = FakeEndpoint(
        [
            """<<<<<<< SEARCH main.py
value = 1
=======
value = 2
>>>>>>> REPLACE
<<<<<<< SEARCH missing.py
missing
=======
replacement
>>>>>>> REPLACE
"""
        ]
    )

    proposal = DiffOperator().propose(_bundle(), _context(endpoint, tmp_path))

    assert proposal.files["main.py"] == (
        "frozen = 1\n"
        "# EVOLVE-BLOCK-START\n"
        "value = 2\n"
        "# EVOLVE-BLOCK-END\n"
        "tail = 2\n"
    )
    assert proposal.notes == "diff: applied=1 failed=1"
    prompt = endpoint.calls[0][1]["content"]
    assert "Goal: increase value" in prompt
    assert "ONLY content" in prompt


def test_diff_raises_when_zero_blocks_apply(tmp_path):
    endpoint = FakeEndpoint(
        [
            """<<<<<<< SEARCH main.py
not present
=======
changed
>>>>>>> REPLACE
"""
        ]
    )

    with pytest.raises(OperatorError, match="zero blocks"):
        DiffOperator().propose(_bundle(), _context(endpoint, tmp_path))
