import random

import pytest

from autoevolve.core.types import Budget, Contract, EvalOutcome, ParentBundle, Program
from autoevolve.mutate.base import OperatorContext, OperatorError
from autoevolve.mutate.crossover import CrossoverOperator


def _program(program_id: str) -> Program:
    return Program(program_id, "r1", None, "seed", program_id, 0, None, "now")


def _context(tmp_path, seed: int = 1) -> OperatorContext:
    return OperatorContext(
        contract=Contract(
            "combine",
            "general",
            "score",
            True,
            0.0,
            None,
            "correct",
            Budget(max_evals=10),
        ),
        rng=random.Random(seed),
        endpoint_cheap=None,
        endpoint_strong=None,
        evaluate_locally=lambda files: EvalOutcome(True, {"score": 1.0}, 0),
        workdir=tmp_path,
    )


def test_crossover_uses_seeded_region_choices_and_primary_frozen_text(tmp_path):
    primary = (
        "primary header\n"
        "# EVOLVE-BLOCK-START\n"
        "a1\n"
        "# EVOLVE-BLOCK-END\n"
        "primary middle\n"
        "# EVOLVE-BLOCK-START\n"
        "a2\n"
        "# EVOLVE-BLOCK-END\n"
        "primary footer\n"
    )
    other = (
        "other header\n"
        "# EVOLVE-BLOCK-START\n"
        "b1\n"
        "# EVOLVE-BLOCK-END\n"
        "other middle\n"
        "# EVOLVE-BLOCK-START\n"
        "b2\n"
        "# EVOLVE-BLOCK-END\n"
        "other footer\n"
    )
    bundle = ParentBundle(
        _program("pa"),
        {"main.py": primary, "only-a.txt": "primary only\n"},
        crossover_parent=_program("pb"),
        crossover_files={"main.py": other, "only-b.txt": "other only\n"},
    )

    proposal = CrossoverOperator().propose(bundle, _context(tmp_path, seed=1))

    assert proposal.files["main.py"] == (
        "primary header\n"
        "# EVOLVE-BLOCK-START\n"
        "a1\n"
        "# EVOLVE-BLOCK-END\n"
        "primary middle\n"
        "# EVOLVE-BLOCK-START\n"
        "b2\n"
        "# EVOLVE-BLOCK-END\n"
        "primary footer\n"
    )
    assert proposal.files["only-a.txt"] == "primary only\n"
    assert "only-b.txt" not in proposal.files
    assert proposal.notes == "crossover: main.py#region1=A, main.py#region2=B"


def test_crossover_requires_second_parent(tmp_path):
    bundle = ParentBundle(_program("pa"), {"main.py": "plain\n"})

    with pytest.raises(OperatorError, match="requires a crossover parent"):
        CrossoverOperator().propose(bundle, _context(tmp_path))
