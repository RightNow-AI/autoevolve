"""Public mutation operator interfaces."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autoevolve.core.types import Contract, EvalOutcome, ParentBundle, Proposal
from autoevolve.mutate.models import ModelEndpoint


@dataclass
class OperatorContext:
    """Runtime services supplied by the evolution loop to an operator."""

    contract: Contract
    rng: random.Random
    endpoint_cheap: ModelEndpoint | None
    endpoint_strong: ModelEndpoint | None
    evaluate_locally: Callable[[dict[str, str]], EvalOutcome]
    workdir: Path
    #: The pack's spec.md. It states what is measured, how much compute a
    #: candidate may use, and what a valid certificate looks like, which is the
    #: most informative thing anyone can hand a mutation operator. Operators
    #: were guessing at all of it. Empty when the pack ships no spec.
    spec_text: str = ""


class OperatorError(Exception):
    """An expected operator failure that should skip the current cycle.

    skip_cycle is the structural contract the core loop checks without
    importing this package: any exception carrying a truthy skip_cycle is a
    skipped cycle, not a crash.
    """

    skip_cycle = True

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Operator(Protocol):
    """One proposal strategy used by the evolution loop."""

    name: str

    def propose(self, bundle: ParentBundle, ctx: OperatorContext) -> Proposal:
        """Produce a candidate or raise OperatorError when no candidate is usable."""
