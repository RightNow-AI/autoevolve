"""Shared seam types. Every package imports these from here and nowhere else.

This module is orchestrator-owned. Lanes may read it and import from it but
never modify it; interface change requests go through the lane report. The
shapes here are normative per docs/ARCHITECTURE.md section 4.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


class EvalError(Exception):
    """Raised by evaluators on gate failure. The reason is user-facing."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class StageSpec:
    """One stage of the evaluation cascade, cheap to expensive."""

    name: str
    timeout_s: float
    mem_mb: int | None = None
    cpu_s: float | None = None


@dataclass(frozen=True)
class Budget:
    """Run budget. At least one bound is required; unbounded runs are refused."""

    max_evals: int | None
    wall_clock_s: float | None = None
    max_cost_usd: float | None = None

    def is_bounded(self) -> bool:
        return any(v is not None for v in (self.max_evals, self.wall_clock_s, self.max_cost_usd))


@dataclass(frozen=True)
class Descriptor:
    """A MAP-elites behavior descriptor binned over a metric range."""

    name: str
    metric: str
    bins: int
    lo: float
    hi: float


@dataclass
class Contract:
    """The locked scoring contract. Immutable for the life of a run."""

    goal: str
    domain: str
    metric: str
    maximize: bool
    baseline: float | None
    target: float | None
    gate: str
    budget: Budget
    descriptors: list[Descriptor] = field(default_factory=list)
    feasibility: dict[str, Any] | None = None
    plateau_n: int = 150

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Contract:
        d = json.loads(raw)
        d["budget"] = Budget(**d["budget"])
        d["descriptors"] = [Descriptor(**x) for x in d.get("descriptors", [])]
        return cls(**d)


@dataclass(frozen=True)
class Program:
    """One candidate in the population. Code lives in the content store."""

    id: str
    run_id: str
    parent_id: str | None
    operator: str
    code_ref: str
    island: int
    cell_key: str | None
    created_at: str


@dataclass(frozen=True)
class EvalOutcome:
    """Result of running a candidate through the cascade."""

    gate_passed: bool
    scores: dict[str, float]
    stage_reached: int
    error: str | None = None


@dataclass(frozen=True)
class Proposal:
    """A mutation proposal: full file contents keyed by relative path."""

    files: dict[str, str]
    notes: str = ""


@dataclass
class ParentBundle:
    """Everything a mutation operator needs about the parent and context."""

    parent: Program
    parent_files: dict[str, str]
    inspirations: list[tuple[Program, dict[str, float]]] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)
    operator_hint: str | None = None
    crossover_parent: Program | None = None
    crossover_files: dict[str, str] | None = None
    parent_sample_seq: int | None = None
