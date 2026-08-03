"""Deterministic archive sampling and discovery ranking."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from autoevolve.core.archive import Elite

LOCAL_PROBABILITY = 0.8


@dataclass(frozen=True)
class ParentChoice:
    """A selected parent plus the exact deterministic replay trace."""

    elite: Elite
    trace: dict[str, Any]


def seeded_rng(run_seed: int, kind: str, event_seq: int) -> random.Random:
    """Create the only allowed random stream for a run decision."""

    return random.Random(f"{run_seed}:{kind}:{event_seq}")


def rank_elites(elites: list[Elite]) -> list[Elite]:
    """Sort elites by descending fitness with a stable program-id tie break."""

    return sorted(elites, key=lambda item: (-item.fitness, item.program.id))


def rank_weights(elites: list[Elite]) -> list[float]:
    """Give the highest-ranked elite the largest integer weight."""

    return [float(weight) for weight in range(len(elites), 0, -1)]


def weighted_index(rng: random.Random, weights: list[float]) -> int:
    """Choose an index using one reproducible uniform draw."""

    if not weights or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("weights must contain at least one positive value")
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(weights) - 1


def choose_from_pool(
    elites: list[Elite],
    rng: random.Random,
    source: str,
) -> ParentChoice:
    """Choose one rank-weighted elite and retain its ordered replay pool."""

    ranked = rank_elites(elites)
    if not ranked:
        raise ValueError("cannot sample an empty elite pool")
    weights = rank_weights(ranked)
    index = weighted_index(rng, weights)
    return ParentChoice(
        elite=ranked[index],
        trace={
            "source": source,
            "candidate_ids": [elite.program.id for elite in ranked],
            "weights": weights,
            "chosen_index": index,
            "chosen_parent_id": ranked[index].program.id,
        },
    )


def choose_parent(elites: list[Elite], island: int, rng: random.Random) -> ParentChoice:
    """Sample with an 80 percent own-island bias and a global fallback."""

    if not elites:
        raise ValueError("run archive is empty")
    local = [elite for elite in elites if elite.program.island == island]
    branch_draw: float | None = None
    if local:
        branch_draw = rng.random()
        pool = local if branch_draw < LOCAL_PROBABILITY else elites
        source = "local" if pool is local else "global"
    else:
        pool = elites
        source = "global"
    choice = choose_from_pool(pool, rng, source)
    trace = dict(choice.trace)
    trace["branch_draw"] = branch_draw
    trace["local_probability"] = LOCAL_PROBABILITY
    trace["local_candidate_ids"] = [elite.program.id for elite in rank_elites(local)]
    trace["global_candidate_ids"] = [elite.program.id for elite in rank_elites(elites)]
    return ParentChoice(choice.elite, trace)


def inspirations(elites: list[Elite], parent_id: str, limit: int = 3) -> list[Elite]:
    """Return the top elites from distinct cells, excluding the parent."""

    result: list[Elite] = []
    seen_cells: set[str] = set()
    for elite in rank_elites(elites):
        if elite.program.id == parent_id or elite.cell_key in seen_cells:
            continue
        seen_cells.add(elite.cell_key)
        result.append(elite)
        if len(result) == limit:
            break
    return result


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def rank_discoveries(
    rows: list[dict[str, Any]],
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rank discoveries by keyword overlap, then recency and identifier."""

    query_words = _keywords(query)
    return sorted(
        rows,
        key=lambda row: (
            len(query_words & _keywords(str(row["text"]))),
            str(row["created_at"]),
            str(row["id"]),
        ),
        reverse=True,
    )[:limit]
