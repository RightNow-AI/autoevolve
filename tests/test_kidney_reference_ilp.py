from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns" / "kidney-exchange"
EVALUATOR_PATH = CAMPAIGN / "evaluators" / "kidney" / "evaluate.py"
REFERENCE_PATH = CAMPAIGN / "reference_ilp.py"


def _load_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
    name: str,
) -> ModuleType:
    monkeypatch.setenv("AUTOEVOLVE_CELL", cell)
    spec = importlib.util.spec_from_file_location(name, EVALUATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _load_reference(name: str) -> ModuleType:
    pytest.importorskip("scipy.optimize")
    spec = importlib.util.spec_from_file_location(name, REFERENCE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _adjacency_hash(adjacency: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    for row in adjacency:
        digest.update(len(row).to_bytes(8, "little"))
        for target in row:
            digest.update(target.to_bytes(8, "little"))
    return digest.hexdigest()


def _legacy_build_adjacency(
    evaluator: ModuleType,
    spec: object,
    pairs: tuple[object, ...],
    altruist_blood: tuple[str, ...],
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed ^ 0x4B49_444E_4559)
    donor_blood = tuple(pair.donor_blood for pair in pairs) + altruist_blood
    rows: list[tuple[int, ...]] = []
    for donor_vertex, blood in enumerate(donor_blood):
        targets: list[int] = []
        for patient_vertex, pair in enumerate(pairs):
            if donor_vertex == patient_vertex:
                continue
            if not evaluator._abo_compatible(blood, pair.patient_blood):
                continue
            if rng.randrange(100) < evaluator._POSITIVE_CROSSMATCH_PERCENT[pair.pra_tier]:
                continue
            targets.append(patient_vertex)
        rows.append(tuple(targets))
    assert len(rows) == spec.pair_count + spec.altruist_count
    return tuple(rows)


def _legacy_generate_instance(evaluator: ModuleType, spec: object) -> object:
    for attempt in range(1_000):
        seed = spec.seed + attempt
        rng = random.Random(seed)
        pairs = tuple(
            evaluator._sample_incompatible_pair(rng) for _ in range(spec.pair_count)
        )
        altruist_blood = tuple(
            evaluator._draw_weighted(rng, evaluator._BLOOD_WEIGHTS)
            for _ in range(spec.altruist_count)
        )
        instance = evaluator.Instance(
            cell=spec,
            generation_seed=seed,
            pairs=pairs,
            altruist_blood=altruist_blood,
            adjacency=_legacy_build_adjacency(
                evaluator,
                spec,
                pairs,
                altruist_blood,
                seed,
            ),
        )
        if not spec.require_validation_shape or evaluator._validation_shape_is_useful(instance):
            return instance
    raise AssertionError("legacy validation graph generation exhausted its attempt cap")


@pytest.mark.parametrize(
    "cell",
    ["small-validation", "pairs-80-frontier", "pairs-160-frontier"],
)
def test_existing_cells_keep_legacy_adjacency_hash(
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
) -> None:
    evaluator = _load_evaluator(monkeypatch, cell, f"test_kidney_legacy_{cell}")
    legacy = _legacy_generate_instance(evaluator, evaluator._CELLS[cell])

    assert evaluator.INSTANCE.generation_seed == legacy.generation_seed
    assert evaluator.INSTANCE.pairs == legacy.pairs
    assert evaluator.INSTANCE.altruist_blood == legacy.altruist_blood
    assert _adjacency_hash(evaluator.INSTANCE.adjacency) == _adjacency_hash(legacy.adjacency)


def test_large_frontier_cell_has_requested_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_large_cell_shape",
    )
    cell = evaluator._CELLS["pairs-5000-frontier"]

    assert cell.pair_count == 5_000
    assert cell.altruist_count == 100
    assert cell.cycle_cap == 3
    assert cell.chain_cap == 8
    assert cell.timeout_s == 60.0
    assert cell.require_validation_shape is False


def test_evaluator_does_not_import_reference_solver() -> None:
    source = EVALUATOR_PATH.read_text(encoding="utf-8")

    assert "reference_ilp" not in source
    assert "scipy" not in source


def test_picef_matches_independent_validation_dynamic_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_picef_validation",
    )
    reference = _load_reference("test_kidney_reference_validation")

    result = reference.solve_picef(
        evaluator.INSTANCE,
        evaluator._enumerate_cycles,
        time_limit_s=30.0,
    )
    optimum = evaluator.exact_validation_optimum()
    solution = evaluator.Solution(cycles=result.cycles, chains=result.chains)

    assert result.optimality_proven is True
    assert result.incumbent_objective == optimum
    assert result.dual_bound == pytest.approx(float(optimum))
    assert result.integer_upper_bound == optimum
    assert evaluator._verify_solution(solution).transplants == optimum


def test_picef_tiny_two_cycle_and_two_edge_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator(
        monkeypatch,
        "small-validation",
        "test_kidney_picef_tiny_evaluator",
    )
    reference = _load_reference("test_kidney_reference_tiny")
    cell = evaluator.CellSpec(
        key="tiny-reference",
        seed=1,
        pair_count=4,
        altruist_count=1,
        cycle_cap=2,
        chain_cap=2,
        timeout_s=5.0,
    )
    pair = evaluator.PairProfile("O", "O", "low")
    instance = evaluator.Instance(
        cell=cell,
        generation_seed=1,
        pairs=(pair, pair, pair, pair),
        altruist_blood=("O",),
        adjacency=((1,), (0,), (3,), (), (2,)),
    )

    result = reference.solve_picef(
        instance,
        evaluator._enumerate_cycles,
        time_limit_s=10.0,
    )
    solution = evaluator.Solution(cycles=result.cycles, chains=result.chains)

    assert result.optimality_proven is True
    assert result.incumbent_objective == 4
    assert result.dual_bound == pytest.approx(4.0)
    assert result.integer_upper_bound == 4
    assert evaluator._verify_solution(solution, instance).transplants == 4
