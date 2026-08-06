"""Reference PICEF integer program for generated kidney exchange cells.

This module is deliberately outside the evaluator gate path. SciPy is imported
only when the reference solve is requested.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_EVALUATOR_PATH = Path(__file__).parent / "evaluators" / "kidney" / "evaluate.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PicefResult:
    """Measured incumbent, bound, certificate, and selected packing.

    ``dual_bound`` is converted back to maximization sense, so it is an upper
    bound on the best possible transplant objective.
    """

    cell: str
    incumbent_objective: int | None
    dual_bound: float | None
    integer_upper_bound: int | None
    optimality_proven: bool
    status: int
    message: str
    mip_gap: float | None
    model_build_s: float
    solve_s: float
    variable_count: int
    constraint_count: int
    enumerated_cycle_count: int
    chain_edge_variable_count: int
    cycles: tuple[tuple[int, ...], ...]
    chains: tuple[tuple[int, ...], ...]

    def wire(self) -> dict[str, object]:
        """Return a JSON-safe report without changing any measured value."""

        return {
            "cell": self.cell,
            "incumbent_objective": self.incumbent_objective,
            "dual_bound": self.dual_bound,
            "integer_upper_bound": self.integer_upper_bound,
            "optimality_proven": self.optimality_proven,
            "status": self.status,
            "message": self.message,
            "mip_gap": self.mip_gap,
            "model_build_s": self.model_build_s,
            "solve_s": self.solve_s,
            "variable_count": self.variable_count,
            "constraint_count": self.constraint_count,
            "enumerated_cycle_count": self.enumerated_cycle_count,
            "chain_edge_variable_count": self.chain_edge_variable_count,
            "cycles": [list(cycle) for cycle in self.cycles],
            "chains": [list(chain) for chain in self.chains],
        }


def _finite_result_value(result: object, name: str) -> float | None:
    raw = getattr(result, name, None)
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _rounded_objective(value: float) -> int:
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError(f"HiGHS returned a nonintegral incumbent objective {value!r}")
    return int(rounded)


def _decode_chains(
    values: Sequence[float],
    cycle_count: int,
    altruists: tuple[int, ...],
    adjacency: Sequence[Sequence[int]],
    pair_count: int,
    chain_cap: int,
) -> tuple[tuple[int, ...], ...]:
    selected: dict[tuple[int, int], int] = {}
    selected_edge_count = 0
    variable = cycle_count

    for altruist in altruists:
        for patient in adjacency[altruist]:
            if values[variable] > 0.5:
                selected[(altruist, 1)] = patient
                selected_edge_count += 1
            variable += 1

    for position in range(2, chain_cap + 1):
        for donor in range(pair_count):
            for patient in adjacency[donor]:
                if values[variable] > 0.5:
                    key = (donor, position)
                    if key in selected:
                        raise RuntimeError(
                            f"PICEF incumbent selected two outgoing edges for {key}"
                        )
                    selected[key] = patient
                    selected_edge_count += 1
                variable += 1

    chains: list[tuple[int, ...]] = []
    decoded_edge_count = 0
    for altruist in altruists:
        first = selected.get((altruist, 1))
        if first is None:
            continue
        chain = [altruist, first]
        decoded_edge_count += 1
        donor = first
        for position in range(2, chain_cap + 1):
            patient = selected.get((donor, position))
            if patient is None:
                break
            chain.append(patient)
            donor = patient
            decoded_edge_count += 1
        chains.append(tuple(chain))

    if decoded_edge_count != selected_edge_count:
        raise RuntimeError(
            "PICEF incumbent contained a selected chain edge outside a decoded chain"
        )
    return tuple(chains)


def solve_picef(
    instance: Any,
    enumerate_cycles: Callable[[Any], tuple[tuple[int, ...], ...]],
    time_limit_s: float,
) -> PicefResult:
    """Solve one instance with position-indexed chain edge variables.

    Cycles are explicit binary set-packing variables. Chain variables use
    ``y[u, v, k]`` for a transplant edge at position ``k``. Position one edges
    leave altruists, later positions leave pair vertices, and flow constraints
    connect consecutive positions.
    """

    if not math.isfinite(time_limit_s) or time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be a positive finite number")

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    build_started = time.perf_counter()
    pair_count = int(instance.cell.pair_count)
    chain_cap = int(instance.cell.chain_cap)
    if chain_cap < 1:
        raise ValueError("chain_cap must be at least one")
    altruists = tuple(int(vertex) for vertex in instance.altruists)
    cycles = enumerate_cycles(instance)

    pair_edge_count = 0
    for donor in range(pair_count):
        for raw_patient in instance.adjacency[donor]:
            patient = int(raw_patient)
            if not 0 <= patient < pair_count or patient == donor:
                raise ValueError(f"invalid pair edge {donor}->{patient}")
            pair_edge_count += 1

    altruist_edge_count = 0
    for altruist in altruists:
        for raw_patient in instance.adjacency[altruist]:
            patient = int(raw_patient)
            if not 0 <= patient < pair_count:
                raise ValueError(f"invalid altruist edge {altruist}->{patient}")
            altruist_edge_count += 1

    cycle_count = len(cycles)
    chain_edge_variable_count = altruist_edge_count + (chain_cap - 1) * pair_edge_count
    variable_count = cycle_count + chain_edge_variable_count
    flow_positions = max(chain_cap - 1, 0)
    altruist_row = {vertex: pair_count + index for index, vertex in enumerate(altruists)}
    flow_row_start = pair_count + len(altruists)
    constraint_count = flow_row_start + pair_count * flow_positions

    def flow_row(vertex: int, position: int) -> int:
        return flow_row_start + vertex * flow_positions + position - 1

    cycle_nnz = sum(len(cycle) for cycle in cycles)
    altruist_edge_nnz = altruist_edge_count * (3 if chain_cap > 1 else 2)
    pair_edge_nnz = 0
    if chain_cap >= 2:
        pair_edge_nnz = pair_edge_count * (3 * max(chain_cap - 2, 0) + 2)
    nonzero_count = cycle_nnz + altruist_edge_nnz + pair_edge_nnz

    rows = np.empty(nonzero_count, dtype=np.int32)
    columns = np.empty(nonzero_count, dtype=np.int32)
    coefficients = np.empty(nonzero_count, dtype=np.int8)
    objective = np.empty(variable_count, dtype=np.float64)
    cursor = 0

    for variable, cycle in enumerate(cycles):
        objective[variable] = -float(len(cycle))
        for vertex in cycle:
            rows[cursor] = vertex
            columns[cursor] = variable
            coefficients[cursor] = 1
            cursor += 1

    variable = cycle_count
    for altruist in altruists:
        for patient in instance.adjacency[altruist]:
            objective[variable] = -1.0
            rows[cursor] = patient
            columns[cursor] = variable
            coefficients[cursor] = 1
            cursor += 1
            rows[cursor] = altruist_row[altruist]
            columns[cursor] = variable
            coefficients[cursor] = 1
            cursor += 1
            if chain_cap > 1:
                rows[cursor] = flow_row(patient, 1)
                columns[cursor] = variable
                coefficients[cursor] = -1
                cursor += 1
            variable += 1

    for position in range(2, chain_cap + 1):
        for donor in range(pair_count):
            for patient in instance.adjacency[donor]:
                objective[variable] = -1.0
                rows[cursor] = patient
                columns[cursor] = variable
                coefficients[cursor] = 1
                cursor += 1
                rows[cursor] = flow_row(donor, position - 1)
                columns[cursor] = variable
                coefficients[cursor] = 1
                cursor += 1
                if position < chain_cap:
                    rows[cursor] = flow_row(patient, position)
                    columns[cursor] = variable
                    coefficients[cursor] = -1
                    cursor += 1
                variable += 1

    if cursor != nonzero_count or variable != variable_count:
        raise RuntimeError("internal PICEF sparse matrix size mismatch")

    matrix = coo_matrix(
        (coefficients, (rows, columns)),
        shape=(constraint_count, variable_count),
    ).tocsr()
    del rows, columns, coefficients
    upper_bounds = np.zeros(constraint_count, dtype=np.float64)
    upper_bounds[:flow_row_start] = 1.0
    constraints = LinearConstraint(matrix, lb=-np.inf, ub=upper_bounds)
    integrality = np.ones(variable_count, dtype=np.uint8)
    model_build_s = time.perf_counter() - build_started

    solve_started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"time_limit": float(time_limit_s), "presolve": True},
    )
    solve_s = time.perf_counter() - solve_started

    minimized_incumbent = _finite_result_value(result, "fun")
    incumbent = (
        _rounded_objective(-minimized_incumbent) if minimized_incumbent is not None else None
    )
    minimized_dual_bound = _finite_result_value(result, "mip_dual_bound")
    dual_bound = -minimized_dual_bound if minimized_dual_bound is not None else None
    integer_upper_bound = (
        math.floor(dual_bound + 1e-5) if dual_bound is not None else None
    )
    optimality_proven = bool(
        incumbent is not None
        and (
            int(result.status) == 0
            or (integer_upper_bound is not None and integer_upper_bound <= incumbent)
        )
    )

    values = getattr(result, "x", None)
    selected_cycles: tuple[tuple[int, ...], ...] = ()
    selected_chains: tuple[tuple[int, ...], ...] = ()
    if values is not None and incumbent is not None:
        selected_cycles = tuple(
            cycle for variable, cycle in enumerate(cycles) if values[variable] > 0.5
        )
        selected_chains = _decode_chains(
            values,
            cycle_count,
            altruists,
            instance.adjacency,
            pair_count,
            chain_cap,
        )
        decoded_objective = sum(len(cycle) for cycle in selected_cycles) + sum(
            len(chain) - 1 for chain in selected_chains
        )
        if decoded_objective != incumbent:
            raise RuntimeError(
                "decoded PICEF packing does not match the measured incumbent objective"
            )

    return PicefResult(
        cell=str(instance.cell.key),
        incumbent_objective=incumbent,
        dual_bound=dual_bound,
        integer_upper_bound=integer_upper_bound,
        optimality_proven=optimality_proven,
        status=int(result.status),
        message=str(result.message),
        mip_gap=_finite_result_value(result, "mip_gap"),
        model_build_s=model_build_s,
        solve_s=solve_s,
        variable_count=variable_count,
        constraint_count=constraint_count,
        enumerated_cycle_count=cycle_count,
        chain_edge_variable_count=chain_edge_variable_count,
        cycles=selected_cycles,
        chains=selected_chains,
    )


def _load_evaluator(cell: str) -> tuple[str, ModuleType]:
    repo_root = str(_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    previous_cell = os.environ.get("AUTOEVOLVE_CELL")
    os.environ["AUTOEVOLVE_CELL"] = cell
    module_name = f"_autoevolve_kidney_reference_{cell.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, _EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the kidney exchange evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if previous_cell is None:
            os.environ.pop("AUTOEVOLVE_CELL", None)
        else:
            os.environ["AUTOEVOLVE_CELL"] = previous_cell
    return module_name, module


def solve_cell(cell: str, time_limit_s: float) -> PicefResult:
    """Load one committed generated cell and run the isolated reference solver."""

    module_name, evaluator = _load_evaluator(cell)
    try:
        return solve_picef(
            evaluator.INSTANCE,
            evaluator._enumerate_cycles,
            time_limit_s,
        )
    finally:
        sys.modules.pop(module_name, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="pairs-160-frontier")
    parser.add_argument("--time-limit-s", type=float, default=600.0)
    args = parser.parse_args()
    print(
        json.dumps(solve_cell(args.cell, args.time_limit_s).wire(), indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
