from autoevolve.core.types import Budget, Contract
from autoevolve.eval.feasibility import check_feasibility


def _contract(*, maximize: bool, target: float | None) -> Contract:
    return Contract(
        goal="sort compactly",
        domain="python",
        metric="score",
        maximize=maximize,
        baseline=1.0,
        target=target,
        gate="correct",
        budget=Budget(max_evals=10),
    )


def test_target_inside_ceiling_is_feasible() -> None:
    result = check_feasibility(
        _contract(maximize=True, target=10.0),
        {"metric": "score", "value": 10.0, "method": "toy bound"},
    )

    assert result["feasible"] is True
    assert result["max_plausible"] == 10.0


def test_target_above_maximize_ceiling_is_infeasible() -> None:
    result = check_feasibility(
        _contract(maximize=True, target=11.0),
        {"metric": "score", "value": 10.0, "method": "toy bound"},
    )

    assert result["feasible"] is False
    assert result["max_plausible"] == 10.0


def test_target_below_minimize_ceiling_is_infeasible() -> None:
    result = check_feasibility(
        _contract(maximize=False, target=0.5),
        {"metric": "score", "value": 1.0, "method": "toy bound"},
    )

    assert result["feasible"] is False
    assert result["max_plausible"] == 1.0


def test_no_ceiling_is_always_feasible() -> None:
    result = check_feasibility(_contract(maximize=True, target=1_000_000.0), None)

    assert result == {
        "feasible": True,
        "reason": "evaluator defines no ceiling",
        "max_plausible": None,
    }


def test_no_target_is_always_feasible() -> None:
    result = check_feasibility(
        _contract(maximize=True, target=None),
        {"metric": "score", "value": 10.0, "method": "toy bound"},
    )

    assert result["feasible"] is True
    assert result["max_plausible"] == 10.0
