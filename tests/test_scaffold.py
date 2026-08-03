"""U0 scaffold sanity: the seam types import everywhere and round-trip."""

from autoevolve.core.types import (
    Budget,
    Contract,
    Descriptor,
    EvalError,
    StageSpec,
)


def test_contract_round_trips_through_json():
    c = Contract(
        goal="make it faster",
        domain="python-speedup",
        metric="speedup",
        maximize=True,
        baseline=1.0,
        target=10.0,
        gate="correct",
        budget=Budget(max_evals=200),
        descriptors=[Descriptor(name="len", metric="code_len", bins=8, lo=0, hi=4000)],
    )
    again = Contract.from_json(c.to_json())
    assert again == c


def test_unbounded_budget_is_detectable():
    assert not Budget(max_evals=None).is_bounded()
    assert Budget(max_evals=10).is_bounded()


def test_eval_error_carries_reason():
    err = EvalError("parity mismatch on fixture 3")
    assert err.reason == "parity mismatch on fixture 3"


def test_evaluator_author_import_surface():
    from autoevolve.eval.contract import EvalError as E
    from autoevolve.eval.contract import StageSpec as S

    assert E is EvalError and S is StageSpec
