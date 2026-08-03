"""Persistent UCB1 operator policy tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from autoevolve.core.bandit import (
    relative_gain,
    select_operator,
    states,
    update_operator,
)
from autoevolve.core.db import init_db, transaction
from autoevolve.core.engine import Engine


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    configured = tmp_path / "home"
    monkeypatch.setenv("AUTOEVOLVE_HOME", str(configured))
    return configured


def test_unpulled_operators_are_preferred_with_name_tie_break(home: Path) -> None:
    init_db(home)
    with transaction(home) as conn:
        assert select_operator(conn, "domain", ("zeta", "alpha")) == "alpha"
        update_operator(conn, "domain", "alpha", 1.0, 2.0)
        assert select_operator(conn, "domain", ("zeta", "alpha")) == "zeta"


def test_ucb1_matches_hand_computed_scripted_history(home: Path) -> None:
    init_db(home)
    with transaction(home) as conn:
        update_operator(conn, "domain", "a", 1.0, 2.0)
        update_operator(conn, "domain", "a", 1.0, 1.0)
        update_operator(conn, "domain", "b", 1.0, 1.1)
        chosen = select_operator(conn, "domain", ("a", "b"))
        arms = {state.name: state for state in states(conn, "domain", ("a", "b"))}
    score_a = 0.5 + math.sqrt(2.0 * math.log(3) / 2)
    score_b = 0.1 + math.sqrt(2.0 * math.log(3))
    assert score_b > score_a
    assert chosen == "b"
    assert arms["a"].pulls == 2 and arms["a"].improvements == 1
    assert arms["a"].mean_gain == pytest.approx(0.5)


def test_gain_is_clipped_and_zero_parent_is_safe(home: Path) -> None:
    init_db(home)
    assert relative_gain(1.0, 100.0) == 1.0
    assert relative_gain(1.0, -100.0) == -1.0
    assert relative_gain(0.0, 0.0) == 0.0


def test_bandit_state_persists_across_engine_instances(home: Path) -> None:
    first = Engine(home)
    with transaction(first.home) as conn:
        update_operator(conn, "persistent", "diff", 2.0, 3.0)
    second = Engine(home)
    with transaction(second.home) as conn:
        state = states(conn, "persistent", ("diff",))[0]
    assert state.pulls == 1
    assert state.improvements == 1
    assert state.mean_gain == pytest.approx(0.5)
