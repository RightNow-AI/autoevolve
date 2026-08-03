from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import autoevolve.gh.action as action

FIXTURES = Path(__file__).parent / "fixtures" / "gh"


class FakeGitHub:
    def __init__(
        self,
        *,
        permission: str = "write",
        timeline: list[str] | None = None,
    ) -> None:
        self.permission = permission
        self.timeline = timeline
        self.calls: list[tuple[Any, ...]] = []

    def list_labels(self, issue_number: int) -> list[dict[str, str]]:
        self._record("list_labels")
        self.calls.append(("list_labels", issue_number))
        return [{"name": "evolve"}, {"name": "evolve:approved"}]

    def get_actor_permission(self, username: str) -> dict[str, str]:
        self._record("permission")
        self.calls.append(("permission", username))
        return {"permission": self.permission}

    def post_comment(self, issue_number: int, body: str) -> dict[str, Any]:
        if "### Run milestone" in body:
            self._record("milestone")
        elif "### Run complete" in body:
            self._record("terminal")
        else:
            self._record("comment")
        self.calls.append(("comment", issue_number, body))
        return {"id": len(self.calls)}

    def _record(self, name: str) -> None:
        if self.timeline is not None:
            self.timeline.append(name)


class FakeEngine:
    def __init__(self, events: list[str], *, infeasible: bool = False) -> None:
        self.events = events
        self.infeasible = infeasible
        self.closed = False
        self.submissions = 0

    def open_run(self, goal: str, **kwargs: Any) -> dict[str, Any]:
        self.events.append("open_run")
        contract = {
            "goal": goal,
            "metric": "cases_per_second",
            "maximize": True,
            "baseline": 100.0,
            "target": 5000.0,
            "gate": "syntax parity",
            "budget": {"max_evals": 25, "wall_clock_s": None},
            "feasibility": None,
        }
        if self.infeasible:
            contract["feasibility"] = {
                "infeasible": True,
                "value": 4000.0,
                "method": "measured ceiling",
                "maximum_plausible_target": 4000.0,
            }
            return {"run_id": "r-infeasible", "contract": contract, "status": "infeasible"}
        return {"run_id": "r-happy", "contract": contract}

    def run_status(self, run_id: str) -> dict[str, Any]:
        status = "target_hit" if self.closed else "open"
        return {
            "status": status,
            "curve": [[0, 100.0], [self.submissions, 100.0 + self.submissions]],
            "best_fitness": 100.0 + self.submissions,
            "artifacts": {},
        }

    def best(self, run_id: str, k: int = 5) -> list[dict[str, Any]]:
        return [{"files": {"parser.py": "def parse(value):\n    return value\n"}}]


def test_opened_posts_one_proposal_and_never_touches_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event = _fixture("issue_opened.json")
    client = FakeGitHub()

    def fake_synth(goal: str, workdir: Path) -> Path:
        evaluator = workdir / "evaluator"
        evaluator.mkdir()
        (evaluator / "evaluate.py").write_text(
            "def evaluate(candidate_dir, stage=0):\n    return {'gate': 1.0}\n",
            encoding="utf-8",
        )
        return evaluator

    monkeypatch.setattr(action, "_synthesize_evaluator", fake_synth)
    monkeypatch.setattr(action, "_new_engine", _unexpected)
    monkeypatch.setattr(action, "_run_loop", _unexpected)
    monkeypatch.setattr(action, "_render_run", _unexpected)

    result = action.dispatch(
        "issues",
        event,
        client,
        home=tmp_path / "home",
        workdir=tmp_path,
    )

    comments = [call for call in client.calls if call[0] == "comment"]
    assert result == 0
    assert len(comments) == 1
    assert "baseline: measured after approval" in comments[0][2]
    assert "Applying the `evolve:approved` label" in comments[0][2]


def test_wrong_label_exits_without_api_or_execution_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeGitHub()
    monkeypatch.setattr(action, "_new_engine", _unexpected)

    result = action.dispatch(
        "issues",
        _fixture("issue_labeled_wrong_label.json"),
        client,
        home=tmp_path / "home",
        workdir=tmp_path,
    )

    assert result == 0
    assert client.calls == []


def test_main_ignores_unrelated_event_without_requiring_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(_fixture("issue_labeled_wrong_label.json")),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issues")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(action, "GitHubClient", _unexpected)

    assert action.main() == 0


def test_nonwriter_is_declined_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = FakeGitHub(permission="read")
    monkeypatch.setattr(action, "_new_engine", _unexpected)
    monkeypatch.setattr(action, "_synthesize_evaluator", _unexpected)

    result = action.dispatch(
        "issues",
        _fixture("issue_labeled_nonwriter.json"),
        client,
        home=tmp_path / "home",
        workdir=tmp_path,
    )

    assert result == 0
    assert [call[0] for call in client.calls] == ["list_labels", "permission", "comment"]
    assert "@external-user" in client.calls[-1][2]
    assert "No evaluator or candidate code was executed" in client.calls[-1][2]


def test_approved_happy_path_runs_in_order_and_posts_cadenced_milestones(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event = _fixture("issue_labeled_approved.json")
    evaluator = tmp_path / "evaluators" / "parser-speed"
    evaluator.mkdir(parents=True)
    (evaluator / "evaluate.py").write_text("# reviewed evaluator\n", encoding="utf-8")
    timeline: list[str] = []
    client = FakeGitHub(timeline=timeline)
    engine = FakeEngine(timeline)

    monkeypatch.setattr(action, "_new_engine", lambda home: engine)

    def fake_loop(
        loop_engine: FakeEngine,
        run_id: str,
        **kwargs: Any,
    ) -> dict[str, str]:
        timeline.append("loop")
        callback = kwargs["on_submission"]
        for index in range(1, 26):
            loop_engine.submissions = index
            callback({"best_fitness": 100.0 + index})
        loop_engine.closed = True
        return {"status": "target_hit"}

    def fake_render(home: Path, run_id: str, out_dir: Path) -> dict[str, Path]:
        timeline.append("render")
        gif = out_dir / "evolution.gif"
        poster = out_dir / "lineage_poster.png"
        gif.write_bytes(b"GIF89a")
        poster.write_bytes(b"PNG")
        return {"gif": gif, "poster_png": poster}

    def fake_report(home: Path, run_id: str, out_path: Path) -> str:
        timeline.append("report")
        text = "# Report\n\nThe locked target was reached by a measured candidate.\n"
        out_path.write_text(text, encoding="utf-8")
        return text

    def fake_pr(client_arg: FakeGitHub, run_id: str, home: Path, workdir: Path) -> str:
        timeline.append("pr")
        assert (workdir / "winner" / "parser.py").is_file()
        return "https://github.test/acme/demo/pull/9"

    monkeypatch.setattr(action, "_run_loop", fake_loop)
    monkeypatch.setattr(action, "_render_run", fake_render)
    monkeypatch.setattr(action, "_report_run", fake_report)
    monkeypatch.setattr(action, "build_terminal_pr", fake_pr)

    result = action.dispatch(
        "issues",
        event,
        client,
        home=tmp_path / "home",
        workdir=tmp_path,
    )

    comments = [call[2] for call in client.calls if call[0] == "comment"]
    milestones = [comment for comment in comments if "### Run milestone" in comment]
    assert result == 0
    assert timeline == [
        "list_labels",
        "permission",
        "open_run",
        "loop",
        "milestone",
        "milestone",
        "render",
        "report",
        "terminal",
        "pr",
    ]
    assert len(milestones) == 2
    assert "Evaluations for run `r-happy`: 10 of 25" in milestones[0]
    assert "Evaluations for run `r-happy`: 20 of 25" in milestones[1]
    assert "### Run complete" in comments[-1]
    assert comments.index(milestones[0]) < comments.index(comments[-1])


def test_infeasible_contract_posts_ceiling_success_and_builds_no_pr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event = _fixture("issue_labeled_approved.json")
    evaluator = tmp_path / "evaluators" / "parser-speed"
    evaluator.mkdir(parents=True)
    (evaluator / "evaluate.py").write_text("# reviewed evaluator\n", encoding="utf-8")
    client = FakeGitHub()
    events: list[str] = []
    engine = FakeEngine(events, infeasible=True)
    monkeypatch.setattr(action, "_new_engine", lambda home: engine)
    monkeypatch.setattr(action, "_run_loop", _unexpected)
    monkeypatch.setattr(action, "build_terminal_pr", _unexpected)

    result = action.dispatch(
        "issues",
        event,
        client,
        home=tmp_path / "home",
        workdir=tmp_path,
    )

    comments = [call[2] for call in client.calls if call[0] == "comment"]
    assert result == 0
    assert events == ["open_run"]
    assert len(comments) == 1
    assert "Contract feasibility result" in comments[0]
    assert "maximum plausible target is 4000" in comments[0]


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _unexpected(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("Execution seam must not be called")
