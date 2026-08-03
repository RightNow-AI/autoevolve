from __future__ import annotations

import json
from pathlib import Path

from autoevolve.gh.prpost import build_terminal_pr


class PullRequestClient:
    def __init__(self) -> None:
        self.files: dict[str, str | bytes] = {}
        self.body = ""

    def get_default_branch(self) -> dict[str, str]:
        return {"name": "main", "sha": "base-sha"}

    def create_branch(self, base_sha: str, name: str) -> dict[str, str]:
        assert (base_sha, name) == ("base-sha", "autoevolve/run-r7")
        return {"ref": f"refs/heads/{name}"}

    def put_files(
        self,
        branch: str,
        files: dict[str, str | bytes],
        message: str,
    ) -> dict[str, str]:
        self.files = files
        return {"sha": "commit-sha"}

    def create_pr(
        self,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> dict[str, str]:
        self.body = body
        return {"html_url": "https://github.test/acme/demo/pull/7"}


def test_terminal_pr_contains_winner_artifacts_and_relative_embeds(tmp_path: Path) -> None:
    (tmp_path / "winner").mkdir()
    (tmp_path / "winner" / "solution.py").write_text("answer = 42\n", encoding="utf-8")
    (tmp_path / "report.md").write_text("# Report\n\nMeasured result.\n", encoding="utf-8")
    (tmp_path / "evolution.gif").write_bytes(b"GIF89a")
    (tmp_path / "lineage_poster.png").write_bytes(b"PNG")
    (tmp_path / ".autoevolve-gh.json").write_text(
        json.dumps(
            {
                "target_path": "src/generated",
                "contract": "CONTRACT\ngoal: improve parser",
                "result": "The measured candidate reached the locked target.",
            }
        ),
        encoding="utf-8",
    )
    client = PullRequestClient()

    url = build_terminal_pr(client, "r7", tmp_path / "home", tmp_path)

    assert url.endswith("/pull/7")
    assert "src/generated/solution.py" in client.files
    assert "autoevolve-results/r7/report.md" in client.files
    assert "![Lineage poster](autoevolve-results/r7/lineage_poster.png)" in client.body
    assert "![Evolution](autoevolve-results/r7/evolution.gif)" in client.body
    assert client.body.endswith("Run id: r7")
    assert "Co-authored-by" not in client.body
    assert "Generated with" not in client.body
