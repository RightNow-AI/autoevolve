"""Product UX checks for tool descriptions and agent instructions."""

from __future__ import annotations

import asyncio
from pathlib import Path

from autoevolve.mcp.server import build_server

ROOT = Path(__file__).parents[1]


def test_every_tool_description_is_nonempty_and_states_return_shape() -> None:
    tools = asyncio.run(build_server(engine=object()).list_tools())

    assert len(tools) == 9
    for tool in tools:
        assert tool.description
        assert "Returns:" in tool.description


def test_skill_frontmatter_and_required_worker_guidance() -> None:
    text = (ROOT / "skill" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, body = text.split("---", 2)
    fields = {
        key.strip(): value.strip()
        for line in frontmatter.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }

    assert fields["name"] == "autoevolve-worker"
    assert fields["description"]
    assert "EVOLVE-BLOCK" in body
    assert "submit_child" in body
    assert "run_status" in body


def test_agent_docs_have_no_em_dash_or_placeholders() -> None:
    files = [
        ROOT / "AGENTS.md",
        ROOT / "autoevolve" / "mcp" / "server.py",
        *(path for path in (ROOT / "skill").rglob("*") if path.is_file()),
    ]

    forbidden_placeholder = "TO" + "DO"
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert chr(0x2014) not in text, path
        assert forbidden_placeholder not in text, path


def test_root_agents_mirrors_worker_and_honesty_rules() -> None:
    path = ROOT / "AGENTS.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8").lower()
    assert "worker loop" in text
    assert "honesty rules" in text
    assert "claude.md" in text
    assert "docs/architecture.md" in text
