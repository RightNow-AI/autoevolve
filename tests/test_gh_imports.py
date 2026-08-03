from __future__ import annotations

import ast
from pathlib import Path

import autoevolve.gh.opened as opened


def test_opened_handler_has_no_execution_import_path() -> None:
    source_path = Path(opened.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden = ("autoevolve.core", "autoevolve.eval", "sandbox", "subprocess")
    assert not any(name.startswith(prefix) for name in imports for prefix in forbidden)
