"""Parse goals and the small configuration block accepted by issue mode."""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Any

DEFAULT_BUDGET_EVALS = 150
DEFAULT_WORKERS = 2

_CONFIG_BLOCK = re.compile(
    r"```autoevolve[ \t]*\r?\n(?P<body>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_EVOLVE_PREFIX = re.compile(r"^\s*evolve\s*:\s*", flags=re.IGNORECASE)
_KNOWN_KEYS = {
    "budget_evals",
    "wall_clock_s",
    "workers",
    "evaluator",
    "target",
    "metric",
    "target_path",
}


def extract_goal(issue_title: str, issue_body: str | None) -> str:
    """Return the issue goal, excluding the issue-mode configuration block."""

    title = _EVOLVE_PREFIX.sub("", issue_title, count=1).strip()
    body = issue_body or ""
    match = _CONFIG_BLOCK.search(body)
    if match is not None:
        body = body[: match.start()]
    body = body.strip()

    parts = [part for part in (title, body) if part]
    goal = "\n\n".join(parts)
    if not goal:
        raise ValueError("The issue must contain a goal in its title or body.")
    return goal


def extract_config(issue_body: str | None) -> dict[str, Any]:
    """Parse the optional fenced issue-mode configuration block."""

    config: dict[str, Any] = {
        "budget_evals": DEFAULT_BUDGET_EVALS,
        "workers": DEFAULT_WORKERS,
    }
    match = _CONFIG_BLOCK.search(issue_body or "")
    if match is None:
        return config

    for line_number, raw_line in enumerate(match.group("body").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = raw_key.strip()
        if key not in _KNOWN_KEYS:
            continue
        value = _strip_optional_quotes(raw_value.strip())
        try:
            config[key] = _parse_value(key, value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid autoevolve config value for {key!r} on line {line_number}: {exc}"
            ) from exc

    return config


def _parse_value(key: str, value: str) -> int | float | str:
    if key in {"budget_evals", "workers"}:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("must be a positive integer")
        return parsed

    if key in {"wall_clock_s", "target"}:
        parsed_float = float(value)
        if not math.isfinite(parsed_float):
            raise ValueError("must be a finite number")
        if key == "wall_clock_s" and parsed_float <= 0:
            raise ValueError("must be greater than zero")
        return parsed_float

    if not value:
        raise ValueError("must not be empty")
    if key in {"evaluator", "target_path"}:
        return _repo_relative_path(value, key)
    return value


def _repo_relative_path(value: str, key: str) -> str:
    raw = value.replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ValueError(f"{key} must be a repository-relative path")
    normalized = raw.strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{key} must be a repository-relative path")
    return path.as_posix()


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value
