"""Measured-claim lint shared by campaign reports and the test suite."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

RUN_ID_PATTERN = re.compile(r"r[0-9a-f]{10}")
NO_CLAIM_MARKER = "[no-claim]"
# A published bound we did not measure. It must name its source, so an empty
# or bare marker does not satisfy the lint. See docs/FRONTIER.md section 4.
LITERATURE_PATTERN = re.compile(r"\[lit:\s*\S[^\]]{2,}\]")
FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")
MEASURED_CLAIM_PATTERNS = (
    re.compile(r"(?<![\w.])\d+(?:\.\d+)?x\b", re.IGNORECASE),
    re.compile(
        r"(?:\d+(?:\.\d+)?\s*%.{0,40}\b(?:faster|speedup|improvement)\b|"
        r"\b(?:faster|speedup|improvement)\b.{0,40}\d+(?:\.\d+)?\s*%)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\d+(?:\.\d+)?\s*(?:TFLOPS|tok/s)|"
        r"(?:TFLOPS|tok/s)\s*\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\b(?:faster|speedup)\b.{0,20}(?<![\w.])\d+(?:\.\d+)?|"
        r"(?<![\w.])\d+(?:\.\d+)?.{0,20}\b(?:faster|speedup)\b)",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class ClaimViolation:
    """One ungrounded measured claim in a Markdown source file."""

    path: Path
    line_number: int
    text: str

    def format(self, root: Path | None = None) -> str:
        """Return a stable file, line, and source-text diagnostic."""

        display = self.path
        if root is not None:
            try:
                display = self.path.relative_to(root)
            except ValueError:
                pass
        return f"{display}:{self.line_number}: {self.text}"


def repository_markdown_paths(root: Path) -> tuple[Path, ...]:
    """Return the normative set of Markdown files covered by the claims law."""

    candidates: set[Path] = set()
    readme = root / "README.md"
    if readme.is_file():
        candidates.add(readme)
    docs = root / "docs"
    if docs.is_dir():
        candidates.update(path for path in docs.rglob("*.md") if path.is_file())
    campaigns = root / "campaigns"
    if campaigns.is_dir():
        for name in ("spec.md", "log.md"):
            candidates.update(path for path in campaigns.rglob(name) if path.is_file())
    return tuple(sorted(candidates))


def scan_repository(root: Path) -> list[ClaimViolation]:
    """Scan every repository Markdown path required by the campaigns spec."""

    return scan_claims(repository_markdown_paths(root))


def scan_claims(paths: Iterable[Path]) -> list[ClaimViolation]:
    """Return measured-claim violations outside fenced code blocks.

    A file whose name carries a run id is a verbatim generated run artifact
    (a copied report in a gallery); the artifact itself is the grounding, so
    it is skipped rather than line-checked.
    """

    violations: list[ClaimViolation] = []
    for path in sorted(Path(item) for item in paths):
        if RUN_ID_PATTERN.search(path.name):
            continue
        violations.extend(_scan_file(path))
    return violations


def _scan_file(path: Path) -> Iterator[ClaimViolation]:
    fence: str | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fence_match = FENCE_PATTERN.match(raw_line)
        if fence_match is not None:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None or not _is_measured_claim(raw_line):
            continue
        if (
            RUN_ID_PATTERN.search(raw_line)
            or NO_CLAIM_MARKER in raw_line
            or LITERATURE_PATTERN.search(raw_line) is not None
        ):
            continue
        yield ClaimViolation(path=path, line_number=line_number, text=raw_line.strip())


def _is_measured_claim(line: str) -> bool:
    return any(pattern.search(line) is not None for pattern in MEASURED_CLAIM_PATTERNS)

