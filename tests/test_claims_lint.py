from __future__ import annotations

from pathlib import Path

from autoevolve.cli.claims_lint import scan_claims, scan_repository

ROOT = Path(__file__).resolve().parents[1]


def test_scanner_catches_ungrounded_measured_claim(tmp_path: Path) -> None:
    path = tmp_path / "claim.md"
    path.write_text("The candidate is 10x faster.\n", encoding="utf-8")

    violations = scan_claims([path])

    assert len(violations) == 1
    assert violations[0].path == path
    assert violations[0].line_number == 1
    assert violations[0].text == "The candidate is 10x faster."


def test_scanner_accepts_run_id_no_claim_marker_and_fenced_examples(
    tmp_path: Path,
) -> None:
    path = tmp_path / "grounded.md"
    path.write_text(
        "Measured 12x on run r0123456789.\n"
        "Example 25% faster. [no-claim]\n"
        "```text\n"
        "Illustrative 40 TFLOPS.\n"
        "```\n",
        encoding="utf-8",
    )

    assert scan_claims([path]) == []


def test_real_repository_claims_are_grounded() -> None:
    violations = scan_repository(ROOT)

    assert not violations, "\n".join(item.format(ROOT) for item in violations)

