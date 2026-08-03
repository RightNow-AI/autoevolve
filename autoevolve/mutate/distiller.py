"""Turn successful lineage diffs into persistent transferable discoveries."""

from __future__ import annotations

import difflib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from autoevolve.mutate.base import OperatorError
from autoevolve.mutate.models import ModelEndpoint


@dataclass(frozen=True)
class _Program:
    id: str
    parent_id: str | None
    code_ref: str
    fitness: float | None = None


def distill_run(
    home: Path,
    run_id: str,
    endpoint: ModelEndpoint | None,
    top_k: int = 5,
) -> list[dict]:
    """Distill top gate-passed lineage changes and persist the resulting statements."""

    if endpoint is None or top_k <= 0:
        return []

    db_path = home / "autoevolve.db"
    domain, goal, top_programs, programs = _read_run(db_path, run_id, top_k)
    if not top_programs:
        return []
    evidence = _build_evidence(home, top_programs, programs)
    if not evidence.strip():
        return []

    response = endpoint.chat(
        [
            {
                "role": "system",
                "content": (
                    "You distill empirical code-evolution evidence into concise prior knowledge."
                ),
            },
            {
                "role": "user",
                "content": _distillation_prompt(domain, goal, evidence),
            },
        ],
        max_tokens=1200,
        temperature=0.2,
    )
    statements = _parse_statements(response)
    if len(statements) < 3:
        return []

    source_programs = [program.id for program in top_programs]
    created_at = datetime.now(UTC).isoformat()
    rows = [
        {
            "id": f"d{uuid.uuid4().hex[:10]}",
            "domain": domain,
            "text": statement,
            "source_run": run_id,
            "source_programs": source_programs,
            "created_at": created_at,
        }
        for statement in statements[:7]
    ]
    _write_discoveries(db_path, run_id, rows)
    _append_markdown(home, domain, run_id, rows)
    return rows


def _read_run(
    db_path: Path, run_id: str, top_k: int
) -> tuple[str, str, list[_Program], dict[str, _Program]]:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise OperatorError(f"could not open run database read-only: {exc}") from exc

    try:
        run = connection.execute(
            "SELECT domain, goal_text, contract_json FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise OperatorError(f"run not found: {run_id}")
        domain, goal, contract_json = run
        contract = json.loads(contract_json)
        metric = contract["metric"]
        gate = contract["gate"]
        direction = "DESC" if contract.get("maximize", True) else "ASC"
        query = f"""
            SELECT p.id, p.parent_id, p.code_ref, fitness.value
            FROM programs AS p
            JOIN scores AS fitness
              ON fitness.program_id = p.id
             AND fitness.metric = ?
             AND fitness.stage = (
                 SELECT MAX(latest.stage)
                 FROM scores AS latest
                 WHERE latest.program_id = p.id AND latest.metric = ?
             )
            WHERE p.run_id = ?
              AND EXISTS (
                  SELECT 1
                  FROM scores AS gate_score
                  WHERE gate_score.program_id = p.id
                    AND gate_score.metric = ?
                    AND gate_score.value = 1.0
                    AND gate_score.stage = (
                        SELECT MAX(latest_gate.stage)
                        FROM scores AS latest_gate
                        WHERE latest_gate.program_id = p.id
                          AND latest_gate.metric = ?
                    )
              )
            ORDER BY fitness.value {direction}, p.id ASC
            LIMIT ?
        """
        top_rows = connection.execute(
            query, (metric, metric, run_id, gate, gate, top_k)
        ).fetchall()
        all_rows = connection.execute(
            "SELECT id, parent_id, code_ref FROM programs WHERE run_id = ?", (run_id,)
        ).fetchall()
    except (json.JSONDecodeError, KeyError, sqlite3.Error) as exc:
        raise OperatorError(f"could not read distillation evidence: {exc}") from exc
    finally:
        connection.close()

    programs = {
        row[0]: _Program(id=row[0], parent_id=row[1], code_ref=row[2]) for row in all_rows
    }
    top_programs = [
        _Program(id=row[0], parent_id=row[1], code_ref=row[2], fitness=row[3])
        for row in top_rows
    ]
    return str(domain), str(goal), top_programs, programs


def _build_evidence(
    home: Path, top_programs: list[_Program], programs: dict[str, _Program]
) -> str:
    sections: list[str] = []
    for top in top_programs:
        chain = _parent_chain(top.id, programs)
        diffs: list[str] = []
        for index in range(1, len(chain)):
            parent = chain[index - 1]
            child = chain[index]
            diff = _diff_programs(home, parent, child)
            if diff:
                diffs.append(f"Parent {parent.id} to child {child.id}:\n{diff}")
        if diffs:
            lineage = " -> ".join(program.id for program in chain)
            sections.append(
                f"Top program {top.id}, fitness={top.fitness}, lineage={lineage}\n"
                + "\n".join(diffs)
            )
    return "\n\n".join(sections)


def _parent_chain(program_id: str, programs: dict[str, _Program]) -> list[_Program]:
    chain: list[_Program] = []
    seen: set[str] = set()
    current_id: str | None = program_id
    while current_id is not None:
        if current_id in seen:
            raise OperatorError(f"program parent cycle detected at {current_id}")
        seen.add(current_id)
        current = programs.get(current_id)
        if current is None:
            break
        chain.append(current)
        current_id = current.parent_id
    chain.reverse()
    return chain


def _diff_programs(home: Path, parent: _Program, child: _Program) -> str:
    parent_files = _read_store_files(home, parent.code_ref)
    child_files = _read_store_files(home, child.code_ref)
    chunks: list[str] = []
    for path in sorted(parent_files.keys() | child_files.keys()):
        before = parent_files.get(path, "")
        after = child_files.get(path, "")
        if before == after:
            continue
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"{parent.id}/{path}",
                tofile=f"{child.id}/{path}",
            )
        )
    return "".join(chunks)


def _read_store_files(home: Path, code_ref: str) -> dict[str, str]:
    store_root = (home / "store").resolve()
    program_root = (store_root / code_ref).resolve()
    try:
        program_root.relative_to(store_root)
    except ValueError as exc:
        raise OperatorError(f"unsafe code_ref in content store: {code_ref}") from exc
    if not program_root.is_dir():
        return {}
    return {
        path.relative_to(program_root).as_posix(): path.read_text(
            encoding="utf-8", errors="replace"
        )
        for path in program_root.rglob("*")
        if path.is_file()
    }


def _distillation_prompt(domain: str, goal: str, evidence: str) -> str:
    return f"""Domain: {domain}
Goal: {goal}

Observed top-program lineage diffs:
{evidence}

Write 3 to 7 falsifiable, transferable, one-sentence statements about what worked.
Tie every statement to an observed change and result. Do not speculate beyond the evidence.
Return only one statement per line, each prefixed with '- '."""


def _parse_statements(response: str) -> list[str]:
    statements: list[str] = []
    seen: set[str] = set()
    for line in response.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        statement = stripped[2:].strip()
        if statement and statement not in seen:
            statements.append(statement)
            seen.add(statement)
    return statements[:7]


def _write_discoveries(db_path: Path, run_id: str, rows: list[dict]) -> None:
    try:
        connection = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        raise OperatorError(f"could not open run database for discovery writes: {exc}") from exc
    try:
        connection.execute("BEGIN IMMEDIATE")
        next_seq = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        for offset, row in enumerate(rows):
            source_programs_json = json.dumps(row["source_programs"], separators=(",", ":"))
            connection.execute(
                """
                INSERT INTO discoveries(
                    id, domain, text, source_run, source_programs, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["domain"],
                    row["text"],
                    row["source_run"],
                    source_programs_json,
                    row["created_at"],
                ),
            )
            payload = json.dumps(
                {
                    "discovery_id": row["id"],
                    "domain": row["domain"],
                    "text": row["text"],
                    "source_programs": row["source_programs"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            connection.execute(
                """
                INSERT INTO events(run_id, seq, kind, payload_json, created_at)
                VALUES (?, ?, 'discovery_added', ?, ?)
                """,
                (run_id, next_seq + offset, payload, row["created_at"]),
            )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise OperatorError(f"could not persist discoveries: {exc}") from exc
    finally:
        connection.close()


def _append_markdown(home: Path, domain: str, run_id: str, rows: list[dict]) -> None:
    discovery_dir = home / "discoveries"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    target = discovery_dir / f"{domain}.md"
    source_ids = ", ".join(rows[0]["source_programs"])
    date = rows[0]["created_at"][:10]
    section = [f"## {date} | run {run_id}", ""]
    section.extend(f"- {row['text']} [sources: {source_ids}]" for row in rows)
    prefix = "\n" if target.exists() and target.stat().st_size else ""
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(prefix + "\n".join(section) + "\n")
