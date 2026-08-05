"""Insert derived knowledge into a remote run's discovery ledger.

The ledger is the designed channel for transferable knowledge and it feeds
every operator prompt. Nothing here weakens a gate: the evaluator still
verifies every candidate exhaustively and exactly. This only tells the search
where to look, which measurably redirected the Ramsey run earlier.
"""

from __future__ import annotations

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"

app = modal.App("autoevolve-seed-discoveries")
store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(f"git clone {REPO} {REPO_ROOT}")
)


@app.function(image=image, volumes={"/store": store}, timeout=900)
def seed(store_name: str, domain: str, texts: list[str], tag: str) -> dict:
    import sqlite3
    from datetime import UTC, datetime
    from pathlib import Path

    store.reload()
    db_path = Path(f"/store/{store_name}/autoevolve/autoevolve.db")
    if not db_path.is_file():
        return {"error": f"no store at {db_path}"}
    conn = sqlite3.connect(db_path)
    now = datetime.now(UTC).isoformat()
    inserted = 0
    for index, text in enumerate(texts):
        ident = f"d-{tag}-{index}"
        if conn.execute("SELECT 1 FROM discoveries WHERE id = ?", (ident,)).fetchone():
            continue
        conn.execute(
            "INSERT INTO discoveries(id, domain, text, source_run, source_programs, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (ident, domain, text, "derived", "[]", now),
        )
        inserted += 1
    conn.commit()
    total = conn.execute(
        "SELECT COUNT(*) FROM discoveries WHERE domain = ?", (domain,)
    ).fetchone()[0]
    domains = [row[0] for row in conn.execute("SELECT DISTINCT domain FROM discoveries")]
    conn.close()
    store.commit()
    return {"inserted": inserted, "domain_total": total, "domains_present": domains}


@app.local_entrypoint()
def main(store_name: str, domain: str, tag: str, text_file: str) -> None:
    import json
    from pathlib import Path

    # One entry per blank-line separated block, so a brief stays readable.
    blocks = [
        block.strip()
        for block in Path(text_file).read_text(encoding="utf-8").split("\n\n")
        if block.strip()
    ]
    print(json.dumps(seed.remote(store_name, domain, blocks, tag), indent=2))
