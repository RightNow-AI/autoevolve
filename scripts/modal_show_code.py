"""Print the source of one program from a Modal store.

A number alone never says whether it was searched or recalled. This project
has already logged three results that turned out to be a published constant
pasted into the source, so reading the winning program is part of judging it.
"""

from __future__ import annotations

import modal

REPO = "https://github.com/RightNow-AI/autoevolve"
REPO_ROOT = "/root/autoevolve"

app = modal.App("autoevolve-show-code")
store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(f"git clone {REPO} {REPO_ROOT}")
)


@app.function(image=image, volumes={"/store": store}, timeout=900)
def show(store_name: str, program_id: str) -> dict:
    import sqlite3
    from pathlib import Path

    store.reload()
    db_path = Path(f"/store/{store_name}/autoevolve/autoevolve.db")
    if not db_path.is_file():
        return {"error": f"no store at {db_path}"}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT code_ref, operator, parent_id FROM programs WHERE id = ?",
        (program_id,),
    ).fetchone()
    if row is None:
        return {"error": f"no program {program_id}"}
    code_ref, operator, parent_id = row
    root = Path(f"/store/{store_name}/autoevolve/store/{code_ref}")
    files: dict[str, str] = {}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(root))] = path.read_text(
                    encoding="utf-8", errors="replace"
                )[:12000]
    return {
        "program_id": program_id,
        "operator": operator,
        "parent_id": parent_id,
        "code_ref": code_ref,
        "files": files,
    }


@app.local_entrypoint()
def main(store_name: str, program_id: str) -> None:
    result = show.remote(store_name, program_id)
    if "error" in result:
        print(result["error"])
        return
    print(f"program {result['program_id']} operator={result['operator']}")
    for name, text in result["files"].items():
        print(f"----- {name} -----")
        print(text)
