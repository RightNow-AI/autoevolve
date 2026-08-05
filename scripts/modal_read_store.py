"""Print text files from the shared store, for reading a research campaign.

The journal and the best file are the campaign's record. Reading them is how a
human checks what the agent actually did rather than what it reported.
"""

from __future__ import annotations

import modal

app = modal.App("autoevolve-read-store")
store = modal.Volume.from_name("autoevolve-store", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={"/store": store}, timeout=600)
def read(relative_path: str, tail_chars: int = 8000) -> str:
    from pathlib import Path

    store.reload()
    path = Path("/store") / relative_path
    if path.is_dir():
        entries = sorted(str(item.relative_to(path)) for item in path.rglob("*"))
        return "DIRECTORY\n" + "\n".join(entries[:200])
    if not path.is_file():
        return f"no such file: {path}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unreadable: {exc}"
    return text[-tail_chars:]


@app.local_entrypoint()
def main(relative_path: str, tail_chars: int = 8000) -> None:
    print(read.remote(relative_path, tail_chars))
