"""Report which agent runtimes could authenticate inside a Modal container.

Prints environment variable NAMES and never values, so it is safe to run and
safe to paste. The question it answers is whether the agentic operator, which
is the only operator that has ever produced a frontier result here, can run
remotely instead of on a laptop.
"""

from __future__ import annotations

import modal

app = modal.App("autoevolve-probe-agent")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code @openai/codex",
    )
)


@app.function(image=image, secrets=[modal.Secret.from_name("autoevolve-model")], timeout=600)
def probe() -> dict:
    import os
    import shutil
    import subprocess

    interesting = sorted(
        name
        for name in os.environ
        if any(
            token in name.upper()
            for token in ("KEY", "TOKEN", "SECRET", "ANTHROPIC", "OPENAI", "AUTOEVOLVE")
        )
    )
    report: dict = {"env_names_present": interesting}

    for runtime in ("claude", "codex"):
        path = shutil.which(runtime)
        entry: dict = {"on_path": bool(path)}
        if path:
            try:
                version = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                entry["version"] = (version.stdout or version.stderr).strip()[:120]
            except Exception as exc:  # noqa: BLE001 - probe must never crash
                entry["version_error"] = str(exc)[:200]
        report[runtime] = entry

    report["anthropic_key_present"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    report["openai_key_present"] = bool(os.environ.get("OPENAI_API_KEY"))
    print(report, flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(probe.remote(), indent=2))
