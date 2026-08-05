"""Find a codex transport that works through this base URL, by testing them.

The agentic operator failed on Modal with `responses_websocket: failed to
connect`. Codex 0.146 prefers a websocket transport for the responses API, and
a proxy that only serves HTTP chat completions cannot answer it. Codex exposes
the transport through a model_provider block, so rather than guess the right
setting this runs the matrix in one container and reports which one completes
an edit.

Each variant is judged the only way that means anything here: did the file on
disk actually change. A clean exit proves nothing, as this project has now
learned four separate times.
"""

from __future__ import annotations

import modal

app = modal.App("autoevolve-probe-codex-wire")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "ca-certificates", "bubblewrap")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex",
    )
)

TASK = (
    "Edit the file target.py in the current directory so the line reading "
    "VALUE = 1 instead reads VALUE = 2. Make the edit with your file tools, "
    "then stop."
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("autoevolve-model")],
    timeout=2400,
)
def probe() -> list[dict]:
    import os
    import shutil
    import subprocess
    from pathlib import Path

    codex = shutil.which("codex") or "codex"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    model = os.environ.get("AUTOEVOLVE_MODEL_STRONG") or os.environ.get(
        "AUTOEVOLVE_MODEL", "gpt-5.6-sol"
    )

    def provider_block(wire_api: str) -> str:
        return "\n".join(
            (
                'model_provider = "autoevolve"',
                f'model = "{model}"',
                'approval_policy = "never"',
                "",
                "[model_providers.autoevolve]",
                'name = "autoevolve"',
                f'base_url = "{base_url}"',
                'env_key = "OPENAI_API_KEY"',
                f'wire_api = "{wire_api}"',
            )
        )

    variants: list[tuple[str, str | None]] = [
        ("default-no-config", None),
        ("provider-wire-chat", provider_block("chat")),
        ("provider-wire-responses", provider_block("responses")),
    ]

    home = Path(os.path.expanduser("~"))
    config_dir = home / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"

    results: list[dict] = []
    for name, config in variants:
        if config is None:
            config_path.write_text('approval_policy = "never"\n', encoding="utf-8")
        else:
            config_path.write_text(config + "\n", encoding="utf-8")

        workspace = Path(f"/tmp/wire-{name}")
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        target = workspace / "target.py"
        original = "VALUE = 1\n"
        target.write_text(original, encoding="utf-8")

        env = dict(os.environ)
        env["CODEX_API_KEY"] = api_key
        command = [
            codex,
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            TASK,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
                env=env,
            )
            code, out, err = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired:
            code, out, err = -1, "", "timed out"
        except OSError as exc:
            code, out, err = -2, "", f"could not start: {exc}"

        changed = target.read_text(encoding="utf-8") != original
        results.append(
            {
                "variant": name,
                "changed": changed,
                "exit_code": code,
                "stdout_tail": (out or "").strip()[-700:],
                "stderr_tail": (err or "").strip()[-700:],
            }
        )
        print(f"{name}: changed={changed} exit={code}", flush=True)
    return results


@app.local_entrypoint()
def main() -> None:
    for row in probe.remote():
        print(f"=== {row['variant']} ===")
        print(f"  changed={row['changed']} exit={row['exit_code']}")
        print(f"  stdout: {row['stdout_tail'][-350:]}")
        print(f"  stderr: {row['stderr_tail'][-350:]}")
        print()
