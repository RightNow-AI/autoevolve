"""Find an invocation of codex that actually edits a file inside a container.

The agentic preflight reported exit code 0 with no file change, which is the
failure this project keeps meeting: a process that finishes cleanly having done
nothing. This tries several documented invocations in one container and reports,
for each, whether the file on disk actually changed, plus the raw output.

Nothing here is a gate. It exists to answer one question with evidence.
"""

from __future__ import annotations

import modal

app = modal.App("autoevolve-probe-codex")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("autoevolve-model")],
    timeout=1800,
)
def probe() -> list[dict]:
    import os
    import shutil
    import subprocess
    from pathlib import Path

    codex = shutil.which("codex") or "codex"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

    home = Path(os.path.expanduser("~"))
    config_dir = home / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = ['approval_policy = "never"', 'sandbox_mode = "workspace-write"']
    if base_url:
        config.append(f'openai_base_url = "{base_url}"')
    (config_dir / "config.toml").write_text("\n".join(config) + "\n", encoding="utf-8")

    task = (
        "Edit the file target.py in the current directory so the line reading "
        "VALUE = 1 instead reads VALUE = 2. Make the edit directly with your "
        "file tools and then stop."
    )

    variants: list[tuple[str, list[str]]] = [
        ("workspace-write", [codex, "exec", "--skip-git-repo-check", "-s", "workspace-write", task]),
        (
            "danger-bypass",
            [codex, "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", task],
        ),
        (
            "full-auto",
            [codex, "exec", "--skip-git-repo-check", "--full-auto", task],
        ),
    ]

    results: list[dict] = []
    for name, command in variants:
        workspace = Path(f"/tmp/probe-{name}")
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        target = workspace / "target.py"
        original = "VALUE = 1\n"
        target.write_text(original, encoding="utf-8")

        env = dict(os.environ)
        env["CODEX_API_KEY"] = api_key
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=420,
                check=False,
                env=env,
            )
            stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, code = "", "timed out", -1
        except OSError as exc:
            stdout, stderr, code = "", f"could not start: {exc}", -2

        now = target.read_text(encoding="utf-8")
        results.append(
            {
                "variant": name,
                "exit_code": code,
                "changed": now != original,
                "content": now.strip()[:80],
                "stdout_tail": (stdout or "").strip()[-1200:],
                "stderr_tail": (stderr or "").strip()[-1200:],
            }
        )
    return results


@app.local_entrypoint()
def main() -> None:
    import json

    for row in probe.remote():
        print(f"=== {row['variant']} ===")
        print(f"  exit={row['exit_code']} changed={row['changed']} content={row['content']!r}")
        print(f"  stdout: {row['stdout_tail'][-600:]}")
        print(f"  stderr: {row['stderr_tail'][-400:]}")
        print()
    print(json.dumps([{k: v for k, v in r.items() if k != "stdout_tail"} for r in probe.remote()], indent=2)[:1500])
