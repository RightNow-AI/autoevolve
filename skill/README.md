# autoevolve-worker skill

This directory ships the agent-facing autoevolve worker skill. `SKILL.md` teaches the measured
worker loop. `reference.md` defines every MCP argument, result shape, and error convention.

## Install in a Claude Code project

Copy or symlink this directory to `.claude/skills/autoevolve-worker/`. The installed path must be:

```text
.claude/skills/autoevolve-worker/SKILL.md
```

PowerShell copy from the repository root:

```powershell
New-Item -ItemType Directory -Force .claude\skills | Out-Null
Copy-Item -Recurse -Force skill .claude\skills\autoevolve-worker
```

PowerShell junction from the repository root:

```powershell
New-Item -ItemType Directory -Force .claude\skills | Out-Null
New-Item -ItemType Junction -Path .claude\skills\autoevolve-worker -Target (Resolve-Path skill)
```

POSIX symlink from the repository root:

```sh
mkdir -p .claude/skills
ln -s ../../skill .claude/skills/autoevolve-worker
```

## Register the MCP server

Stdio:

```text
claude mcp add --transport stdio autoevolve -- uv run autoevolve serve
```

Streamable HTTP:

```text
claude mcp add --transport http autoevolve http://127.0.0.1:8747/mcp
```

For project scope, `.mcp.json` accepts `http` as the Streamable HTTP type:

```json
{
  "mcpServers": {
    "autoevolve": {
      "type": "http",
      "url": "http://127.0.0.1:8747/mcp"
    }
  }
}
```

The stdio command and HTTP server entry point are provided by the CLI unit. The MCP server itself
lives in `autoevolve/mcp/server.py`.
