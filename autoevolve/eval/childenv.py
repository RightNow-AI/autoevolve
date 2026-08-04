"""The one environment every evaluator child process gets.

Two child processes read this: the describe probe that loads an evaluator
module to learn its contract, and the sandbox that judges a candidate. They
used to build their environments from two separate copies of the allowlist,
and the copies drifted. The sandbox learned to pass AUTOEVOLVE_ prefixed
workload configuration so a campaign could select a cell; the describe probe
never did. A pack that reads its cell at import time, which docs/FRONTIER.md
requires so a candidate cannot choose the instance it is judged against, was
therefore impossible to describe and could not run at all. The Ramsey pack did
exactly that and sat at zero programs while its store looked merely idle.

One definition, imported by both, so the two can no longer disagree.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Host variables a child needs to start Python and find a temporary directory.
ALLOWED_ENV = frozenset(
    {
        "HOME",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
    }
)

#: AUTOEVOLVE_ names that configure the engine rather than the workload.
#: AUTOEVOLVE_HOME points at the run database, so a candidate holding it could
#: edit its own scores. Endpoint, model, and agent runtime settings are equally
#: none of a candidate's business.
ENGINE_ONLY_ENV = frozenset(
    {
        "AUTOEVOLVE_HOME",
        "AUTOEVOLVE_ARTIFACTS_DIR",
        "AUTOEVOLVE_LOCAL_BASE_URL",
        "AUTOEVOLVE_LOCAL_MODEL",
        "AUTOEVOLVE_AGENT_RUNTIME",
        "AUTOEVOLVE_AGENTIC_TIMEOUT_S",
        "AUTOEVOLVE_AGENTIC_TOOLS",
        "AUTOEVOLVE_AGENTIC_TURNS",
    }
)


#: Substrings that mark a name as credential shaped wherever it appears.
_CREDENTIAL_TOKENS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def is_workload_config(name: str) -> bool:
    """Whether an AUTOEVOLVE_ name describes the workload, not the engine.

    Credential shaped names are excluded even under the project prefix, so
    configuration can never smuggle a key into a child's environment.
    """

    if any(token in name for token in _CREDENTIAL_TOKENS):
        return False
    return name not in ENGINE_ONLY_ENV and not name.startswith("AUTOEVOLVE_MODEL")


def build_child_env() -> dict[str, str]:
    """Build the scrubbed environment shared by describe and sandbox children."""

    env = {name: value for name, value in os.environ.items() if name in ALLOWED_ENV}
    for name, value in os.environ.items():
        if name.startswith("AUTOEVOLVE_") and is_workload_config(name):
            env[name] = value
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = str(_REPO_ROOT)
    # Belt and braces with the -P launch flag: either alone keeps the candidate
    # directory off sys.path, and both together survive a launch site that
    # forgets the flag.
    env["PYTHONSAFEPATH"] = "1"
    return env
