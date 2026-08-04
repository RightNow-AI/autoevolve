import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import autoevolve.eval.contract as contract_module
from autoevolve.core.types import EvalError, StageSpec
from autoevolve.eval.contract import load_evaluator

FIXTURES = Path(__file__).parent / "fixtures"
TOY_EVALUATOR = FIXTURES / "eval_toy"
SOCKET_IMPORT_EVALUATOR = FIXTURES / "eval_socket_import"
SOCKET_CEILING_EVALUATOR = FIXTURES / "eval_socket_ceiling"
SPAWN_IMPORT_EVALUATOR = FIXTURES / "eval_spawn_on_import"


def test_load_evaluator_reads_child_reported_contract() -> None:
    evaluator = load_evaluator(TOY_EVALUATOR)

    assert evaluator.dir == TOY_EVALUATOR.resolve()
    assert evaluator.stages == [
        StageSpec(name="smoke", timeout_s=20.0),
        StageSpec(name="full", timeout_s=30.0),
    ]
    assert evaluator.gate == "correct"
    assert evaluator.has_ceiling is False
    assert "source compactness" in evaluator.spec_text
    assert evaluator.ceiling() is None


def test_load_evaluator_rejects_missing_evaluate_py(tmp_path: Path) -> None:
    evaluator_dir = tmp_path / "missing-entrypoint"
    (evaluator_dir / "baseline").mkdir(parents=True)

    with pytest.raises(EvalError, match="missing evaluate.py"):
        load_evaluator(evaluator_dir)


def test_load_evaluator_surfaces_import_failure(tmp_path: Path) -> None:
    evaluator_dir = tmp_path / "broken-import"
    (evaluator_dir / "baseline").mkdir(parents=True)
    (evaluator_dir / "evaluate.py").write_text(
        'raise RuntimeError("import exploded")\n',
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="RuntimeError: import exploded"):
        load_evaluator(evaluator_dir)


def test_load_evaluator_blocks_network_during_module_import() -> None:
    with pytest.raises(EvalError, match="network disabled in sandbox"):
        load_evaluator(SOCKET_IMPORT_EVALUATOR)


def test_evaluator_ceiling_blocks_network_call() -> None:
    evaluator = load_evaluator(SOCKET_CEILING_EVALUATOR)

    with pytest.raises(EvalError, match="network disabled in sandbox"):
        evaluator.ceiling()



def _process_is_alive(pid: int) -> bool:
    """Whether a process id still names a running process on this platform."""

    if sys.platform == "win32":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_describe_timeout_kills_spawned_grandchild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(contract_module, "_RUNNER_TIMEOUT_S", 0.5)
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setenv("AUTOEVOLVE_GRANDCHILD_PID_FILE", str(pid_file))

    with pytest.raises(EvalError, match="evaluator describe timed out after 30s"):
        load_evaluator(SPAWN_IMPORT_EVALUATOR)

    # Assert the process died, rather than inferring it from how quickly the
    # call returned. Elapsed time measures machine load as much as it measures
    # the kill, and this test failed on a busy machine where the kill worked.
    pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if not _process_is_alive(pid):
            break
        time.sleep(0.1)
    assert not _process_is_alive(pid), f"grandchild {pid} outlived the describe timeout"


def test_loader_carries_metric_declaration() -> None:
    evaluator = load_evaluator(TOY_EVALUATOR)
    assert evaluator.metric == "score"
    assert evaluator.maximize is True


def test_loader_carries_behavior_descriptors():
    """Without descriptors the archive is one cell and search is hill climbing."""

    from pathlib import Path

    from autoevolve.eval.contract import load_evaluator

    pack = Path(__file__).parents[1] / "campaigns" / "golomb-ruler" / "evaluators" / "golomb"
    evaluator = load_evaluator(pack)

    assert evaluator.descriptors
    names = {item["metric"] for item in evaluator.descriptors}
    assert names == {"max_gap", "gap_spread"}
    for item in evaluator.descriptors:
        assert item["bins"] >= 1
        assert item["hi"] > item["lo"]


CELL_AT_IMPORT_EVALUATOR = FIXTURES / "eval_cell_at_import"


def test_describe_receives_workload_configuration(monkeypatch) -> None:
    """A pack that reads its cell at import time must still be describable.

    Frontier packs are required to select their instance at import, before any
    candidate code runs. The describe probe is an import, so if it does not
    receive AUTOEVOLVE_CELL the pack cannot be loaded at all. That is what
    silently kept the Ramsey campaign at zero programs.
    """

    monkeypatch.setenv("AUTOEVOLVE_CELL", "large")
    evaluator = load_evaluator(CELL_AT_IMPORT_EVALUATOR)

    assert evaluator.gate == "correct"
    assert evaluator.metric == "size"


def test_describe_rejects_a_pack_whose_cell_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("AUTOEVOLVE_CELL", raising=False)

    with pytest.raises(EvalError, match="AUTOEVOLVE_CELL must be one of"):
        load_evaluator(CELL_AT_IMPORT_EVALUATOR)


def test_child_environment_passes_workload_config_but_never_engine_or_secrets(
    monkeypatch,
) -> None:
    """One rule, shared by the describe probe and the sandbox.

    These used to be two copies of one allowlist and they drifted, which is
    why the describe probe lost the cell while the sandbox kept it.
    """

    from autoevolve.eval.childenv import build_child_env

    monkeypatch.setenv("AUTOEVOLVE_CELL", "large")
    monkeypatch.setenv("AUTOEVOLVE_KERNEL_ELEMENTS", "65536")
    monkeypatch.setenv("AUTOEVOLVE_HOME", "/engine/private")
    monkeypatch.setenv("AUTOEVOLVE_AGENT_RUNTIME", "claude")
    monkeypatch.setenv("AUTOEVOLVE_MODEL_STRONG", "secret-model")
    monkeypatch.setenv("AUTOEVOLVE_API_KEY", "sk-should-never-pass")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-pass")

    env = build_child_env()

    assert env["AUTOEVOLVE_CELL"] == "large"
    assert env["AUTOEVOLVE_KERNEL_ELEMENTS"] == "65536"
    for withheld in (
        "AUTOEVOLVE_HOME",
        "AUTOEVOLVE_ARTIFACTS_DIR",
        "AUTOEVOLVE_AGENT_RUNTIME",
        "AUTOEVOLVE_MODEL_STRONG",
        "AUTOEVOLVE_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        assert withheld not in env
    assert env["PYTHONSAFEPATH"] == "1"
