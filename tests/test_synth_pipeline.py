from pathlib import Path

import pytest

import autoevolve.eval.contract as eval_contract
from autoevolve.core.types import EvalError
from autoevolve.synth.pipeline import _validated_files, synthesize
from tests.fakes import FakeEndpoint

GOOD_EVALUATOR = """### FILE: spec.md
```markdown
Metric: score in points. Gate: correct. Target: maximize. Hardware: CPU.
Fixtures are fixed examples stored in fixtures/cases.json.
```
### FILE: evaluate.py
```python
from pathlib import Path

from autoevolve.eval.contract import EvalError, StageSpec

STAGES = [StageSpec(name="quick", timeout_s=1.0)]
GATE = "correct"

def evaluate(candidate_dir: Path, stage: int = 0) -> dict[str, float]:
    return {"correct": 1.0, "score": 1.0}
```
### FILE: baseline/main.py
```python
value = 1
```
### FILE: fixtures/cases.json
```json
{"expected": 1}
```
"""

BAD_MISSING_GATE = """### FILE: spec.md
```
Metric: score.
```
### FILE: evaluate.py
```python
STAGES = []

def evaluate(candidate_dir, stage=0):
    return {"score": 1.0}
```
### FILE: baseline/main.py
```python
value = 1
```
"""


def test_synthesize_writes_and_validates_good_evaluator(monkeypatch, tmp_path):
    loaded: list[Path] = []
    monkeypatch.setattr(
        eval_contract,
        "load_evaluator",
        lambda evaluator_dir: loaded.append(evaluator_dir),
        raising=False,
    )
    endpoint = FakeEndpoint([GOOD_EVALUATOR])

    evaluator_dir = synthesize("make Python faster", tmp_path, endpoint)

    assert evaluator_dir == tmp_path / "evaluator"
    assert loaded == [evaluator_dir]
    assert (evaluator_dir / "evaluate.py").is_file()
    assert (evaluator_dir / "baseline" / "main.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    prompt = endpoint.calls[0][1]["content"]
    assert "Classified domain: python-speedup" in prompt
    assert "Correctness is checked before score" in prompt


def test_synthesize_retries_once_with_validation_error(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_contract, "load_evaluator", lambda path: None, raising=False)
    endpoint = FakeEndpoint([BAD_MISSING_GATE, GOOD_EVALUATOR])

    evaluator_dir = synthesize("improve code", tmp_path, endpoint)

    assert evaluator_dir.is_dir()
    assert len(endpoint.calls) == 2
    retry_prompt = endpoint.calls[1][1]["content"]
    assert "missing GATE" in retry_prompt
    assert "Previous response" in retry_prompt


def test_synthesize_raises_after_two_invalid_responses(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_contract, "load_evaluator", lambda path: None, raising=False)
    endpoint = FakeEndpoint([BAD_MISSING_GATE, BAD_MISSING_GATE])

    with pytest.raises(EvalError, match="missing GATE"):
        synthesize("improve code", tmp_path, endpoint)
    assert len(endpoint.calls) == 2
    assert not (tmp_path / "evaluator").exists()


def test_structural_validation_rejects_missing_gate():
    with pytest.raises(EvalError, match="missing GATE"):
        _validated_files(BAD_MISSING_GATE)
