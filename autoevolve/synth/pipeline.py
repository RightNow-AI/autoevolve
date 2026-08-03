"""English goal to evaluator-folder synthesis pipeline."""

from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath

from autoevolve.core.types import EvalError
from autoevolve.mutate.models import ModelEndpoint
from autoevolve.mutate.parsing import parse_file_blocks
from autoevolve.synth.domains import classify_domain
from autoevolve.synth.prompts_synth import repair_prompt, synthesis_prompt


def synthesize(
    goal_text: str,
    workdir: Path,
    endpoint: ModelEndpoint,
    *,
    load: bool = True,
) -> Path:
    """Generate, structurally validate, and optionally load one evaluator directory.

    load=False skips executing the generated evaluate.py entirely. The GitHub
    opened handler uses it so no issue-derived code runs before the
    evolve:approved consent label; only text generation and structural checks
    happen pre-approval.
    """

    evaluator_dir = workdir / "evaluator"
    if evaluator_dir.exists() and any(evaluator_dir.iterdir()):
        raise EvalError(f"evaluator directory already exists and is not empty: {evaluator_dir}")

    domain = classify_domain(goal_text)
    output = endpoint.chat(
        [
            {
                "role": "system",
                "content": "You design deterministic, measured code optimization evaluators.",
            },
            {"role": "user", "content": synthesis_prompt(goal_text, domain)},
        ],
        max_tokens=8192,
        temperature=0.2,
    )

    last_error: EvalError | None = None
    for attempt in range(2):
        try:
            files = _validated_files(output)
            _replace_evaluator_dir(evaluator_dir, files)
            if load:
                _load_evaluator(evaluator_dir)
            return evaluator_dir
        except EvalError as exc:
            last_error = exc
            if evaluator_dir.exists():
                shutil.rmtree(evaluator_dir)
            if attempt == 1:
                break
            output = endpoint.chat(
                [
                    {
                        "role": "system",
                        "content": "You repair evaluator contracts exactly as instructed.",
                    },
                    {
                        "role": "user",
                        "content": repair_prompt(goal_text, domain, exc.reason, output),
                    },
                ],
                max_tokens=8192,
                temperature=0.1,
            )

    if last_error is None:
        raise EvalError("evaluator synthesis failed without a validation error")
    raise last_error


def _validated_files(output: str) -> dict[str, str]:
    parsed = parse_file_blocks(output)
    if not parsed:
        raise EvalError("model response contained no valid file blocks")

    files: dict[str, str] = {}
    for raw_path, content in parsed.items():
        path = PurePosixPath(raw_path.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise EvalError(f"generated evaluator contains unsafe path: {raw_path}")
        normalized = path.as_posix()
        allowed = normalized in {"spec.md", "evaluate.py"} or (
            len(path.parts) > 1 and path.parts[0] in {"baseline", "fixtures"}
        )
        if not allowed:
            raise EvalError(f"generated evaluator contains unexpected path: {raw_path}")
        files[normalized] = content

    if "spec.md" not in files:
        raise EvalError("generated evaluator is missing spec.md")
    evaluate_source = files.get("evaluate.py")
    if evaluate_source is None:
        raise EvalError("generated evaluator is missing evaluate.py")
    if re.search(r"\bSTAGES\s*[:=]", evaluate_source) is None:
        raise EvalError("generated evaluate.py is missing STAGES")
    if re.search(r"\bGATE\s*[:=]", evaluate_source) is None:
        raise EvalError("generated evaluate.py is missing GATE")
    if re.search(r"\bdef\s+evaluate\s*\(", evaluate_source) is None:
        raise EvalError("generated evaluate.py is missing def evaluate")
    if not any(path.startswith("baseline/") for path in files):
        raise EvalError("generated evaluator baseline/ is empty")
    if not any(path.startswith("fixtures/") for path in files):
        raise EvalError("generated evaluator fixtures/ is empty")
    return files


def _replace_evaluator_dir(evaluator_dir: Path, files: dict[str, str]) -> None:
    if evaluator_dir.exists():
        shutil.rmtree(evaluator_dir)
    evaluator_dir.mkdir(parents=True)
    for relative_path, content in files.items():
        target = evaluator_dir / PurePosixPath(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _load_evaluator(evaluator_dir: Path) -> None:
    try:
        from autoevolve.eval.contract import load_evaluator

        load_evaluator(evaluator_dir)
    except EvalError:
        raise
    except Exception as exc:
        raise EvalError(f"generated evaluator failed contract validation: {exc}") from exc
