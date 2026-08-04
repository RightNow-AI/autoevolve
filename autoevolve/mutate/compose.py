"""Composition helpers binding mutate operators to the core loop.

Every surface that drives the in-process loop (CLI run and join, the GitHub
action) uses these instead of guessing at wiring. The loop supplies only the
locked contract, a per-cycle RNG, and the engine home in its context; this
module adds the resolved model endpoints and an optional real sandboxed
evaluator callable.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoevolve.mutate.base import OperatorContext
from autoevolve.mutate.models import resolve_endpoint
from autoevolve.mutate.registry import get_operator


def build_local_evaluator(
    evaluator_dir: Path | None,
) -> Callable[[dict[str, str]], Any] | None:
    """A callable evaluating candidate files through the real sandbox cascade."""

    if evaluator_dir is None:
        return None
    from autoevolve.eval.cascade import run_cascade
    from autoevolve.eval.contract import load_evaluator

    evaluator = load_evaluator(evaluator_dir)

    def evaluate_locally(files: dict[str, str]) -> Any:
        with tempfile.TemporaryDirectory(prefix="autoevolve-local-eval-") as tmp:
            root = Path(tmp)
            for relative, content in files.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            return run_cascade(evaluator, root)

    return evaluate_locally


def load_spec_text(evaluator_dir: Path | None) -> str:
    """Read a pack's spec.md, which is read directly rather than through the
    describe probe because only the text is wanted and a subprocess is not."""

    if evaluator_dir is None:
        return ""
    path = Path(evaluator_dir) / "spec.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def build_get_operator(
    operator_names: list[str] | None,
    evaluate_locally: Callable[[dict[str, str]], Any] | None = None,
    spec_text: str = "",
) -> Callable[[str], object]:
    """Return the loop's get_operator with endpoints resolved once.

    A hint outside the allowlist is substituted with the first allowed
    operator; the returned operator reports its own name so the loop records
    what actually ran and bandit stats stay truthful.
    """

    endpoint_cheap = resolve_endpoint("cheap")
    endpoint_strong = resolve_endpoint("strong")

    class _BoundOperator:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.name = str(getattr(inner, "name", "unknown"))

        def propose(self, bundle: Any, ctx: Any) -> Any:
            full = OperatorContext(
                contract=ctx.contract,
                rng=ctx.rng,
                endpoint_cheap=endpoint_cheap,
                endpoint_strong=endpoint_strong,
                evaluate_locally=evaluate_locally,
                workdir=Path(ctx.workdir),
                spec_text=spec_text,
            )
            return self._inner.propose(bundle, full)

    def factory(name: str) -> object:
        effective = name
        if operator_names and name not in operator_names:
            effective = operator_names[0]
        return _BoundOperator(get_operator(effective))

    return factory
