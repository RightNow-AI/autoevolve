"""Public import surface for evaluator authors.

Evaluators do `from autoevolve.eval.contract import StageSpec, EvalError`.
The loader itself (load_evaluator) is implemented in unit U2 per
docs/ARCHITECTURE.md section 7 and re-exported here.
"""

from autoevolve.core.types import EvalError, StageSpec

__all__ = ["EvalError", "StageSpec"]
