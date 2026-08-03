import random

from autoevolve.core.types import Budget, Contract, EvalOutcome, ParentBundle, Program
from autoevolve.mutate.base import OperatorContext
from autoevolve.mutate.rewrite import RewriteOperator
from tests.fakes import FakeEndpoint


def test_rewrite_replaces_marker_file_and_ignores_unknown_path(tmp_path):
    parent = Program("p1", "r1", None, "seed", "ref", 0, None, "now")
    bundle = ParentBundle(
        parent,
        {
            "main.py": "# EVOLVE-BLOCK-START\nvalue = 1\n# EVOLVE-BLOCK-END\n",
            "readme.txt": "read-only\n",
        },
    )
    endpoint = FakeEndpoint(
        [
            """### FILE: main.py
```python
# EVOLVE-BLOCK-START
value = 9
# EVOLVE-BLOCK-END
```
### FILE: extra.py
```python
unexpected = True
```
"""
        ]
    )
    context = OperatorContext(
        contract=Contract(
            "increase value",
            "general",
            "score",
            True,
            1.0,
            9.0,
            "correct",
            Budget(max_evals=10),
        ),
        rng=random.Random(1),
        endpoint_cheap=None,
        endpoint_strong=endpoint,
        evaluate_locally=lambda files: EvalOutcome(True, {"score": 9.0}, 0),
        workdir=tmp_path,
    )

    proposal = RewriteOperator().propose(bundle, context)

    assert proposal.files["main.py"] == (
        "# EVOLVE-BLOCK-START\nvalue = 9\n# EVOLVE-BLOCK-END\n"
    )
    assert proposal.files["readme.txt"] == "read-only\n"
    assert proposal.notes == "rewrite: files=1 ignored_unknown=extra.py"
