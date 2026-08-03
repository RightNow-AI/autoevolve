"""Cheap SEARCH/REPLACE mutation operator."""

from __future__ import annotations

from autoevolve.core.types import ParentBundle, Proposal
from autoevolve.mutate.base import OperatorContext, OperatorError
from autoevolve.mutate.parsing import apply_search_replace, parse_search_replace
from autoevolve.mutate.prompts import build_diff_prompt


class DiffOperator:
    """Ask the cheap endpoint for exact edits against the parent."""

    name = "diff"

    def propose(self, bundle: ParentBundle, ctx: OperatorContext) -> Proposal:
        endpoint = ctx.endpoint_cheap
        if endpoint is None:
            raise OperatorError(
                "diff requires an endpoint; set AUTOEVOLVE_LOCAL_BASE_URL or set "
                "OPENAI_API_KEY with AUTOEVOLVE_MODEL_CHEAP or AUTOEVOLVE_MODEL"
            )
        prompt = build_diff_prompt(bundle, ctx.contract)
        messages = [
            {"role": "system", "content": "You are a precise code mutation operator."},
            {"role": "user", "content": prompt},
        ]
        response = endpoint.chat(messages, temperature=0.2)
        blocks = parse_search_replace(response)
        if not blocks:
            correction = (
                "Your reply contained no valid SEARCH/REPLACE blocks. Reply again with "
                "ONLY blocks in exactly this shape, nothing else:\n"
                "<<<<<<< SEARCH relative/path.py\n"
                "exact lines copied from the file\n"
                "=======\n"
                "replacement lines\n"
                ">>>>>>> REPLACE"
            )
            messages = [
                *messages,
                {"role": "assistant", "content": response},
                {"role": "user", "content": correction},
            ]
            response = endpoint.chat(messages, temperature=0.2)
            blocks = parse_search_replace(response)
        files, applied, failed = apply_search_replace(bundle.parent_files, blocks)
        if applied == 0:
            raise OperatorError(f"diff applied zero blocks; failed={failed}")
        return Proposal(files=files, notes=f"diff: applied={applied} failed={failed}")
