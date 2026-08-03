"""Strong-model full-file rewrite mutation operator."""

from __future__ import annotations

from autoevolve.core.types import ParentBundle, Proposal
from autoevolve.mutate.base import OperatorContext, OperatorError
from autoevolve.mutate.parsing import parse_file_blocks
from autoevolve.mutate.prompts import build_rewrite_prompt


class RewriteOperator:
    """Ask the strong endpoint to rewrite marker-bearing files."""

    name = "rewrite"

    def propose(self, bundle: ParentBundle, ctx: OperatorContext) -> Proposal:
        endpoint = ctx.endpoint_strong
        if endpoint is None:
            raise OperatorError(
                "rewrite requires an endpoint; set AUTOEVOLVE_LOCAL_BASE_URL or set "
                "OPENAI_API_KEY with AUTOEVOLVE_MODEL_STRONG or AUTOEVOLVE_MODEL"
            )
        prompt = build_rewrite_prompt(bundle, ctx.contract)
        # A rewrite emits whole files, so the default 4096 cap truncates the
        # response mid-file. The unclosed fence is then rejected by the parser
        # and the cycle silently skips, which made this operator near-inert on
        # anything but tiny programs. Size the budget to the input instead.
        mutable_chars = sum(
            len(content)
            for content in bundle.parent_files.values()
            if "EVOLVE-BLOCK-START" in content
        )
        max_tokens = max(8192, min(32768, (mutable_chars // 2) + 4096))
        response = endpoint.chat(
            [
                {"role": "system", "content": "You are a disciplined full-file code rewriter."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        parsed = parse_file_blocks(response)
        mutable_paths = {
            path
            for path, content in bundle.parent_files.items()
            if "EVOLVE-BLOCK-START" in content and "EVOLVE-BLOCK-END" in content
        }
        accepted = {path: content for path, content in parsed.items() if path in mutable_paths}
        if not accepted:
            raise OperatorError("rewrite returned no recognized marker-bearing files")

        files = dict(bundle.parent_files)
        files.update(accepted)
        unknown = sorted(path for path in parsed if path not in bundle.parent_files)
        read_only = sorted(
            path for path in parsed if path in bundle.parent_files and path not in mutable_paths
        )
        notes = [f"rewrite: files={len(accepted)}"]
        if unknown:
            notes.append(f"ignored_unknown={','.join(unknown)}")
        if read_only:
            notes.append(f"ignored_read_only={','.join(read_only)}")
        return Proposal(files=files, notes=" ".join(notes))
