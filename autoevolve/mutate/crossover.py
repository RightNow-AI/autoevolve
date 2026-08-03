"""Deterministic model-free crossover operator."""

from __future__ import annotations

from dataclasses import dataclass

from autoevolve.core.types import ParentBundle, Proposal
from autoevolve.mutate.base import OperatorContext, OperatorError


@dataclass(frozen=True)
class _SplitFile:
    frozen: list[str]
    mutable: list[str]


class CrossoverOperator:
    """Combine mutable regions while preserving primary-parent frozen text."""

    name = "crossover"

    def propose(self, bundle: ParentBundle, ctx: OperatorContext) -> Proposal:
        if bundle.crossover_parent is None or bundle.crossover_files is None:
            raise OperatorError("crossover requires a crossover parent and crossover files")

        files = dict(bundle.parent_files)
        choices: list[str] = []
        for path, primary_text in bundle.parent_files.items():
            other_text = bundle.crossover_files.get(path)
            if other_text is None:
                continue
            primary = _split_evolve_regions(primary_text, path)
            if not primary.mutable:
                continue
            other = _split_evolve_regions(other_text, path)
            if len(primary.mutable) != len(other.mutable):
                raise OperatorError(
                    f"crossover marker region count differs for {path}: "
                    f"{len(primary.mutable)} != {len(other.mutable)}"
                )

            selected: list[str] = []
            for region_index, (from_primary, from_other) in enumerate(
                zip(primary.mutable, other.mutable, strict=True), start=1
            ):
                choose_primary = ctx.rng.random() < 0.5
                selected.append(from_primary if choose_primary else from_other)
                choice = "A" if choose_primary else "B"
                choices.append(f"{path}#region{region_index}={choice}")
            files[path] = _join_regions(primary.frozen, selected)

        if files == bundle.parent_files:
            # Every region coin landed on the primary, so this is the parent
            # verbatim. Submitting it would spend an evaluation to re-measure
            # something already in the archive.
            raise OperatorError("crossover produced the parent unchanged")
        note = ", ".join(choices) if choices else "no shared mutable regions"
        return Proposal(files=files, notes=f"crossover: {note}")


def _split_evolve_regions(text: str, path: str) -> _SplitFile:
    frozen: list[str] = []
    mutable: list[str] = []
    frozen_buffer: list[str] = []
    mutable_buffer: list[str] = []
    inside = False

    for line in text.splitlines(keepends=True):
        if "EVOLVE-BLOCK-START" in line:
            if inside:
                raise OperatorError(f"nested EVOLVE-BLOCK-START in {path}")
            frozen_buffer.append(line)
            frozen.append("".join(frozen_buffer))
            frozen_buffer = []
            inside = True
            continue
        if "EVOLVE-BLOCK-END" in line:
            if not inside:
                raise OperatorError(f"EVOLVE-BLOCK-END without start in {path}")
            mutable.append("".join(mutable_buffer))
            mutable_buffer = []
            frozen_buffer.append(line)
            inside = False
            continue
        if inside:
            mutable_buffer.append(line)
        else:
            frozen_buffer.append(line)

    if inside:
        raise OperatorError(f"EVOLVE-BLOCK-START without end in {path}")
    frozen.append("".join(frozen_buffer))
    return _SplitFile(frozen=frozen, mutable=mutable)


def _join_regions(frozen: list[str], mutable: list[str]) -> str:
    parts = [frozen[0]]
    for index, region in enumerate(mutable):
        parts.extend((region, frozen[index + 1]))
    return "".join(parts)
