# Lossless compression evaluator

## Task

The candidate implements a self-contained lossless byte compressor in `codec.py`.
`compress(data: bytes) -> bytes` produces the stored representation.
`decompress(blob: bytes) -> bytes` must recover the exact original bytes.

The bundled seed is a pure-Python byte-aligned LZ77 codec. It uses literal runs and
fixed-window backreferences. This simpler seed was chosen over an arithmetic coder because exact
roundtrip correctness is more important than an aggressive initial ratio.

## Certificate and search shape

This pack has an exact certificate. The `lossless` gate compares the decompressed bytes with the
original bytes. The comparison is deterministic, total, and cheap at the bundled corpus size.
The metric is graded because every saved byte improves `compression_ratio`. In the taxonomy from
`docs/DOMAINS.md`, this is an exact certificate and matches the tractable Hutter Prize shape.

## Gate and metrics

The `lossless` gate requires
`decompress(compress(sample)) == sample` for every selected corpus sample. Equality is exact byte
equality. A mismatch raises `EvalError` with the sample name and the first differing byte offset.
A non-empty sample may not produce an empty compressed blob.

The primary metric is `compression_ratio`, defined as
`total_original_bytes / total_compressed_bytes` over the selected samples. It is unitless and its
target semantics are maximize. The evaluator computes both totals from exact bytes that it reads
or snapshots itself. It also returns `compressed_bytes` and `original_bytes` as byte counts.
Candidate-reported sizes and scores are never accepted.

Stage 0 uses the two smaller samples and has a 15 second timeout. Stage 1 uses all three samples
and has a 30 second timeout. The gate is checked at both stages before any metrics are returned.

`ceiling()` returns `None`. Shannon entropy of a fixed sample is a useful soft reference, but it
is not a certified ceiling for a general compressor. Claiming a hard ceiling here would be
dishonest.

## Candidate isolation and cheat deterrence

Candidate source is scanned with Python `ast` before it is imported. Imports of `zlib`, `lzma`,
`bz2`, `gzip`, `brotli`, archive compression modules, and their low-level helpers are rejected.
Imports of `pathlib` and `os` are rejected. Calls to `open`, `read_bytes`, and `read_text` are also
rejected. The resulting `EvalError` names the forbidden module or file operation.

Compression and decompression use separate module namespaces. Every decompression call loads a
fresh process-local candidate namespace and receives no argument other than the immutable blob.
This prevents an ordinary candidate from saving the input in module state during compression and
reading it back during decompression.

The AST scan is a deterrent for accidental and lazy cheating. It is not a security boundary.
Candidates already run inside the AutoEvolve evaluator subprocess, whose sandbox limits and
verdict protections are documented in `docs/CONTRACT.md`.

Candidate return values are accepted only when their exact type is `bytes`. The evaluator takes
one immutable snapshot and all later gate and metric clauses read only that snapshot. Container
subclasses cannot change their answers between the gate and the metric.

## Hardware and dependencies

This evaluator needs only a CPU and the Python standard library. It is deterministic and offline.
It does not use wall-clock measurements in the gate or metric.

## Fixture provenance

`fixtures/make_fixtures.py` uses seed `1618033`. It writes three byte-stable samples:

- `natural-language.txt` is 8,320 bytes from a fixed word list and a seeded word-level Markov
  process.
- `repetitive.txt` is 8,000 bytes of repeated phrases.
- `near-random.txt` is 4,096 bytes of seeded near-random printable text.

Regenerate the corpus with:

```text
python evaluators/lossless-compression/fixtures/make_fixtures.py
```

The script rewrites byte-identical files for unchanged source and seed.

## Candidate guidance

Agents may change only code between each `# EVOLVE-BLOCK-START` and
`# EVOLVE-BLOCK-END` pair in `baseline/codec.py`. They must preserve the documented
`compress(data: bytes) -> bytes` and `decompress(blob: bytes) -> bytes` signatures.
