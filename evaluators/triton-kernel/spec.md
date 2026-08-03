# Triton kernel evaluator

## Task

The candidate computes `x + alpha * y` for float32 vectors. `AUTOEVOLVE_CELL` selects
one of `add-1k`, `add-8k`, `scale-1k`, or `scale-8k`. The add cells use `alpha=1.0`;
the scale cells use non-unit alpha values. The suffix selects 1024 or 8192 elements.
When `AUTOEVOLVE_CELL` is unset, all four groups run. Any other value is rejected.

The bundled seed contains a Triton vector-add-and-scale kernel. Triton and Torch are
imported only inside the real execution path. The same file contains a pure NumPy
`ref` implementation outside the mutable block.

## Metrics and gate

The `mock_parity` gate checks every output against the bundled NumPy reference with
relative tolerance `1e-5` and absolute tolerance `1e-6`. Every case must pass. The
gate key is stable in both execution modes.

Real mode returns `tflops` and `candidate_ms`. `tflops` counts one multiply and one
add per element. Timing uses Torch CUDA events on the current GPU during the current
evaluation, taking the minimum of three batches of twenty launches after three warmup
launches. The target semantics for `tflops` are maximize.

Throughput is measured on a separate device-resident workload, not on the parity
fixtures. The fixtures are deliberately tiny so correctness stays fast, and timing
them would measure kernel launch overhead and the host transfer instead of
throughput. For a memory-bound operation a host transfer over PCIe can never approach
device memory bandwidth, so such a number could never be compared against the roofline
and would be meaningless. The benchmark workload defaults to 2^24 elements, is
generated once on the device with a fixed seed, and its size can be overridden with
`AUTOEVOLVE_KERNEL_ELEMENTS`.

In real mode the candidate must return a CUDA tensor. A candidate that falls back to
the CPU still produces correct values and would pass parity, so without this rule a
CPU measurement could be published as a GPU throughput number. The evaluator checks
the device of the returned object and fails the gate otherwise.

Mock mode returns only metrics with a `mock_` prefix. `mock_score` is a deterministic
CPU cost-model value. If the candidate provides `mock_schedule(n)`, its `score` value
is averaged across the cases. Otherwise the evaluator parses integer `BLOCK` and
`num_warps` assignments from the candidate source and applies the documented launch
utilization model. A mock score is a CI signal and is never a throughput claim.

## Hardware and ceiling

Real mode needs a CUDA GPU, Torch, and a working CUDA runtime. It does not need
Triton. The certificate is numerical parity against the reference plus elapsed time on
the device, and a candidate may reach that through Triton, plain Torch, CuPy, or any
other backend it can import. Requiring Triton specifically would strand working
hardware in mock mode, which is exactly the situation on Windows, where Triton ships
no wheel. If Torch or a CUDA device is unavailable the evaluator uses mock mode
without importing GPU packages. Neither Torch nor Triton is a project dependency;
install the GPU stack yourself, for example
`uv pip install torch --index-url https://download.pytorch.org/whl/cu128`.

In real mode, `ceiling()` reads memory clock and bus width from
`torch.cuda.get_device_properties`. It returns a memory-bound roofline for two FLOPs
and twelve transferred bytes per element. Its method is `roofline: memory-bound`.
It returns `None` in mock mode or when the runtime does not expose those properties.

## Fixture provenance

`cases.json` contains one group for each campaign cell. The add groups use `alpha=1.0`
at sizes 1024 and 8192. The scale groups use `alpha=0.375` at size 1024 and
`alpha=-1.25` at size 8192. Groups of the same size reuse identical vectors so only
the operation changes. Python `random.Random` seed `65537` generates every value.
Values are rounded to seven decimal places.

Regenerate the fixture with:

```text
python evaluators/triton-kernel/fixtures/make_fixtures.py
```

The script is deterministic and rewrites byte-identical JSON.

## Candidate guidance

Agents may change only code between `# EVOLVE-BLOCK-START` and
`# EVOLVE-BLOCK-END` in `kernel.py`. They must preserve the `run` signature and the
optional `mock_schedule(n)` hook contract. They must not change `ref`.
