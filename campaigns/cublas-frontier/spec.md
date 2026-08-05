# cuBLAS frontier campaign

## Goal and claim boundary

This campaign searches for shape-specific CUDA kernels that beat the cuBLAS path
measured in the same process, on the same GPU, during the same evaluator run. It does
not target large square FP32 GEMM near the hardware roofline. A win on one listed
shape and GPU is evidence only for that cell, device, software image, and run id.

The answer cannot be recalled from a published table. A kernel for one exact shape on
one exact GPU has to be measured. The campaign still labels the simple odd-square cell
as validation so harness success is not confused with frontier search capability.

| cell | role | workload |
| --- | --- | --- |
| `odd-1000-validation` | validation | FP32 GEMM, M=N=K=1000 |
| `skinny-4096x8-frontier` | frontier | FP32 GEMM, M=4096, N=8, K=4096 |
| `batched-1024x32-frontier` | frontier | 1024 FP32 GEMMs of 32 by 32 by 32 |
| `fused-bias-relu-frontier` | frontier | FP32 GEMM, M=1024, N=256, K=1024, then bias and ReLU |

`AUTOEVOLVE_CELL` selects one cell. An unset value selects the validation cell. Any
unknown value fails before candidate code runs.

## Correctness gate

The gate is `correctness`. Every candidate output is compared with a float64
`torch.matmul` reference computed from the original float32 inputs. The reference is
computed before candidate import, so a candidate cannot make a wrong answer pass by
mutating its inputs or monkeypatching the baseline. The fused reference adds bias and
applies ReLU in float64.

The tolerance is `rtol=5e-5` and `atol=2e-5`. Each input includes a precision sentinel
whose dot product uses the float32 value 1.0003. TF32 rounds away the low mantissa bits
and moves that output by roughly 6e-4 relative, outside the relative tolerance. IEEE
FP32 accumulation retains enough precision to pass. The evaluator also disables TF32
for its cuBLAS baseline and checks the baseline against the same float64 references.
A candidate that silently uses TF32, fp16 inputs, or fp16 accumulation fails the gate.

Real mode requires a float32 CUDA tensor on the exact device holding the inputs. A CPU
tensor, a tensor on another GPU, a wrong shape, a mapping containing a claimed score,
or a module-level claimed timing metric fails. The evaluator computes every metric.

The setup input, launch-count input, and timed inputs are distinct. Timed inputs are
generated on the GPU before candidate import and used once by the candidate. Float64
references for every timed input are also computed before candidate import. Every
timed output is checked after timing. Returning a cached setup answer or mutating the
timed inputs cannot produce a passing speedup.

## Baseline, timing, and metric

The baseline is eager `torch.matmul` in the evaluator process. The fused cell then
runs eager bias addition and eager ReLU, so its baseline includes the separate
epilogue launches that a fused candidate may remove. No stored latency, vendor table,
or previous run enters the score.

Both paths receive three warmup launches. Both paths then run five timed rounds with
three one-use workloads per round and use the minimum per-launch time. Inputs are
already device resident. A CUDA event is recorded and completed before each round.
The evaluator synchronizes the whole device before recording the ending event, then
completes that event. This makes work launched on a non-default stream part of the
measured interval. Host transfer and float64 reference work are outside the interval.
The Modal entrypoint requires one worker so another evaluation cannot contend for the
same GPU while either path is timed.

`METRIC = "speedup"` and `MAXIMIZE = True`. The metric is:

```text
speedup = cublas_ms / candidate_ms
```

The division direction is deliberate. A faster candidate has a smaller denominator
and therefore a larger score. Real results also return `cublas_ms` and `candidate_ms`
so the ratio can be audited. A candidate that calls `torch.matmul` is legal and should
remain near the 1.0 floor after measurement noise. It cannot submit its own score.

## Structural descriptors

MAP-elites uses two non-quality descriptors:

- `tile_area_log2` comes from the candidate's positive integer `BLOCK_M` and
  `BLOCK_N` declarations.
- `kernel_launches` is counted from CUDA device events in a post-warmup Torch profiler
  pass on a fresh input. It must match the positive integer `KERNEL_LAUNCHES`
  declaration. A candidate that performs no CUDA kernel launch fails.

Mock mode parses the declared launch count because no CUDA profiler exists. Both
descriptor metrics are returned in both modes, so the archive never collapses to one
cell.

## Candidate compute contract

The candidate entrypoint is `kernel.py` and exports:

```python
def run(
    a,
    b,
    bias,
    activation,
    *,
    real=False,
    deadline=None,
):
    ...
```

`real=True` receives CUDA-resident float32 tensors. `bias` is either a CUDA vector of
length N or `None`. `activation` is `"none"` or `"relu"`. `real=False` receives NumPy
arrays for the CI mock.

Candidates may compile and autotune during setup. The evaluator passes an absolute
`time.monotonic()` deadline equal to the stage timeout minus 90 seconds of reporting
headroom. It uses `inspect.signature` and omits the keyword for a candidate that does
not accept it. The outer evaluator timeout still applies to a candidate that ignores
the deadline.

Only content inside the seed's EVOLVE-BLOCK markers may change. The seed is a small,
untuned tiled Triton GEMM written from first principles. It uses IEEE input precision,
small fixed tiles, and a separate epilogue or copy kernel. It deliberately performs
two launches and does not call a tuned GEMM library. Scoring below 1.0 is expected and
desirable for the seed.

## CPU mock

`AUTOEVOLVE_FORCE_CUBLAS_MOCK=1`, missing Torch, or missing CUDA selects the CPU mock.
The mock runs reduced shapes and applies the same float64 parity rule without importing
Torch or Triton. It returns the primary metric key with value 0.0 plus `mock_mode=1.0`.
That zero is an explicit non-measurement sentinel, not a GPU speed claim. Mock results
must never appear in a performance report.

## Bounds and promotion

`bounds.json` contains no literature record. Each entry points to the dynamic
`cublas_ms` that the evaluator measures on specific hardware in a specific run and
says plainly that it is not a published bound. Rechecking means rerunning the cell and
citing the new run id, GPU model, and raw times.

A single frontier score is a candidate. A performance claim requires three completed
improving seeds on the same GPU and software image, followed by an independent rerun.
Budget exhaustion or a plateau remains a negative result and never becomes a win.
