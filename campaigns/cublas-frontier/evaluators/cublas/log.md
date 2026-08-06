# cublas-frontier log

Append-only. Negative results get the same block format as wins.

## 2026-08-06 skinny-4096x8, agent-controlled campaign, store research-cublas-gpu

Reported: speedup 1.0196, cublas_ms 0.18909, candidate_ms 0.18546, on an A10G
with mock_mode false. Tier: NOT a win. Recorded as a harness defect, not a
result.

The candidate does not contain a kernel. It loads libcublas.so.12 by ctypes
and calls cublasSgemm_v2, which is the identical kernel the baseline calls
through torch.matmul. Its 2 percent comes from three things: bypassing the
PyTorch dispatcher, tuning cublasSetSmCountTarget to 70 with a 4 MiB
workspace, and reusing preallocated output buffers.

So the honest statement is that it beat `torch.matmul` by about 2 percent, not
that it beat cuBLAS. At this shape the output is 4096 by 8 and the whole call
is 0.19 ms, so dispatcher and allocator overhead are a real fraction of the
total. That is a genuine engineering observation and it is worth keeping. It is
simply not the claim this pack advertises.

Two harness defects made the number possible, and both are now closed.

The baseline allocated a fresh output tensor on every call while a candidate
was free to reuse one, so the comparison partly measured the allocator rather
than the kernel. `_baseline_output` now accepts a caller buffer and uses it,
giving both sides the same advantage.

The timed launch count was fixed and knowable at three warmups plus five rounds
of three repeats, about eighteen calls. The candidate held a pool of exactly 32
preallocated outputs and popped from it without ever returning one, so it
covered the measured calls and would raise IndexError on call 33. It fitted the
measurement rather than the problem. The repeat count is now drawn per run from
the cell seed between 11 and 29, so no fixed pool can quietly cover it.

The pattern is worth stating plainly because it has now happened six times in
this project in different costumes: the gate was correct every time, and the
question it was asking was subtly not the question intended.
