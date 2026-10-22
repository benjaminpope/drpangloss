# XLA CPU run-to-run nondeterminism (0.4.1 upstream report)

Status: investigated 2026-10-22. Not filed. Ben files; the upstream tracker already has a matching open issue
([jax-ml/jax#39741](https://github.com/jax-ml/jax/issues/39741)), so the draft below is a comment, not a new issue.

## Evidence in virgil

The case is the compiled Levenberg–Marquardt solve, not a primitive op. Trail:

- virgil-validation#138 found that `fit` (LM, the default for uniform priors) on a noise-free limb-darkened star plus
  companion sometimes ended at a garbage optimum: flux about 0 and the offsets at the prior bound.
- virgil#443 (`dd819df`) fixed the symptom: optimistix's `ClassicalTrustRegion` accepts a step when
  `Δf <= 0.01 * predicted_reduction`, so a garbage damped solve with a positive predicted reduction was accepted. LM now
  rejects any step that raises the loss.
- The cause of the garbage solves was not fixed. The same inputs and compiled code give different results from call to
  call. Eager stepping of the same LM steps never does (0 of 100); the compiled `while_loop` does.
  `--xla_cpu_enable_concurrency_optimized_scheduler=false` makes the unfixed code pass 60 of 60 (notes in the 0.4.0 release
  review session; no copy in the repo, which has no `XLA_FLAGS` setting except `likelihood.py:729-743`, a device-count helper).
- Unrelated and not this bug: commit `b1158ae` (CLEAN test, first χ² 27.1/28.7/29.3 across CI runners) is a cross-platform
  float32 difference in the noise draw, and `jnp.hypot` NaN gradient at the origin on JAX 0.12.0 (#455).

## What differs

Reproduced on the pre-fix code (`c4464858e406e2d070fefbb8013bc3dc872c3fea`, parent of `dd819df`), x64, CPU, scene of
`test_a_fit_near_the_truth_does_not_run_away_from_it`: 20 consecutive `fit` calls in one process gave 15 distinct results
(best loss 14.5; others 2e5 to 2e6). With the scheduler flag off: 1 distinct result.

Narrowed (all in one process, default flags unless stated):

| piece | repeats | distinct bitwise results |
|---|---|---|
| jitted `residuals`, `jacfwd`, `jacrev` of the virgil objective | 200 each | 1 |
| `optx.least_squares`, plain `LevenbergMarquardt` + `lx.QR`, virgil residuals | 30 | 1 |
| same with a `terminate` that calls `jax.grad` of the loss (what `_GradientStoppedLM` does) | 30 | 3 |
| the same, Gaussian star + point companion | 30 | 3 |
| the same, uniform-disk star + point; limb-darkened star alone; two points | 30 | 1 each |

The post-optimization HLO of the compiled solve contains one `lapack_dgeqrf_ffi`, one `lapack_dormqr_ffi` (the damped QR solve) and
eight `lapack_dtrsm_ffi` (the per-group 3×3 whitening of the closure phases, forward and backward, batched over 42 groups),
several of which the concurrency-optimized scheduler can run at the same time. This is a hypothesis for where the race is;
I have not localized it.

Seven jax-only reconstructions (no virgil) stayed bit-identical over 100 to 1500 calls: dependent and independent QR, SVD,
eigh, Cholesky and LU in and out of a `while_loop`; optimistix LM with a synthetic visibility model, with and without
the gradient-norm `terminate`, with closure phases by gather, with batched `solve_triangular` whitening, and a
42-group four-telescope replica of the likelihood's structure. A virgil-free repro therefore needs the exported module
(as #39741 found), which is what the replay below does.

## Minimal repro (jax only, via an exported module)

`export_module.py` (needs virgil at `c446485`, run once) exports the whole jitted LM solve of the failing scene with
`jax.export` (92 KB, f64, input `x0` is the 4-vector of unconstrained starting coordinates). The replay uses only
jax, numpy and flatbuffers:

```python
# replay.py
import argparse, jax, numpy as np
jax.config.update("jax_enable_x64", True)
from jax import export
ap = argparse.ArgumentParser(); ap.add_argument("--calls", type=int, default=100); a = ap.parse_args()
fn = export.deserialize(bytearray(open("lm_step.jaxexport", "rb").read())); x0 = np.load("x0.npy")
outs = [tuple(np.asarray(o) for o in fn.call(x0)) for _ in range(a.calls)]
print(f"jax {jax.__version__}  calls={a.calls}  distinct bitwise={len({b''.join(o.tobytes() for o in r) for r in outs})}  steps={outs[0][1]}")
```

Files (not committed, 92 KB binary): `~/Library/Application Support/Claude/side-session-notes/local_9a7256c5-8eb2-4f23-9030-5d03771abfc0/xla-cpu-repro/`
(`lm_step.jaxexport`, `x0.npy`, `replay.py`, plus the virgil-side `export_module.py`, `lm_models.py`, `lmrepro.py`).
Attach `lm_step.jaxexport`, `x0.npy` and `replay.py` to the comment, or put them in a gist.

## Results

Apple M4 (4 performance + 6 efficiency cores), macOS 26.7.1 arm64, Python 3.12, jax = jaxlib = 0.11.2, numpy from the
virgil venv, x64. Each row: three separate processes, 200 calls each; the cell is the number of distinct bitwise results
in that process.

| `XLA_FLAGS` | process 1 | 2 | 3 |
|---|---|---|---|
| (default) | 4 | 2 | 2 |
| `--xla_cpu_enable_concurrency_optimized_scheduler=false` | **1** | **1** | **1** |
| `--xla_cpu_use_thunk_runtime=false` | 1 | 6 | 3 |
| `--xla_cpu_multi_thread_eigen=false` | 1 | 3 | 5 |
| `--xla_cpu_enable_fast_math=false` | 4 | 5 | 2 |
| `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1` | 3 | 5 | 3 |
| `--xla_force_host_platform_device_count=4` | 5 | 5 | 2 |
| scheduler flag off and `--xla_cpu_use_thunk_runtime=false` | 1 | 1 | 1 |

`JAX_DEFAULT_MATMUL_PRECISION=highest` gave 1 distinct result in one run, which is inside the default's range (1 to 5); the
module is f64 so the setting should not matter. No `taskset` on macOS, so the single-core case is untested here.
On this machine the scheduler flag is the only one that always restored determinism, unlike #39741 (x86 Linux, f32), where it
only reduced the rate.

## Existing upstream issues

- [jax-ml/jax#39741](https://github.com/jax-ml/jax/issues/39741) (open, 2026-08-05): "[CPU] One compiled module returns
  different results across calls when the process has more than one thread." Same signature (one compiled module, identical
  inputs, different results, scheduler flag helps, thunk-runtime and Eigen flags only reduce the rate; reporter could not
  rebuild it jax-only either and used an exported module). The one comment is an unrelated suggestion that blames reduction order
  and proposes a nonexistent `jax_cpu_enable_concurrency_optimized_scheduler` config; it does not apply here (f64, no
  large reductions; the QR/trsm LAPACK calls are the only candidates).
- [openxla/xla#50284](https://github.com/openxla/xla/issues/50284): CPU scheduling memory bound; mentions the same flag, not this.
- No other matches for `nondeterministic`, `different results across calls`, `thunk`, `lapack` in jax-ml/jax or openxla/xla.

## Draft: comment for jax-ml/jax#39741 (Ben to post)

> A second independent instance, with the smallest module I have so far, and a case where the scheduler flag *does* restore
> determinism.
>
> **Setup.** Apple M4, macOS 26.7.1 arm64, Python 3.12, jax = jaxlib = 0.11.2, float64, CPU. The module is one `jax.jit` of
> `optimistix.least_squares` (Levenberg–Marquardt, `lineax.QR`, a custom `terminate` that evaluates `jax.grad` of the loss each step,
> 100 steps) on a 4-parameter interferometric model fit (about 550 residuals). Post-optimization HLO has one
> `lapack_dgeqrf_ffi`, one `lapack_dormqr_ffi` and eight batched `lapack_dtrsm_ffi` custom calls (42 × 3×3). Stepping the
> same LM steps eagerly never differs (0 of 100); the compiled while-loop sometimes returns a garbage damped solve.
>
> **Repro (jax, numpy, flatbuffers only).** Exported with `jax.export` (92 KB); `replay.py` deserializes it and calls it on
> identical buffers, counting distinct bitwise results (attached: `lm_step.jaxexport`, `x0.npy`, `replay.py`).
>
> ```
> pip install jax==0.11.2 flatbuffers
> python replay.py --calls 200
> ```
>
> Distinct bitwise results in 200 calls, three processes per row:
>
> | `XLA_FLAGS` | |
> |---|---|
> | default | 4, 2, 2 |
> | `--xla_cpu_enable_concurrency_optimized_scheduler=false` | 1, 1, 1 |
> | `--xla_cpu_use_thunk_runtime=false` | 1, 6, 3 |
> | `--xla_cpu_multi_thread_eigen=false` | 1, 3, 5 |
> | `--xla_cpu_enable_fast_math=false` | 4, 5, 2 |
> | `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1` | 3, 5, 3 |
> | scheduler flag off and `--xla_cpu_use_thunk_runtime=false` | 1, 1, 1 |
>
> The effect is large: before we guarded our solver against accepting a step that raises the loss, about half of the full fits
> ended at a wrong optimum (flux about 0, parameters on the prior bound); with the scheduler flag off, 0 of 60. Only the
> exported module reproduces it. Seven hand-built jax-only reconstructions (QR/SVD/eigh/Cholesky/LU, optimistix LM with a
> synthetic model, batched `solve_triangular`) stayed bit-identical over up to 1500 calls, as the reporter here also found.
> The only non-elementwise ops are the LAPACK FFI calls, so a race between concurrently scheduled LAPACK thunks
> (for example the in-place `dtrsm` outputs, `output_to_operand_aliasing`) is my guess; I have not localized it.
