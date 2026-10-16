# virgil 0.4.0 core review (origin/main 420d1b7)

Scope: likelihood, fitting, oidata, gains, _closure, _linear, inference, svi, _flat, _precision, priors, angles, _deprecate, oifits, amigo, observables, spectra.
Read-only. "VERIFIED-RUN" = reproduced with a small float32/float64 snippet. "VERIFIED-READ" = confirmed in code, not run. "UNVERIFIED" = suspicion only.
Not fully read: observables.py (grepped for static fields/wavel use, `_template`), spectra.py (grepped), oifits.py writers (`build_hdulist`, `_array_hdu`).
Findings from `design/codebase_review_2026-10b.md` that are still open are marked "(0.3.0 carry-over)".

## Correctness

### C1  should-fix-before-0.4.0  VERIFIED-RUN
`src/virgil/oidata.py:880-940` (`OIData._diagonalized`, called from `__init__` at :1044 and :1060)
- Evidence: the rank test `keep = variance > 1e-10 * max(variance)` (and the same 1e-10 in the placeholder check) is applied to an `eigh` done in `jax.numpy`, i.e. float32 unless the caller enabled x64. Float32 eps is 1.2e-7, so the numerically zero eigenvalues of a rank-deficient projected covariance sit above the threshold and are kept as real directions with tiny errors (huge weight).
- Repro: `op = C @ B` with C (40x15), B (15x60) random, sigma = 0.01*(1+U), `OIData._diagonalized(op, sig)`. Expected rank 15. float32: 26-28 directions kept in 20/20 seeds, minimum error ~6e-5 to 1.6e-4 against ~1e-2 for real directions. float64 (`jax.enable_x64`): 15 in 20/20 seeds.
- Why it matters: this is the DISCO / kernel-phase / projected-closure-phase path, which is exactly where operators are rank deficient. The spurious rows carry numerical noise as data with weight ~1e4 times a real one, so chi2 is garbage. The AGENTS.md convention is float32-safe library code. A 1e-10 relative tolerance is not float32-safe.
- Fix: build the operator on the host in float64 (`onp.linalg.eigh`) before casting back. This function is not traced, so there is no cost. Alternatively use `max(1e-10, n * eps(dtype))`. Add a float32 regression test.
- Effort: S.

### C2  should-fix-before-0.4.0  VERIFIED-READ
`oidata.py:162-165` (docstring) vs `:421-424` (code); attribute at `:126`, `:426`; `amigo.py:143`
- Evidence: the docstring says the record key `observable_kind` is an alias of `vis_mode`. The code only reads `data.get("vis_mode", data.get("observable_vis_mode", "auto"))`, so `observable_kind` as an input key is ignored. The undocumented key `observable_vis_mode` is what is actually accepted. Separately, the attribute `observable_kind` means "split" vs "mixed_log_complex" (layout), and `amigo.py:143` sets it in the record. `OIData.__init__` then overwrites it with "split" at :426, so a record key never reaches the attribute either. There are ~25 uses of it in aliases/imaging/detection/gains. So one name means two things, and the docstring is wrong.
- Fix: correct the docstring now. Before the names freeze, rename the attribute (for example `layout`, or an enum) and drop `observable_vis_mode`.
- Effort: S for the docs; M for the rename (mechanical, ~25 sites).

### C3  should-fix-before-0.4.0  VERIFIED-READ
`oidata.py:1624-1659` (`with_wavelength_scale`), `observables.py:502` (`FluxSpectrum._template`), `likelihood.py:490`
- Evidence: the docstring says the scaled wavelengths "move the spectra together". The method only replaces `self.wavel` (and `bandwidth`). `FluxSpectrum`, `DifferentialPhase` and their continuum bases hold their own `wavel` (observables.py:302, :652), and `_template` evaluates `model.total_spectrum(self.wavel)` at the unscaled wavelengths. With `noise={"wavel_scale": ...}` and `extras=("flux",)`, the visibility model sees the scaled wavelengths and the flux template does not.
- Fix: either scale the extras' `wavel` in `with_wavelength_scale` (a `tree_at` over the blocks), or say in the docstring that extras are not scaled. Add a test that checks one or the other.
- Effort: S.

### C4  0.4.x  UNVERIFIED
`svi.py` (`variational`)
- `regularizers` and `likelihoods` are not passed through `cast_tree(.., dtype)`, whereas `fit` casts the whole `_Objective`. This may give mixed precision in an x64 run.
- "Flows need at least two parameters" is documented but I did not see a check. A single-parameter flow probably gives an obscure numpyro/flowjax error.
- Fix: cast the same way as `fit`, and raise a clear `ValueError`. Effort: S.

### C5  0.4.x  VERIFIED-READ
`fitting.py`: `chi2_red` in `fit` is `sum(chi2)/scale`, with `scale` from `n_independent`, but chi2 includes the periodic penalty rows of correlated closure phases (`n_residuals > n_independent`). For >=4-station closure-phase data, `chi2_red` is slightly biased. Document it, or subtract the penalty rows. (This is consistent with the "report raw chi2/N" policy only if the definition is stated.) Effort: S.

## API / technical debt (names that freeze at 0.4.0)

### A1  should-fix-before-0.4.0  VERIFIED-READ
`inference.py:69, 80, 242-243, 296-297, 348-349`
- `fisher()` returns the observed information (Hessian of -log L), and `gaussian_fisher` returns the expected Fisher matrix. Same word, different objects.
- `laplace_covariance(objective, x, ridge)` and `laplace_cov(values, params, model, data)` are near names for different levels (flat objective vs model/data).
- `laplace_cov` and `fisher` take `dtype="float64"`. `laplace_parameter_uncertainty` takes no `dtype` and does `np.asarray(values, dtype=float)` in ambient precision (float32 by default), which is inconsistent. The Hessian of a float32 likelihood gives noisy second derivatives.
- None of the three accept `noise=` or `**options`, so fixed noise terms, gain widths or wavelength terms cannot be passed. `fit` and `loglike` accept them.
- `values` comes before `model` and `data`, the opposite of the library convention. It is kept via `old_order` (so it is a deliberate 0.3 compatibility), but the new order is also not the documented primary form in `laplace_parameter_uncertainty`, whose parameter list still uses `params` after `values`. `inference.py` has no `__all__`.
- `regularized_inverse(ridge=1e-10)` and `laplace_covariance(ridge=1e-10)` use an absolute ridge that is unit-dependent.
- Fix: rename `fisher` to `observed_information` (keep an alias via `renamed`); give all three `dtype` and `noise`; add `__all__`.
- Effort: M.

### A2  should-fix-before-0.4.0  VERIFIED-READ
`_deprecate.py`, `likelihood.py:1141-1142`
- `posterior_predictive_summary` has `@renamed()`, but its first parameter is `samples` and nothing in `NAME_ALIASES` maps to it, so the decorator is a no-op there. `NAME_ALIASES` maps `model_fn->model` globally, so any function with a `model_fn` parameter accepts it silently.
- The shims warn in 0.4 and `removed=True` / alias removal is scheduled for 0.5. The same list should be in CHANGELOG as a removal schedule. (`Tabulated` carries `FutureWarning` "removed in a later release" at `spectra.py:665` with no version. Say 0.5.)
- `gauss_newton_mass` already had the new argument order in 0.3.0 (checked with `git show v0.3.0`), so no shim is needed. `variational` and `laplace_*` forms are new or covered.
- Effort: S.

### A3  0.4.x  VERIFIED-READ
`likelihood.py`: `is_tied`, `tied_log_prior`, `noise_sites`, `noise_for`, and the wrapper chain `_whiten`, `_whitened_and_errors`, `_whitened_and_log_norm`, `_whitened_errors_and_log_norm` (four near-identical thin wrappers around `_whiten_with_log_norm`).
- Public-looking internals used by `fitting`. If they are not meant for users, prefix with `_` before 0.4.0 or they become API. Collapse the wrappers into one function with flags.
- `posterior_predictive_summary` ignores `extras`, and its `phi_mean` is an arithmetic mean of wrapped phases (not circular). Use a circular mean, or document it.
- `_host_devices` uses `jax._src.xla_bridge` (guarded by try/except, so degrades, but is fragile across JAX releases; the project policy is to track latest JAX).
- Effort: S-M.

### A4  0.4.x  VERIFIED-READ
`fitting.py`: `NoLeastSquaresForm(TypeError)` is documented "for backward compatibility", but it did not exist in 0.3.0 (grep count 0 in `v0.3.0:fitting.py`), so there is nothing to be compatible with. Make it a `ValueError` subclass or drop the compat claim. Effort: S.

### A5  0.4.x  VERIFIED-READ
`amigo.py`: `load_oi_data` uses `onp.load(..., allow_pickle=True)`. The docstring warns about it, which is adequate. Consider a `.npz`/JSON route in 0.5. Note only.

## Legibility

### L1  should-fix-before-0.4.0 (for `__init__`), others 0.4.x  VERIFIED-RUN (scanner)
Function length (total lines / lines of code), measured by a scratch script:
- `oidata.py` `OIData.__init__`: 310 / 237 (was 210 at 0.3.0; grows with every feature). Suggested split: parse record -> arrays -> flags -> operators -> extras. Highest-value refactor in scope.
- `oidata.py` `_build_extras` 120, `_subset` 107, `select` 81.
- `observables.py` `DifferentialPhase.build` 163 / 115, `FluxSpectrum.build` 81.
- `svi.py` `variational` 246 / 105. `oifits.py` `_read_closure_phases` ~120 (777-897; contains an inline `coords` lazy-read and a 3-level loop).
- `likelihood.numpyro_model` has a ~197-line docstring (longer than the body). Move the parameter tables into the guide, and link.
- `oifits.py:44` stale `TODO(Stage 6a.0)` (the work is done: legs are built from OI_T3, see `_read_closure_phases`). The comment block above `_WAVE_RTOL` is misplaced (it describes the frame tolerance). `oidata.py:2134` `TODO: support reversed baselines` while `oifits` now handles reversed legs: check whether still true.
- `_read_visibilities`, `_read_absolute_phases`, `_read_visamp` have no docstrings (private, low priority).
- Effort: M for `__init__`, S each for the rest.

## Performance

### P1  0.4.x  VERIFIED-READ (0.3.0 carry-over)
Python loops over groups in builders: `gains.gain_modes` (`for f in onp.unique(frame): frame == f`, O(n_frames * n)), `ClosureNoise.from_indices` and `_incidence` (per group), `observables.DifferentialPhase.build` (per row, nested). Quadratic for GRAVITY-sized data (10^4-10^5 samples). `oifits._BaselineLookup.times` is O(n^2) within an MJD window. Fix: `onp.unique(return_inverse)` plus `searchsorted`/`bincount`. Effort: M. Not a 0.4.0 blocker (reader speed, not fit speed).

### P2  0.4.x  VERIFIED-READ (0.3.0 carry-over)
Static fields that make a jit recompile when a value changes: `OIData.t_ref` (`oidata.py:130`), `FluxSpectrum.widths/scale/poly_width` (`observables.py:317-319`), `DifferentialPhase.prior_width` (`:667`), `GainModes.groups` (`gains.py:110`). A scan over prior widths or t_ref recompiles each point. Fix: make the numeric ones traced leaves, keep only structure static. Effort: S-M.

### P3  0.4.x  VERIFIED-READ
`fitting._covariance` builds a dense n x n for n <= 8192 (537 MB in float64), plus `eigvalsh` in `_correlation_condition`. `svi._start_scale` builds an n x n Cholesky regardless of n. Fine for GRAVITY after averaging, risky for imaging-size parameter vectors. Add a size guard or a matrix-free path. Effort: M.

### P4  0.4.x  VERIFIED-READ
`fitting._lm`: the last chunk's `max_steps` is static, so the final short chunk recompiles. Pad the chunk or pass it as a traced loop bound. Effort: S.

### P5  note  VERIFIED-READ
`oidata.py:1331`: `jax.lax.map` over samples for time-dependent models is deliberate (commented). OK.

## Notes (looked sound)
- `_flat`, `priors`, `angles`, `_precision`, `_linear`, `_closure`, `gains` block whitening: no defects found on reading.
- `_deprecate._INTERNAL` frame skipping is fine.
- `with_model` raises "noise_scale must be non-negative." without the value (0.3.0 carry-over). Add `got {value}`.
- `fit`'s `_Objective.init`: `unit = term.endswith("scale") or term.endswith("_modes")` is a string-suffix rule. It works but is fragile if a new term ends in "scale". Use an explicit set.
- `oifits.py` reader handles reversed T3 legs and checks coordinates; error messages are good.

## Suggested order before tagging 0.4.0
1. C1 (float64 eigh, plus test). 2. C2 and C3 (docs or tiny code). 3. A1 and A2 renames/aliases while the shim window is open. 4. A3 underscore the internals. 5. `OIData.__init__` split (can slip to 0.4.x if names do not change).
