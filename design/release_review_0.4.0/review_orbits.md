# virgil 0.4.0 review: orbits, epochs, aliases, orbit_search, detection, grid_fit, limits, _grid, pipeline

Checkout: origin/main 420d1b7 (detached). Read-only review; nothing edited.
Line numbers are approximate (+-5) and refer to that checkout.
"verified" = I ran it (scratch scripts). "read" = from reading code only, not run.

Not examined in depth (gaps): grid_fit.linear_flux_grid, detection.injection_recovery, pipeline binary.py / star.py / _checks.py / _quicklook.py (cli.py and _io.py/_core.py read), aliases `_fit_band` / `_refine` / `_flip_modes` / `_resample`, `estimate_primary_diameter`, design/codebase_review_2026-10b.md only skimmed.

Sizes: orbits 1700, epochs 2644, aliases 2947, orbit_search 1184, detection 1787, grid_fit 861, limits 804, _grid 182 lines. epochs.py is new in 0.4.0 (whole file in the diff).

## Summary of the overlap question

There are three overlapping ways to start / score / fit orbits and nothing tells a user which to use:

1. epochs.py: `rank_orbits` (fixed flux, one orbit list, quoted or marginal scales) and `epoch_positions` + `estimate_primary_diameter` + `start_from_positions` -> `OrbitStart` (positions -> starting_orbits -> rank_orbits -> `fit`).
2. orbit_search.py: `score_orbits` (flux-marginalized, extra RV/position terms) -> `OrbitScores`. Nothing else in the package uses it.
3. aliases.py: `fit_orbit_aliases` -> `AliasResult`. Own engine (`_UDBinary`, own parameterization and priors, own Laplace/IS evidence). Takes a list of OIData, not `Epochs`.

They share the `_Surface` marginal-likelihood algebra (good) but import it across private names (B2). Result types are incoherent: `RankedOrbits` holds orbit objects; `OrbitScores` holds arrays only; `OrbitStart` wraps `RankedOrbits` + `FitResult`; `AliasResult.best` is a plain dict of elements with no bridge back to `start_from_positions` / `chain_values` / `fit`.

---

## Findings

### A1. `_axis` misreads a 3-value flux grid as (lo, hi, n)
- Severity: should-fix-before-0.4.0. Effort: small. Verified.
- orbit_search.py ~252 (`_axis`). The test is `isinstance(spec, tuple) and len(spec)==3 and float(spec[2])>=1`.
- Evidence: `_axis((0.01, 0.1, 1.0), True, 'flux')` returns the single flux [0.01]; `(0.01, 0.1, 2.0)` returns [0.01, 0.1]; the list `[0.01, 0.1, 1.0]` works. The docstring promises "or the flux values themselves". A natural 3-point grid ending at 1.0 silently becomes one flux.
- Fix: require an int `n` in the third slot, or take a dedicated type / keywords (`flux_range=(lo, hi, n)`).

### A2. `alias_bands` docstring says the opposite order
- Severity: note. Effort: trivial. Verified.
- aliases.py `alias_bands`: "Ordered by N (longest period last)". Actual output is ascending N, so N=0 (longest period) is first.

### A3. `_cap_bands(..., n_max=0)` flags capped but returns the window unchanged
- Severity: note. Effort: trivial. Verified. Edge case, no real effect.

### A4. `_thiele_innes_fit_sign_free` regularization is a float32 no-op
- Severity: note. Effort: small. Read, not run.
- orbits.py: `u + 1e-12*eye` vanishes in float32. Near-singular pairs (e.g. a single epoch gives pair (0,0)) yield inf/nan; only the isfinite filter in the argmin protects it. Fix: scale the ridge to the matrix norm, or solve with `lstsq`.

### A5. Orbit core checked and correct
- Verified. `ThieleInnesOrbit.to_kepler` round trips for 300 random orbits incl. inc 0, 90, ~180; velocity via jvp works. `PositionData.from_sep_pa` covariance/rotation conventions read as correct. `PositionData.term` / `RVData.term` agree with `.loglike` (6.7859917 both).

### A6. Pre-release history baked into new public API (epochs scales / gap)
- Severity: should-fix-before-0.4.0. Effort: small. Read.
- epochs.py is entirely new in 0.4.0, yet `EpochPositions.gap` is documented "unchanged for one release, after which gap will hold the marginal value", and `rank_orbits` / `start_from_positions` default `scales=None` -> "quoted" with a FutureWarning that the default will become "marginal". `score_orbits` already defaults to "marginal".
- Fix: choose "marginal" now, drop the warning, define `gap` once.

### A7. rank_orbits vs score_orbits: inconsistent signatures
- Severity: should-fix-before-0.4.0 (cheap now, a breaking change later). Effort: small-medium. Read.
- `rank_orbits(model, data, orbits, *, scales=None, s_max=None, ...)` vs `score_orbits(epochs, model, orbits, *, scales="marginal", s_max=5.0, ...)`: the latter is data-first, violating the AGENTS.md model-before-data rule in a brand-new function (no `_deprecate` shim needed pre-release).
- `model(orbit)` vs `model(orbit, flux)`; different defaults for `scales` and `s_max`; both take `dtype`; return types `RankedOrbits` vs `OrbitScores`.

### A8. AliasResult JSON has no format version and carries legacy handling
- Severity: should-fix-before-0.4.0. Effort: small. Read.
- aliases.py `to_json` / `save` / `load`: no format field (the checkpoint has `_CHECKPOINT_FORMAT = 3`, the result does not). `load` handles "a file written before they were saved" and uses an `UNRECORDED` string sentinel exported in `__all__`, plus a `primary_diam` / `vis_amp` override: dev history in a new public API.
- Fix: add `"format": 1`, check it on load, drop `UNRECORDED` from `__all__` (or make private).

### B1. Three overlapping orbit entry points, no decision tree, incoherent result types
- Severity: should-fix-before-0.4.0 for a documented decision tree plus the bridges below; full unification 0.4.x. Effort: docs small, bridges medium. Read.
- `fit_orbit_aliases` takes a sequence of OIData and computes its own mean dataset times, ignoring `Epochs.at="epoch"`. The epoch tools take `Epochs`.
- `AliasResult.best` is a dict, not an orbit object: no route into `start_from_positions`, `chain_values` or `fit` (`start_values(orbit, flux)`).
- `AliasResult.scene()` returns the private `_UDBinary`, not a public class, so it cannot go in a pipeline model spec.
- `OrbitScores` has no orbit objects; `RankedOrbits` does; `OrbitStart` wraps one.
- `position_profile` (aliases) wraps `starting_orbits` per band: a fourth path to a starting grid.
- Fix: a "Which orbit tool?" page (quick rank -> `rank_orbits`; flux-marginalized with RV/position terms -> `score_orbits`; period aliases with evidences -> `fit_orbit_aliases`; hand-off to `fit` -> `start_from_positions`); give `AliasResult` a `to_orbit()` / `best_orbit`; accept `Epochs` in `fit_orbit_aliases`.

### B2. Private cross-imports and a lazy cyclic import
- Severity: should-fix-before-0.4.0 if modules are being restructured anyway, otherwise 0.4.x. Effort: medium. Read.
- orbit_search imports `_Surface, _blocks, _check_dof, _dof_for, _orbit_batch_size, _stack, _orbit_list` from epochs and `_Term` from orbits. aliases imports `_Surface` from epochs and `_fingerprint` from detection. `epochs._ranked` lazily imports `rank_scores` from orbit_search to avoid a cycle.
- Fix: a private `_marginal.py` holding `_Surface`, the scoring helpers and `rank_scores`.

### B3. score_orbits (and epochs._loglikes) recompile on every call
- Severity: 0.4.x (performance). Effort: medium. Verified for score_orbits.
- The jitted `every` is defined inside `score_orbits` and closes over model, epochs, bands. With 5 candidates and 4 datasets, calls 2 and 3 each cost 2.7-5.8 s (`jax.log_compiles`: "Finished XLA compilation of jit(every) in 5.78 sec"). Same pattern in `epochs._loglikes` (`every_marginal` / `every`, ~772, ~794) and the jit at orbit_search ~1098. A comment in `_Surface` claims one kernel serves every dataset of one shape via a static jit argument; these closures do not do that. Datasets/bands are unrolled in Python inside jit, so compile cost scales with their number.
- Fix: module-level jit with `_Surface` static and data traced; cache per-shape.

### B4. starting_orbits memory blow-up (no batching)
- Severity: should-fix-before-0.4.0 or 0.4.x. Effort: small. Verified up to 410k points; larger extrapolated (unverified).
- orbits.py `starting_orbits`: one `vmap` over the whole periods x eccs(19) x phases(36) grid, built with a Python list comprehension.
- 10 epochs, 205,200 grid points: peak RSS ~547 MB unsigned, ~1.9 GB sign_free (7.7 s). 410,400 points sign_free: ~2.4 GB, 33 s.
- Extrapolation (unverified): default `period_grid` for T=400 d, P = 1-30 d gives ~3500 periods, 2.4M points: ~6 GB unsigned, >20 GB sign_free. `position_profile` uses narrow bands so is spared; `start_from_positions` with a user period list is exposed.
- Fix: `lax.map` with `batch_size`, vectorized grid construction.

### B5. epoch_positions default batch_size=4096
- Severity: 0.4.x. Effort: small. Read, not run.
- epochs.py `epoch_positions` defaults `batch_size=4096` rather than `batch_size_or_default(None, data)` as the other grid functions do (`_grid.batch_visibilities()`). Passed to `lax.map` in `_grid_scores`: memory ~4096 x n_vis, which could OOM on AMIGO-like data.

### B6. Epochs.spread_days never checked
- Severity: note. Effort: small. Read.
- A dataset spanning many days silently gets a mean-time snapshot; only the docstring warns. Suggest a warning when spread exceeds a fraction of the resolution-limited motion or the period.

### B7. Pickle checkpoints in aliases
- Severity: 0.4.x (note if folders are never shared). Effort: small. Read.
- `_load_band` calls `pickle.load` on `band_<N>.pkl` before its key check; a bare `except Exception` silently refits. Arbitrary-code risk for shared directories; fragile across dataclass changes; contradicts the pipeline's "nothing is pickled" claim. Fix: use the `.json` already written, or npz.

### B8. Pipeline stability and model-spec issues
- Read unless stated.
- Verified good: schema `virgil-pipeline-run-v1` is versioned and checked by `Result` and the CLI (`_io.py` ~624, `cli.py` `_record_problems`); model specs are pickle-free (`allow_pickle=False`, `_io.py` ~354, ~372). `info` / `validate` do no JAX work as claimed.
- B8a (0.4.x; should-fix if run folders are shared): `model_from_spec` does `importlib.import_module(info["module"])` + `getattr` with no `virgil.` prefix or eqx.Module check (the save side has `_public_class`), and builds with `object.__new__` + `setattr`, bypassing `__init__` / `__check_init__`. A hostile manifest.json can import arbitrary importable modules.
- B8b (0.4.x, but it breaks the stable contract): `model_from_spec` raises `ModelSpecError` if a class's dataclass field set differs from the stored one, so any release adding/removing a model field breaks reloading old runs, contradicting "change only by addition". Fix: tolerate added fields with defaults.
- B8c (policy, note): both pipelines are `STABILITY = "stable"` (binary.py:165, star.py:326) in their first release, freezing setting names, defaults, check names, thresholds and file names while the functions beneath them changed argument order/deprecations in 0.4. Consider "provisional" for 0.4.0.
- B8d (docs, small): OrbitPipeline is not built (design/pipeline.md), so the headline orbit feature is not reachable through the stable layer; the docs should say so.
- B8e (low): `_Pipeline.from_config` resolves a relative `output` override against the config file's directory when the config also has an `output` key.

### B9. Documentation hygiene
- Severity: should-fix-before-0.4.0. Effort: small. Read.
- Public docstrings/comments cite internal plan numbering ("problem 5/8/9/10/13", "orbit-consolidation plan") and specific targets/nights (Gl 229 2024-02-27/28, AL Dor, iota Peg 1.05 mas), against the project rule that science targets appear only as worked examples in design notes.
- "disc" (epochs, 11 hits) vs "disk" (aliases, 7 hits) inconsistent.

### B10. detection / limits (mostly fine)
- Note. `DetectionMC.threshold` uses the plain 1-fap quantile while `false_alarm_probability` is (k+1)/(n+1), so the realized FAP at the threshold is slightly above the requested one. `nsigma` is the documented two-sided equivalent (`nsigma(2.0, 1.0, 5)` = 1.779 vs one-sided 1.438); `local_nsigma` matches sqrt for 1 dof. Verified: `chi2ppf`, `ruffio_upperlimit`, `nsigma` match scipy in float32 (1e-3 relative at p=1-1e-6 is input quantization). `numpy>=2.0` is required so `onp.trapezoid` is fine. `detection_statistics` and `limits` follow model-first order with `old_order` shims.

### B11. Legibility: over-long functions
- Severity: 0.4.x. Effort: medium each. Read.
- Lines >120: `aliases.fit_orbit_aliases` 2437 (511 lines, ~260 docstring; body mixes window, V^2 amplitude, auto primary, seeding, settings, checkpoint loop), `epochs.epoch_positions` 1619 (360), `epochs.start_from_positions` 2358 (263), `orbit_search.score_orbits` 918 (267), `epochs.estimate_primary_diameter` 2041 (225), `detection.injection_recovery` 757 (224), `grid_fit.linear_flux_grid` 554 (181), `orbit_search._band_marginal` 631 (141), `detection.detection_statistics` 111 (121).
- In `score_orbits`, `log_flux` leaks from the band loop into the `extra` zeros dtype (fragile). `_orbit_list` decides batched vs single from the first leaf only.

### B12. Imports
- Verified. Orbit modules import jaxoplanet lazily (`orbits._jaxoplanet()`), so the base install is unaffected.

---

## Suggested before-0.4.0 shortlist
A1, A6, A7, A8, B1 (decision-tree doc + `AliasResult` bridge), B4, B9; B2 if cheap. Everything else 0.4.x or notes.
