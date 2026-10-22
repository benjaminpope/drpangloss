# Code shape audit (0.4.1)

Report only: nothing under `src/` or `tests/` changes here. Measured on `origin/main` at `52cbfea` with [`scripts/code_shape.py`](../scripts/code_shape.py) (`python scripts/code_shape.py [--since REF]`), which prints every list below: definitions, per-module call sites (own-module uses, `from x import _name`, `mod._name` elsewhere, and registering decorators such as `@f.defjvp`), statement counts, the tests/docs/examples grep, normalized-AST hashes, same-name body similarity (`difflib`) and line-shingle overlap. Two steps were done by hand: the `~/data` analysis-script grep (`rg` for `virgil… import _name`) and reading the bodies behind the proposals. Spelling follows Oxford -ize.

## Summary

The count of private helpers is real but is mostly **decomposition, not triviality**. Of 877 module-level private functions (15,914 lines), 467 have one call site, but only 113 of those have ≤3 statements, and only **76** can be inlined (the other 37 are jit/registered boundaries or mentioned by tests). Those 76 total 661 lines, so inlining them saves about 350 lines. The other ~350 single-use helpers are 4–86 statements long: named steps of large functions, and they should stay named unless their parent is deleted. Same-named helpers across modules are **not** duplicates (the highest body similarity is 0.48, mostly <0.2). The genuine duplication is in differently named code: a profiled-amplitude residual, union-find closure grouping, and the per-pipeline `_nuts`/`_posterior`/`_summarize`/`_fit_checks`/quick-look-cell skeletons. Realistic net saving from the six PRs below is **about 1,300–1,700 `src` lines (2–3%)**; the large lever is the 0.5 removal of the argument-order shim and `legacy/`, not helper hygiene.

| Category | Finding | Size |
|---|---|---|
| (a) trivial single-use wrappers | 76 to inline; 37 to keep (17 jit/registered, 25 mentioned by tests/docs, 5 both) | 661 lines, about −350 net |
| (b) true duplicates | 5 clusters (see below); same-name helpers are not duplicates | about −250 lines |
| (c) oversized modules | `pipeline/disk.py`, `aliases.py`, `epochs.py`, `oifits.py`, `pipeline/orbit.py`: simplify first; split only `aliases.py` and (0.5) `models.py` | about −600 lines |
| (d) dead code | none: no private module-level function has zero uses (the one candidate, `limits._gammaincinv_jvp`, is registered by `@_gammaincinv.defjvp`) | 0 lines |

The coordinator's "937 private vs 490 public" counts module-level functions and classes (now 933 and 494); module-level private *functions* alone are 877. Methods add 286 private and 666 public.

## (a) Trivial single-use wrappers

Rule used: one call site in `src/`, ≤3 statements, no decorator, and no mention in `tests/`, `docs/` or `examples/`. The full list is in A1; the script reproduces it with no manual step. Concentrations: `pipeline/star.py` (11), `pipeline/_quicklook.py` (6), `pipeline/disk.py` (5), `ensemble.py`, `imaging.py`, `pipeline/binary.py` (4 each). Pure `return f(...)` wrappers among them: `_log_radial_norm`, `_field_latents`, `_kepler`, `_n_trials`, `_n_free`, `_robustness_of`.

**Keep** (A2): `jit`/`vmap`/`custom_jvp` boundaries (`_jitted_chi2`, `_jitted_residuals`, `_jitted_log_norm`, `_neg_loglike_hessian`, `_fit_batch`, `_m2loglike_grid_jit`, `_batch`, `_leggauss`, ...), and names that tests import (`_signs_trusted`, `_channels`, `_load_disk`, `_load_star`, `_fit_batch`, `_require_finite`, ...). Never inline a helper that analysis scripts import from `virgil`: `plotting._draw_beam`, `likelihood._whitened_and_errors`, `spectra._planck_ratio`, `oidata._valid_samples`, `models._cvis_gaussian_envelope`, `models._as_flux` (found by grepping `~/data`).

## (b) True duplicates

Same names, different bodies (not duplicates): `_chi2`, `_moments`, `_positive`, `_jsonable`, `_candidates`, `_polish`, `_refine`, `_profile`, `_evaluate`, `_limits`, `_parts`, `_check`, `_distinct`. Largest body similarity is 0.48 (`_quicklook`, a wrapper each) and 0.35 (`_moments`, `_evaluate`). No two private defs of ≥3 statements have the same normalized AST. The real duplicates:

1. **Profiled V² amplitude residual.** `aliases._epoch_residuals` (with `_scale_vis`) and `epochs._amp_residuals` repeat the same 9-line weighted-least-squares amplitude and clip; the docstring of one cites the other. Home: `epochs` (which `aliases` imports), as one `amp_profile(r, data, n, amp_range)` returning `(amp, scaled residuals)`. About −25 lines.
2. **Distinct-mode selection.** `aliases._distinct` (tracks of orbit modes) and `epochs._distinct` (positions of ranked orbits) are the same greedy "best first, keep if some time differs by more than d". Home: `epochs`, `_distinct_tracks(tracks, d, n=None)`. About −15 lines.
3. **Union-find over triangles.** `_closure._groups` and `oifits._closure_stations` each define a local `find`. Home: `_closure` (nothing below `oifits` may import `oidata`; `_closure` imports nothing, so `oifits` importing it keeps the order), or `_utils` if a third user appears. About −20 lines.
4. **Per-pipeline sampling and report skeletons.** `star._nuts` and `disk._nuts` (numpyro model, `_run_chains`, `_draws`); `_posterior`, `_summarize`, `_fit_checks`, `_chi2_section`, `_chi2_checks` and `_bound_fractions` in four pipelines each; `_quicklook._star_cells`, `_disk_cells` and `_binary_cells` share 40–46% of their lines (shingle overlap). Home: `pipeline/_common` (already holds `_run_chains`, `_draws`, `_diagnostics`, `_sampler_report`, `_bound_samples`). About −150 lines.
5. **`disk._ring_start` / `disk._fill_new`** overlap 50%; fold into one in the disk PR (about −10).

Not duplicates, despite the guess: `_closure_stations`/`_stations_from_triangles` (write side) versus `_baselines_from_triangles` (read side); they share a concept but not code.

## (c) Oversized modules

Lines: `models` 6129, `aliases` 3567, `epochs` 3084, `imaging` 2895, `pipeline/disk` 2546, `oidata` 2077, `fitting` 1955, `pipeline/_checks` 1929, `oifits` 1865, `plotting` 1849. Prefer deletion, then (for `aliases`, `models`) a private split that keeps the public path.

- **`pipeline/disk.py`** (2546; 88 private behind 4 public). Parallel *ring* and *GP* paths are written twice: `_compare_rings` (84) / `_compare_gp` (54), `_profile_rings` (72) / `_profile_gp` (57), `_fold_fit` / `_fold_gp_values` / `_fold_pa`, `_ring_priors` / `_planet_priors` / `_geometry_priors`. Drive both from one family table (builder, priors, start, fold, profile) and delete the second copy of each driver. Five trivial wrappers inline. Target −350 to −450. No split.
- **`aliases.py`** (3567; 79 private behind 7). Duplication (b1, b2) and 3 inlines (`_senses`, `_seed_motion`, `_skipped_codes`; `_batch` is a jit boundary and `_pinned`, `_failed_band`, `_dof_pairs`, `_signs_trusted` are called by tests, so they stay) first (−100). The short-arc path (`_short_arc_profile`, `_short_arc_bands`, `_seed_scores`, `_seeding_positions`, `_signs_trusted`, `_seed_motion`, ~250 lines) is a seeding strategy separate from the band fit. If still above 3000 lines after that, split private submodules at: `_alias_problem.py` (`_UDBinary` … `_Draws`, lines 376–1000), `_alias_seed.py` (seeding, lines ~2380–2700), leaving `fit_orbit_aliases` and the result classes in `aliases.py`. Pin the split with `tests/lint/test_package_layout.py` and the import-order paragraph of `AGENTS.md`. The split is size-neutral; only do it if the worker judges it clarifies.
- **`epochs.py`** (3084; 42 private). `_coarse_then_exact` (85) and `_flag_one_night` (85) are the only long helpers; dedupe (b1, b2); `_loglike_static`/`_loglike_kernel` and `_grid_scores`/`_grid_marginal`/`_grid_fixed`/`_grid_batch` are four near-sibling evaluators to check for a shared kernel. Target −80. No split.
- **`oifits.py`** (1865; 55 private behind 3). Read side: `_read_closure_phases` (112), `_read_visibilities` (69), `_concat_records` (62), `_concat_extras`; write side: `_baseline_layout` (59), `_closure_baselines`, `_closure_stations`, `_stations_from_triangles`, `_repeated`, `_scattered`, `_unsupported_reason` (about 190 lines reconstructing stations for closure-only files). Dedupe (b3); check whether the reconstruction can be replaced by writing the stations the reader already knew (`_BaselineLookup`). Target −120 to −200. No split.
- **`pipeline/orbit.py`** (1564; 44 private behind 1). Four `_robustness*`/`_without`/`_outlying_night` helpers, `_single_star`, `_modes`, `_null_verdict`: check each against `aliases` results for already-reported numbers; `_positions_json`/`_positions_from_json` (78 lines) re-serialize what `pipeline/_io` handles. Target −100. No split.
- **`fitting.py`** (1955). `_lm`, `_lbfgs`/`_lbfgs_loop`/`_lbfgs_run`/`_lbfgs_round`/`_capped_lbfgs` (five layers, 200 lines) and `_adam` share the loss/gradient/tolerance scaffolding (`_loss_and_gradient`, `_tolerance`, `_lm_tolerance`, `_largest`, `_scaled_loss`); four trivial wrappers inline. Target −80. No split.
- **`models.py`** (6129, Wave 3) should split in 0.5, not 0.4.1 (public path `virgil.models` must stay stable, and the split is size-neutral). If done: `_models_components.py`, `_models_cvis.py` (the analytic `cvis_*`), `_models_disk.py` (`MultipoleDisk` and `multipole_disk_cvis`), `_models_system.py` (`System`, binaries, `HarmonixModel`), re-exported by `models`.
- `imaging.py` (2895), `oidata.py` (2077), `plotting.py`, `_checks.py`: no change proposed; `imaging`'s private helpers are mostly the jit boundaries.

## (d) Dead code

None found. The script's only zero-use candidate, `limits._gammaincinv_jvp`, is the rule of a `custom_jvp` registered by decorator (`@_gammaincinv.defjvp`), so it is live; the script now counts decorated defs as used. No private module-level function is unreferenced in `src/`. Not measured: unused private methods and unused public API. 0.5 candidates (not 0.4.1): `_deprecate.py` (337 lines) and its decorators in 7 modules once `removed=True` lands, and `legacy/oifits_implaneia.py` (430 lines; no analysis script in `~/data` mentions it; confirm with Ben).

## PR plan

Order matters: nothing else shares files, except that PR 3 reuses helpers PR 4 adds to `pipeline/_common`, so land **PR 4 before PR 3**. Net lines are estimates; PR 1's inlining figure is measured, the rest are not.

| # | PR | Files | Net `src` | Risk | Pinning tests | Wave 3? |
|---|---|---|---:|---|---|---|
| 1 | oifits and closure | `oifits.py`, `_closure.py`, `oidata.py` (only if `_row_labels` inlines) | −150 to −250 | medium: readers feed everything | `test_oifits.py`, `test_closure.py`, `test_closure_offsets.py`, `test_regressions.py`, round-trip fixtures | no |
| 2 | orbit family | `aliases.py`, `epochs.py`, `orbit_search.py`, `orbits.py`, `tempering.py` | −200 to −300 | high: numerics | `test_aliases.py`, `test_epochs.py`, `test_epoch_coarse.py`, `test_orbit_search.py`, `test_orbit_positions.py`, `test_orbits.py`; run the alias gate once at the final release commit | no |
| 3 | disk pipeline | `pipeline/disk.py` | −350 to −450 | medium | `test_pipeline_disk.py`, `test_pipeline_robust_disk.py`, `test_flared_disk_*.py` | no (after 4) |
| 4 | other pipelines | `pipeline/_common.py`, `_core.py`, `binary.py`, `star.py`, `orbit.py`, `image.py`, `_quicklook.py`, `cli.py`, `_io.py` | −300 to −450 | medium | `test_pipeline_*.py` (binary, star, orbit, image, io, checks, robust_*), `test_concrete_checks_under_jit.py` | no |
| 5 | numerics and sweep | `fitting.py`, `likelihood.py`, `imaging.py`, `inference.py`, `ensemble.py`, `plotting.py`, `limits.py`, `detection.py`, `arcs.py`, `crossval.py`, `simulate.py`, `_gauss_newton.py`, `angles.py`, `metrics.py` | −200 to −300 | medium-low | `test_fitting.py`, `test_imaging.py`, `test_detection.py`, `test_plotting_detection.py`, `test_sparse_imaging.py`, `test_regressions.py` | no |
| 6 | Wave 3 files | `models.py`, `fields.py`, `radio.py`, `radio_clean.py`, `_geometry.py` | −60 to −120 | low | `test_models_core.py`, `test_multipole_disk*.py`, `test_composition.py`, radio and field tests | **yes: wait for the Wave 3 PR to merge** |

No two PRs share a file. PRs 1, 2, 4 and 5 can run in parallel now; 3 follows 4; 6 follows the Wave 3 merge. Each worker brief: read `AGENTS.md` "Keeping the code small" and the import-order paragraph, list the categories above for its files from this note, keep every name tests or `~/data` import (A2), and report net lines from the CI size comment.

## Guard

Recommended: **an `AGENTS.md` rule plus a non-failing CI report**, no hard lint. Add to "Keeping the code small": "Don't add a private function that is called once and has ≤3 statements (the audit's threshold) unless it is a `jit`/`vmap` boundary, a test seam or a named step; inline it." Then have the existing size comment in `.github/workflows/lint.yml` also print, from `scripts/code_shape.py`, the added private defs with one call site and ≤3 statements (the audit's threshold) (names only). The report is `python scripts/code_shape.py --since origin/main`, which lists only the single-use, undecorated, unmentioned private functions added since the base; `lint.yml` calls it after checking out the base ref, and the audit and guard both use the 3-statement cut (`--max-stmts`). It never blocks a legitimate helper, and reviewers see the list. Rejected: a failing lint test (the 33 legitimate keepers would need an allowlist that rots) and a private/public ratio (it penalizes splitting a long function, which is the good case).

## Appendix

### A1. Inline candidates (one call site, ≤3 statements, undecorated, not mentioned in tests/docs/examples)

- `_gauss_newton.py`: `_decrement`
- `aliases.py`: `_senses`, `_seed_motion`, `_skipped_codes`
- `angles.py`: `_log_radial_norm`
- `arcs.py`: `_ecc_step`, `_intersect`
- `crossval.py`: `_is_number`
- `detection.py`: `_seed_record`, `_fingerprint_data`
- `ensemble.py`: `_default_weight_ranges`, `_start_model`, `_member_image`, `_chi2_red`
- `epochs.py`: `_check_marginal_data`
- `fitting.py`: `_print_progress`, `_check_dense_size`, `_lm_residuals`
- `imaging.py`: `_base_weight`, `_spectral_shape`, `_field_latents`, `_rotated_180`
- `legacy/oifits_implaneia.py`: `_scalar`
- `likelihood.py`: `_whitened_and_log_norm`, `_term_loglike`
- `models.py`: `_unit_flux`, `_rim_angle`, `_cvis_binary_angular`
- `oidata.py`: `_row_labels`
- `oifits.py`: `_exposure_time`, `_check_amptyp`
- `orbits.py`: `_kepler`, `_thiele_innes`
- `pipeline/_common.py`: `_mcmc`, `_sample_stats`, `_wrap`
- `pipeline/_quicklook.py`: `_fmt`, `_closing`, `_binary_posterior_cells`, `_star_posterior_cells`, `_disk_geometry_cell`, `_disk_posterior_cell`
- `pipeline/binary.py`: `_n_trials`, `_derived_diagnostics`, `_chi2_section`, `_search_checks`
- `pipeline/cli.py`: `_parse_value`
- `pipeline/disk.py`: `_wavel0`, `_n_free`, `_single_ring`, `_mackay_scale`, `_best_at_bound`
- `pipeline/image.py`: `_edges`, `_members`, `_edge_flux`
- `pipeline/orbit.py`: `_search_report`, `_robustness_of`, `_orbit_section`
- `pipeline/star.py`: `_uniform_disk`, `_quadratic_derived`, `_quadratic_priors`, `_registry`, `_freq_max`, `_diameter_grid`, `_lobe_table`, `_fit_plots`, `_chi2_section`, `_fit_section`, `_posterior_checks`
- `plotting.py`: `_one_sided_fap`, `_rates_at`, `_position_covariances`
- `radio.py`: `_relabel`
- `radio_clean.py`: `_sidelobe`, `_model_image`
- `simulate.py`: `_shift_times`

### A2. Keep: one call site, ≤3 statements, but a jit/registered boundary or mentioned in tests/docs/examples

`_flat.py:_` (jit, tests), `_marginal.py:_leggauss` (jit), `aliases.py:_batch` (jit), `aliases.py:_pinned` (tests), `aliases.py:_flip_is_relabelling` (tests), `aliases.py:_failed_band` (tests), `aliases.py:_signs_trusted` (tests), `aliases.py:_dof_pairs` (tests), `arcs.py:_jittered` (tests), `epochs.py:_loss_order` (tests), `fields.py:_radial_factor` (tests), `fitting.py:_gradient_norm` (jit), `fitting.py:_state_gradient_norm` (jit), `fitting.py:_lm_tolerance` (jit), `imaging.py:_pad_brightness` (tests), `imaging.py:_jitted_log_norm` (jit), `imaging.py:_jitted_residuals` (jit), `imaging.py:_jitted_chi2` (jit), `inference.py:_neg_loglike_hessian` (jit), `inference.py:_neg_loglike_curvature` (jit), `limits.py:_significance_ceiling` (tests), `metrics.py:_ncc_table` (jit, tests), `orbits.py:_fit_batch` (jit, tests), `pipeline/_common.py:_m2loglike` (tests), `pipeline/_common.py:_m2loglike_grid_jit` (jit), `pipeline/binary.py:_require_finite` (tests), `pipeline/disk.py:_channels` (tests), `pipeline/disk.py:_load_disk` (tests), `pipeline/disk.py:_geometry_grid` (jit), `pipeline/disk.py:_dynamic_range_sigma` (tests), `pipeline/disk.py:_compare` (tests), `pipeline/orbit.py:_engine_defaults` (tests), `pipeline/orbit.py:_positions_json` (tests), `pipeline/star.py:_load_star` (tests), `pipeline/star.py:_scan` (jit, tests), `pipeline/star.py:_refine` (jit, tests), `pipeline/star.py:_bound_fractions` (tests)

### A3. Per-module counts

| module | lines | public | private | public methods | private methods |
|---|---:|---:|---:|---:|---:|
| pipeline/disk.py | 2546 | 4 | 88 | 6 | 7 |
| aliases.py | 3567 | 7 | 79 | 35 | 2 |
| oifits.py | 1865 | 3 | 55 | 9 | 2 |
| models.py | 6129 | 51 | 49 | 214 | 111 |
| pipeline/_common.py | 729 | 0 | 49 | 0 | 0 |
| imaging.py | 2895 | 31 | 48 | 27 | 0 |
| fitting.py | 1955 | 8 | 47 | 29 | 9 |
| pipeline/orbit.py | 1564 | 1 | 44 | 3 | 7 |
| epochs.py | 3084 | 16 | 42 | 45 | 9 |
| pipeline/star.py | 1221 | 5 | 42 | 4 | 5 |
| detection.py | 1540 | 8 | 30 | 19 | 7 |
| pipeline/binary.py | 897 | 1 | 30 | 2 | 8 |
| plotting.py | 1849 | 15 | 29 | 0 | 0 |
| orbit_search.py | 1137 | 4 | 25 | 8 | 0 |
| pipeline/_quicklook.py | 653 | 3 | 23 | 0 | 0 |
| orbits.py | 1629 | 14 | 21 | 48 | 8 |
| likelihood.py | 1008 | 17 | 17 | 0 | 0 |
| arcs.py | 687 | 5 | 16 | 4 | 0 |
| pipeline/image.py | 590 | 1 | 14 | 2 | 4 |
| ensemble.py | 1214 | 14 | 13 | 5 | 0 |
| grid_fit.py | 689 | 9 | 9 | 0 | 0 |
| pipeline/cli.py | 259 | 2 | 9 | 0 | 0 |
| radio.py | 1567 | 5 | 9 | 27 | 15 |
| radio_clean.py | 651 | 4 | 9 | 5 | 0 |
| _geometry.py | 704 | 20 | 8 | 0 | 0 |
| inference.py | 432 | 8 | 8 | 0 | 0 |
| limits.py | 670 | 10 | 8 | 0 | 0 |
| observables.py | 1017 | 6 | 8 | 28 | 10 |
| oidata.py | 2077 | 3 | 8 | 27 | 21 |
| _flat.py | 205 | 1 | 7 | 19 | 2 |
| _gauss_newton.py | 251 | 5 | 7 | 0 | 0 |
| gains.py | 735 | 5 | 7 | 10 | 3 |
| svi.py | 468 | 2 | 7 | 1 | 1 |
| crossval.py | 462 | 7 | 6 | 0 | 0 |
| pipeline/_checks.py | 1929 | 51 | 6 | 3 | 0 |
| _deprecate.py | 337 | 5 | 5 | 0 | 0 |
| _utils.py | 209 | 8 | 5 | 0 | 0 |
| fields.py | 645 | 5 | 5 | 10 | 1 |
| pipeline/_core.py | 525 | 3 | 5 | 7 | 23 |
| spectra.py | 692 | 10 | 5 | 21 | 19 |
