# virgil size sweep: v0.3.0 -> origin/main (9717dcc), read-only static analysis

Trees: origin/main (src, tests, docs, notebooks, plus examples/scripts/benchmarks exported for the sweep) and v0.3.0. The sweep scripts are not kept.

Confidence tags: **[V]** = verified by reading the code; **[H]** = heuristic (AST/token/diff analysis, not individually read).

Headline: the 23.5k net new src lines are almost all new features (16.2k of them are 11 new modules). There is very little copy-paste (about 180 lines of exact-structure duplication in src) and very little dead code (about 100 lines dead everywhere). The real levers are (a) a parallel orbit scorer that nothing wires in, (b) docstring prose that belongs in docs/design notes, (c) test consolidation, (d) deprecation shims that fall due, and (e) hand-written image code beside the PolarDisk engine.

Line-count basis (AST): src 48,813 lines = 25,253 code + 16,670 docstring + 1,122 comment + 5,768 blank. Docstring lines by section: prose 8,253, Parameters 4,661, Returns 971, Attributes 810, Notes 622, Examples 499, References 100.

---------------------------------------------------------------------

## 1. Dead code

Method: AST references (Name, Attribute, keyword, import alias, identifier-like string constants, so getattr/registries count as uses; docstrings and `__all__` do not) in src, then token counts in tests, docs (api listing counted separately), notebooks (not archive), examples, scripts, benchmarks. Then a reachability pass (`reach.py`): roots = anything referenced outside src defs or by tests/docs/notebooks/examples, plus `__all__`; transitive closure through same-named defs. Pipeline `_LAZY`, `__getattr__` hooks, the star model registry (`_registry`/`_entry_for` by string), numpyro/optimistix/equinox protocol methods were checked and excluded.

### 1a. Dead everywhere [V]

| item | file:line | lines | note |
|---|---|---|---|
| `_azimuthal_spectrum` | models.py:3302 | 11 | no reference anywhere (its sibling `_log_azimuthal_spectrum` is used) |
| `_NAMES` constant | aliases.py:92 | 11 | unused tuple of coordinate names |
| `image_harmonics_from_latents` | harmonics.py:136 | 3 | only listed in docs/api/harmonics.md |
| `lag1_correlation` | tempering.py:238 | 15 | exported, listed in docs/api/tempering.md, no caller in src/tests/docs |
| `Format_STAINDEX_V2` / `_T3` | legacy/oifits_implaneia.py:98,110 | 20 | only dead after the shim; module says removal in 0.6 |
| `rad2mas`, `GetWavelength`, `__getattr__` (cp_indices) | legacy/oifits_implaneia.py:55-96 | ~35 | FutureWarning shims, "removed in virgil 0.6" |
| unused parameters (not interface-forced) | aliases.py:2164 `_physical(t_ref)`; models.py `_n_needed(q_max)`; pipeline/_quicklook.py:38 `_intro(summary)` | ~3 | AST unused-arg scan found 43 total, almost all are interface-conforming overrides |

Total dead everywhere: about 40 lines of core code (about 95 with the legacy shims). Kept deliberately (look dead, are not): `_FlatTransform.codomain/_inverse` (numpyro Transform API), `_GradientStoppedLM.terminate` (optimistix API), `_gammaincinv_jvp` (`@defjvp`), `pipeline.__getattr__/__dir__` (`_LAZY`), `virgil.__getattr__`.

### 1b. Used only by tests (no src caller, no tutorial/notebook/example use) [H, names grep-verified]

| item | file:line | lines | tests |
|---|---|---|---|
| `chi2ppf` + `_gammaincinv`, `_gammaincinv_guess`, `_gammaincinv_jvp` | limits.py:161,201,223,255 | 94 | test_regressions.py (~60 lines) |
| `cvis_radial_dirac_delta_modulated` | models.py:6519 | 77 | test_models_sources.py (~40) |
| `plot_hmc_fisher_chainconsumer` | plotting.py:628 | 58 | one mention |
| `PolarDisk.freeze_geometry`, `suggest_n_r` | models.py:5121, 5016 | 58 + 38 | test_radial_gp (documented in api only) |
| `apply_elliptical_transf_spat_freq` | _geometry.py:382 | 36 | test_utils.py |
| `chain_starts` | epochs.py:1026 | 38 | test_epochs (api only) |
| `polar_field_priors`, `PolarField.from_disk` | fields.py:701, 588 | 39 + 30 | api only |
| `from_legacy` / `to_legacy` | harmonics.py:388, 410 | 42 | test_harmonics.py ~60 |
| `residual_image` | imaging.py:3030 | 45 | api only |
| `bias_test` | simulate.py:72 | 36 | api only |
| `RadialLinear`, `RadialPower`, `_RadialFunction` | models.py:3216, 3242, 3208 | 54 | api only |
| `AxialVonMises`, `position_angle_prior` | orbits.py:1659, 1611 | 42 + 23 | api only |
| `KeplerOrbit.relative_velocity/to_thiele_innes/from_jaxoplanet/to_jaxoplanet`, `PositionData.from_sep_pa`, `StateVectorOrbit.from_kepler` | orbits.py:392,439,474,446,634,1063 | ~90 | tests only |
| `spectral_window` | aliases.py:168 | 13 | api only |
| `LorentzianLine` | spectra.py:304 | 20 | api only |

32 items (1,333 lines incl. model classes `TruncatedCone` 384, `GravityDarkenedStar`-class members etc. that are public science features and are not removal candidates). Excluding public model classes and documented user API, the removable tests-only set is about 270 src lines (chi2ppf chain 94, cvis_radial_dirac 77, plot_hmc_fisher 58, apply_elliptical_transf_spat_freq 36) plus about 150 test lines. These are public names, so each needs an owner decision.

### 1c. Options never exercised [H]
74 optional arguments of public functions are never passed by any call in src/tests/examples/notebooks/docs (e.g. `epoch_positions` flag tuning knobs `gap_mas, flag_chi2_factor, flag_sep_fraction, flag_flux_tol`; `start_from_positions(refine_positions, flag_options)`; `fit(lbfgs_memory, cg_steps)`; `clean(stall_window, stall_tolerance, refresh_norms)`; `variational(optimizer, window)`; `HarmonixModel(render_method, expects_wavelength_units)`). Removing them is about 250 lines incl. docstring entries, but several are intentional knobs; low value, not recommended before 0.4.0.

---------------------------------------------------------------------

## 2. Duplication

Tooling: AST statement windows (identifiers and literals normalised, window of 4-8 statements, nested defs excluded) over src and tests, plus function-pair similarity (difflib on normalised lines). Finding: **exact-structure duplication in src totals only about 182 lines**; near-duplicates are conceptual (parallel algorithms), not pasted blocks. Estimates below for the parallel clusters are [H] unless stated.

### 2a. Orbit modules (epochs.py 2672, aliases.py 3236, orbit_search.py 1184, tempering.py 352, orbits.py 1700)
All new since 0.3.0. They already share `_Surface` (epochs.py:519, 119 lines), `_gauss_newton`, `_precision`. Parallel pieces [V by read]:

| cluster | locations | lines | one shared helper would remove |
|---|---|---|---|
| "log integral over a bounded scalar by adaptive peak/Gauss-Legendre" implemented 3 times | epochs.py:441 `_log_integral` (39); orbit_search.py:550 `_newton` (27), :579 `_scale` (7), :588 `_nodes` (28), :618 `_marginal_1d` (11), :631 `_band_marginal` (141); grid_fit.py:111-145 `_refine_flux_grid` Newton step | 250 | ~80 |
| distinct-orbit filter (best-first, track distance > threshold) | epochs.py:1005 `_distinct` (19) and aliases.py:1092 `_distinct` (16) | 35 | ~15 |
| IRLS kernel problem classes with identical `loss/rows/diag` shape | epochs.py:1480 `_RefineProblem` (29) and aliases.py:534 `_KernelProblem` (34) | 63 | ~25 |
| "static binary with optional UD primary" | epochs.py:1282 `_binary` (14); aliases.py:427 `_UDBinary` (16) + :445 `_model` (11); `BinaryModelCartesian` | 41 | ~15 |
| `_epoch_residuals` vs `_amp_residuals` (V² amplitude profile) | aliases.py:458 (28) and epochs.py:1318 (19) | 47 | ~15 |
| three ways to go from data to starting orbits | `epochs.start_from_positions` 2386 (263) + `rank_orbits` 907 (96) + `chain_starts` 1026 (38); `aliases._candidates/_init_orbit/_init_starts/_score_pool/_refine` 1000-1150 (~110) + `position_profile` 364 (46); `orbit_search.score_orbits` 918 (267) | ~900 | see section 3 |

Subtotal if only the helpers are shared: about 150 lines. Note `start_from_positions` is the only path from an orbit search into the general `fit()` with an arbitrary `model` (extended components, noise terms) and `OrbitStart.chain_values` into numpyro; `fit_orbit_aliases` is restricted to a UD primary + point companion. So they are not duplicates of each other, only of the shared machinery.

### 2b. pipeline/binary.py (1048) vs star.py (1473) vs _common.py (384) [V]
- NUTS wrapper repeated: binary.py:795 `_run_nuts` (21) + :750 `_posterior` (43) + :818 `_write_samples` (14) + :849 `_plot_posterior` (11) vs star.py:1136 `_nuts` (53) + :1191 `_posterior` (68). Same sequence: `numpyro_model`, key split, `chain_init_params`, `_mcmc`, run, `get_samples`, `_sample_stats`, `_diagnostics`, `write_h5`, corner/trace plot. A `_sample(posterior, start, settings, dense_mass)` and `_save_posterior_plots` would remove about 55.
- `_validate` repeated sampler/error_scale/chain_method/wavel_range/error_floor checks: binary.py:215 (34) and star.py:378 (30): about 17 shared (dup3: 21 matched).
- `_overview` stage (8 lines each, binary.py:367, star.py:547) and `_stages` boilerplate (binary.py:316, star.py:425): about 15.
- `_quicklook._binary_posterior_cells` (22) and `_star_posterior_cells` (29): 19 matched lines.
Total pipeline: about 110 lines. Confidence [V] for the NUTS block, [H] for the rest.

### 2c. imaging.py (3316) vs ensemble.py (1320) [H]
ensemble.py builds on imaging (`l_curve`, `log_evidence`, `starting_image`, `_chi2`); no pasted blocks found (best similarity: imaging `_field_images`/`_evidence_latents` against each other, not against ensemble). At most 30 lines of small helper overlap (`_parts`, `_chi2`, `_member_image` vs `_map_images`). Not a target.

### 2d. PolarDisk section of models.py vs hand-written classes [V]
`_centred_cvis` of GaussianArc, TruncatedCone and ModulatedGaussianRim already runs on `polar_disk_cvis`. What remains hand-written is the image path:
- TruncatedCone `_centred_image` models.py:1066 (75 lines), plus `rings` 1026, `_modulation` 994, `_ring_angle` 1003, `_orders` 985 (about 35 more) vs PolarDisk `_lifted_image` models.py:5221 (89). Both contain the same orthogonal-grid row/column Gaussian splat (about 35 lines each) and the same blocked fallback. A shared `_splat_rings` helper removes about 35 even if neither class changes; moving the cone's image to `_lifted_image` removes about 60 to 90.
- ModulatedGaussianRim `_centred_image` models.py:2091 (43) repeats `PolarDisk._blurred` models.py:5330 (18): identical 12-line kernel block (dup3 matched lines 2126-2133 vs 5340-5347). A `_blur_in_plane` helper removes about 15; making the rim image a `PolarDisk(ThinRing, Harmonics, blur)` removes about 35 more (design/radial_profiles.md plans this; the planned old-vs-new check has not been run, so numerics are not verified).
- GaussianArc `_centred_image` (5) + `curve` 698 (21) + `_quadrature_cvis` 720 (11): the quadrature path for very narrow arcs is not on the engine; keep.
- FlaredDisk `_centred_image` 2521 (24): stays (rule: never deprecate classes not reproducible as special cases).
Subtotal: about 100 lines with no numerical change (helpers), about 200 if the images are re-routed (needs the numerical check).

### 2e. plotting.py (2078) [H]
dup3 finds only 4-line clusters (plot_model/plot_residual_map 400-403/488-491, plot_contrast_curve/plot_completeness). Cost is in size, not duplication: `plot_roc` 196, `plot_grid_map` 184, `plot_completeness` 156, `plot_null_distribution` 154, `plot_orbit_ensemble` 150, `plot_data_model_correlation` 150 lines. A shared figure/axes helper would save at most about 50.

### 2f. Other exact clusters in src [V], all small
`__getattr__` identical 11-12 lines in `System` (models.py:5607), `RadialSum` (3153), `spectra.Sum` (596): about 22; `__check_init__` shape checks in EllipticalGaussian 487, TruncatedCone 942, EllipticalLimbDarkenedDisk 1394: about 20; grid_fit.py:246 and 282 wrappers: 12; gains.py `_pack`/`_pack_offsets` (375/676): 9, `gain_modes`/`closure_offsets`: 5; oifits.py `_read_closure_phases`/`_read_absolute_phases`: 7+6+5; pipeline/_checks.py r_hat/divergences/chi2_reduced: 9; limits.py `_absil_limits`/`_injection_limits` 709/750: 13. Total about 120.

---------------------------------------------------------------------

## 3. Superseded or parallel APIs

| item | file:line | lines | status |
|---|---|---|---|
| `orbit_search.score_orbits` + `OrbitScores`, `SharedFlux`, `_Scorer`, `_band_marginal`, ... | orbit_search.py:1-1184 (keep `rank_scores` 209, 36 lines, moved into epochs) | **1,148** | [V] nothing in src calls it; only tests (test_orbit_search.py 598, test_broadcast_in_time.py:131-172), docs/api/orbit_search.md, CHANGELOG, design notes. It is stage D1 of design/automatic_orbits.md (D2+ not built). Added by PR #298 as one file (+1160/0 src, +572 tests, nothing else touched), so it can be held back cleanly. Unreleased, so no deprecation needed. Feature cost: shared-flux marginalisation across epochs and RV/position terms in the orbit score; no other API does this. |
| `rank_orbits` + `RankedOrbits` + `_ranked` + `_loglikes` | epochs.py:907 (96), 828 (62), 892 (13), 755 (70) | 241 | [V] used by `start_from_positions` and benchmarks; documented in docs/orbit_fitting.md and orbit_fitting.ipynb. Not superseded by `fit_orbit_aliases` (fixed-flux, general model function, different result type). |
| `start_from_positions` + `OrbitStart` + `chain_starts` + `_distinct` + `_loss_order` + `_warn_if_coarse` | epochs.py:2386 (263), 2297 (87), 1026 (38), 1005 (19), 2651 (4), 2657 (16) | 427 | [V] parallel to `fit_orbit_aliases` for the well-sampled case but it is the only route from search to `fit()` of an arbitrary model and into numpyro (`chain_values`). Could become a thin wrapper if `AliasResult` gave an orbit/start (release review B1), saving about 150 to 250 lines [H]; not before 0.4.0. |
| `aliases.position_profile` + `epochs.epoch_positions` seeding | aliases.py:364 (46) | 46 | wraps `starting_orbits`: a fourth path to a starting grid [V]. |
| `Tabulated` | spectra.py:624 | **96** | [V] CHANGELOG 0.3.0 says "removal is planned for 0.4.0"; still present with FutureWarning. Tests in test_chromatic.py, test_spectra_lines.py, test_deprecate.py (~40 lines). Removable now. |
| `GaussianArc` FutureWarning on construction + "Change of shape" paragraph | models.py:540-556, 613-624 | ~30 | [V] text says "removed in the next release". Keep through 0.4.0, remove at 0.5. |
| `_check_scales` FutureWarning for `rank_orbits`/`start_from_positions` default | epochs.py:719-735, docstrings at 951-956, 2474-2475 | ~25 | [V] epochs.py is new in 0.4.0, so the default can be `"marginal"` now with no warning (release review A-item). |
| `EpochPositions.gap` "unchanged for one release" | epochs.py:1162 | ~10 | [V] same logic; new module. |
| argument-order shim: `old_order`, `renamed`, `_apply_aliases`, `_alias_map`, `NAME_ALIASES` | _deprecate.py:71-226 (156 lines) + 18 `@old_order` and 22 `@renamed` decorator lines across 12 modules | 156 + 40 | [V] 0.4 adds the shim; per _deprecate.py:6 and CHANGELOG:162-180 the aliases go in 0.5 and `old_order(..., removed=True)` (a TypeError naming the new call) in 0.5, deleted later. Honest split: 0.5 removes `renamed`, `NAME_ALIASES`, aliases (about 85 src); the rest (about 125 src) at 0.6. Tests: test_deprecate.py 325 + part of test_argument_order.py 359 (about 450 lines over both releases). |
| legacy ImPlaneIA shims | legacy/oifits_implaneia.py:46-125 | ~80 | [V] "removed in virgil 0.6". The module itself is 430 lines. |
| `harmonics.from_legacy/to_legacy` + `_legacy_check` | harmonics.py:383-431 | ~50 | [V] only tests call them; needed only while TruncatedCone/ModulatedGaussianRim keep az_amps/az_pas. |

Estimates: removable **now if declared provisional / unreleased**: score_orbits (1,148 src, about 640 tests), `Tabulated` (96 + 40 tests), the `scales` FutureWarning and `gap` note (about 35). Total about 1,280 src + 680 tests. At **0.5**: about 85 (renamed) + 30 (GaussianArc) + 50 (legacy harmonics converters if the legacy classes are re-based) = about 165 src, about 250 tests. At **0.6**: about 125 + 80 src.

---------------------------------------------------------------------

## 4. Docstring and comment bloat

Counts [V]: 81 docstrings of 40 lines or more total 6,241 lines; the part that is lead prose beyond 12 lines or a Notes section totals 1,583 lines. Design-history markers are few: `problem N` / `plan phase` appear 19 times in aliases.py (e.g. 92-150, 2800-2860) plus one each in fitting, epochs, `_gauss_newton`, `_flat`. Named science targets in docstrings/comments: Gl 229 (4), GRAVITY (9), MATISSE (7), PDS 70 (2), AL Dor (2), iota Peg (1.05 mas, epochs.py:1286, 1700), WR 137/104, nu Hor, a MATISSE N-band contest file (imaging.py:2690). Few, but they conflict with "science targets only as worked examples in design notes".

The 20 longest docstrings (lines = docstring only) with movable estimate [H, section sizes verified]:

| # | name | file:line | doc lines | breakdown | what moves | est. saved |
|---|---|---|---|---|---|---|
| 1 | fit_orbit_aliases | aliases.py:2606 | 283 | Parameters 201, Notes 59 | Notes, `temper`, `sign_free`, `p_range` paragraphs with "solved problem 8/13/14" history, AL Dor, 5-night rules (design/orbit_aliases.md) | 90 |
| 2 | fit | fitting.py:638 | 184 | Parameters 161 | shared `noise`, `likelihoods`, `flat_coordinates` text (see below) | 40 |
| 3 | linear_flux_grid | grid_fit.py:554 | 159 | prose 50, Params 74 | derivation prose | 30 |
| 4 | gauss_newton_mass | fitting.py:1055 | 151 | prose 80 | mass-matrix derivation | 55 |
| 5 | clean | imaging.py:1471 | 151 | prose 78 | algorithm description | 50 |
| 6 | PolarDisk | models.py:4737 | 144 | Notes 41, Params 59, Examples 18 | design notes, conventions | 35 |
| 7 | error_scale | imaging.py:2630 | 144 | prose 102 | MacKay derivation, two $$ equations, MATISSE contest numbers (chi2/pt 0.005 vs 0.49) to docs page | 90 |
| 8 | epoch_positions | epochs.py:1637 | 138 | prose 55, Params 78 | night-flag rules, Gl 229 dates, iota Peg | 30 |
| 9 | numpyro_model | likelihood.py:863 | 129 doc (197 for the function) | Params 95, Examples 25 | `flat_coordinates` SBC narrative (1000 replicates), noise-term catalogue (duplicated in `fit`) | 40 |
| 10 | PolarField | fields.py:342 | 128 | prose 50, Notes 22 | design, conventions | 50 |
| 11 | TruncatedCone | models.py:757 | 123 | prose 67 | geometry derivation | 45 |
| 12 | start_from_positions | epochs.py:2386 | 119 | Params 81 | steps description duplicates docs/orbit_fitting.md:426 | 20 |
| 13 | GravityDarkenedStar | models.py:1593 | 118 | Notes 58 | physics notes | 45 |
| 14 | variational | svi.py:282 | 117 | prose 43 | method discussion | 25 |
| 15 | read_oifits | oifits.py:55 | 116 | Notes 58 | format-handling notes | 40 |
| 16 | RadialGP | models.py:3362 | 111 | prose 57 | kernel maths | 35 |
| 17 | EnsembleSpec | ensemble.py:102 | 109 | Attributes 106 | table of settings, keep | 0 |
| 18 | detection_statistics | detection.py:111 | 106 | Returns 30, Notes 23 | notes | 25 |
| 19 | score_orbits | orbit_search.py:918 | 104 | Params 57, prose 33 | (goes with section 3) | 25 |
| 20 | Harmonics | models.py:3952 | 102 | prose 50 | conventions | 30 |

Top-20 movable about 800 lines; leaving a 4 to 8 line summary plus a link costs about 100, so **net about 700 saved from the top 20 [H]**; applying the same rule to all 81 long docstrings (1,583 candidate lines, about half genuinely movable) gives **about 700 to 900 lines [H]**.

Second source [V]: parameter text repeated across public functions. `noise` is documented in 6 places (81 lines), `flat_coordinates` in 3 (42), `priors` in 3 (37), `data` in 58 (200 lines, short each). Pointing `fit`, `numpyro_model`, `variational`, `gauss_newton_mass`, `laplace_*` at one shared "Noise terms" and "Flat coordinates" docs section would save about 100 to 150 lines.

Also note module docstrings in the orbit modules carry long constant-explanation comment blocks: aliases.py:92-150 (about 60 lines of comments on `LABELS_FREE_NATS`, `_PERIOD_BOUND_FRACTION`, ...) cite "problem 5/8/10/13" and "plan phase 1d"; moving to design/orbit_aliases.md saves about 40 more.

---------------------------------------------------------------------

## 5. Tests (36,962 lines; 1,532 test functions; 229 parametrize uses)

Largest files [V]: test_aliases 2123, test_models_sources 1513, test_orbits 1505, test_fitting 1113, test_detection 1036, test_pipeline_star 1015, test_epochs 979, test_oifits 968, test_radial_gp 950, test_orbit_positions 949, test_observables 949, test_evidence 817, test_harmonics 808, test_pipeline_binary 799, test_regressions 769, test_polar_field 742, test_grid_and_limits 688, test_composition 651, test_sparse_imaging 620, test_orbit_search 598.

Tests of dead/superseded code [V]:
- test_orbit_search.py 598 + test_broadcast_in_time.py:125-172 (about 45): score_orbits.
- test_deprecate.py 325 (22 tests) + test_argument_order.py 359 (18 tests): shim tests, due 0.5/0.6.
- Tabulated usage in test_chromatic.py, test_spectra_lines.py, test_deprecate.py: about 40.
- test_harmonics.py:586-638 from_legacy/to_legacy: about 60; test_regressions.py chi2ppf: about 60; cvis_radial_dirac tests: about 40.
Sum if all above went: about 1,500 lines.

Near-duplicate tests [H, similarity >= 0.7 on normalised lines; 47 non-overlapping pairs, 539 matched lines in the top 80]:
- test_grid_and_limits.py:79,120,163,226: four `*_axis_order_tracks_key_order` tests, 21-25 lines each, 0.8-0.9 similar: parametrize, about 55 saved.
- test_pipeline_binary.py / test_pipeline_star.py: `test_run_folder_layout` (22/23 lines, 0.91), `test_quicklook_notebook` (31/24), `test_cli_*`, `full_run`, `_signature`, `data`: shared helper in `_shared.py`, about 70.
- test_oifits.py 251/780, 362/447, 686/913 pairs: parametrize, about 60.
- test_aliases.py 483/528 (failed-start counting), test_tempering.py 296/333/360, test_gauss_newton.py 231/247/409/435, test_observables.py 830/868 and 378/707, test_models_sources.py 61/137/250 (`*_oidata_and_render`) and 158/178: about 120.
- test_elr 65/142 (f64/f32 solver), test_orbits 241/260 (float32 variants): parametrize on dtype, about 25.

Duplicated helpers/fixtures [V]: `_night` (test_aliases 50, test_epochs 64, test_orbit_search 42, test_orbit_positions 40, test_oifits 706), `_tables` (test_observables 46 (58 lines), test_pa_round_trip 32 (58), test_oifits 26 (31), test_bandwidth_smearing 131), `_scene` ×9, `_data` ×8, `_uv` ×5, `_cvis` ×4 (2 identical), `_render_visibilities` ×3 (1 identical pair), `_no_model` ×3 (2 identical), `_baselines` ×6, `full_run`/`data` in the two pipeline tests. Moving the clearly shared ones (`_night`, `_tables`, `_render_visibilities`, `_cvis`, `_uv`, `_no_model`) into tests/_shared.py: about 250 lines.

Estimated savings: parametrization 400 + shared helpers 250 = **about 650 lines (1.8%)** [H]; tests of removed/superseded code about 1,500 (overlaps section 3). Exact structural duplication in tests is only 415 lines (dup.py).

---------------------------------------------------------------------

## 6. Growth attribution (git log --first-parent --numstat v0.3.0..origin/main -- src tests; 93 first-parent commits, each diffed against its first parent)

Totals: src +28,608 / -5,253 gross over merges (net +23,355; `git diff --shortstat v0.3.0 origin/main -- src` says +25,570/-2,215 for the endpoints); tests +18,537 / -897 (net +17,640).

New modules account for 16.2k of the src growth: aliases 3,236, epochs 2,672, pipeline/* 5,765 (star 1,473, binary 1,048, _io 808, _checks 787, _core 529, _quicklook 450, _common 384, cli 235, __init__ 51), ensemble 1,320, orbit_search 1,184, harmonics 521, metrics 400, tempering 352, _gauss_newton 295, _deprecate 226, _flat 205. Big growth in existing files: models.py +3,304 (3,306 -> 6,610), imaging +686, fitting +686, fields +514, likelihood +263, orbits +222.

Top 10 features (net lines; src / tests), PRs grouped by feature:

| rank | feature | PRs | src | tests |
|---|---|---|---|---|
| 1 | Orbit tools: epochs (rank/start/positions/flags), orbit_search scorer, aliases, tempering, result API | #257 #262 #281 #292 #298 #319 #321 #324 #331 #359 #366 #368 #372 #295 #280 #278 | +7,689 | +4,948 |
| 2 | virgil.pipeline (core+binary, star, quiet) | #305 #307 #308 | +5,051 | +1,782 |
| 3 | PolarDisk, harmonics, PolarField, radial profiles, cone modulation | #303 #336 #340 #355 #357 #367 | +4,218 | +4,100 |
| 4 | Image ensembles, metrics, image-field-from-data, imaging diagnostics, evidence helpers | #258 #259 #272 #304 #371 #361 #370 #330 #274 | +2,407 | +1,377 |
| 5 | Pre-0.4.0 review fixes (rework of pipeline + aliases + `_common`; churn +3,396/-1,977 gross) | #323 #325 #327 | +1,419 | +1,197 |
| 6 | Gauss-Newton kernel and curvature rows (`_gauss_newton`, aliases/orbit kernels) | #349 #352 #350 | +766 | +1,218 |
| 7 | Sampler flat coordinates (`_flat.py`, svi) | #301 | +725 | +534 |
| 8 | Bandwidth smearing, chromatic grids, broadcast-in-time, spectro select | #345 #346 #339 #265 | +480 | +833 |
| 9 | Dedup, convention renames, small fixes | #316 #311 #312-#315 #291 #293 #282 #283 #300 #332 | +412 | +735 |
| 10 | API argument-order shim | #256 #261 #266 | +188 | +549 |

Per-PR top (src, tests): #305 pipeline-core-binary 3,111/802; #307 pipeline-star 1,722/813; #323 review-0.4.0-fixes 1,273/1,114 (net, with +3,214/-1,941 gross src); #340 polar-p6 1,355/1,454; #336 polar-p2 1,360/1,121; #355 polar-p3 1,117/947; #298 orbits-scorer 1,160/572; #368 orbit-1c-defaults 890/647; #301 sampler-flat-coords 725/534; #372 orbit-1d-tempering 669/565; #292 orbit-search 698/406; #319 orbit-aliases 900/174.

Orbit tools plus the Gauss-Newton kernel that serves them (feature groups 1 and 6) are 36% of src growth and 35% of tests growth. Feature groups 1-4 are 74% of src growth.

---------------------------------------------------------------------

## Ranked reduction opportunities

(lines are src+tests unless noted; "pre-0.4.0" means can land before the release without a deprecation cycle)

| rank | opportunity | lines saved | effort | risk | when |
|---|---|---|---|---|---|
| 1 | Hold back `orbit_search.score_orbits` (revert #298 + later fixes; move `rank_scores` into epochs) | 1,150 src + 640 tests = about 1,790 | S | feature loss: shared-flux score with RV/position terms, unwired; unreleased | pre-0.4.0 (if declared provisional or held back); otherwise 0.5 |
| 2 | Move docstring design/derivation/history prose to docs/design notes, leave summary + link; consolidate repeated `noise`/`flat_coordinates` parameter text | 800 to 1,000 src (top 20 alone about 700) | M (mechanical, link check) | low (tests/test_docstrings.py, docs build) | pre-0.4.0 for top 20; rest 0.4.x |
| 3 | Test consolidation: parametrize near-duplicates, shared helpers into `_shared.py` | about 650 tests | M | low | pre-0.4.0 or 0.4.x |
| 4 | Remove `Tabulated` (promised for 0.4.0), unwrap `scales` FutureWarning (default marginal) and `EpochPositions.gap` note in the new epochs module | about 140 src + 60 tests | S | low | pre-0.4.0 |
| 5 | Re-route hand-written images onto the PolarDisk engine (TruncatedCone and ModulatedGaussianRim `_centred_image`, shared splat/blur helpers) and drop `cvis_radial_dirac_delta_modulated` | 100 (helpers only) to 230 src, about 40 tests | M-L (plan's old-vs-new numerical check is outstanding) | medium (rendered-image numerics) | helpers pre-0.4.0; re-routing 0.5 |
| 6 | Deprecation shims falling due: `renamed`+aliases 0.5 (85 src, about 250 tests), `old_order` 0.6 (125 src, about 200 tests), legacy ImPlaneIA shims 0.6 (80 src), GaussianArc warning 0.5 (30) | about 320 src + 450 tests | S each | low | 0.5 / 0.6, not before |
| 7 | Orbit-module helper unification (adaptive log-integral/Newton ×3, `_distinct` ×2, `_RefineProblem`/`_KernelProblem`, `_binary`/`_UDBinary`) | about 150 src | M | medium (numerics of evidences; keep tests as guard) | 0.4.x |
| 8 | Pipeline dedup (shared NUTS/posterior writer, validators, overview, quicklook cells) | about 110 src | S-M | low (stable contract tests exist) | pre-0.4.0 possible |
| 9 | Tests-only public helpers (`chi2ppf` chain 94, `plot_hmc_fisher_chainconsumer` 58, `apply_elliptical_transf_spat_freq` 36, `from_legacy/to_legacy` 42) and dead code (about 40) | about 270 src + 150 tests | S | low-medium (public names in api listings) | pre-0.4.0 for dead; 0.5 for public names |
| 10 | Exact copy-paste clusters (`__getattr__` ×3, `__check_init__` ×3, gains, oifits readers, grid_fit/limits wrappers) | about 120 src | S | low | any time |
| 11 | `start_from_positions` as a thin wrapper over alias machinery (needs `AliasResult.to_orbit`) | 150 to 250 src + about 150 tests | L | high | 0.5+ |
| 12 | Unused options (74 args) | about 250 src | M | medium | not recommended |

Sum of ranks 1-10 that are pre-0.4.0 feasible: about 3,900 lines (src about 2,700, tests about 1,200), roughly 5% of 85.8k total; the remaining about 1,500 are 0.5/0.6 shims and the PolarDisk re-routing. Growth was +41k, so no clean-up recovers more than about a tenth of it without removing features; the only feature-bearing item is rank 1.
