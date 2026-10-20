# Pipeline parsimony review: measurement and plan

Status (2026-10-19): **phase A done** (disk, star, binary and `_common`);
**phase B done** (`_core`, `_checks`, `_io`, `_quicklook`, `cli`, `orbit.py`; see below).
Measured on the tree of PR #422 (`origin/pr422`, merged with `main` at
4dafa23), because `orbit.py` is part of what is being measured; re-measure
after `git merge origin/main`.

The question (Ben, 2026-10-18): are the pipelines parsimonious? Short
answer: **mostly yes.** Literal copy-paste is low; the savings are
structural, about 300 lines of about 11,200 (under 3 per cent). `disk.py` is
big because it is two science workflows (`rings` and `gp`) with their own
profile, compare and checks, not because of scaffolding: perhaps 70 of its
2,905 lines are generic. Nothing here changes the public API, the summary
JSON keys, the check names or order, or the run-folder layout.

## Phase A outcome (disk, star, binary, `_common`)

Net `src/virgil/pipeline` against `main` 4dafa23: **-90 lines** (409 added,
499 removed), tests untouched in behaviour. Per module (lines, before to after):

| Module | Before | After |
| --- | ---: | ---: |
| `disk.py` | 2905 | 2809 |
| `star.py` | 1441 | 1367 |
| `binary.py` | 1040 | 952 |
| `_common.py` | 516 | 684 |

What landed: shared validation (`_positive`, `_integer`, `_pair`,
`_validate_common`), shared posterior scaffolding (`_draws`, `_chain_health`,
`_sampler_report`, `_site_summary`, `_prior_bounds`, `_load_start`,
`_write_samples`, `_corner_trace`, `_sampler_checks`, `_save_fit`, `_pick`,
`_save_map_correlation`), one `grids.h5` writer in binary, and in disk the
removal of the stub `_DiskFit`, the double model loads in `compare`,
unused parameters and defensive code that could not trigger, and repeated
reshape/argmax in `_geometry_stage`. Items 8 and 10 of the plan were skipped
(`_DEFAULTS` blocks and `Stage(...)` tuples read better spelled out), D2
(`DiskModel.disk/.candidate`) is left to Ben, and the `star._freq_max`
rewrite was dropped because it is not bit-identical (rounding).

Honest conclusion: duplication was low, so the saving is small. The helpers
cost about 160 lines in `_common`; the pipelines shed about 250. The value is
that the sampler checks, validation messages and posterior files now come
from one place, and from the tests added in this pass.

Verified by golden comparison (summary, stage reports and check lists byte for
byte on the binary, star, disk-gp and disk-rings fixtures, ignoring timings,
provenance and paths) after each commit, plus an HDF5 structure and PNG-hash
comparison for the binary `grids.h5` change. The comparison showed that the
fit report stores `repr(prior)`, which contains a memory address: a latent
non-reproducibility to fix with a stable `repr`.

### Findings for follow-up

- **Disk, `error_scale="fit"`: the "fitted" `error_scale` check can never
  fire.** `compare["error_scales"]` is `None` for rings under `"fit"` (the
  compare stage fits with quoted errors and has no fitted scales), so the
  `kind="fitted"` branch in `_chi2_checks` is unreachable. Fixing it needs a
  decision on where the fitted scales come from; left as is.
- **Instrument-specific code:** none in `disk.py`, `star.py`, `binary.py` or
  `_common.py`. `wavel_range` and `error_floor` are generic `OIData`
  operations. One docstring remark names PIONIER (`disk.py:252`, "PIONIER-like
  errors are often conservative"); it describes a typical case, no result
  depends on it. Shared modules are audited in phase B.
- `repr(prior)` memory address in the fit report (above).

## Phase B outcome (`_core`, `_checks`, `_io`, `_quicklook`, `cli`, `orbit.py`)

Lines, before to after (against `main` 9ce6578):

| Module | Before | After |
| --- | ---: | ---: |
| `orbit.py` | 1456 | 1411 |
| `_checks.py` | 1692 | 1732 |
| `_common.py` | 707 | 728 |
| `_core.py` | 533 | 538 |
| `binary.py`, `star.py` | 2327 | 2329 |
| `_io.py`, `_quicklook.py`, `cli.py`, `disk.py` | unchanged | unchanged |
| **Total** | 11275 | 11298 (+23) |

`orbit.py` moved onto the shared validation, `_draws`, `_chain_health`,
`_site_summary`, `_sampler_report`, `_sampler_checks`, `_write_samples`,
`_corner_trace` and `_pick` (-45 lines, and its band summaries are now NaN
aware). `at_bound` and `start_at_bound` merged into one function. The net is
positive because the robustness fixes are code: NaN guards in six checks,
`ring_count` and `inclination` edge cases, the empty-list error and a stable
prior description (`_prior_repr`). `_io.py`, `_quicklook.py` and `cli.py` have
no repeated work and no unused function, so they are unchanged. A shared
"value against two thresholds" helper (plan item 7) was tried and dropped: the
ladders differ in meaning text and non-finite message, so the helper was no
shorter than the code it replaced.

## 1. Measurement

Lines per module (`src/virgil/pipeline`, PR #422 tree), functions counting
nested ones, and the largest functions (lines):

| Module | Lines | Funcs | Largest functions |
| --- | ---: | ---: | --- |
| `disk.py` | 2905 | 105 | `_compare_rings` 95, `_geometry_stage` 88, `plot_summary` 85, `_profile_rings` 72, `_profile_gp` 67 |
| `_checks.py` | 1616 | 47 | `length_scale` 96, `ring_count` 84, `inclination` 55, `resolution_regime` 54 |
| `star.py` | 1441 | 54 | `_posterior` 68, `_find_lobes` 50, `_fit` 47, `plot_v2_models` 46, `_nuts` 46 |
| `orbit.py` (#422) | 1414 | 56 | `_validate` 81, `_summarize` 79, `_plot_sky` 56 |
| `binary.py` | 1040 | 50 | `_stages` 43, `_posterior` 43, `_validate` 34 |
| `_io.py` | 830 | 49 | `model_spec` 59, `_baseline_layout` 40 |
| `_quicklook.py` | 601 | 23 | `_orbit_cells` 70, `execute_notebook` 51 |
| `_core.py` | 533 | 36 | `run` 43, `_run_stage` 34 |
| `_common.py` | 516 | 35 | `_interior_start` 35, `_run_chains` 33 |
| `cli.py` | 256 | 11 | `build_parser` 53 |
| `__init__.py` | 64 | 2 | |
| **Total** | **11216** | | tests: about 4,900 lines (`tests/test_pipeline_*.py`, orbit included) |

Of 229 distinct function names in the four pipeline modules only 16 occur in
more than one, and those are the stage entry points (`_fit`, `_posterior`,
`_summarize`, `_search`) plus the helpers listed in section 3. #419 already
moved `_quicklook`, `_overview`, `_scales_of`, `_chi2_64` and
`_m2loglike_64` into `_common`.

### What a stage does that another pipeline's stage also does

| Repeated work | Where | Lines each | Now |
| --- | --- | ---: | --- |
| `_validate` tail: `num_samples`/`num_chains` ≥ 1, `num_warmup` ≥ 0, `chain_method`, `wavel_range`, `error_floor` (+ `batch_size`, `error_scale` in two) | binary, star, disk, orbit | 12-20 | 4 copies, identical text |
| `_validate` integer/positive/pair helpers | binary and orbit define local closures; star and disk spell each test out | 25-60 | 2 styles |
| Posterior: draws to NumPy, `_sample_stats`, per-site `_diagnostics`, the `r_hat_max`/`ess_bulk_min`/`divergence_fraction`/`num_*` block | binary, star, disk, orbit | 10-14 | 4 copies |
| Posterior: per-parameter `{quantiles, (std), diagnostics}` summary | binary, star, disk (orbit inside `_band_summary`) | 8-12 | 3 copies; `std` differs (binary has none) |
| Posterior: `write_h5` with `schema` and `sampler` attributes | binary, star, disk, orbit | 8-12 | 4 copies |
| Posterior: corner and trace figure (`plot_chainconsumer_diagnostics`, `_save` x2) | binary `_plot_posterior`, star inline, orbit `_plot_posterior` | 11-14 | 3 copies |
| Posterior: start values from a saved model (`load_model_values`, `if k in priors`) | binary, star (disk, orbit build their own) | 5-6 | 2 copies |
| `_posterior_checks`: `prior_bound`, `r_hat`, `ess`, `divergences` | binary, star (+`walls`), disk, orbit (inline) | 6-19 | 4 copies |
| `_save_fit`: `save_model` then `write_json(info.json)` | star, disk (binary inline) | 3-4 | 3 copies |
| `_pick(report, KEYS)` | binary and orbit helper; star and disk comprehensions | 1-2 | 2 spellings |
| `_summarize` skeleton: unpack `reports.get(k)` for six stage names, `if load: sections["data"] = pick(...)` | all four | 6-10 | 4 copies |
| `_stages`: `load`, `overview` and `quicklook` `Stage(...)` tuples | binary, star, disk (orbit similar) | 7 | 3-4 copies |
| `_DEFAULTS`: the sampler and data blocks (`num_warmup`... `seed`, `wavel_range`, `error_floor`, `batch_size`) | all four | 8-10 | 4 copies |
| `_checks`: `at_bound` and `start_at_bound` | star, binary, disk | 20 + 13 | two checks, one idea |
| `_checks`: the "not computable → one warn `Check`" opening | inner_edge, outer_edge, render_consistency, structure_unconstrained, chi2_reduced, r_hat, ess, ... | 5-7 | 8-10 copies |
| `_checks`: a value against `warn`/`fail` thresholds into a status and a meaning | chi2_reduced, error_scale, r_hat, ess, divergences | 10-14 | 5 copies |

Not repeated (leave alone): the scans (`_search`, `_scan_new_ring`,
`_geometry_grid`), the model registries' content, `_fit_report`s, and the
per-check message text, which differ in substance.

## 2. Why `disk.py` is 2,905 lines

By region (lines, approximate):

| Region | Lines | Disk-specific? |
| --- | ---: | --- |
| Registry (`DiskModel`, `CHECKS`, `models()`), class docstring and settings | 505 | docstring (about 165) and `_DEFAULTS` are the public documentation; `DiskModel.disk/.candidate/.checks` and `CHECKS` are dead (review D2, about 30 lines) |
| Context, shapes, scenes, priors, `_fit_rings`, `_fold_fit` | 430 | yes |
| `geometry` stage and the point-source search | 265 | yes |
| `profile`: rings (about 350) and GP (about 385) | 735 | yes: two algorithms |
| `compare`: rings (about 190) and GP (about 65), inner edge, stability, render | 370 | yes |
| Plots (`plot_summary`, `_profile_curves`) | 140 | yes, but `plot_summary` is the one figure of a disk run |
| `posterior`: `_fold_pa`, `_posterior_setup` | 165 | `_fold_pa` (31) and `_posterior_setup` (50) yes; `_nuts` and `_posterior` (about 85) are scaffolding |
| Summary and checks (`_summarize`, `_disk_section`, `_fit_checks`...) | 220 | mostly yes; the `_posterior_checks` and `_pick`-shaped parts are scaffolding |

So scaffolding that belongs in `_core`/`_common`/`_io`/`_checks` is about
70 lines of `disk.py` (the validation tail 15, the posterior wrapper 50, the
checks wrapper 7, `_save_fit` 3), plus about 30 dead lines in the registry.
**The module is large because a ring-count search with back-fitting and a
length-scale-grid Gaussian process are each a pipeline's worth of science.**
Moving that into a `disk/` subpackage (`rings.py`, `gp.py`, `checks.py`)
would help navigation, not line count; it is a separate decision for Ben
and not part of this PR.

## 3. Plan

One PR, no behaviour change. Estimates are net lines of `src/virgil/pipeline`
(after adding the helpers, with their docstrings). Order is by value per
risk; each item is a separate commit so a reviewer can drop one.

| # | Change | Est. net | Notes |
| --- | --- | ---: | --- |
| 1 | **Shared validation.** `_core` gets `_positive`, `_integer(name, minimum)`, `_pair(name, low=0)` and `_validate_common(s, *, batch_size=..., error_scale=...)` for the sampler and data settings; the four `_validate`s call them and keep only their own settings. | -85 | Messages are pinned only by substring (`match="error_scale"`, `"max_grid"`, `"n_r"`, ...). Keep each message's wording; where `binary` says "must be positive; got X" and `star` says "must be positive", keep one and check the tests. |
| 2 | **Shared posterior summary** in `_common`: `_draws(mcmc)` (get_samples to NumPy plus `_sample_stats`), `_param_summary(samples, names, diagnostics, *, std, angle, period)`, `_sampler_report(settings, parts)` for the `num_*`, `r_hat_max`, `ess_bulk_min`, `divergence_fraction` block (from per-model summaries or from one fit's diagnostics). | -55 | The deferred "shared `_posterior` summary (40-60 lines)". `binary`'s summary has no `std` and disk's skips `_vec` sites: keep as arguments. Key order of the JSON must not change (snapshot tests compare key sets, so also compare key order in the proof below). |
| 3 | **`_write_samples(p, groups, **attrs)`** (writes `schema` and `sampler`) and **`_plot_corner_trace(output, samples, columns)`** in `_common`; `_load_start(folder, priors)`. | -40 | `orbit._empty_plot` stays in `orbit.py`. |
| 4 | **`_checks.posterior(post, *, bounds=None, walls=None)`** returning the four sampler checks in order; four call sites. | -20 | star passes `bounds` with `{model}.{param}` prefixes and the worst model's wall probe. |
| 5 | **`_io.save_model(..., info=)`** replacing the three `_save_fit`s (the deferred `_save_fit`); `_pick` to `_common` and use it in star and disk. | -15 | `info.json` is written by the same `write_json`; `models/<name>/info.json` stays in the stage outputs. |
| 6 | **Merge `at_bound` and `start_at_bound`** into one `_checks.at_bound(names, *, start=False)`: `start=True` gives the name `"start_at_bound"` and the NUTS-start message, and (as now) no pass branch. | -8 | Two check names and two messages stay, so the summaries do not change. If the merged body reads worse than two short functions, keep two and share only the `Check` construction. |
| 7 | **Checks: shared `_unavailable(name, value, threshold, what, status="warn")`** for the "could not be computed" opening and **`_graded(...)`** for the value-against-two-thresholds ladder, used where the code shrinks (`chi2_reduced`, `error_scale`, `r_hat`, `ess`). | -30 | Time-box: if `_graded` needs more parameters than the code it replaces has branches, stop at `_unavailable` only (about -15). Check names, values, thresholds and messages are pinned by `test_pipeline_checks.py`. |
| 8 | **Shared `_DEFAULTS` blocks:** `_core._SAMPLER_DEFAULTS` and `_DATA_DEFAULTS` merged with `{**...}` in each class; `_Pipeline._unpack(reports, *names)` for the `reports.get` unpacking in `_summarize`. | -25 | The dict order only matters for `defaults()`; keep the pipeline's own keys first and compare `defaults()` with the snapshot. Not changing the docstrings. |
| 9 | **Carry-over dead code from the 0.4.0 review**, if not yet done: `DiskModel.disk/.candidate/.checks` and `CHECKS` (D2), `_compare_rings` computing the Laplace covariance twice (P4, also halves its Hessian cost), `star._freq_max` (S3), `_profile_curves` identical branches (D3), `_chi2_checks(settings)` unused argument and `_RIVAL_DELTA_CHI2` alias (D4), `from_oifits`' `__new__` (S2). Re-grep on the merged `main`; some may already be gone. | -60 | P4 changes no result (same covariance); say so in the commit and test the log-determinant unchanged. |
| 10 | **Stage scaffolding.** `_stages` repeats three `Stage(...)` tuples (`load`, `overview`, `quicklook`) per pipeline. A `_common.standard_stage(name)` would save about 15 lines but hide the stage list from the reader of `_stages`. | -15 | **Decide at implementation time; default is to skip.** A pipeline's `_stages` is the thing a reader opens first. |
| | **Total** | **about -350** (-300 without 9 and 10) | |

Expected result after the PR: `src/virgil/pipeline` from 11,216 to about
10,900 lines, `disk.py` about 2,800, `star.py` about 1,320, `binary.py` about
940, `orbit.py` about 1,320, `_common.py` about 600, `_checks.py` about 1,560
(the shared code moves into `_common`, `_core`, `_checks` and `_io`). The
report in the PR gives the exact before and after per module.

### Not planned

- **`StarModel` parent-chain generality** (review S3, about 50-80 lines):
  the registry is a documented extension point, and the review said to ask
  Ben. Out of scope.
- **Performance** (P1 `_builder` cache, P2 `_scan_new_ring` partial, P3
  binary `_search_maps`, tempering dead helpers D1): not parsimony of
  pipeline code and they change what is compiled or removed from the public
  `tempering` module; separate PRs.
- **Splitting `disk.py` into a subpackage:** navigation, not parsimony; Ben's
  call.
- **`_io.py`, `_quicklook.py`, `cli.py`:** the review found them clean and
  the measurement shows no repeated work worth a helper (`_quicklook`'s cell
  builders differ by pipeline in content).
- **A base class for stages or posteriors** (a `_BayesianStage` mixin): the
  four posteriors differ in what they sample (one model; several models;
  flat coordinates with an `_fold_pa`; per-alias bands). Shared functions
  with explicit arguments are easier to follow than a hook-laden class.

## 4. Proof of no behaviour change

1. **Existing tests.** `tests/test_pipeline_{binary,star,disk,orbit,checks,io,start_at_bound}.py`
   pin signatures, defaults, summary key sets, check order and names.
   Run the files locally with `OMP_NUM_THREADS=2`, one at a time, each file
   inside the 30 s per-test budget outside the slow tier; the slow tier and
   anything heavy go to CI (`full-ci` label).
2. **Golden comparison, because key-set snapshots do not pin values.** Before
   the first commit, run each pipeline on its test fixture on `main` and save
   `summary.json`, `stages/*/report.json` and the check list to the scratchpad
   (a short script that reuses the fixtures of the existing tests, so no new
   compute beyond a normal test run). After each commit compare them
   **byte for byte, ignoring timings and the provenance block**. A difference
   fails the commit. Binary and star fixtures are cheap; the disk and orbit
   fixtures only if their tests are already quick locally, else the CI run of
   the same script on the PR (a one-off job in the PR, not committed).
3. **Public surface.** `inspect.signature` of each pipeline class and
   `defaults()` against the snapshot; `virgil.pipeline.__all__` unchanged;
   `docs/api/pipeline.md` unchanged.
4. **Error-message substrings.** Grep `match=` in `tests/test_pipeline_*.py`
   (the list is in section 3, item 1) and run `pytest -k "validate or raises"`
   after item 1.
5. **Adversarial Sonnet review** of the final diff before asking for merge,
   told to look for changed key order, changed check messages, and a helper
   whose parameters outgrow the code it replaced.

## 5. Order of work

1. After v0.4.0 and #422: `git merge origin/main`; re-run the measurement
   (the script is a 20-line `ast` walk; keep it in the scratchpad, not the
   repo); re-grep item 9 candidates.
2. Golden outputs on the merged `main` before any edit.
3. Items 1, 2+3, 4, 5, 6, 7, 8, 9 as separate commits, each followed by the
   targeted tests and the golden comparison; item 10 only if it reads better.
4. Before/after line counts per module in the PR description.
