# Release review for 0.4.0 (main at `420d1b7`, 2026-10-16)

Status: this note is a record. Of "Revisions I recommend", revisions 1, 3 and 5 are applied in `design/` (`orbit_pipeline.md`, the status table of `imaging_plan.md`, the refreshed headers and `design/history/`); revision 2 (replace the Gl 229 gate) was not adopted, and the gate stays (see Decisions); revision 4 (a release checklist) is a CI and contributor-docs change.

This review covers the 522 commits since `v0.3.0` (2026-10-06). In ten days
`src/` grew from 25k to 48k lines and the tests from 59 to 89 files. Most of
that is new: orbit search (`epochs`, `aliases`, `orbit_search`), PolarDisk,
the pipelines, ensembles with GP members, SVI and bandwidth smearing.

**Method.** Seven Sonnet reviewers ran in parallel on a detached checkout of
main: core inference, models and imaging, orbits/detection/pipeline, docs,
CI, faithfulness to the Plan, and in-flight work. I checked every finding
used below as a blocker against the code myself, and read the public API,
the module growth and the live orbit plan by hand. No heavy JAX was run. The
detailed reports, each finding with file:line and evidence, are in
[`release_review_0.4.0/`](release_review_0.4.0/) (`review_core.md`,
`review_models.md`, `review_orbits.md`, `review_docs.md`, `review_ci.md`,
`review_plan.md`, `review_inflight.md`).

## Decisions (Ben, 2026-10-16)

- **The Gl 229 gate stays.** 0.4.0 waits for the unattended Gl 229 orbit
  search, which is close on the orbit branch. Orbit 1d (#372) is merged.
- **Stability tiers are adopted** (PR #373), with the Gl 229 gate also kept.
- **R2 is downgraded.** DISCO operators are orthonormal by construction and
  validated as whitening, so they never reach the `eigh` branch. The float32
  rank cut affects only a user operator over correlated closure phases from
  four or more telescopes. It is fixed as robustness in the correctness PR.
- **The new names are fixed before the tag.**
  - Core names: the core API-freeze PR.
  - PolarDisk names: the PolarDisk API-freeze PR. `PolarDisk` → `MultipoleDisk`,
    `PolarKernel` → `MultipoleKernel`, `PolarField` → `MultipoleField`.
  - Orbit names: the "Orbit 0.4.0 API fixes" PR after #372.
- **Main red (R1):** PR #374.
- **Code growth:** see
  [`release_review_0.4.0/size_overhead.md`](release_review_0.4.0/size_overhead.md).
- **`orbit_search` ships as provisional.** A removal was briefly planned
  (tag `archive/score-orbits` at 9717dcc remains), then reversed. The Gl 229
  mode-ranking scripts (`score_modes.py`, `refine_modes.py`,
  `nested_modes.py` in the Gl 229 analysis) use `score_orbits` and
  `SharedFlux`, and orbit search is a core feature. The R4 `_axis` fix stays
  in #375, and the model-before-data order goes in the orbit API
  fixes PR.

## Verdict (as first written)

**Not ready to tag today.** The numerical core is sound and the docs build
cleanly. But:

1. **Main is red.**
2. **Four silent-wrong-answer or crash bugs** are cheap to fix.
3. **The release criterion you set on 2026-10-08 is not met**: "0.4.0 waits for
   the full orbit search working (Gl 229 found unattended)".
4. **New public names would freeze in a state the in-flight orbit plan already
   intends to change.**

I recommend changing the criterion, not waiting for it. Tag 0.4.0 within a
few days as a release with explicit **stability tiers**: a stable core, and a
set of clearly provisional new features. That covers the orbit search,
PolarDisk, the pipelines, GP ensembles and SVI. `OrbitPipeline` and the
unattended Gl 229 search then become 0.4.x (provisional), as you suggested
for future features.

The one orbit change I would land **before** the tag is phase 1d (tempering).
Untempered, `fit_orbit_aliases` reports band probabilities of 1.000 at
χ²/N ≈ 12. That is a statistical-correctness defect in a function 0.4.0 ships,
not a missing feature.

## Release blockers

| # | Finding | Evidence (checked) | Effort |
|---|---|---|---|
| R1 | **Main CI is red.** x64 shard 1/3 has failed on three consecutive main pushes (`1ee23a1`, `76ee8d6`, `420d1b7`) in two `slow` tests of `tests/test_aliases.py`: the night-flag test gets `miss[5] = 0.067` and expects > 3; the skip-flagged test gets 3 nights where it expects 5. float32 passes. jaxoplanet turns on x64, so orbit users run exactly this path. Treat it as a possible real bug in night flagging under float64 until shown otherwise. It reached main because the PR gate skips `slow`. | `gh run view` on main | 2–4 h |
| R2 | **`OIData._diagonalized` keeps spurious directions in float32.** `eigh` runs in `jax.numpy` (float32 by default), with a rank cut at `1e-10 × max`, which is below float32 eps. For a rank-15 operator, 26–28 directions are kept, the extra ones with tiny σ and so huge weight. This is the DISCO, kernel-phase and projected-closure-phase path, where rank deficiency is normal. Fix: do the construction-time `eigh` in host `onp` float64, plus a float32 regression test. | `oidata.py:917-918`; reviewer reproduced 20/20 seeds | 1–2 h |
| R3 | **`PolarDisk.is_physical()` raises `AttributeError` with `VonMises` harmonics.** It reads `self.harmonics.prior`, which `VonMises` lacks, and so breaks `model_loglike(reject_unphysical=True)` and `System.is_physical`. | `models.py:4973` | 30 min |
| R4 | **`SharedFlux.grid=(0.01, 0.1, 1.0)` is read as `(lo, hi, n=1)`**, which silently gives a single flux. A 3-tuple must not be ambiguous: take `(lo, hi, n)` only as a named form, or require `n` to be an `int`. | `orbit_search.py:254` | 30 min |
| R5 | **Release hygiene.** `pyproject.toml` and `CITATION.cff` say 0.3.0. `CITATION.cff` has `doi: "TBD"` (invalid CFF) and no `date-released`. The CHANGELOG is still "Unreleased", misses #370 and #371, contradicts itself on ensemble `field_factors` ((1, 2, 4) vs the code's (0.75, 1, 1.5)), and wrongly claims Oxford spelling throughout. | `review_docs.md` D1–D4 | 2 h |
| R6 | **A broken promise.** The 0.3.0 CHANGELOG says `Tabulated` is removed in 0.4.0, but it still only warns. Either remove it, or amend the note to 0.5 (my preference, given the shims all go in 0.5). | `CHANGELOG.md:714-716`, `spectra.py:665` | 15 min |
| R7 | **`publish.yml` has no CI gate.** 10 of the last 19 main pushes were cancelled, so a tagged SHA may never have been tested. Add a check that `tests.yml` succeeded on the tagged commit. | `review_ci.md` CI-04 | 1 h |

## Freeze these names now, or mark them provisional

Every item below is new in 0.4.0. Changing it now is free; after the tag it
needs a deprecation cycle.

- **`score_orbits(epochs, model, …)` is data-first**, against the convention
  that 0.4.0's own migration enforces everywhere else. Its `scales`/`s_max`
  defaults and model signature (`model(orbit, flux)`) also differ from
  `rank_orbits` (`model(orbit)`).
- **`epochs.py` ships deprecations in its first release.** These are
  `EpochPositions.gap` ("unchanged for one release") and a FutureWarning that
  the `scales` default will flip from quoted to marginal. Choose marginal now,
  as `score_orbits` already does, and delete both.
- **`GaussianArc` emits a FutureWarning on every construction** about a
  shape change that has never been released. Delete the warning.
- **`likelihood_grid` keyword-only data and grid** is finished on branch
  `likelihood-grid-kwonly` (a415d24, no PR). Land it before the tag or drop
  it. Narrowing the signature in 0.4.1 would itself be a break.
- **Result files without a format version:** the `AliasResult` JSON (which
  will change with 1d) and `DetectionMC.meta`. Add a version field.
  Pipeline folders are fine (`virgil-pipeline-run-v1`, checked on load).
- **PolarDisk names.** `RadialGP` plays two roles, profile and function, and
  has unused `rule_x`/`rule_w`. Every radial profile carries a meaningless
  `weight`. `r0`/`sigma`/`width`/`fwhm`/`scale`/`diam`, `pa` vs
  `rotation_deg` and `ratio` vs `inc` are mixed. Either settle them in one PR
  now, or mark the whole PolarDisk family provisional. The second is
  realistic, since the profile parametrization is still an open decision in
  `radial_profiles.md` and validation against frank, vis-r and FRAP has not
  been done.
- **Keyword-only arguments** for `imaging.clean` (about 20 parameters),
  `starting_image` and the later arguments of `Harmonics.__init__`.
- **`inference` naming.** `fisher` (the observed Hessian) and
  `gaussian_fisher` (the expected Fisher) are confusable, and so are
  `laplace_cov` and `laplace_covariance`, which do different things.
  `laplace_parameter_uncertainty` has no `dtype` and runs in float32, unlike
  its siblings. Rename or alias them now.
- **`OIData.observable_kind`** collides with a documented record key of the
  same name that the code ignores (`oidata.py:162` vs `:421`). Fix the
  docstring and rename the attribute.
- **Pipelines.** `BinaryPipeline` and `StarPipeline` declare
  `STABILITY = "stable"` in their first release. `design/pipeline.md` itself
  says to settle `global_nsigma` before the first stable release, and
  `model_from_spec` raises on any change to a dataclass's fields, which
  contradicts the "change only by addition" contract. **Recommendation:** ship
  them as `provisional` in 0.4.0 and promote them in 0.5 after one release of
  use. The alternative is to settle both points now.

## Should fix before the tag (not blocking)

- **Bandwidth smearing** (`with_bandwidth`) raises a `TypeError` for
  `BinaryModelCartesian`/`Angular`, `Rotated`, `OrbitalBinary` and the
  documented `System(star, bg=Resolved())`, so the flagship binary models
  cannot be fitted with smearing. Support them (`to_system()`, a zero offset,
  rotated u, v), or document the limitation in the error and the docs.
- **`Precision.HIGHEST` gaps** that only bite in float32 on A100/H100:
  - the limb-darkening coefficient matmul (`models.py:1323`, binomials up to
    C(22, 11) cancelling under TF32);
  - the RadialGP/PolarKernel contractions (`models.py:3597, 3620, 3820`);
  - the DISCO operator (`oidata.py:1240`).

  Each is a one-line change.
- **The t_ref warning** (`orbits.py:135-155`) doesn't say it becomes an
  error in 0.5, which was your 2026-10-06 decision.
- **CI:**
  - run the `slow` tests on PRs that touch `aliases`, `epochs`, `orbit_search`
    or `orbits` (this would have caught R1);
  - open an issue when a main push goes red (today only the nightly does);
  - install `.[test,orbits]` in the lowest-dependencies job, which now skips
    the whole orbit stack and so the jaxoplanet floor too (1885 passed vs
    2067).
- **Docs:**
  - a five-line quickstart and a "where to start" block on `README.md` and
    `docs/index.md`;
  - a short "Which orbit tool?" page (`rank_orbits`/`start_from_positions`
    vs `score_orbits` vs `fit_orbit_aliases`), because a user cannot tell
    today;
  - a provisional badge on every page from the tier list below;
  - extras notes: `[orbits]`, `[plots]` and `[pipeline]`;
  - the dead zodiax link (`binary_search.md:213`);
  - the bogus table row in the `aliases` API page (`aliases.py:1480`);
  - the Drevon et al. year (2026);
  - pyproject URLs and classifiers.
- **Public docstrings** cite internal plan problem numbers and specific
  targets (Gl 229 nights, AL Dor, ι Peg), against the "design docs are
  general" rule. Mostly in `aliases.py`/`epochs.py`.

## Maintainability and technical debt

The core is in good shape. `_linear`, `_closure`, `_flat`, `priors`,
`angles`, `_precision`, `fields`, `scenes`, `simulate` and the orbit
conversions (300 random round trips) all read cleanly and showed no defects.
No public function has data before model except `score_orbits`. Docs
snippets use no deprecated calls.

The debt is in **size and overlap, accumulated at speed**:

1. **`models.py` is 6.6k lines with 48 classes; it doubled in the last three
   days** with PolarDisk P1–P6. `imaging.py` is 3.3k. AGENTS.md's rule "New
   model code goes in `models.py`" caused this. For 0.4.1, do one mechanical PR
   that turns `models` into a package (`_base`, `components`, `polar/`,
   `scene`, `cvis`) re-exported from `models/__init__.py`. Do the same for
   `imaging/`. Add an I/O round-trip test for pickled `__module__` paths, and
   change the AGENTS.md rule. It doesn't need to block 0.4.0, because import
   paths stay the same.
2. **Three orbit entry points with incompatible result types.**
   - `epochs` gives `OrbitStart`;
   - `orbit_search` gives `OrbitScores`, and nothing else in the package uses
     it;
   - `aliases` gives `AliasResult`, whose `.best` is a plain dict and whose
     `.scene()` returns the private `_UDBinary`.

   They are joined by lazy-import cycles: `orbit_search` and `aliases` import
   private names from `epochs`, and `epochs` imports from `orbit_search`
   lazily. The 2026-10-15 orbit plan already resolves this by growing
   `OrbitPipeline` out of `aliases.py`. Meanwhile, move the shared marginal
   algebra to a private `_marginal.py` and add the bridge
   `AliasResult` → `KeplerOrbit`/`System`.
3. **Several very long functions** need splitting along their stages, ideally
   while 1d and 1e touch them anyway:
   - `fit_orbit_aliases` (511 lines);
   - `epoch_positions` (360);
   - `OIData.__init__` (310, up from 210);
   - `start_from_positions` (263);
   - `score_orbits` (267);
   - `svi.variational` (246).
4. **Velocity versus review.** 522 commits in ten days, with stacked
   cumulative branches merging other branches repeatedly. R1 shows that the PR
   gate no longer covers what lands. Stability tiers are the cheapest guard:
   provisional code can move fast without each change becoming a deprecation.
5. **Carry-over from 0.3.0:**
   - quadratic Python loops in `gain_modes`, `ClosureNoise.from_indices`,
     `DifferentialPhase.build` and `_BaselineLookup.times`;
   - static fields that recompile on a value change (`OIData.t_ref`, the
     `FluxSpectrum` widths, `DifferentialPhase.prior_width`).

## Performance

**Ben, 2026-10-16:** fix all of these now. Assignments, split by file:
- orbit work: B3 recompiles, B4 `starting_orbits` batching and B5
  `epoch_positions` batch default (epochs, orbits, orbit_search);
- the MultipoleDisk PR (#378): `freeze_geometry`;
- perf-builders: P1 builder loops and P2 static fields (gains, `_closure`,
  observables, oifits, oidata);
- perf-fitting: P3 dense covariances and P4 the `_lm` chunk (fitting, svi);
- perf-imaging: #328 evidence SVD, `dirty_image` and `metrics.align`
  (imaging, metrics).

As first written:

- **`score_orbits` recompiles on every call**: 2.7–5.8 s even for 5
  candidates. The closure-jit pattern also appears in `epochs._loglikes`. Use
  a module-level jit with a static `_Surface`.
- **`starting_orbits`** is one unbatched `vmap` over periods × 19 e × 36
  phases: 2.4 GB at 410k points. `fit_orbit_aliases` calls it per band, which
  keeps the default path smaller, but a direct call on a long baseline can
  OOM. Use `lax.map` with a batch size.
- **`freeze_geometry`** uses `lax.cond`, which becomes `select` under `vmap`
  and so loses its speed-up.
- **`dirty_image`** builds dense complex (k, npix) matrices.
- **`metrics.align`** makes O(n²) un-jitted NCC calls.
- **Issue #328:** `log_evidence` runs its SVD in NumPy on the CPU, which takes
  hours on MATISSE-sized data.

## CI

CI is in much better shape than a week ago:

- the PR gate takes a median 11.6 min (it was 20–23);
- every failure was deterministic, so I found no flaky tests;
- the Claude review succeeded 9 times in 9 and leaves useful comments;
- trusted publishing worked for 0.2.0 and 0.3.0.

Beyond R1, R7 and the three CI items above, these can wait for 0.4.x (see
`review_ci.md`):

- **py3.13:** an unsharded 23–31 min job, the longest on every push.
- **Per-test budget:** it disables itself when a PR adds a test file.
- **Cache:** 10.3 GB against a 10 GB quota.
- **Untested paths:** the nightly, canary and `full-ci` paths have never run,
  and their labels don't exist yet.
- **Dtype and sampler test PRs:** these follow-ups from the CI brief have not
  landed.

## Faithfulness to the Plan, and revising it

**Implemented as planned, and checked against the code:**

- the standing decisions of `imaging_plan.md`: one residual vector, `fit`
  and `numpyro_model` sharing one specification, float64 inside `fit`, the
  NUFFT removed, Jeffreys defaults with the no-data prior test, and opt-in
  flat coordinates;
- the argument-order migration;
- the detection ROC stages;
- PolarDisk P1–P6;
- the Gauss–Newton GN1, GN2 and GN4.

**Diverged, where the Plan is stale rather than the code:**

- the `[sampling]` extra and blackjax are gone;
- the default field of view now comes from the data (`image_field`);
- LM does materialize a Jacobian, with QR for few coordinates;
- `sampler_flat_coordinates.md` still says "default" in its body.

**Diverged, where the code should follow the Plan:** the `Precision.HIGHEST`
gaps above, and `radial_profiles.md` §5, which promised deprecations for
subsumed classes. Following your FlaredDisk rule (never deprecate classes
that are not special cases), amend §5 rather than deprecate `TruncatedCone`
or `ModulatedGaussianRim`.

**Revisions I recommend:**

1. **The live orbit plan is not in the repo.** It exists only as
   the unmerged plan of 2026-10-15 (now `design/orbit_pipeline.md`). Commit
   it now as `design/orbit_pipeline.md`, and mark these as superseded in
   their headers: `automatic_orbits.md` (whose header still says "Nothing
   here is implemented"), `orbit_aliases.md`, `thiele_innes_marginalisation.md`
   and `start_from_positions`. Today an agent reading `design/` would follow
   the dead D2→G route.
2. **Replace the Gl 229 gate on 0.4.0** with the tier policy below. Move the
   gate to the release that promotes `OrbitPipeline` out of provisional.
3. **Collapse `imaging_plan.md`'s orchestration, "order of work" and Stage 2
   sections into a status table**, and fix the five stale statements above.
   It should read as a current plan, not a log.
4. **Add a release checklist.** Branch `release-evidence-gate` already has
   one: tier A of virgil-validation passing, evidence pinned to one commit,
   no open `bug` findings, and a mutation kill rate. Add to it: main green on
   the tagged SHA, the version strings, the CHANGELOG heading, and the tier
   list.
5. **Update or archive stale headers.**
   - Update `visibility_orbits.md`, `pipeline.md`, `detection_roc.md`,
     `gauss_newton_plan.md`, `image_reconstruction.md` and
     `sparse_imaging.md`.
   - Archive the two `codebase_review_2026-10*.md` files and
     `orbit_fitting_review.md` under `design/history/`.

## Stability tiers proposed for 0.4.0

- **Stable** (deprecation cycle before any change): `OIData` and its readers
  and writers; the core components and `System`; `likelihood`; `fit`;
  `grid_fit`; `limits`; `detection`; `inference` (after the renames);
  `priors`; `angles`; `spectra`; `observables`; `gains`; `orbits` (the
  Keplerian classes); `simulate`.
- **Provisional** (may change in any 0.4.x, which the CHANGELOG says and each
  page shows with a badge):
  - `epochs`, `aliases` and `orbit_search`;
  - the PolarDisk family (`PolarDisk`, the radial profiles, `RadialGP`,
    `PolarKernel`, `Harmonics`, `VonMises`, the heights);
  - `ensemble`'s GP members and `evidence_grid`;
  - `svi`;
  - `virgil.pipeline` (Binary, Star and the coming Disk, Image and Orbit
    pipelines);
  - `imaging.clean`;
  - `metrics`.

A single `virgil._stability.provisional` marker (a docstring admonition, plus
a `__provisional__` set in each module) would make this checkable by a test.
It should not warn at runtime.

## In flight: what this release must not block

These are safe as long as the tiers above are adopted:

- **#369 DiskPipeline** is already provisional.
- **Orbit phase 1d (tempering)** is uncommitted in a local worktree, and has no tests yet. Land it before the tag
  (see Verdict).
- **Orbit 1e/1f (`OrbitPipeline`)** needs a change to the `_core.py` base
  class for multi-night inputs. That stays possible only if the pipelines are
  provisional.
- **`ImagePipeline`** is #370/#371 territory. Its name differs from the
  design note's `ImagingPipeline`; settle the name before it lands.
- **`ensemble-window`** (WIP, no PR) changes `EnsembleSpec` defaults.
- **GN3, GN5 and GN6** are additive.
- **Issues #309** (StarPipeline `multimodal`), **#297**, **#268**, **#328**
  and **#329** are additive or internal.
- **Issue #39** (notebooks to a separate repo) should wait until after the tag.

## Proposed order to the tag

1. Make main green (R1).
2. One correctness PR: R2, R3, R4, the HIGHEST gaps and the smearing
   message.
3. One API-freeze PR: `score_orbits` model-first; drop the `epochs`
   deprecations and the `GaussianArc` warning; versioned `AliasResult` JSON;
   the `inference` renames; keyword-only `clean`, `starting_image` and
   `Harmonics`; `observable_kind`. Land `likelihood-grid-kwonly` alongside.
4. Orbit 1d (tempering) through its own pr-worker.
5. Tiers: the marker, the badges, the pipeline `STABILITY`, and a CHANGELOG
   "Stability" section.
6. CI: the publish gate, `slow` on orbit PRs, the red-main issue, and lowest
   deps with `[orbits]`.
7. Docs: the quickstart, "Which orbit tool?", the extras notes and the fixes.
8. **Every docs example run on an OzSTAR GPU under the profiler** (Ben,
   2026-10-16), using the `jax-profiling` skill. There are about 200 python
   blocks on 27 pages.
   - **Extract.** A deterministic extractor turns each docs page into one
     `.py` script, keeping its python blocks in order. Each block is wrapped
     in `jax.profiler.StepTraceAnnotation(page, step_num=i)` and ends with
     `block_until_ready`. Print statements report each block's key results
     (fitted values, χ²/N, shapes) next to the value the page quotes, if it
     quotes one. Pages that need external data are listed and staged from
     `$MYFRED`, never silently skipped.
   - **Run.** A Slurm array job runs one page per task on one A100, at the
     default `highest` precision, with `env.txt` and the `nvidia-smi`
     telemetry from the skill. Each page runs twice in a fresh namespace:
     once cold, untraced, for wall time and compile time and count (a
     `jax.monitoring` listener); then once traced, for steady state.
   - **Summarise in the job.** `jax_trace_summary.py --json` runs on the
     node. Only the small JSON summaries, the stdout and the telemetry CSV
     come back to the Mac. A pure-Python aggregator (no JAX, no trace
     reading) turns them into one markdown table: page, pass/fail, printed
     values against the quoted ones, cold wall time, compile time, compile
     count, steady time per block, GPU busy fraction and the top 3 ops. That
     table is all the coordinator reads; raw traces never go through an LLM.
   - **Before and after.** Take a baseline on main before the performance
     PRs merge and a final run on the release candidate, compared with
     `--baseline`. A page that fails, disagrees with its quoted numbers, or
     takes more than about 5 min, blocks the tag until it is fixed or the
     page is changed.
   - **Where.** The tooling lives in a separate scripts repository, as the
     `docs_examples_prof` job. Ben submits and pulls the results; agents only
     read the downloaded JSON.
9. The release PR:
   - bump the version (`pyproject`, `uv lock`, `CITATION.cff` with its date
     and DOI fields);
   - date the CHANGELOG `## 0.4.0` and restructure it into highlights,
     migration, added, changed, fixed and provisional;
   - label it `full-ci`;
   - after the merge, confirm the push run on the merge commit was not
     cancelled, then tag.

Steps 2, 3, 6 and 7 are independent and can run in parallel; step 3 touches
the files 1d touches, so it goes after step 4 or on the same worker. I
estimate two to three days of agent work, plus your decisions on the tiers
and the pipeline stability.
