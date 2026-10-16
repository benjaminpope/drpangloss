# Reducing virgil's growth overhead (2026-10-16)

Ben asked how to reduce the overhead of `src/` doubling in the ten days
since 0.3.0. The evidence is in [`size_sweep.md`](size_sweep.md), a static
analysis of main at `9717dcc` against `v0.3.0`.

## Diagnosis: the growth is features, not bloat

| | v0.3.0 | main | ratio |
|---|---|---|---|
| src code lines | 13.2k | 26.6k | 2.0× |
| src docstring lines | 9.2k | 16.6k | 1.8× |
| src comments + blanks | 2.7k | 5.2k | 1.9× |
| tests | 19.8k | 37.3k | 1.9× |

The mix has not changed. Docstrings were about 35% of `src` at 0.3.0 and are
still about 35%. Exact copy-paste is about 180 lines, and dead code is about
60. Net growth by feature (src / tests):

1. orbit tools: +7.7k / +4.9k;
2. pipelines: +5.1k / +1.8k;
3. PolarDisk, harmonics and the polar field (now `MultipoleDisk`): +4.2k / +4.1k;
4. ensembles, metrics and imaging: +2.4k / +1.4k;
5. the pre-0.3.0 review fixes, the Gauss–Newton kernel, flat coordinates,
   smearing and the shims: the remaining ~4k.

**So refactoring cannot fix it.** All the deduplication and dead code put
together is under 10% of `src`. The overhead comes from three things:

- **what gets built:** code written for a plan stage before anything needed
  it;
- **what never gets removed:** paths kept after something supersedes them;
- **where the explanation lives:** design history and derivations in
  docstrings.

I first flagged `orbit_search.score_orbits` (1,148 `src` lines and about 640
of tests) as unused, but that search missed the Gl 229 analysis scripts,
which rank alias modes with it. So it stays. The lesson for rule 2: look in
the analysis folders, not only the repos, before calling code unused.

## One-off cuts before 0.4.0 (about 3.9k lines, 5%)

| Cut | Lines (src + tests) | Needs |
|---|---|---|
| ~~Hold `score_orbits`/`orbit_search` back from 0.4.0~~ Rejected: the Gl 229 mode-ranking scripts use it; it ships provisional | (~1,790) | none |
| Docstring diet on the 20 longest docstrings, moving derivations and history to docs pages or design notes and leaving a summary and a link: `fit_orbit_aliases` (283 lines), `gauss_newton_mass` (151), `error_scale` (144), `numpyro_model` (~197), … Share the repeated `noise`/`priors`/`flat_coordinates` parameter text. | ~800–1,000 | after the freeze PRs merge, since they edit the same files |
| Shared test helpers and parametrization (`_scene` ×9, `_data` ×8, `_night` ×5, `_tables` ×4) | ~650 | any time; 0.4.x is fine |
| Remove `Tabulated` as the 0.3.0 CHANGELOG promised, and drop the `epochs` warnings (in hand) | ~200 | your decision on `Tabulated` (currently set to "removed in 0.5") |
| Pipeline: one shared NUTS/posterior wrapper in `_common` | ~110 | before DiskPipeline and OrbitPipeline add a third and fourth copy |
| Dead code (`_azimuthal_spectrum`, `aliases._NAMES`, `lag1_correlation`, `image_harmonics_from_latents`, legacy shims) and the copy-paste clusters | ~180 | none |

These go to 0.5, when the shims expire:

- the `renamed`/`old_order` shims and their tests (~770);
- test-only public helpers such as `chi2ppf` and
  `cvis_radial_dirac_delta_modulated` (~420, each your call).

## Rules going forward (the real fix)

These change how agents work, so they go in AGENTS.md and the briefs.

1. **Delete before you add.** A feature PR names what it supersedes and
   removes it in the same PR, or opens a removal issue for the next minor.
   Nothing is kept merely because it exists.
2. **Build on demand.** A plan stage lands in `src/` when a test system,
   validation case or user needs it. Exploratory engines stay in
   virgil-validation or a notebook until then, and land provisional. The new
   stability tier makes provisional code cheap to change, but it does not
   make unused code free to maintain.
3. **Docstrings say what, docs say why.**
   - A public docstring has a summary, parameters, returns and a short
     example, aiming at 60 lines at most.
   - Derivations, design history, plan problem numbers and named targets go
     in a docs page or design note, with a link.
   - Private helpers get one or two lines.
   - Revise the orbit plan's Principle 5 to match: "every solved problem is
     pinned by a named test and one row in the solved-problems table, and the
     code carries a one-line comment pointing at the row", instead of an
     essay at each site.
4. **Reuse before writing a numerical helper.** Check `_linear`, `_grid`,
   `_geometry` and `_utils` first. Three separate adaptive log-integral
   routines exist today: `epochs._log_integral`,
   `orbit_search._band_marginal` and `grid_fit._refine_flux_grid`. Give the
   orbit modules one private `_marginal.py`.
5. **Tests share helpers.** Shared fixtures go in `tests/conftest.py` or
   `tests/_helpers.py`. Near-identical tests are parametrized. A test file
   should not redefine `_scene`/`_data` helpers that already exist.
6. **New pipelines are thin.** `DiskPipeline`, `ImagePipeline` and
   `OrbitPipeline` subclass `_core` and reuse `_common`. Each aims at about
   500 lines, against 1.0–1.5k for Binary and Star today.
7. **Make growth visible, without gating it.**
   - A CI step posts each PR's net `src`/`tests` line delta as a comment.
   - Above +1,000 `src` lines, the PR description says what the size buys
     and what it supersedes.
   - Each release review reports the size table above.

## What this buys

The one-off cuts take `src` back by about 3k lines. The rules matter more:
at the current pace, 0.4.x would add OrbitPipeline, DiskPipeline,
ImagePipeline, GN3–GN6 and visibility-orbit stages 3–6. Without rules 1, 2
and 6 that is another ~10k lines; with them, plausibly half.
