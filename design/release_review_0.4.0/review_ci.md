# CI and test-suite review for virgil 0.4.0

Reviewer scope: `.github/workflows/*`, pytest config, `tests/` organisation, lint scripts, uv.lock / lock-upgrade, publish.yml.
Checked out: origin/main `420d1b7`. Read-only; one-shot `gh` queries only. Nothing was re-run.
Data window: the last 100 runs of each workflow (about 2026-10-14 11:00Z to 10-15 11:00Z). The tiered `tests.yml` landed at 10-14 23:28Z, so the "tiered era" below is about 37 completed PR runs and 19 main pushes. Per-job numbers come from `gh run view --json jobs` on 47 completed tiered-era runs; log excerpts come from main run 38723896496 and the failed-job logs of the 11 failing runs.

## 0. Headline

The earlier CI review's workflow recommendations (2.2, 2.3, 2.4, most of 2.7 and 2.9) have landed, and they work. The PR gate is now 6 sharded jobs (py3.12 f32 and x64, no slow tests). Median PR run wall time is 11.6 min (p90 12.1, n=13) since the budget was raised to 30 s, against 20-23 min before. After the raise, 13 of 13 completed PR runs were green.

Four things matter for 0.4.0:

1. **Main is red.** The x64 shard 1/3 job has failed on three consecutive pushes to main (`1ee23a1`, `76ee8d6`, `420d1b7`, all 10-15). Two `slow` tests in `tests/test_aliases.py` fail, and the PR gate cannot see them because it skips `slow`. Nothing notifies anyone.
2. **publish.yml has no gate.** It checks the tag against `pyproject.toml` and nothing else. It does not require a green CI run on the tagged SHA, and 10 of the last 19 main pushes were cancelled, so a given SHA may never have run CI at all. CITATION.cff and the CHANGELOG heading are unchecked.
3. **The lowest-dependencies job skips about 180 tests**, the entire orbit stack, because it never installs jaxoplanet.
4. **The nightly, canary and `full-ci` paths have never run in production.** They are untested code, and the nightly is what cleans the over-quota cache.

PyPI publishing itself is fine. Trusted publishing with the `pypi` environment worked for 0.2.0 and 0.3.0 (runs 37162531343 and 37397916139, both success), and PyPI lists 0.2.0 and 0.3.0.

## 1. Findings

Severity: blocker = do not tag until fixed; SF = should fix before 0.4.0; 0.4.x; note.

### CI-01. Main is red on x64: two orbit-alias slow tests (blocker)

- **Evidence.** Runs 38707965208 (`1ee23a1`, the merge of #366), 38723828319 (`76ee8d6`) and 38723896496 (`420d1b7`) all fail only in `ubuntu-latest py3.12 x64 1/3`, each with `2 failed, 7xx passed, 2 skipped`.
  - `test_a_single_night_alias_is_flagged_and_the_alias_fit_finds_the_band` (test_aliases.py:1348/1350): `miss[5] = 0.0666 > 3.0` fails. The expected alias peak is not found.
  - `test_skip_flagged_can_be_switched_off_and_unseeded_survives_save_load` (:1421/1423): `(3,) == (5,)` fails, so three nights are seeded where five are expected.
  - The float32 shards passed with slow tests on all three commits.
  - PR #366 merged with all six shard checks green. Both tests are `@pytest.mark.slow`, and the PR gate runs `-m "not slow"`.
- **Why it matters beyond CI.** Importing jaxoplanet switches JAX to x64 globally (see `tests/test_docstrings.py:30` and the earlier review, section 1). Anyone using the orbit stack therefore runs the x64 numerics. Night flagging, `epoch_positions` and the alias search are headline 0.4.0 features. A deterministic x64-only difference in which nights get flagged or seeded is probably a real behaviour difference, not only a test artefact.
- **Fix.** Investigate with x64 on, via a one-off OzSTAR or CI dispatch (no heavy JAX on the laptop). Either fix the flagging or seeding logic, or fix the fixture if it relies on f32 rounding. Then confirm main goes green. Do not tag over it.
- **Effort.** M.

### CI-02. The PR gate skips `slow`, so slow-only regressions merge green (SF)

- **Evidence.** 57 `@pytest.mark.slow` functions (18 in `test_aliases.py`, 6 in `test_epochs.py`, 4 in `test_fitting.py`, 3 each in `test_evidence`, `test_models_sources`, `test_radial_gp`, `test_sampling`). They run only on main pushes, the `full-ci` label and the nightly. CI-01 is the demonstration.
- The orbit-alias code is the area under most churn this week (PRs #360-#371), and `slow` is concentrated there.
- The earlier review's decision to defer branch protection (2.1) is not the issue here. Required checks would not have helped, because the gate does not run the failing tests.
- **Fix.** Pick one:
  - (a) Have `tests.yml`'s `matrix` job (which already runs `gh pr diff`) set `slow: true` for PRs touching `src/virgil/orbits/**` or the orbit test files.
  - (b) Add the `full-ci` label by convention to orbit PRs.
  - (c) Accept the gap but make main-red loud (CI-03).
  - Whichever is chosen, put `full-ci` on the 0.4.0 release-prep PR so macOS, py3.13 and slow tests run before merge.
- **Effort.** S.

### CI-03. Nobody is told when main goes red (SF)

- **Evidence.** `nightly_report.py` runs only when `github.event_name == 'schedule'`. The three red main pushes above produced no issue and no comment. The `nightly-failure` label does not exist yet in the repo (labels: bug, documentation, duplicate, enhancement, good first issue, help wanted, invalid, question, wontfix, awaiting-ozstar, review-response).
- **Fix.** Extend the report job to `push` on main (label `main-red`, same open/update/close logic), or send a notification. The script already does the work, and JUnit artifacts are uploaded on every run.
- **Effort.** S.

### CI-04. publish.yml publishes without any CI gate (SF)

- **Evidence.** `publish.yml` runs `release: published` → check tag equals pyproject version → `uv build` → `twine check --strict` → upload. There is no `needs` on tests and no check that the tagged commit passed `tests.yml`.
- Main pushes are frequently cancelled. `tests.yml` pushes share the concurrency group `tests-push-refs/heads/main` with `cancel-in-progress: false`, so a newer pending run replaces an older pending one. Of the 19 main pushes since the tiering landed, 10 were cancelled, 6 succeeded and 3 failed. Example: the merges of #367 and #370 (`1438db3`, `627ed35`) were cancelled within about a minute.
- The earlier review's 2.8 item (a release gate on virgil-validation) is also not landed (see the table).
- **Fix.** In `publish.yml`'s `build` job, add a step that queries `gh run list -w tests.yml --commit $GITHUB_SHA --event push --json conclusion` and fails unless one run is `success`. It needs `actions: read`. If none exists, the release procedure is to `gh workflow run tests.yml` on the tag commit and wait for it. Add the same to the CONTRIBUTING "Releasing" steps. Optional: also require a green validation run.
- **Effort.** S.

### CI-05. Release metadata is checked in one place of three (SF)

- **Evidence.**
  - `pyproject.toml:3` says 0.3.0.
  - `uv.lock:3805` says 0.3.0. This is guarded: the `package` job runs `uv lock --check`.
  - `CITATION.cff:6` says `version: 0.3.0`. Nothing checks it.
  - `CHANGELOG.md:8` is `## Unreleased`, with the 0.3.0 heading at :668. Nothing checks that a `## 0.4.0 (date)` heading exists.
  - The CONTRIBUTING "Releasing" list says only "Bump version in pyproject.toml, run uv lock".
  - `tests/lint/test_package_layout.py:26` checks only that `virgil.__version__` equals the installed distribution version.
- **Fix.** Add a check in `tests/lint` (cheap, runs in lint.yml): `CITATION.cff` version equals the pyproject version. Repeat in publish.yml's tag step, plus a check that the CHANGELOG has `## <version>`. Update CONTRIBUTING with the CITATION.cff and CHANGELOG steps.
- **Effort.** S.

### CI-06. The lowest-dependencies job never runs the orbit stack (SF)

- **Evidence.** `tests.yml` installs the floor job with `uv pip install --resolution lowest-direct -e ".[test]"`, with no `orbits` or `integrations` extra. A job log (run 38679094500, job 116239394360) shows `1885 passed, 23 skipped`, against 2067 passed in the normal jobs. The skips include every `importorskip("jaxoplanet")` module: `test_aliases`, `test_epochs`, `test_orbit_search`, `test_orbits`, `test_attached`, `test_benchmark_harness`, `test_broadcast_in_time`, parts of `test_no_data_priors`, `test_north_angle`, `test_orbit_positions`, `test_simulate` and `test_bandwidth_smearing`.
- So the declared floor `jaxoplanet>=0.1.0` (the `orbits` extra), and the combination jax 0.8.0 + jaxoplanet 0.1.0, is untested. The orbit stack is the headline of 0.4.0. The harmonix tests are skipped there too.
- **Fix.** Install `".[test,orbits]"` at lowest-direct. If it fails to resolve or run, raise the jaxoplanet floor (or the jax floor) to what works. The lowest job runs automatically on the version-bump PR, since `pyproject.toml` changes, so check it there.
- **Effort.** S to M, depending on what breaks.

### CI-07. py3.13 is the longest job on every main push (0.4.x)

- **Evidence.** py3.13 is a single unsharded job running 2067 tests: 22.9-30.7 min (run 38723896496: 1816 s; timeout is 45 min). macOS py3.11 runs the same 2067 tests in 16-21 min. The three f32 shards take 9-17 min, and the x64 shards 8-18 min. A whole main run is 36.7 min median, 43 min p90. Runner cost is about 110 runner-minutes per main run against 56 per PR run.
- **Fix.** Shard py3.13 three ways like py3.12, or run py3.13 only in the nightly and on `pyproject.toml` PRs. Effort S.

### CI-08. The per-test budget is off for exactly the PRs that add test files (0.4.x)

- **Evidence.** `VIRGIL_TEST_BUDGET` is set only if `steps.cache.outputs.cache-matched-key` starts with `prefix`, which includes `pack<sha of tests/test_durations.json + git ls-files 'tests/*test_*.py'>`. A PR that adds a test file changes the pack hash, its restore falls back to an older prefix, and the budget is off. A new test file is also where a new heavy test is most likely. On main pushes with a cold cache the budget is off too, and py3.13 and macOS never have it (`budget` is set only on the Linux py3.12 jobs; run 38723896496 logged `per-test budget: off s`).
- **Fix.** Enable the budget when the cache matched the JAX-version prefix of the job (JAX keys cache entries by program anyway) and keep the pack hash only for the saved key. Or accept the gap.
- **Effort.** S.

### CI-09. JAX cache store is over quota, and each pack hash adds about 0.9 GB (0.4.x)

- **Evidence.** 32 caches, 10.33 GB (limit 10 GB). About 7.6 GB are legacy keys from before the "save from empty" change (`jax-ubuntu-latest-py3.12-x641-8647…` 1.6 GB, three `…lowest/py3.12/py3.13-…` at 1.1-1.45 GB, two macOS at 0.76-0.8 GB). The new per-week sets are 68-247 MB per job. There are already two pack sets in week W42 (`pack9c6647c6` and `packf8f05495`, about 0.9 GB each).
- The nightly deletes every `jax-*` cache not ending in the current ISO week, so the legacy keys go on its first run. That run has not happened yet (cron 03:00 UTC; the last scheduled runs listed are 10-05 and 10-12).
- Every test-file addition produces a new pack set on the next main push.
- **Fix.** Check `gh cache list` after the first nightly. Longer term, drop the pack hash from the key (see CI-08), or delete older packs of the same week on save.
- **Effort.** S.

### CI-10. Nightly, canary and `full-ci` are untested in production (0.4.x)

- **Evidence.**
  - No `schedule` run exists since the tiering commit `471903d` (the listed schedule runs, 10-05 success and 10-12 failure, are of the old weekly workflow). The nightly adds lowest, the canary (`uv lock --upgrade` then `uv sync`), slow tests on everything, the report job, `nightly_report.py` and the cache clean-up. None has executed.
  - `full-ci.yml` has 2 runs, both `skipped` (other labels). The label `full-ci` does not exist in the repo's label list, so it must be created before it can be applied.
  - The release PR can exercise `full-ci`. The nightly needs a day.
- **Fix.** Create the `full-ci` label. Read the first nightly result before tagging. Effort S.

### CI-11. A serial floor remains in the sampler-heavy modules (0.4.x)

- **Evidence.** `tests/over_budget.txt` still holds 48 exemptions. In run 38723896496 the py3.13 and macOS jobs' slowest items were `test_pipeline_star::test_summary_keys_snapshot` (setup 135 s, the `full_run` fixture, unchanged since the earlier review), `test_aliases::test_random_starts_and_s_max_without_positions` (66-78 s), `test_divergences_at_hard_walls_are_explained_not_hidden` (50-63 s), `test_pipeline_binary` setup 36-47 s, `test_orbit_search::test_independent_of_batch_size` (17-57 s) and the `test_svi` NUTS-vs-SVI comparisons (20-33 s each).
- `test_durations.json` totals about 2730 s over 77 files. The three heaviest files: aliases 300 s, orbit_search 289 s, pipeline_star 260 s.
- Shards pack whole files, so shard 1 stays the critical path (PR median 8.9 min f32 and 11.2 min x64 for shard 1; shard 3 about 8-9.6).
- **Fix.** This is brief A's PR 2 (not landed): shrink fixtures, build the key-snapshot test from a stub run, mark one realistic run `slow`. Effort M.

### CI-12. x64 / dtype state not made explicit (brief A PR 1 not landed) (0.4.x)

- **Evidence.** `tests/conftest.py` has no autouse x64 guard. `tests/_tolerances.py` does not exist. `tests/_shared.py` has no tiny-NUTS helper. AGENTS.md "Testing notes" has no testing rules (budget, oracles, one file per subject). Only one PR is open (#369, a pipeline). Two tests are still skipped on x64 by import-time config (`test_chromatic.py:416`, `test_fitting.py:1063`), and `test_sampling.py:28` skips when x64 is off.
- This relates to CI-01: x64 state decides which numerics the orbit tests see.
- **Fix.** As in the earlier review's 2.5, 2.6 and the AGENTS.md rules of 2.7. Effort M.

### CI-13. The Claude review workflow works and is worth its cost, but is noisy (note)

- **Evidence.** 9 of 9 `pull_request` reviews succeeded in 2.5 min median (p90 4.6). Reviews of #366, #368, #370, #371 and #367 left 3-5 inline comments each. Those on #366 are substantive (`primary_diam="fit"` always warns with advice that cannot be followed; a reason string wrong for `min_gap=None`; no test reaching the lower-bound branch; a test that would pass without the change).
- The `review response label` workflow ran 43 of 43 times successfully in 0.1 min and tracks unanswered Claude threads.
- Cost: `--model claude-opus-5-5`, max 40 turns, on Ben's OAuth token, for every non-draft PR (memory note: usage was at 88% on 10-08). The value per PR looks high. Consider Sonnet for routine PRs and Opus on a label.
- Noise: the workflow also listens to `issue_comment` and `pull_request_review_comment`. Of the last 100 runs, `pull_request_review_comment` accounts for 82 (51 skipped, 31 cancelled) and `issue_comment` for 9 (8 cancelled, 1 skipped), all triggered by Claude's own comments. A per-PR concurrency group with `cancel-in-progress: false` replaces older pending runs, so a human `@claude` request made while a review runs could be dropped. I did not find an instance.
- **Fix.** Move the concurrency to job level or skip bot-authored comments before it applies. Pin `anthropics/claude-code-action` to a commit, as `publish.yml` pins its actions, since it holds the OAuth token and `id-token: write`. Effort S.

### CI-14. Docs build is unlocked; docs and notebook sync are covered (note)

- **Evidence.** `zensical-pages.yml` builds with `zensical build --clean --strict` on every PR and main push (0.4 min on PR, 0.7 median on main). It installs `-e ".[docs]"` unlocked, with `zensical>=0.0.62,<0.1`. 4 of 76 PR runs failed (not diagnosed; the logs of two had expired). Deploy on main works (20 of 24 push runs succeeded, 4 cancelled), and `ZENSICAL_DEPLOY_ENABLED` is `true`. There is no deploy on tag; the 0.4.0 docs go live when the version-bump PR merges, which is fine.
- The `lint` workflow's `checks` job (docs sync, package layout, notebook-script unit tests) caught a stale `docs/polar_disk_rings.md` on 10-15 (run 38701469041) in about a minute, so docs-sync failures no longer fail 4-5 test jobs.
- The real notebook-output staleness check (`scripts/check_notebook_outputs.py`) lives in `ozstar-sync.yml` as the job `notebooks`, under an unrelated workflow name. `tests/lint/test_notebook_outputs.py` only unit-tests the script.
- **Fix.** Optional: install docs from the lock (a `docs` dependency group) and move the notebook check to lint.yml. Effort S.

### CI-15. Small items (note)

- `lint_local.sh` says "same scope as CI", but CI runs `ruff check .` over the whole repo (including `benchmarks/`, `design/`, `docs/`, `notebooks/tutorial_helpers.py`), while the local script checks `src tests examples scripts` and the tracked notebooks. A lint error in `benchmarks/` passes the pre-push hook and fails CI.
- `copilot-setup-steps.yml` still does `uv pip install --upgrade jax jaxlib`, contradicting the lock model. Harmless.
- The repository default workflow permission is `write` (`gh api …/actions/permissions/workflow`). Every workflow declares its own permissions, so it is not exploitable today, but it should be `read`.
- The `pypi` environment has no required reviewer and no branch policy, so "Publish release" uploads immediately. Fine for a solo maintainer; adding yourself as reviewer gives a last pause.
- `pyproject.toml` has no `--strict-markers` and no `-W error` beyond the three shim-warning filters, so new JAX deprecation warnings go unnoticed (about 115 warnings per job).
- The lock gives JAX 0.10.2 for Python 3.11 and 0.11.2 for 3.12+. On Linux, 3.11 is exercised only at the floors; at the locked versions only macOS py3.11 on main covers it.
- No flaky tests observed: across the 100 tests.yml runs, no commit has a run that failed and then passed. Failures in this window were deterministic: the 15 s budget (all PR failures I read, before the raise to 30 s at `4921fac`) and CI-01.

## 2. Measurements

### Per-workflow (last 100 runs of each; wall = startedAt to updatedAt, completed runs only)

| workflow | PR runs | main-push runs | notes |
|---|---|---|---|
| tests.yml, PR runs since 10-14 23:30Z | median 12.8 min, p90 26.9 (early pre-shard runs); 29 success / 8 failure / 13 cancelled | median 36.7 min, p90 43.3; 6 success / 3 failure / 10 cancelled | |
| tests.yml, PR runs after the 30 s budget (`4921fac`, 07:25Z) | median 11.6 min, p90 12.1, max 13.0; 13 success / 0 failure / 10 cancelled | | |
| lint.yml | median 0.4 min; 73 success / 2 failure / 1 cancelled of 76 | 0.4 min; 15 success / 8 cancelled of 23 | failures: stale docs sync (caught), deps/uv-lock-upgrade |
| zensical-pages.yml | median 0.4 min; 70 success / 4 failure / 2 cancelled of 76 | 0.7 min; 20 success / 4 cancelled | |
| claude-review.yml | pull_request: 9 of 9 success, 2.5 min median; comment events: 91 runs, all skipped or cancelled | | |
| review-response-label.yml | 43 of 43 success, 0.1 min | | |
| ozstar-sync.yml | 81 success / 19 failure of 100 (deliberate `awaiting-ozstar` failures) | | |
| full-ci.yml | 2 runs, both skipped | | never exercised |
| lock-upgrade.yml | 1 run (10-15 05:04), success; produced PR #360, merged | | works |
| publish.yml | 2 releases (0.2.0, 0.3.0), both success, 0.6 min | | |

### Per-job (tiered era, completed jobs, minutes)

| job | n | median | p90 | max | failed |
|---|---|---|---|---|---|
| PR: py3.12 1/3 | 29 | 8.9 | 12.2 | 15.6 | 3 (budget) |
| PR: py3.12 2/3 | 29 | 9.1 | 12.3 | 14.9 | 0 |
| PR: py3.12 3/3 | 29 | 9.6 | 10.5 | 11.1 | 2 (budget) |
| PR: py3.12 x64 1/3 | 29 | 11.2 | 13.2 | 15.1 | 6 (budget) |
| PR: py3.12 x64 2/3 | 29 | 8.6 | 11.3 | 14.1 | 0 |
| PR: py3.12 x64 3/3 | 29 | 8.3 | 10.8 | 11.6 | 3 (budget) |
| PR: py3.11 lowest (only on `pyproject.toml` PRs) | 10 | 16.9 | 22.9 | 26.3 | 0 |
| main: macOS py3.11 | 9 | 21.1 | 27.9 | 34.4 | 0 |
| main: py3.13 (unsharded) | 9 | 29.0 | 30.7 | 30.7 | 0 |
| main: py3.12 x64 1/3 | 7 | 14.5 | 17.4 | 17.9 | 3 (CI-01) |
| main: py3.12 1/3, 2/3, 3/3 | 7 each | 15.2 / 9.2 / 8.8 | 16.4 / 11.7 / 10.8 | 19.3 / 12.5 / 11.6 | 0 |
| build and install the wheel | 36-45 | 0.4 | 0.4 | 0.5 | 0 |

Runner minutes per run: median 56 on a PR, 110 on main. Before tiering, the earlier review recorded four 16-26 min jobs per PR (about 80-100).

### Test suite (static count; I did not run it)

- 80 `test_*.py` files in `tests/` (plus `tests/lint` with 3), about 1537 `def test_`, 2067 non-slow cases collected per the CI log, 36,078 lines.
- 57 `slow` functions; 3 modules use `xdist_group` (`test_pipeline_star`, `test_pipeline_binary`, `test_orbit_search`).
- Skips and xfails: no `xfail` anywhere. Skips are `importorskip` of optional packages (jaxoplanet, harmonix, h5py, nbclient, scipy.interpolate; see CI-06), two import-time x64 skips, and one dtype `pytest.skip` in `test_grid_and_limits.py:547`.
- Slowest tests: listed in CI-11.
- The six PR jobs are two configurations, not redundant in value (the x64 job is what exposed CI-01), but x64 shard 1 is the heaviest.
- Does CI catch what matters?
  - Newest JAX: yes. Installing from the lock, the daily lock PR (#360 merged) and the nightly canary (unproven).
  - Lowest deps: partly (CI-06). Only `[test]` is covered at the floor, and only on `pyproject.toml` PRs and nightly.
  - float32 and x64: both on every PR. The x64 result for the slow orbit tests arrives only after merge (CI-01).
  - Docs build: yes, strict, on every PR.
  - Notebook and doc sync: yes, in lint.yml and ozstar-sync.yml.
  - macOS, py3.13: only after merge (by design).
  - Built wheel: yes, on every run (`package` job: `uv lock --check`, build, `twine check --strict`, wheel import and smoke test in a clean environment). publish.yml builds the same way but does not repeat the smoke test.

## 3. What 0.4.0 needs before tagging

1. Fix CI-01 and get main green (blocker).
2. Bump `pyproject.toml`, run `uv lock`, update `CITATION.cff`, rename the CHANGELOG heading to `## 0.4.0 (date)`. A lint/publish check (CI-05) would stop a miss.
3. Label that PR `full-ci` (create the label first) and make sure the lowest job runs (it will, since `pyproject.toml` changes); read its result (CI-06: orbit tests are skipped there today).
4. After merging, confirm the push run of the merge commit was not cancelled (CI-04), or dispatch `tests.yml` on it.
5. Tag `v0.4.0` on that commit and publish the release. Trusted publishing is configured and works.

## 4. Status of the earlier CI review's recommendations

| # | Recommendation (earlier review) | Status | Evidence |
|---|---|---|---|
| 1 / 2.1 | Required status checks on main, no force pushes | Not landed (deliberately deferred 2026-10-15) | `branches/main/protection`: `allow_force_pushes: true`, no required status checks, `required_approving_review_count: 0`. CI-01 shows main red for three pushes, but required checks alone would not have caught it, because slow tests are outside the PR gate. |
| 2 / 2.2 | Move non-JAX checks (docs sync, layout, notebook outputs) to lint.yml | Landed | lint.yml `checks` job runs `tests/lint` (about 0.4 min); tests.yml has `--ignore=tests/lint`; the compile-every-file test is gone from `test_docstrings.py`. The real notebook-staleness script remains in ozstar-sync.yml (CI-14). |
| 3 / 2.3 | Tier the matrix (gate / full / nightly) | Landed, with variations | `matrix` job chooses entries. PR: py3.12 f32 and x64 in 3 shards each; lowest on `pyproject.toml` PRs; full: + macOS py3.11 and py3.13; nightly: + lowest, canary, slow everywhere. PR wall 11.6 min median (target about 8-10). py3.13 unsharded (CI-07). |
| 2.3 note | Nightly failure reporting through an issue | Landed, unproven; no main-push reporting | `nightly_report.py`; no scheduled run since (CI-03, CI-10). |
| 2.3 note | Sharding with committed durations | Landed | `tests/test_durations.json` (77 files, 2730 s), `--shard K/N` in conftest, whole-file packing. |
| 4 / 2.4 | Install from uv.lock, daily lock-upgrade PR | Landed and working | `uv sync --locked …`; lock-upgrade.yml opened #360 on 10-15 (merged). The `package` job runs `uv lock --check`. Canary for ahead-of-lock breaks is unproven. |
| 4 / 2.5 | Autouse x64 guard, `tests/_tolerances.py`, explicit dtype | Not landed | No guard in conftest; no `_tolerances.py`; import-time skips remain (CI-12). |
| 5 / 2.6 | Oracles or n_eff margins instead of single-seed assertions; shared tiny-NUTS helper | Not landed | `tests/_shared.py` has no sampler helper (CI-12). |
| 6 / 2.7 | Per-test time budget enforced in CI; pytest-timeout | Landed, relaxed | `VIRGIL_TEST_BUDGET=30` (calls, plus setup charged per file); `tests/over_budget.txt` has 48 exemptions; `timeout = 300` in pyproject, pytest-timeout in the `test` extra. The 15 s first version failed every PR run I read (8 PR runs) and was raised to 30 s; no budget failures since. Off for new test files and cold caches (CI-08). |
| 2.7 | AGENTS.md testing rules; shrink pipeline fixtures; consolidate files; delete shim tests at 0.5 | Not landed | AGENTS.md "Testing notes" unchanged (3 lines). `test_pipeline_star` setup still 135 s. Shim-test deletion belongs to 0.5. |
| 7 / 2.8 | Calibration moved to virgil-validation; nightly validation against main; release gate | Not landed (virgil side) | `publish.yml` has no gate (CI-04). I did not review virgil-validation. |
| 2.9 | JAX cache key on (os, python, x64, JAX, week); save once; clean old caches | Landed, with caveats | Key as specified plus pack hash; save from empty; nightly deletes older weeks. Store still 10.33 GB until the first nightly runs (CI-09). |
| 2.9 | `jax.clear_caches()` only above an RSS threshold | Not landed (low priority) | `conftest.py` `_clear_jax_caches` unchanged. |
| 2.9 | pmap subprocess tests marked slow | Not landed | `test_gains.py` subprocess use unchanged (:304, :322), no `slow` mark. |
| 2.9 | `count_compiles` private-API fallback | Not landed | `tests/_compiles.py` still falls back to `jax._src.monitoring`. |
| 2.9 | Job timeouts 30 min on main, 90 nightly | Landed (differently) | 30 min PR non-slow, 45 on main and slow, 90 nightly. py3.13 on main takes up to 30.7 of 45 min. |
| Briefs A | Two test PRs (dtype safety; samplers) | Not landed | Only PR #369 is open; none touches `tests/_tolerances.py`. |
| Briefs | Workflow tiering PR | Landed | Commits `471903d`, `4c237c1`, `b30fe4d`, `c81c074`, `80a0f98`, `4921fac`. |
