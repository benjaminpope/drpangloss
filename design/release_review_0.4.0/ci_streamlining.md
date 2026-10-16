# Streamlining CI without losing robustness (2026-10-16)

The numbers come from [`review_ci.md`](review_ci.md), which used `gh` over
the last 100 runs of each workflow on 10-14 and 10-15.

## Where the cost is

| | runner-min | wall | notes |
|---|---|---|---|
| PR gate | ~56 | 11.6 min median | py3.12 f32 × 3 shards + x64 × 3 shards; no `slow` |
| main push | ~110 | ~37 min median | gate + slow + macOS py3.11 (21 min) + py3.13 unsharded (29 min) |
| Whole suite, serial | 2,730 s | | 77 files; 48 tests or files exempt from the 30 s budget |

**Cancelled runs.** 10 of the last 19 main pushes were cancelled by the next
push, so most of the expensive main-push runner time is thrown away, and the
tagged SHA may never have been tested.

**Size.** 10 workflows hold about 1,000 lines of YAML. `tests.yml` alone
is 385, with a Python matrix chooser embedded in it. Tests are 37k lines, 0.76
lines per src line.

## Proposals

Ordered by value; none drops a check that has caught a real bug.

1. **Main pushes run the PR gate plus `slow`, not the full matrix.**
   - macOS py3.11, py3.13 and the dependency floors run nightly, and on PRs
     labelled `full-ci` (always on the release PR).
   - **Robustness:** platform and version breaks are caught within a day, and
     before any release, by the publish gate (R7), which needs a completed
     `tests.yml` run on the tagged SHA.
   - **Saves** about 50 runner-min a push, plus the cancelled half-runs.
2. **x64 runs only where dtype matters.** The x64 shards repeat the whole
   suite, including I/O, plotting, pipeline orchestration, CLI and metadata
   tests, which do not depend on dtype.
   - **Change:** mark the dtype-independent test files once (a list in
     conftest, or a `dtype_independent` marker) and skip them in x64, with
     the numerical files (models, likelihood, fitting, orbits, aliases,
     epochs, imaging, inference, detection, grids, limits, observables,
     gains, svi, fields, ensemble) staying in.
   - **Robustness:** R1 (the float64-only failure) sits in a numerical file,
     so it would still be caught.
   - **Saves** about a third of the x64 shards: two shards instead of three,
     or shorter ones.
3. **Shrink the slowest fixtures before adding shards.**
   - `test_pipeline_star`'s 135 s setup and the 48 budget exemptions set the
     wall time of the slowest shard.
   - **Change:** tiny datasets, fewer optimizer and sampler steps, and
     `scope="session"` fixtures shared across a file.
   - **Target:** no exemption above 60 s, and a gate wall time of about 8
     min.
4. **Keep the additions lean.**
   - The slow tests on orbit PRs (#374) run only the orbit test files' slow
     tests, not all 57; I have asked for this.
   - The PR size comment goes in `lint.yml`'s existing job, not a new
     workflow (growth-rules PR).
5. **Make the YAML readable.**
   - Move the matrix chooser out of `tests.yml` into
     `.github/scripts/ci_matrix.py`, with a small unit test in `tests/lint`.
   - Fold `full-ci.yml`, which has never run, into the label trigger in
     `tests.yml`.
   - Merge `claude-review.yml` and `review-response-label.yml` into one
     workflow, and restrict its triggers: 91 comment-event runs were all
     skipped or cancelled.
   - **Target:** `tests.yml` under 200 lines.
6. **Stop deliberate red crosses.** `ozstar-sync.yml` fails on purpose while
   a PR waits for OzSTAR (19 of 100 runs). Report that state as a neutral
   check or in the job summary, so red always means broken.
7. **Get the cache under quota.** It is at 10.3 GB of 10 GB, with 7.6 GB of
   old-format keys, so evictions cause cold caches, which also turn off the
   per-test budget.
   - **Change:** run the nightly cleanup once by hand (`workflow_dispatch`),
     then let the schedule keep it under quota.
8. **Trim test code.**
   - Parametrize near-duplicate tests and share helpers (~650 lines; see
     `size_sweep.md` §5).
   - Delete the shim tests when the shims expire at 0.5 (~450 lines).

## A CI budget, so it doesn't creep back

These go in AGENTS.md next to the growth rules:

- **Targets:**
  - PR gate wall time of 12 min or less;
  - 60 runner-min or less per PR;
  - no single test over 30 s without a listed exemption.
- **Cost statement.** A PR that adds a job, a matrix entry or a fixture over
  30 s states the cost in its description.
- **Release reviews** report the table above.

## Order

1. Merge #374 first, since `tests.yml` conflicts.
2. Then one PR for items 1, 2, 5 and 6.
3. Then one PR for item 3.
4. Item 7 is a single manual dispatch of the nightly workflow.
5. Item 8 joins the test parametrization in 0.4.x.
