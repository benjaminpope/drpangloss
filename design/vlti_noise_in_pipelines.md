# Instrument noise models in pipelines

Status (2026-10-18): **design, nothing implemented.** Target virgil 0.4.1.
Counterpart in virgil-vlti: `design/noise_spec_emission.md` (how the pair test
emits the spec).

## Problem

virgil's pipelines assume independent errors per channel: the quoted σ on V²
and on closure phase, optionally rescaled by one factor. For PIONIER that is
the wrong error structure, and the fit pays for it.

- **The pair test** (virgil-vlti, `design/pionier_systematics.md`: IW Car,
  PIONIER, ESO 1103.D-0797, 11 disjoint pairs of consecutive frames) finds that
  only about 0.32σ of the quoted V² error is white. The rest is an achromatic
  gain per frame. The pair test fits widths on V² (1.1 % per telescope and
  0.9 % per baseline); on log |V|, the unit of `with_gains` and of the spec,
  that is half: about 0.55 % per telescope and 0.45 % per baseline
  (ΔlnL = +250 to +275 over independent errors). Closure-phase errors are about
  1.5× too large (white scale 0.65), with an offset of about 0.21° per frame and
  triangle once the target's own change is subtracted. These are lower bounds,
  since consecutive frames share calibrators.
- **The post-AGB imaging** of nine PIONIER discs (virgil-validation,
  `scripts/postagb_imaging.py`) shows the consequence. 42–87 % of the V²
  residual variance sat in the per-(frame, baseline) channel mean, against 17 %
  expected for white noise, and the adjacent-channel correlation of the
  residuals was 0.4–0.94. With one MacKay scale (`error_scale="rescale"`) the
  GP fits absorbed the calibration offsets into the image: χ²/pt was about 0.6,
  hyperparameters landed at the grid corners, and ΔlnZ was inflated.
- **The widths differ between targets.** The per-target pair test
  (`scripts/postagb_noise_widths.json` on virgil-validation branch
  `claude/postagb-noise-model-06daae`, not yet on main) gives a white V² scale of
  0.24–0.61, a telescope gain of 0.46–1.7 % and a baseline gain of 0.30–0.91 %
  on |V|, and a closure-phase white scale of 0.65–0.89. A single default would be
  wrong for most targets; the spec has to be measured per target or per night.

The machinery to *use* such a model mostly exists. What is missing is a
standard way to carry it from virgil-vlti into a pipeline run and keep it there.

## Current state (verified in the code, 2026-10-18)

- `OIData.with_gains(telescope=, baseline=, chromatic=, modes=)`
  (`oidata.py`) and `OIData.with_closure_offsets(baseline=, triangle=, modes=)`
  exist. The gains are marginalized analytically (`virgil.gains`), so the
  covariance depends on the model and fits use L-BFGS.
- `imaging.log_evidence` takes the fixed gain widths of the data. Fitted noise
  terms (`vis_gain_telescope`, `vis_gain_baseline`, `vis_gain_chromatic`,
  `vis_gain_modes`, `phi_offset_*`) exist as `noise=` sites of `fit`, but the
  evidence does not fit them.
- `imaging.error_scale` refuses data with gains or closure offsets
  (`_check_scalable_blocks`, only for `by_observable=True`; the single-scale path
  uses the quoted errors and says it does not support `noise=` terms).
- `pipeline/_io._unsupported_reason` returns "calibration gains or closure
  offsets" for such data, so `processed.oifits` is skipped with that reason and a
  run cannot be reloaded as OIFITS.
- `data_fingerprint` hashes every field including `gains` and `phase_offsets`,
  so the resume check already notices them; but the *recipe* that made them
  (widths, provenance) is not recorded anywhere.
- DiskPipeline's `error_scale` is `"auto"` (`"rescale"` for `"gp"`, `"quoted"`
  otherwise), `"quoted"`, `"fit"` (log-uniform scales on V² and closure phase;
  not for `"gp"`) or `"rescale"`. Star and Binary offer `"quoted"` and `"fit"`.
- `read_oifits(..., bandwidth=False)` by default.
- `virgil_vlti.pairs.fit_noise` fits the widths (`Fit.widths`: term name to
  width) but emits no standard record.

## 1. The noise spec

### Format

A plain, versioned dict, stored as JSON. It is a *measurement* of an
instrument's error structure, independent of any dataset.

```json
{
  "schema": "virgil.noise_spec/1",
  "instrument": "PIONIER",
  "white": {"v2": 0.32, "t3": 0.65},
  "gains": {
    "telescope": 0.0055,
    "baseline": 0.0045,
    "chromatic": null
  },
  "closure_offsets": {"triangle": 0.0036, "baseline": null},
  "provenance": {
    "tool": "virgil-vlti 0.x (pairs.pair_test)",
    "run": "IW Car pair test, 11 disjoint pairs",
    "date": "2026-10-04",
    "target": "IW Car",
    "night": null,
    "program": "1103.D-0797",
    "bound": "lower"
  }
}
```

Rules:

- `white` are multiplicative scales on the *quoted* errors per observable
  (`v2`, `t3`; `t3` follows the closure-phase naming of `OIData`). Missing key
  means 1.
- `gains` are 1σ widths on log |V| (the unit of `with_gains`). `Fit.widths` of
  the pair test are on V² (IW Car: 1.1 % and 0.9 %), so virgil-vlti halves them
  when emitting (the example's 0.0055 and 0.0045); the spec has one unit, and
  the halving happens once, in the emitter only. `chromatic` is the width of the (λ_ref/λ)² shape of `with_gains`.
  `null` leaves a group out.
- `closure_offsets` are in **radians** (the unit of `with_closure_offsets`);
  virgil-vlti converts from degrees. `triangle` and `baseline` as in
  `with_closure_offsets`.
- `provenance` is free-form but must hold `run` and `date`; it is carried into
  `run.json` and the quicklook notebook and is never interpreted. `bound` is
  `"lower"` or `"estimate"`: the pair test is a lower bound.
- Unknown top-level keys are an error; unknown `provenance` keys are kept. A
  future `schema` major version is refused with a message naming the version.

virgil does not import virgil-vlti. The spec is the whole interface; virgil
documents it and any tool, by hand or script, can write one.

### Functions

New module `virgil.noise_spec` (public, small):

- `validate_noise_spec(spec) -> dict`: returns a normalized copy (floats,
  defaults filled, `null` groups kept) or raises `ValueError` listing every
  problem: wrong schema, negative or non-finite widths, unknown keys, white
  scale ≤ 0, offsets in implausible units. Realistic closure-offset widths are about
  4e-3 rad (0.21°), so a width above 0.05 rad is refused as probably degrees
  (0.21 written in degrees would otherwise pass as 12°), and one above 0.02 rad
  warns. Gain widths above 0.1 on log |V| are refused likewise.
- `load_noise_spec(source)`: path or dict, then `validate_noise_spec`.
- `apply_noise_spec(data, spec, *, white=True, gains=True, offsets=True)
  -> OIData`: the one function that applies a spec. In order: scale the quoted
  errors by `white` (`with_error_scale` with the per-observable dict), then
  `with_gains(telescope=, baseline=, chromatic=)`, then
  `with_closure_offsets(baseline=, triangle=)`. It refuses data that already
  have gains or offsets (applying twice is a bug), and skips a group whose
  precondition fails with a warning that says why (closure offsets need four or
  more telescopes). It records nothing on the data; the caller keeps the spec.
- `instrument_match(data, spec)`: a warning, not an error, if `data`'s
  instrument (from `insname`, if the record has it) differs from the spec's.

`apply_noise_spec` is a pure function of `(data, spec)`, so it is the only thing
persistence needs to replay.

### Design choice: widths are fixed, not fitted

The spec fixes the widths. Fitting them (`noise=` terms) remains a separate,
explicit analysis and is not what a pipeline does by default (see §4). A
measured width from calibrators or pair tests is more trustworthy than one the
fit can trade against the image, which is exactly what failed in the post-AGB
work.

## 2. The pipeline `noise_spec=` setting

The setting is `noise_spec`, not `noise`: `noise=` already means *fitted
noise-term priors* in `fit`, `numpyro_model` and `Epochs` (`conventions.md`), a
different object under a shared-parameter name. All pipelines take `noise_spec`:

- `None` (default): current behaviour, independent errors.
- A spec dict, or a path (`str`/`PathLike`) to a spec JSON.

Settings resolve as defaults, then the JSON config, then keyword arguments,
unchanged. In the **config file** `noise_spec` may be a path (relative to the config
file) or an inline object. The **resolved** setting stored in `run.json` is
always the full validated spec inline (plus `noise_path`, the path it came
from, for humans), so a run folder is self-contained and a moved config does
not change the meaning of a resumed run.

**CLI:** `virgil-pipeline ... --noise-spec spec.json` (and `--noise-spec none` to override
a config). No flags for individual widths: a width not in a spec file has no
provenance.

**Default stays `None`.** The spec is instrument- and target-specific, and a
default drawn from one target (IW Car) would be wrong for most of the nine we
saw. A pipeline given PIONIER data and no spec instead *warns* (via the
residual-correlation check, §5) when the residuals show the structure.

Applied when: `load` stage, after selections and before any stage uses the
data. Order matters: the pair test scaled the *raw quoted* σ, so the white
scale is applied first and `error_floor` (an absolute floor) after it, so a
floor stays a floor instead of becoming 0.32 × floor. Because the floor and
the gains both model excess error, a run that sets both `error_floor` and
`noise_spec` gets a warning that the floor is probably double counting.
`apply_noise_spec` therefore takes un-floored data and the `load` stage
reorders accordingly: spec white scale, then floor, then gains. The stage
report records the spec's provenance and the applied groups.

**Model compatibility:**

- `StarPipeline`, `BinaryPipeline`, `DiskPipeline` with any model: gains and
  offsets enter through the likelihood, so all work. Fits with gains need the
  gradient-based solvers; the pipeline already routes to them, and the
  pipeline's `grid` and `profile` stages call `log_evidence`, which supports
  fixed widths.
- `OrbitPipeline` (planned, `orbit_pipeline.md`) will take the same setting.
- Data types unsupported by the gain machinery (vis amplitudes, `uv_grid`,
  extras) raise at `load` with the existing reason string.

## 3. Persistence and resume

Today `processed.oifits` is the data *as fitted*, so gains and offsets, which
OIFITS cannot hold, block it. The new rule is to store the **raw processed data
and the spec**, and rebuild the fitted data by replay.

- `data/processed.oifits`: the data after selections, *before* the spec's white scale, the error
  floor and the gains (which `Result.data()` replays in that order; without a
  spec the floor is applied as today). It is always writable for supported data, so the file never carries
  the nuisance model.
- `data/noise_spec.json`: the validated spec, byte-stable (sorted keys).
- `run.json`: `inputs` gains `noise_sha256` (hash of the canonical spec JSON;
  `null` with no spec) and the resolved settings hold the spec inline.
- **Resume hash.** `data_fingerprint` is unchanged and is computed on the
  *fitted* data (after `apply_noise_spec`), so it already covers the gains. The
  resume check additionally compares `noise_sha256` first, so the error message
  can say "the noise spec differs" rather than "the data differ". Changing the
  spec invalidates every stage from `load`.
- **Replay lives in `Result.data()`** (`_io.py`), whose docstring promises the
  data actually fitted; `virgil.pipeline.load` only builds the `Result`. It reads
  `processed.oifits`, replays `apply_noise_spec` with the stored spec, and
  checks two things: the pre-spec fingerprint plus `noise_sha256` (both exact,
  recorded in `run.json`), and the replayed gain modes against the stored
  `data_sha256` *with a tolerance*, not byte equality: the modes come from JAX,
  whose last bits change with version, backend and float32/float64, which the
  routine lock upgrades would turn into spurious refusals. A tolerance failure
  is a warning; an exact-hash failure is an error.
- **`_unsupported_reason`.** The gains/offsets branch is deleted from the
  *writer* check (the writer only ever sees pre-spec data), but a data object
  that reaches the writer *with* gains still gets the same reason string, so a
  hand-built `OIData.with_gains(...)` passed straight to a pipeline keeps working
  as today: no OIFITS written, reason in the load report. The `noise_spec=` route is
  the supported way and loses that limitation. (A pipeline given pre-gained data
  *and* a spec refuses: `apply_noise_spec` will not stack them.)

## 4. Interaction with `error_scale`

Three questions: what each mode means with a spec, whether to disable
`rescale`, and how evidence stays comparable.

### Recommendation: with a spec, `rescale` acts on the white part only

Two options were considered.

*A. Disable `rescale` when a spec is given.* Simple, and it cannot absorb
calibration offsets. But the white scale in the spec is a pair-test number,
itself uncertain (0.24–0.61 across targets), and the model-dependent part of
the misfit (a missing component) is exactly what MacKay's `s` catches. Disabling
it removes a useful diagnostic.

*B. `rescale` estimates one extra factor `s` on the white part only.* The
covariance becomes `s² · σ_white² · D + G G^T` with the gain term *not* scaled,
which is what the measurement says: the gain term has its own fixed width.
`imaging.error_scale` then needs to be generalized so the nuisance block does
not scale. This is the physically right thing and is what the existing refusal
(`_check_scalable_blocks`) says is missing: "their nuisance widths do not scale
with the quoted errors". We recommend **B**.

Consequences for B, to settle in the implementation PR:

- With gains the covariance is `C(s) = s² W + G Gᵀ` (W the diagonal quoted
  white term, G the fixed gain modes). The MacKay fixed point `s² = χ²/(N − γ)`
  holds only when the whole covariance scales with `s²`; here setting
  d ln Z / d s² = 0 has no closed form and reduces to it only for G = 0.
  Reusing it with gain-aware χ² and γ would bias `s` by treating the
  gain-dominated directions as if they scaled. So `s` is found by a 1-D
  maximization of `log_evidence` over `s` (bracketed, log-spaced), and
  "χ²_white" is not used. The current refusal is lifted only
  for the single-scale path; `by_observable=True` with gains is allowed *per
  observable* only if the gains touch one observable (V² here, so `vis`
  and `phi` scales are well defined). Otherwise it still raises.
- `s` is reported separately from the spec's white scale, never multiplied into
  it: two numbers, two meanings (a measured instrument property and a
  per-target model-adequacy factor).
- A large `s` (existing thresholds, warn 2, fail 5) with a spec on is still a
  failed fit, not a noise measurement.

**Fallback if B proves fragile:** ship A first (error_scale `"rescale"` raises
`ValueError` when `noise_spec` is set, with a message pointing here) and add B later.
It is a strictly smaller first PR and the default for GP fits would then be
`"quoted"` when a spec is given.

### What `"fit"` means with a spec

`"fit"` fits log-uniform scales on V² and closure phase. With a spec, it fits
those scales *on the white part*, with the spec's gains fixed. The fitted
`vis_scale` multiplies the already-spec-scaled errors, so the start is `1` and
the log-uniform bounds are centred on 1: `1` means "the spec is right". (The
start is never the spec's white scale, which would apply it twice.) The gain widths are
never fitted by `"fit"`; fitting them is the existing explicit `noise=`
site-level API, and is out of pipeline scope for 0.4.1.

### Evidence on one footing

`log_evidence` and the pipeline's ℓ / hyperparameter scans must be compared
under one covariance. Rules:

- Within one pipeline run the spec is fixed, so every grid point, every ℓ and
  every model in a comparison sees the same `data`. This already holds for
  `rescale` (the factor is estimated once at the best-fit ℓ and applied to all;
  `disk.py` `_scaled`).
- Across runs, evidence is comparable only with the same `noise_sha256`; the
  quicklook comparison tables carry it and the compare helpers refuse to rank
  runs whose hashes differ. A run without a spec is never comparable with one
  with a spec; the log-determinant of the gain covariance is part of the
  evidence (that is a feature: it is what penalizes absorbing offsets into the
  image).
- Evidence between flat and structured GP means is addressed in §8.

## 5. Checks

All checks are pure functions in `_checks.py` returning `Check`, as now.

### Residual correlation (new, `residual_correlation`)

Inputs are the whitened V² residuals of the best fit, laid out as in
`_baseline_layout` (`n_frames × n_baselines × n_channels`). Two statistics:

1. **Common-mode fraction per (frame, baseline).** The fraction of the V²
   residual variance carried by the channel mean of each (frame, baseline),
   `f = n_ch · mean(r)² / Σ r²`, averaged over (frame, baseline). The white
   expectation is `1/n_ch` (17 % for six channels), with a small-sample
   standard error from the number of (frame, baseline) groups.
2. **Adjacent-channel ρ.** The mean correlation of the residuals in neighbouring
   channels, after removing nothing, against 0 for white noise (standard error
   `1/√N_pairs`).

Status: warn when `f` exceeds the white expectation by more than `k` standard
errors *and* `f > 2/n_ch`, or when `ρ > 0.3`; fail when `f > 0.6`. Defaults from
the nine post-AGB targets (42–87 % and 0.4–0.94 respectively) are conservative;
tune on them in the test plan. The warning text says: "The V² residuals are
correlated across channels (…); this is the signature of per-frame calibration
gains. Measure a noise spec with virgil-vlti's pair test and pass `noise_spec=`; see
the noise-spec section of `docs/conventions.md`." The check is useful **without a spec**: it is the
detector that tells a user they need one. With a spec it should *drop*; staying
high means the spec's widths are too small, and the text says that instead.

Data without a (frame, baseline, channel) layout (`_baseline_layout` returns a
reason: uv-grid, flagged, projected or single-channel data, where n_ch = 1
makes f ≡ 1) skip the check with that reason; they neither raise nor fail. It
applies to V² only. Closure-phase offsets share channels the same way but
the independent-triangle bookkeeping (`closure_noise.md`) means a first version
should cover V² and add closure phase only once the null expectation is
derived.

### `at_bound` and render, as they stand

- `at_bound(names)` (and `start_at_bound` for the starting point) warns that a
  fitted parameter sits at a bound. Post-AGB GP fits hit grid corners precisely
  because the offsets were absorbed. With a spec we expect fewer; the check is
  unchanged. A new use: if a spec is on and hyperparameters *still* sit at the
  corner, the message adds a line about checking the spec's `bound` field (it is
  a lower bound).
- `render_consistency(chi2_model, chi2_render)` compares model and rendered
  image χ²/N. With gains, both must use the same covariance, so the render χ²
  is computed on `apply_noise_spec`-ed data and the check is otherwise
  unchanged. Add a unit test so a spec cannot be silently dropped from the render
  path.

## 6. Bandwidth smearing

`read_oifits(..., bandwidth=True)` is off by default. PIONIER's six channels
are broad (R≈40); smearing at long baselines is a few percent in V², the same
order as the gains we are trying to model. Smearing is a deterministic, smooth function of (u, v, λ), nearly the same in
two consecutive frames, so it cancels in the pair difference and cannot inflate
the measured per-frame gain widths. In a fit, unmodelled smearing instead shows
up as a static, baseline-length-dependent V² deficit that the model or image
absorbs, which the random per-frame gains do not. So it is a separate bias
from the one this note addresses, worth fixing on its own account but not a
reason to doubt the spec.

Recommendation: do **not** change the global default in 0.4.1 (it is a public
`read_oifits` behaviour change, and a pipeline's `data_sha256` would change for
every existing run). Instead:

- the pipelines gain a setting `bandwidth` (default `None` = leave as read),
  which, when `True`, requires that the data were read with `bandwidth=True` and
  errors otherwise with the exact `read_oifits` call;
- the spec's provenance may record `bandwidth` (bool) for information; there is
  no mismatch warning, since the mechanism above does not operate;
- the 0.5 question is whether `bandwidth=True` becomes the default; collect the
  IW Car comparison (below) first.

## 7. Migration, back-compat, tests, order

**Back-compat.** `noise_spec=None` leaves the fit and its hash unchanged: same
settings resolution, same data fingerprint. Existing run folders resume. `run.json` gains `noise_sha256`
(absent = no spec); the schema version bumps minor (additive). Hand-built
`with_gains` data keep working through the pipelines exactly as now.

**Behaviour changes for the CHANGELOG.** The residual-correlation check runs on
every pipeline, with or without a spec, so `summary.json`'s check list and
worst status change, and most existing PIONIER runs (f of 42–87 %) would go
from pass to fail at f > 0.6. List it as a behaviour change, or ship it as warn
only (no fail) in 0.4.1 and raise it to fail in 0.5.

**Stability.** Per `AGENTS.md`, new public features start provisional:
`virgil.noise_spec` gets `__provisional__ = ("*",)` and the `ADMONITION`; the
pipelines' `noise_spec` and `bandwidth` settings are listed in
`docs/stability.md` and the CHANGELOG Stability section. The noise spec is
documented in `docs/conventions.md` and the pipeline docs, not on a new page.

**Tests (no heavy fits; synthetic OIData).**

- `validate_noise_spec`: each rejection, unit-confusion guard, round trip.
- `apply_noise_spec` equals the manual `with_error_scale().with_gains()
  .with_closure_offsets()` chain; refuses double application; skips offsets
  below four telescopes with a warning.
- Persistence: run → `load` reproduces the fingerprint; changing one width
  raises `ConfigMismatchError` naming the spec; `processed.oifits` is written for
  specced data and is the pre-spec data.
- `error_scale` with gains (option B): recovers the white scale of synthetic
  data with known gains and white level; reduces to today's result with zero
  widths.
- Checks: `residual_correlation` pass on white noise (many seeds, false-alarm
  rate), warn on synthetic gains, quieter after `apply_noise_spec`; the nine
  post-AGB cases as a regression table (kept as numbers, no data).
- CLI: `--noise-spec` path, inline config, `none` override.

**Implementation order.**

1. **virgil-vlti** (`vlti-noise-spec-emit`): `pairs` emits the spec;
   unit conversions; `bound` and `bandwidth` provenance. No virgil dependency;
   its test validates against a copy of the schema (or against virgil's
   validator when virgil is installed, skipped otherwise).
2. **virgil PR 1**: `virgil.noise_spec` (validate, load, apply), `docs/conventions.md` section (no new page), provisional markings, tests.
3. **virgil PR 2**: pipeline `noise_spec=` setting through `_core`/`_io`, `load`,
   persistence, resume hash, CLI/config, `_unsupported_reason` split. Star,
   Binary and Disk together (they share `_core`), with the evidence-footing
   guard.
4. **virgil PR 3**: `residual_correlation` check in all pipelines (independent of
   the spec; can land first if wanted) and the render/at_bound text.
5. **virgil PR 4**: `error_scale` with a spec (option B, or the disabling
   fallback if B is deferred).
6. **virgil PR 5** (optional): the `bandwidth` pipeline setting.

PR 3's check is the cheapest and most valuable on its own; if time is short for
0.4.1, ship PRs 3 and 2 (with fallback A) and defer B and bandwidth.

## 8. Open questions

- **Flat versus blob GP means.** The post-AGB imaging used both. A flat mean
  puts more of the large-scale structure in the GP, so calibration offsets
  (which look like large-scale amplitude errors) have more room to hide there
  than with a blob mean that already carries the bulk. Does the spec change the
  *ranking* of flat versus blob? Test on the nine targets once the spec is
  available: compare ΔlnZ(flat − blob) with and without it. If the sign flips for
  some targets, the earlier model comparisons need redoing.
- **Whose widths?** Per-target (pair test of the target's own run), per-night,
  or pooled by programme. The nine targets span a factor of ~3 in the gain
  widths; pooling needs a test that it is not worse than per-target.
- **Lower bounds.** Consecutive-frame pairs share calibrators. A spec with
  `bound: "lower"` should perhaps carry a user-set inflation (`gains.inflate`)
  rather than have the pipeline invent one.
- **Closure phases.** Offsets per triangle need four telescopes and are small-angle
  (≲ 0.3 rad). Should the first release carry closure offsets at all, or only
  V² gains plus the closure white scale?
- **Chromatic gains.** The pair test did not fit the (λ_ref/λ)² term; keep the
  field, leave it `null` in emitted specs until measured.
- **Other instruments.** GRAVITY has a different structure (calibrator PCA,
  `gravity_calibrator_pca.md`). The spec's `gains.modes` is deliberately not in v1;
  a v2 can add them without breaking v1 readers.
- **IW Car validation (placeholder).** A validation of this scheme on IW Car is
  running in virgil-validation. Result to add here: χ²/pt with and without the
  spec, ΔlnZ for flat versus blob, the residual-correlation statistics before and
  after, and whether the fitted `s` under option B moves toward 1.
  *Result: pending.*
