# Instrument data and noise treatments in pipelines

Status (2026-10-18): **design, nothing implemented.** Target virgil 0.4.1; it
must not block 0.4.0. Replaces `vlti_noise_in_pipelines.md` (virgil#420), which
specified the same machinery for VLTI only. Counterpart in virgil-vlti:
`design/noise_spec_emission.md`.

## 1. The boundary

virgil's pipelines (`StarPipeline`, `BinaryPipeline`, `DiskPipeline`,
`OrbitPipeline`) are lean and instrument-agnostic. They **never** preprocess
for an instrument or model its noise: no choosing channels, trimming bands,
screening exposures, `insname` branches, or settings that only one instrument
needs. The instrument packages do all of that and hand over two things:

1. **Prepared `OIData`**: the right channel, the usable band, bad exposures left
   out, flags set. The pipeline treats it as the data and runs with its default
   settings.
2. **Optionally, a noise-treatment object** (§2): a measured description of the
   instrument's error structure beyond the quoted σ.

Current owners: virgil-vlti for GRAVITY, PIONIER and MATISSE. CHARA, AMI/JWST
(or ground-based SAM) and others come later as their own packages (or a module
of the instrument's reduction package); none of them changes virgil's
pipelines.

Settings stay about the *analysis* (model, grid, `error_scale`, `n_bands_max`).
`wavel_range` and `error_floor` are generic cuts a user may apply to any data;
an instrument's own values for them belong in the prepared data, not in the
pipeline's call. This was tried the other way once (a coordinator asked the
pipeline to select the GRAVITY SC channel and the 2.05–2.18 µm window) and
rejected: the pipeline would then grow one branch per instrument.

**Test of the boundary.** If a pipeline source file has to import an instrument
package, name an instrument, or read `insname` to decide what to do, the
change belongs in the instrument package. The one allowed use of the
instrument name is a *warning* when data and noise object disagree (§2).

## 2. One interface: the noise spec

### What the pipeline needs

Today the pipelines assume independent errors: the quoted σ per observable,
optionally rescaled by `error_scale`. `OIData` already represents the common
departures from that, and the likelihood already handles them:

- `with_error_scale(...)` (white scale on the quoted errors),
- `with_gains(telescope=, baseline=, chromatic=, modes=)` (achromatic or
  chromatic per-frame gains, marginalized analytically in `virgil.gains`),
- `with_closure_offsets(baseline=, triangle=)`.

So the interface carries *a measured choice of these three*, no more. It is a
**declarative spec**, not code: virgil never imports an instrument package, and
a spec is a file anyone can write by hand. The measurement belongs to the
instrument package; what a spec *does* belongs to virgil.

### The protocol

A noise object is any of:

- a `dict` in the spec format below;
- a path (`str`/`PathLike`) to its JSON;
- an object with `to_noise_spec() -> dict` (so an instrument package can pass
  its own fit result, e.g. virgil-vlti's `Fit`, without a conversion step).

The pipeline calls exactly three functions, new in a small public module
`virgil.noise_spec`, and nothing else on the object:

| Call | Does |
| --- | --- |
| `load_noise_spec(obj) -> dict` | resolves path / `to_noise_spec()` / dict, then `validate_noise_spec` |
| `validate_noise_spec(spec) -> dict` | normalized copy, or `ValueError` listing every problem |
| `apply_noise_spec(data, spec) -> OIData` | the one place a spec acts on data |

`apply_noise_spec` is a pure function of `(data, spec)`. It is the only thing
persistence has to replay (§4).

### Format

```json
{
  "schema": "virgil.noise_spec/1",
  "instrument": "PIONIER",
  "white": {"vis": 0.32, "phi": 0.65},
  "gains": {"telescope": 0.0055, "baseline": 0.0045, "chromatic": null},
  "closure_offsets": {"triangle": 0.0036, "baseline": null},
  "provenance": {"run": "…", "date": "2026-10-04", "bound": "lower"}
}
```

- `white`: multiplicative scales on the *quoted* errors, keyed by `OIData`'s own
  names for `with_error_scale`: `vis` (`d_vis`, whether the data are V² or
  amplitudes), `phi` (closure or absolute phases) and the kinds in
  `observables.KINDS` (`t3amp`, `visamp`, ...). No translation table: a key the
  data lack is left alone, an unknown key is an error. A missing key means 1.
- `gains`: 1σ widths on log |V| (the unit of `with_gains`). `null` leaves a
  group out.
- `closure_offsets`: widths in **radians** (the unit of `with_closure_offsets`).
- `instrument`: free text; used only for the warning below and for humans.
- `provenance`: free-form, must hold `run` and `date`, never interpreted;
  carried into `run.json`, `summary.json` and the quicklook. `bound` is
  `"lower"` or `"estimate"`.
- Units are converted **once, in the emitter** (the instrument package), never
  in virgil, and the emitter's own tests guard against degrees written as
  radians. `validate_noise_spec` refuses only what is invalid for any
  instrument: non-finite or negative widths, a white scale ≤ 0, a closure
  offset above 0.3 rad (the small-angle limit of `with_closure_offsets`) and a
  gain width above 1 on log |V|. It *warns* about widths plausible for one
  instrument but large for most (offset above 0.02 rad, gain width above 0.1),
  naming the likely unit slip; ground-based SAM (about 3° = 0.05 rad) and
  CHARA transfer-function errors (10–15 % on log |V|) must be able to emit their
  measured widths.
- **Versioning.** The schema string carries a major number only. Unknown keys
  are an error at the top level *and inside every group*; only `provenance`
  is free-form. So a new group, or a new key in a group, is a major bump, and
  an older reader refuses the spec naming the version instead of silently
  dropping a term (a missing calibrator-PCA term would make a fit
  overconfident).

### Extending to other instruments

An instrument whose noise needs a term outside these three does not get a
pipeline branch. The route is two steps, each generic:

1. `OIData` (and the likelihood) learns the new term, so any caller can use it.
2. The spec gains a group or key for it under a new major `schema`
   (`gains.modes` for GRAVITY's calibrator PCA, a covariance block for AMI);
   v1 readers refuse a v2 spec naming the version, they do not misread it.

A spec is therefore always something `OIData` can already express, and the
pipeline's code does not change with the instrument.

### `instrument_match`

`instrument_match(data, spec)` warns (never errors) if the record's `insname`
differs from the spec's `instrument`: a PIONIER spec on GRAVITY data is probably
a mistake, but a user may have a reason.

### Widths are fixed, not fitted

A spec fixes the widths. Fitting them (`noise=` sites of `fit`) stays a separate
explicit analysis, not something a pipeline does by default: a width measured
from calibrators or pair tests is more trustworthy than one the fit can trade
against the image, which is what went wrong in the post-AGB imaging (§5).

## 3. The pipeline setting

Every pipeline takes `noise_spec`, not `noise` (`noise=` already means fitted
noise-term priors in `fit`, `numpyro_model` and `Epochs`):

- `None` (default): independent errors, behaviour and hashes unchanged.
- A noise object (dict, path or `to_noise_spec()` object).

Settings resolve as today (defaults, then the JSON config, then keywords). In a
config file the value may be a path (relative to the file) or an inline object;
the **resolved** setting stored in `run.json` is always the full validated spec
inline, so a run folder is self-contained.

CLI: `virgil-pipeline ... --noise-spec spec.json` (`--noise-spec none` overrides
a config). No flags for individual widths: a width not in a spec file has no
provenance.

**Default stays `None`.** A spec is instrument- and target-specific; a default
drawn from one target would be wrong for most (§5).

**Where it applies.** In `_Pipeline.processed` (`_core.py`), after the
selections and before any stage uses the data. The white scale goes first,
then `error_floor` (an absolute floor stays a floor rather than becoming
0.32 × floor), then gains and offsets. Setting both `error_floor` and
`noise_spec` warns that the floor probably double counts. The `load` report
records the spec's provenance and the groups applied.

**Models.** Gains and offsets enter through the likelihood, so Star, Binary,
Disk and Orbit all work; fits with gains use the gradient-based solvers the
pipelines already select, and `log_evidence` takes the fixed widths. Data the
gain machinery does not support (visibility amplitudes, `uv_grid`, extras)
raise at `load` with the existing reason string. A pipeline given data that
already carry gains *and* a spec refuses: `apply_noise_spec` will not stack
them.

**`error_scale` with a spec, in PR 2.** `DiskPipeline`'s default is `model="gp"`
with `error_scale="auto"`, which resolves to `"rescale"`; `imaging.error_scale`
checks nuisance blocks only on its `by_observable=True` path
(`_check_scalable_blocks`), so Disk's `error_scale(model, data, path="disk")`
would silently compute the MacKay `s² = χ²/(N − γ)` on data with gains, which
assumes the whole covariance scales. So the safe behaviour ships *with* the
setting, not after it: with a spec, `"auto"` resolves to `"quoted"`, an explicit
`"rescale"` raises `ValueError` pointing here, and `_check_scalable_blocks`
also covers the single-scale path so no caller can reach the wrong `s`. The
white-part `s` is the later PR (§7).

## 4. Persistence, resume and the summary

Today `processed.oifits` is the data *as fitted*, so gains and offsets, which
OIFITS cannot hold, block it (`_io._unsupported_reason`). The new rule: store
the **pre-spec processed data and the spec**, and rebuild the fitted data by
replay.

- `data/processed.oifits`: the data after selections, *before* the spec; always
  writable for supported data, so the file never carries the nuisance model.
- `data/noise_spec.json`: the validated spec, byte-stable (sorted keys).
- `run.json`: `inputs.noise_sha256` (hash of the canonical spec JSON; `null`
  without a spec) and the spec inline in the resolved settings.
- **Resume / `ConfigMismatchError`.** `data_sha256` (`_input_record`,
  `_check_resume`) is the fingerprint of the *input* data, before `wavel_range`,
  `error_floor` or any spec, so it does not cover the gains. The inline spec in
  the resolved settings is compared like any other setting, but `noise_sha256`
  must be compared explicitly as well: it is the guard that still works when the
  settings are stored differently, and it gives the message "the noise spec
  differs" rather than a long settings diff. Skipping it would let a run resume
  with different widths and fit later stages under another covariance. Changing
  the spec invalidates every stage from `load`.
- **`summary.json`** gains a shared `noise` section, written by `_write_summary`
  in `_core.py` so no pipeline implements it: `{schema, instrument, sha256,
  applied (groups), provenance}`, or `null`. The quicklook shows it and the
  compare helpers refuse to rank runs whose `sha256` differ (§6).
- **Replay lives in `Result.data()`**, which promises the data actually fitted.
  It reads `processed.oifits` and replays `apply_noise_spec` with the stored
  spec. `data_sha256` is of the raw input, so the `load` stage records what
  replay needs:
  - `inputs.processed_sha256`, taken on the `OIData` *re-read from
    `processed.oifits`* (so the OIFITS round trip is part of the hash), checked
    exactly: an error on failure, as is `noise_sha256`;
  - `data/noise_replay.json`, **numbers, not a hash**: the shape and the
    Frobenius norm of each applied gain-mode matrix and of its Gram matrix
    `G Gᵀ`. Replay compares them with `allclose`, and a mismatch is a warning:
    the modes come from JAX, whose last bits change with version, backend and
    precision, so a digest of them would always fail.
- **`_unsupported_reason`**: the gains/offsets branch leaves the writer check
  (the writer only sees pre-spec data) but still triggers for a hand-built
  `with_gains(...)` object passed straight in, exactly as today.

## 5. Worked examples

### VLTI (main example): PIONIER now, GRAVITY and MATISSE next

virgil-vlti already prepares VLTI data: it picks the instrument channel, the
band window and screens exposures. For PIONIER it also *measures* the error
structure. The pair test (virgil-vlti `design/pionier_systematics.md`; IW Car,
ESO 1103.D-0797, 11 disjoint pairs of consecutive frames) finds only about
0.32σ of the quoted V² error is white; the rest is an achromatic gain per frame
(widths on V² of 1.1 % per telescope and 0.9 % per baseline, so 0.55 % and
0.45 % on log |V|, the unit of the spec). Closure-phase errors are about 1.5×
too large (white scale 0.65), with an offset of about 0.21° per frame and
triangle. These are lower bounds, since consecutive frames share calibrators.
The nine post-AGB PIONIER discs of virgil-validation
(`scripts/postagb_imaging.py`) show the cost of ignoring this: 42–87 % of the
V² residual variance sat in each (frame, baseline) channel mean (17 % expected
for white noise), and with one MacKay scale the GP fits absorbed the
calibration offsets into the image (χ²/pt about 0.6, hyperparameters at the grid
corners, ΔlnZ inflated). Widths differ between targets (white V² scale
0.24–0.61, telescope gain 0.46–1.7 %), so the spec is measured per target or
per night. virgil-vlti emits the spec (halving the V² widths to log |V| and
converting degrees to radians once), `virgil-pipeline --noise-spec` or
`noise_spec=` carries it in. GRAVITY adds a calibrator-PCA term (a v2
`gains.modes` group, §2) and MATISSE its own; none needs a pipeline change.

### CHARA

A CHARA package (a reduction or calibration module for the array's beam
combiners, e.g. MIRC-X, MYSTIC or PAVO) would do the equivalent of virgil-vlti:
select the combiner and spectral channels, drop exposures with poor seeing or
low coherence, and flag them in the `OIData`. Its calibrator-to-calibrator
scatter measures the per-night transfer-function error, which is again a
per-frame achromatic gain on V² (the `telescope` and `baseline` widths) and
a white scale on the quoted errors, so it emits the same `virgil.noise_spec/1`
with `instrument: "MIRC-X"` and its own `provenance`. Its calibration-limited
V² floors enter as `white`; `error_floor` stays an analysis choice. Nothing in
the pipeline knows the instrument is CHARA.

### AMI / JWST (and ground-based SAM)

An AMI package prepares the calibrated OIFITS or `OIData` for the kernel-phase
or closure-phase data (the filter, the integration and the target–calibrator
pair), screens bad integrations and sets the flags. Its dominant correlated
errors are different: closure-phase and V² errors are correlated through
shared baselines and a common detector systematic, and calibrator-to-target
mismatch is a per-observation offset. The offset part is the existing
`closure_offsets` and `gains` groups. A full covariance block is a **v2 group**:
`OIData` learns to carry it (it has `vis_mat`/`phi_mat` projections and the
`closure_noise.md` covariance machinery), then the spec names it. Until
then the AMI package emits what v1 can say (white scale, offsets) or
nothing, and the pipeline runs unchanged. AMICAL covers ground-based SAM
reduction only, never JWST.

Science targets in this note are examples only.

## 6. The job-script pattern

Two steps. The first uses the instrument package; the second is the pipeline
with **no instrument settings**.

The reference is the Gl 229 `OrbitPipeline` release gate in
[ozstar_scripts#33](https://github.com/benjaminpope/ozstar_scripts/pull/33)
(`scripts/orbit_aliases`):

1. `prepare_gl229.py` is the instrument step, on the virgil-vlti side. It reads
   the GRAVITY nights with `insname="GRAVITY_SC"`, flags V² and closure phases
   outside 2.05–2.18 µm (`virgil_vlti.calibrate.channel_window`), leaves out one
   screened exposure with its stated reason, and returns `{night: OIData}`
   and a record of what it did (`insname`, window, exclusions, file counts).
2. `run_pipeline.py` is the analysis step:
   `OrbitPipeline(data, output=run).run()`, no settings. With a noise object it
   becomes `OrbitPipeline(data, output=run, noise_spec=spec).run()`, still no
   instrument setting.

A job script that needs an instrument fact in step two is wrong; move that fact
into step one. Step one's record is the part of the run that the pipeline
cannot see (open question below).

## 7. Open questions

- **Prepared-data record.** Step one's record (window, exclusions) lives
  beside the run, not in `run.json`; `data_sha256` proves identity of the input but not
  meaning. Should the pipelines accept an optional free-form `prepared_by`
  dict (like `provenance`) stored in `run.json` and `summary.json`, never
  interpreted? Cheap; decide in the PR that adds `noise_spec`.
- **Whose widths?** Per target, per night or pooled by programme. The post-AGB
  targets span a factor of about 3 in gain width; pooling needs a test that it
  is not worse than per target.
- **Lower bounds.** Consecutive-frame pairs share calibrators. A spec with
  `bound: "lower"` could carry a user-set inflation (`gains.inflate`) rather
  than have the pipeline invent one.
- **`error_scale` with a spec.** Recommended: `"rescale"` estimates one extra
  factor `s` on the white part only, covariance `C(s) = s² W + G Gᵀ` with the
  gain term unscaled. The MacKay fixed point `s² = χ²/(N − γ)` assumes the
  whole covariance scales, so `s` is found by a bracketed 1-D maximization of
  `log_evidence`, and is reported beside, never multiplied into, the spec's
  white scale (two numbers: a measured instrument property and a per-target
  model-adequacy factor). `imaging.error_scale` must be generalized so the
  nuisance block does not scale; until then PR 2 refuses `"rescale"` with a
  spec (§3).
  `"fit"` fits scales on the white part with the gains fixed, starting at 1
  (never at the spec's scale, which would apply it twice); gain widths are
  never fitted by a pipeline. The refusal is what PR 2 ships.
- **Residual correlation check.** A pure function in `_checks.py`: the
  fraction of V² residual variance in each (frame, baseline) channel mean,
  against 1/n_ch for white noise, and the adjacent-channel correlation.
  Instrument-agnostic and useful *without* a spec (it tells a user they need
  one; with one it should drop). Warn only in 0.4.1 (most existing PIONIER runs
  would fail a hard threshold), fail from 0.5. Data without the
  (frame, baseline, channel) layout skip it with the reason.
- **Evidence footing.** Within a run the spec is fixed, so every grid point and
  model sees one covariance. Across runs, evidence is comparable only with the
  same `noise_sha256`; a run without a spec never compares with one with a
  spec.
- **Closure offsets.** They need four telescopes and are small-angle (≲ 0.3 rad).
  Should the first release carry them, or only V² gains plus the closure white
  scale?
- **Bandwidth smearing** (PIONIER R≈40, a few percent in V² at long baselines)
  is a deterministic, static bias separate from the random gains, so not part
  of the spec. A pipeline `bandwidth` setting could require data read with
  `bandwidth=True`; the global default stays off in 0.4.1 (it would change
  every `data_sha256`). Decide for 0.5.
- **Flat versus blob GP means.** Does a spec change the ranking? Test on the
  nine post-AGB targets once a spec exists: ΔlnZ(flat − blob) with and
  without it.
- **Validation placeholder.** The IW Car run in virgil-validation (χ²/pt with
  and without, ΔlnZ, residual statistics, whether `s` moves toward 1). *Result:
  pending.*

## 8. 0.4.1 implementation order

None of this is in 0.4.0.

1. **virgil-vlti** (`vlti-noise-spec-emit`): `pairs` emits the spec,
   unit conversions, `bound` provenance, a `to_noise_spec()` on the fit result.
   No virgil dependency; its test validates against a copy of the schema (or
   virgil's validator when installed).
2. **virgil PR 1**: `virgil.noise_spec` (`validate`, `load`, `apply`,
   `instrument_match`), `docs/conventions.md` section, `__provisional__` marking
   and tests. Stability per `AGENTS.md`: new public features start provisional.
3. **virgil PR 2**: the `noise_spec` setting through `_core`/`_io`: `processed`,
   persistence, `noise_sha256`, resume message, the summary `noise` section,
   `--noise-spec`, the `_unsupported_reason` split, and the `error_scale`
   guard of §3 (`"auto"` → `"quoted"`, `"rescale"` refused, the check on the
   single-scale path). Star, Binary and Disk together (they share `_core`);
   OrbitPipeline takes it when it lands.
4. **virgil PR 3**: the residual-correlation check (independent of the spec;
   can land first) and the render/`at_bound` text.
5. **virgil PR 4**: `error_scale="rescale"` on the white part with a spec.
6. **virgil PR 5** (optional): the `bandwidth` setting.

PR 3 is cheapest and most valuable alone; if time is short, ship PRs 3 and 2
and defer the rest.

**Tests** (synthetic `OIData`, no heavy fits): each `validate` rejection and the
unit guard; `apply` equals the manual `with_error_scale().with_gains()
.with_closure_offsets()` chain and refuses double application; a run reproduces
its `processed_sha256` and replay numbers after `load`, and changing one width raises `ConfigMismatchError`
naming the spec; `processed.oifits` is written for specced data and is the
pre-spec data; `summary.json` has the `noise` section; the render path cannot
drop the spec; `DiskPipeline(data, noise_spec=spec)` with default settings does
not reach MacKay `rescale`, and explicit `"rescale"` raises; a spec with `phi`
or `t3amp` keys scales the right observable; a pipeline test with a stub instrument (a plain dict, no
instrument package) shows the pipelines run identically whatever `instrument`
says.

**Back-compat.** `noise_spec=None` leaves fits and hashes unchanged; existing
run folders resume. `run.json` gains `inputs.noise_sha256` and `inputs.processed_sha256` (absent = no spec or an older run) and the data folder gains `noise_replay.json`,
a minor, additive schema bump.
