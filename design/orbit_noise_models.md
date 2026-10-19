# Instrument noise models in the orbit search

Status (2026-10-20): **design for review, nothing implemented.** Extends
`instrument_noise_in_pipelines.md` (the `noise_spec` interface, target 0.4.1)
to `OrbitPipeline` and the engine under it (`epochs.epoch_positions`,
`aliases.fit_orbit_aliases`). It replaces nothing in `orbit_pipeline.md`
except row 1g ("hierarchical gains"), which this note supersedes.

## Why

The orbit search treats quoted errors as white, with a free error scale per
block (`epochs._Surface`) and a tempering factor `f` for correlated channels
(`virgil.tempering`). Spectro-interferometric data break that assumption: when
calibration errors are a gain per frame, common to all channels, `f`
approaches the number of channels per (frame, baseline). Three things then
fail together:
- the per-night peak gaps are overconfident by about `f`;
- unmodelled gains move the per-night peaks;
- the night flags, applied to the untempered gaps, do not fire.

The search then seeds from sidelobes, and tempering by `1/f` flattens the
evidence until no orbit stands out.

**Worked example (virgil-validation#147, GRAVITY).** HD 9312 has raw χ²/N of
22–104 on every night and `f` = 854. On 5 of 10 nights the best peak is a
north–south sidelobe, and no flag fires. The orbit beats a single star by
2.4 nats, and the published period ranks 30th of 200 bands.

virgil already has the right likelihood term: `OIData.with_gains` and
`with_closure_offsets`, which `virgil.gains` marginalizes analytically.
virgil-vlti already measures the widths: the pair test (`pairs.pair_test`) and
the calibrator-PCA templates (`templates.template_modes`). The orbit search
uses neither, because `_Surface` refuses data with gains
(`epochs._check_marginal_data`).

## 1. API boundary

**No new `noise=` argument on the engine.** The noise model travels *in the
data*, as everywhere else in virgil. `epoch_positions` and `fit_orbit_aliases`
accept an `OIData` (or a list of epochs) on which `with_gains` and
`with_closure_offsets` have already been called; §2 is what lets them. A
user of the engine applies the widths themselves.

**`OrbitPipeline` takes `noise_spec=`**, exactly as Star, Binary and Disk do
in `instrument_noise_in_pipelines.md` §3: a `virgil.noise_spec/1` dict, a path
to its JSON, or an object with `to_noise_spec()`. `_Pipeline.processed` calls
`apply_noise_spec` once per epoch, and persistence, `noise_sha256` and the
summary `noise` section are shared through `_core`. The orbit pipeline adds one
thing: multi-night data. v1 applies **one spec to every epoch** (widths pooled
over the nights, as the pair test measures them). A mapping `{epoch: spec}` is
an open question (Q3).

**Per-datum modes** (the calibrator-PCA templates) are not in spec v1. They
reach virgil as the `virgil.modes/1` OIFITS tables that virgil-vlti's
architecture review (§2, 2026-10-19) proposes, read into `with_gains(modes=)`
and `with_closure_offsets(modes=)`, and are named by spec v2 `gains.modes`
(`instrument_noise_in_pipelines.md` §2, "Extending"). Until that reader exists,
GRAVITY uses spec v1 widths (§5).

| Piece | Lives in |
| --- | --- |
| Gains, offsets, analytic marginalization | virgil (`gains`, `_linear`); exists |
| `noise_spec` load, validate, apply | virgil `noise_spec` (0.4.1 PR 1, planned) |
| Gain-aware search surface, flags (§2, §4) | virgil `epochs`, `aliases` (new) |
| `OrbitPipeline(noise_spec=)` | virgil `pipeline/orbit.py` via `_core` (0.4.1 PR 2) |
| Pair test, spec emission | virgil-vlti `pairs`, `noise_spec_emission.md` (planned) |
| Calibrator-PCA templates, modes OIFITS | virgil-vlti `calpca`, `templates`; HIGH in virgil-vlti#40 |
| Per-system specs, residual pair tests, gallery runs | virgil-validation |

virgil never imports virgil-vlti. The boundary test of
`instrument_noise_in_pipelines.md` §1 holds: no `insname` branch in the
engine.

## 2. Likelihood

**Analytically marginalized, never sampled.** A GRAVITY night has tens of
frames × (4 telescope + 6 baseline) gains: hundreds of nuisance parameters per
night, thousands per orbit. `virgil.gains` integrates them out as a low-rank
covariance `C = D + U Uᵀ`, block-diagonal by frame, with `U = J τ m` and
`J = 2 V²_model`. Widths are fixed by the spec, not fitted
(`instrument_noise_in_pipelines.md` §2).

**The search surface with gains.** `_Surface.score` integrates a scale `s` per
block out of a likelihood whose covariance is `s² D`. With gains, the proposal
is to scale the **whole** covariance, `C(s) = s² (D + U Uᵀ)`. Then the score
keeps its closed form, applied to the whitened χ², plus one log-determinant
that does not depend on `s`:

    m = −Σ_b (ν_b/2) ln χ²_{w,b} − ½ Σ_b ln det(D + U Uᵀ)_b ,

or the same `ln s` integral with `s_max`. `_linear.whiten_blocks` already
returns both the whitened residuals and the half-log-determinant. The surface
keeps the property that makes it a search surface: an overall error
misquotation still cancels from peak gaps. A white-only scale,
`C(s) = s² D + U Uᵀ`, is the more physical choice (`instrument_noise` §7,
"`error_scale` with a spec"). It loses the closed form, and each node of the
`ln s` quadrature needs its own whitening. It is Q1.

**Where `J` comes from.**
- In the **per-night grid scans** (`epoch_positions`, the grid stage), take `J`
  from the observed V², clipped at zero. `C` is then fixed per night, its
  whitening is precomputed once, and the grid's cost is unchanged apart from
  one `O(n·k)` product per point (k = number of modes per frame).
- In **refinement and the alias-band fits**, take `J` from the model, as
  `virgil.gains` does (Lachaume 2021), so that the covariance does not depend
  on the noisy data. Each evaluation re-whitens: about 10 rank-one steps per
  frame for spec v1, and up to about 40 with PCA modes.

**Joint alias-band fits.** `aliases._loglike` sums the gain-aware per-epoch
score. In the Gauss–Newton kernel, the IRLS rows become the whitened rows. The
log-determinant's curvature is left out of the step, which is safe because a
step is kept only if the log likelihood rises. The Laplace Hessian is the
exact JAX Hessian, so it includes the log-determinant. Rough cost: 3–10× per
likelihood evaluation on GRAVITY MEDIUM. It is to be measured on OzSTAR before
anything else is built; a 200-band GRAVITY run (the worked example) is the benchmark.

**Per-night V² amplitude (`vis_amp`).** Keep it. It covers night-level losses
that the gain prior (about 1 %) does not: anisoplanatism
(`gravity.anisoplanatic_v2_loss`), fibre response and smearing. The variable
projection becomes `A = (m̃ᵀ õ)/(m̃ᵀ m̃)` on whitened model and data. `J` is
evaluated on the unscaled model, an error second order in the gain widths. `A`
and the mean of a night's telescope gains are partly degenerate, but `A` is
free in (0.5, 1.2) while the gains carry a 1 % prior, so `A` takes the mean.

**UD primary.** The model is unchanged. Gains act on the UD model through `J`.
One degeneracy matters: a `chromatic` gain, (λ_ref/λ)² per baseline, has the
shape of a small disk's V² deficit at one baseline. So when
`primary_diam="auto"` or `"fit"`, `chromatic` is used only if the spec
measured it, and `estimate_primary_diameter` and `RESOLVE_NATS` run on the
gain-aware score.

**Closure offsets** enter the phase blocks in the same way
(`with_closure_offsets`, small-angle). With the PIONIER widths (0.21°) they
barely change positions. They are kept because the pair test measured them.

## 3. Tempering

Tempering stays as the safety net, but it is estimated *after* the noise
model. The adjacent-channel estimator in `virgil.tempering` runs on residuals
whitened by `C`, so that a gain the spec explains is not counted twice. If the
spec is adequate, `f` should fall from hundreds to about 1–2. Tempering then
covers only what the spec misses (chromatic structure, wavelength-calibration
ripple).

**Adequacy checks**, all reported per night in the summary, none per system:
1. **χ²/N on the specced covariance** (whitened, raw: not rescaled, per
   `report-raw-chi2`) and the profiled `ŝ`. A spec is adequate when `ŝ` is
   within a factor of 2 of 1 on most nights.
2. **`f` within `FLAG_RANGE`** (0.5–2) on the whitened residuals. The existing
   `tempering` flag already says so.
3. **Residual-correlation check** (`instrument_noise` §7, PR 3): the fraction of
   residual variance in each (frame, baseline) channel mean against 1/n_ch,
   computed on whitened residuals. It applies without a spec as well, and then
   says that one is needed.
4. **Pair test on the orbit residuals** (virgil-validation, using virgil-vlti
   `pair_test` on data − best model): the widths it finds must be consistent
   with the spec's. An excess means the spec is too narrow for that target.

## 4. Peak picking

**Is a noise model expected to remove sidelobe picks?** Partly. Three effects
are mixed up in them:
- The gaps are inflated by about `f`. A per-frame gain that the model does not
  include makes a sidelobe look decisive by tens to hundreds of nats. Whitening
  by `C` fixes this, and so the `alias-peaks` and `low-gap` flags fire.
- Unmodelled gains bias the peak heights. Smooth V²(B) offsets per frame are
  fitted by moving the companion. Marginalizing the gains leaves the position
  to the within-frame chromatic modulation and to the closure phases, which is
  what tells sidelobes apart.
- Coverage. A single night's uv tracks leave a real north–south ambiguity at
  about λ/B. Companions within a few λ/B_max are in this regime (the worked example: ≲ 5.5 mas, λ/B_max ≈ 3.5 mas), and
  no noise model removes it.

So the expectation is that the spec turns wrong-but-confident nights into
flagged ones. It is not expected to make every night pick the right peak.

**The safeguard, independent of any noise model:**
1. **Tempered night flags.** `epoch_positions` applies `flag_alias_nats` and
   `flag_min_gap` to the gap divided by the night's `f`, as the short-arc cut
   already does (`docs/orbit_fitting.md`, "Short arcs"). Without a spec, this
   alone would have flagged the worked example's ambiguous nights.
2. **Cross-night consistency flag `sep-outlier`.** A night's best separation
   above `k ×` the median of the other unflagged nights (relative, `k` about 2,
   and no prior from any published orbit) *and* with a rival peak inside that
   bound is flagged. The flag names the rival.
3. **Keep the competing peaks.** Flagged nights already seed nothing but still
   count in every band's likelihood. Each band fit is direct to the data, so
   the band decides between a flagged night's peaks. The change: after the
   fit, report for each flagged night which peak the winning orbit passes
   through (`night_resolved_to`). A band whose orbit goes through rivals on
   most flagged nights is visible, not hidden.
4. **Seed floor.** When flagging leaves fewer than two seeding nights, the
   engine falls back to random candidates (existing behaviour). It must never
   seed from a flagged night.

## 5. Instruments

| Data | Noise source | Status |
| --- | --- | --- |
| PIONIER | pair test → spec v1: white V²/CP scales, `telescope`/`baseline` gains, triangle offsets | pair test exists; emission planned |
| GRAVITY MEDIUM | pair test → spec v1 now; calibrator-PCA modes (`gravity_single_medium_combined_2023-2025`) via spec v2 later | template ships; modes reader missing |
| GRAVITY HIGH | the same; the HIGH template is in virgil-vlti#40 (AT only) | open PR |
| GRAVITY (UT, other configurations) | pair test only; `Template.mismatches` (strict) refuses a mismatched template | — |
| No model | today's behaviour: white errors + scale + tempering | default |

Templates and spec gains must not stack. A PCA template already splits
telescope and baseline terms, so with `gains.modes` the `telescope` and
`baseline` groups stay `null`. `validate_noise_spec` should reject the
combination (architecture review §4a: one shared width for external modes
double-counts).

**Without a model**, the pipeline runs as today and says so. The summary
records `noise: null`. When `f > 2`, or when the residual-correlation check
fails, the warning says "correlated noise: supply a noise spec", instead of
only `tempering`.

## 6. Validation plan

Laptop: synthetic unit tests only (seconds, `OMP_NUM_THREADS=2`). All real
data and grids run on OzSTAR (one job array, GPU nodes).

1. **Unit (virgil).** (a) Widths → 0 reproduces the current `_Surface` score
   exactly. (b) On a synthetic binary with injected gains (`OIData.with_model`
   with `key` draws them), the gain-aware gap is within Monte-Carlo scatter of
   the truth, while the white gap is inflated by about n_ch. (c) `f` on
   whitened residuals is within 5 % of 1. (d) A `vis_amp` and gains round trip.
   (e) `sep-outlier` and the tempered flags fire on a two-peak synthetic night.
2. **Release gate (worked example: Gl 229).** A cold alias search at the release commit, run
   with and without a spec, must still rank N = 34 (P = 12.134 d) first with
   Δlog Z > 10 over the runner-up.
3. **A failing system, blind (worked example: HD 9312).** A 2 × 2 ablation: spec on/off × safeguard on/off. Pass:
   36.5 d is the top band, and the orbit beats both a single star and free
   positions decisively. Blind starts only, no published period or positions.
   Record which change does the work.
4. **No regressions** on the other 11 gallery systems: the same top band (or a
   better one against the reference), with χ²/N and `f` reported before and
   after.
5. **Injection–recovery (p-value calibration).** Inject orbits on the failing system's and
   two other systems' (u, v, t) sampling. Draw the noise from the spec
   (`with_model(key=)`) and, separately, from virgil-vlti's resampled-residual
   nulls (`nulls.resampled_null_datasets`, virgil-vlti#39), so that a spec is
   not checked only against itself. Calibrate the band `p`, `alias_ambiguity`
   and the `no-orbit` odds. Null injections set the false-alarm rates.
6. **Cost.** Wall time per band with and without the spec, from the failing-system
   run. Decide from it whether the grid stage needs the fixed-`C` shortcut of
   §2.

## 7. Open questions for Ben

- **Q1. Scale model.** Should the scale be common, `s²(D + UUᵀ)` (closed form,
  cheap, the recommendation for the search surface), or white-only,
  `s²D + UUᵀ` (more physical; quadrature costs about 10× more)? Final
  evidences could use white-only even if the search uses common.
- **Q2. Order of work.** Should the §4 safeguard (tempered flags,
  `sep-outlier`) ship first and alone? It needs no spec and is cheap, and it
  may be enough for the worked example.
- **Q3. Per-night specs.** Should v1 use one spec per orbit run (pooled
  widths), or should `noise_spec` accept `{epoch: spec}`, since years-apart
  nights have different calibrators and conditions?
- **Q4. Data-derived widths without virgil-vlti.** For instruments with no
  package, should the pipeline offer an explicit, opt-in self-spec, with a
  per-(frame, baseline) gain width estimated from the residual channel means?
  It breaks "widths are measured, not fitted", so the default would stay off.
- **Q5. GRAVITY now.** Should GRAVITY use pair-test spec v1 until the modes
  reader and spec v2 land, or should the orbit work wait for the calibrator-PCA
  templates (HIGH is not merged)?
- **Q6. Release target.** Should this be 0.4.1 alongside the `noise_spec` PRs,
  or 0.5? The surface change touches `epochs` and `aliases`, both provisional.
- **Q7. `vis_amp` with a spec.** Should `vis_amp` stay on by default, or go off
  when the spec has telescope gains? It overlaps them, and leaving it on costs
  one profiled parameter per night.
