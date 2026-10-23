# Orbit fits with calibrated data and virgil-vlti noise models

Status (2026-10-20): design reviewed and decided (§6), partly implemented. Scope: virgil
`OrbitPipeline`/`fit_orbit_aliases` and the virgil-vlti data path into them. Instrument noise
modelling stays in virgil-vlti; virgil only consumes the data and modes it hands over. Worked
examples: a GRAVITY orbit gallery (OzSTAR 18821038/18826461/18826486, virgil e3e1f2d, results
in virgil-validation#147), HD 9312 as the failing system, and Gl 229 as the release gate (through 0.4.1; retired 2026-10-23).

## 0. Status on main

| Item | Section | State |
| --- | --- | --- |
| Gains and closure offsets on the marginal surface (R1) | §3.2.1 | done: virgil#437, node test virgil#442 |
| `vis_amp` off with gains | §3.2.2 | done: virgil#437 |
| f > 3 with gains reported as "noise model inadequate" | §3.2.3 | done: virgil#437 |
| Tempering on conditional residuals | §3.2.3 | open |
| `orbit_search` marginalizes gains | §3.2.4 | open |
| Implausible-peak flag, kept peaks | §3.2.5 | open (the short-arc sign gate is tempered by f since virgil#435) |
| Loader from calibrated files and modes to `OIData` | §3.1.4 | done: virgil-vlti#43 (`to_oidata`) |
| HIGH calibrator-PCA template, calibration fixes | §3.1.2 | done: virgil-vlti#40, #42 |
| Calibrators found, downloaded and applied | §3.1.1–3 | in progress (virgil-validation) |
| Rerun and validation | §4 | open |

## 1. What went wrong (checked against code and data)

1. **Uncalibrated GRAVITY data.** ESO's `*_SCI_VIS` products (PRO CATG `SINGLE_SCI_VIS` or
   `DUAL_SCI_VIS`) are reduced, *not* calibrated: their V² carries the transfer function
   T(frame, baseline, λ), a multiplicative bias that a free error scale cannot remove. The
   worked example's gallery fitted V² from such files, with one V² amplitude per night
   (`vis_amp`) as the only correction. One scalar per night cannot absorb a transfer function
   that changes per frame, baseline and channel. This is the dominant cause of HD 9312's χ²/N of
   22–104 per night and its tempering factor of 1/854. (Gl 229's data are `DUAL_SCI_VIS`
   MEDIUM: also uncalibrated. They pass the gate regardless; §4.)
2. **HIGH resolution, unbinned.** HIGH data (about 1,400 channels per baseline in K) were fitted
   at native resolution. Channel-correlated residuals then dominate, and the compiled programs
   are large. The CPU "LLVM … Unable to allocate section memory" aborts seen on three systems
   are plausibly this; not confirmed.
3. **The orbit engine refused the noise machinery** (fixed by virgil#437). virgil marginalizes
   channel-shared calibration gains analytically (`virgil.gains`; `OIData.with_gains`,
   `with_closure_offsets`; the exact log-determinant in `likelihood._whiten_with_log_norm`).
   Until virgil#437, `epochs._Surface` refused data with gains or closure offsets, because its
   analytic error-scale marginalization assumes C = s²D. It now integrates the scale
   numerically for them (§3.2.1). It still refuses extra observables with a model-dependent
   covariance (checked in `_Surface.__init__`), which matters if |V| or T3AMP blocks are ever loaded.
   Two inconsistencies remain:
   - the `vis_amp` profile rebuilds the model as `obs + r·err`, valid only for diagonal errors
     (now refused with gains, §3.2.2);
   - `orbit_search` profiles gains (one dof each) where the alias engine marginalizes them
     (§3.2.4).
4. **virgil-vlti does the data side.**
   - `virgil-vlti-calibrate` calibrates `*_SCI_VIS` with same-night `*_CAL_VIS`, validated
     against ESO's `gravity_viscal` (V² 0.12%, closure phases 0.06°).
   - It windows (`--window`) and bins (`--bin`).
   - It writes calibration-template modes beside each file (`--template`, with
     `--tf-error propagated` so that nothing is counted twice) and calibrator-diameter modes
     (`--cal`/`--diameter`, the `rel-` keys for `with_gains(modes=…)`).
   - It screens calibrators (`calpca.screen_calibrators`).
   - It ships GRAVITY single-field MEDIUM and HIGH templates (virgil-vlti#40).
   - The pair test (`pairs`, `templates.gain_template_from_pairs`) gives PIONIER gain widths.
   - `to_oidata` (virgil-vlti#43) turns calibrated files and their modes into a virgil
     `OIData` with gains, replacing the tutorial `stack_modes`.
5. **Calibrators are only partly found.** A same-programme search covered 6 of the worked
   example's 11 GRAVITY systems (one of them for half its files). Many candidates are
   themselves programme binaries, which are unusable as calibrators. The per-system inventory
   is in virgil-validation's calibrator manifest, not here.

## 2. Physics the design rests on

- **Calibration first, nuisances second.** Calibrated V² = V²_true × (1 + residual gain). The
  residual is small (GRAVITY MEDIUM template: PCs of 1.9%, 0.52%, 0.43%, 0.29% per channel in V²,
  41–81% telescope-based; closure phases 0.33°, 0.25°; PIONIER pair test 0.55%/telescope and
  0.45%/baseline on |V|) and shared by a frame's channels. Marginalizing it is what turns 1,400
  channels into the few numbers they really are.
- **What V² still says once gains are marginalized.** Per-frame grey gains absorb the grey part
  of V². A binary's V² signal survives through its chromatic modulation and its pattern between
  baselines. In the worked example (separation ≤ 5.5 mas; λ/B ≈ 3.5 mas at 2.2 µm on 130 m)
  that is about 0.2 fringe cycles across K, so most position information comes from closure
  phases. This is the honest information content. The evidences will be weaker than white
  noise claims and far stronger than tempering by 1/854.
- **Sidelobes are real ambiguities, not noise.** A single night of AT snapshots has elongated
  (u, v) coverage, so its χ² surface has sidelobes along the poorly sampled direction. The
  worked example's wrong nights are off in Dec by 3–15 mas: that fits. A right noise model sets
  the relative heights of the peaks correctly but does not remove them. The orbit must choose
  between them: keep several peaks per night and flag implausible ones (§3.2.5).
- **Binning HIGH by 8 channels costs nothing for a continuum binary.** About 1,400 channels over
  2.05–2.38 µm (or K, about 0.45 µm) are 2.4–3.2e-4 µm each, so an 8-channel bin is about
  2–2.6e-3 µm: resolving power λ/Δλ ≈ 850–1,100, MEDIUM-like. For s = 15 mas, B = 130 m (about
  4.3 fringe cycles at 2.2 µm) the phase change across one bin is 0.004–0.005 cycle.
  Keep the Brγ window (2.155–2.175 µm) out.
- **Primary diameter.** Keep the UD primary (virgil 1c). Calibrator-diameter uncertainties enter
  as modes from virgil-vlti, not as inflated errors.

## 3. Design

### 3.1 Data (virgil-vlti; no noise code in the analysis repository)

1. **Calibrators.** Search the archive per system (ESO TAP: the same programme and night first,
   then other programmes on the same night and configuration). Exclude every science target of
   the analysis and any star with known companions, and prefer `STD`/`*_CAL_VIS` products.
2. **Calibrate** each night with `virgil-vlti-calibrate --tf-error propagated --template <the
   HIGH or MEDIUM template matching the setup> --bin 8 --window 2.05 2.38` (`--bin 1` for
   MEDIUM), plus calibrator diameters (`--cal`/`--diameter`, so diameter modes are written).
   Screen with `calpca.screen_calibrators`.
3. **Nights without a usable calibrator: closure phases only**, with the template's closure-phase
   modes as closure offsets. V² from `*_SCI_VIS` is never used.
4. **One loader** (virgil-vlti `to_oidata`, done): it reads the calibrated files, applies
   virgil's selection, then sets `with_gains(modes=…)` and `with_closure_offsets(modes=…)` from
   the modes beside each file, aligned sample for sample after selection. Its tests assert exact
   alignment, unchanged template widths, and that a zero-mode load equals plain `OIData`.
5. **PIONIER**: files are already calibrated. Gain widths come from the pair test
   (`gain_template_from_pairs`) through the same loader.

### 3.2 virgil (OrbitPipeline and alias engine; virgil never imports virgil-vlti)

1. **Score data with gains on the marginal surface (R1, done).** Per block,
   C(s) = s²D + UUᵀ (U from the gains or offsets), integrated over ln s with the Jeffreys prior
   on [−ln s_max, ln s_max] (`_NUMERIC_S_MAX` when the caller gives none). R2 was rejected
   (quoted scale s = 1 with the template's overall width fitted: cheaper, but it trusts the
   propagated errors).
   - **Quadrature: a fixed 32-node Gauss–Legendre rule on a peak-centred window**
     (Newton-centred, ±9 curvature-σ scaled by dof, inside the Jeffreys range). A rule over the
     whole range would undersample the peak, whose width in ln s is about 1/√(2ν). Tests
     (`tests/test_marginal_gains.py`, virgil#442):
     - the absolute score matches a 4001-point dense integral with a Cholesky log det (gains on
       V² only, default nodes) to 1e-4;
     - differences between two models agree with dense per-block integrals (2001-point
       trapezoid, vis + phi + gains + offsets) to about 1e-3 at 16 nodes and about 1e-9 at 32,
       asserted at 0.01 and 1e-3. The reference itself is good to about 1e-4, so these are
       agreements of differences, not the absolute accuracy of log L.
     32 is kept because 16 misses the zero-width-gains regression (match to the plain surface at
     1e-6) by about 2e-3.
   - **Optional speed-up: a Woodbury integrand.** Whitened by D, C(s) becomes s²I + ŨŨᵀ with
     Ũ = D^{-½}U (`gains.GainModes._columns`), of K columns. Per node, a Cholesky of the K×K
     matrix s²I + ŨᵀŨ gives χ²(s) and ln det C(s) exactly in O(K³), and stays differentiable
     where an eigendecomposition does not: SVD/eigh gradients divide by λ_i − λ_j, so they are
     NaN at repeated singular values (zero widths, equal widths on symmetric baselines).
     Two constraints:
     - Ũ depends on the model (the columns carry dObs/dlog|V|), so it is rebuilt at every
       likelihood evaluation and gradients flow through it.
     - K is small only per *frame*, when there are no `spanning` modes. With spanning modes K is
       every mode in the block (frames × modes), and the K×K work can exceed the 32 whitenings it
       replaces.
     Worth doing only if profiling shows the per-node whitening dominates.
2. **`vis_amp` is off for data with gains (done).** Telescope and baseline gains supersede the
   per-night amplitude, whose profile also assumes diagonal errors. The default `vis_amp=True`
   resolves to off with an INFO message, so the engine still needs no per-system arguments; an
   explicit amplitude with gains raises (`epochs._refuse_amp_with_gains`). `vis_amp` stays on
   for data without gains (the legacy uncalibrated path, labelled legacy in results).
3. **Tempering after the gains.**
   - **Static fits (done):** the per-night positions that `_temper_factors` fits from are
     refined on the same data, so they carry the gains.
   - **Residuals (open):** today the V² block uses residuals whitened with the full covariance,
     and a closure-phase block with offsets takes the pooled f. Whitening mixes a frame's
     channels, which `channel_chains` (lag-1 and lag-2 along consecutive channels) assumes it
     does not. Proposed: the conditional residuals, r minus the posterior-mean gain
     contribution, divided by the quoted errors. They keep channel order, and the common mode
     the gains absorb is gone before f is estimated, for closure phases too.
   - **What `dof` tempers:** the whole per-block log L(s), the log det included (a power
     likelihood, as before). This is approximate: f is measured on what the gains leave, but
     the power 1/f also discounts the K gain-mode directions, which were never measured to need
     it. The bias is negligible for K ≪ N. Tempering only the complement of the gain modes
     would remove it, but needs the Woodbury integrand below (§3.2.1), which would then no
     longer be optional.
   - **Acceptance (done):** block f ≤ 3 on data with gains or offsets, with f_raw ≥ 0.5.
     f > 3 is reported as "noise model inadequate" (`aliases._INADEQUATE_F`) and is never
     silently tempered away. For 2 < f ≤ 3 the `FLAG_RANGE` (0.5–2) tempering warning is
     expected and accepted: tempering is applied and reported. f_raw < 0.5 (anticorrelated
     residuals, `tempering.out_of_range`) fails. Data without gains keep the `FLAG_RANGE`
     warning as before.
4. **One treatment of gains: marginalize (open).** `orbit_search` profiles gains while the alias
   engine marginalizes them, so with many gain dofs per night the ranking and the evidence can
   disagree about the best orbit. `orbit_search` should marginalize by default.
5. **Implausible-peak flag and kept peaks (open).**
   - `OrbitPipeline` calls `epoch_positions` with `dof=1`, so the night flags
     (`flag_alias_nats`, `flag_min_gap`) see untempered gaps, which a large f inflates. They
     should be divided by the night's f, as the short-arc sign gate is since virgil#435.
   - Flag a night whose best peak's separation exceeds the robust maximum of the other nights'
     by a set factor (relative, no published orbit), or that no orbit in the band visits.
   - Carry the top M peaks per night into the band seeding ("keep each night's modes").
   - Flags are reported, never silently dropped.
6. **Compile size.** Binning should remove the CPU aborts. If not, map the per-night kernel over
   padded nights (`lax.map`/scan) rather than unrolling.

### 3.3 Analysis repository (virgil-validation)

The gallery reads through `to_oidata` only. It labels each night's data path ("calibrated +
template gains", "closure phases only", or "legacy uncalibrated V²"); a system whose nights
differ is labelled "mixed", with the per-night paths in its table. Results from uncalibrated
V² stay labelled legacy until rerun.

## 4. Validation, cheapest first

1. **Unit tests (virgil, virgil-vlti).** Loader alignment; zero-width gains reproduce the plain
   `_Surface`; the integrand matches the dense `GainModes.covariance` (Cholesky χ² and log det)
   for s ≫, ≪ and ≈ the widths; the rule matches a dense 1-D integral (virgil#442). Done except
   the s-regime cases.
2. **One night of the failing system on the laptop (≤ 2 min, bin 8).** After calibration and
   gains: χ²/N ≈ 1 on the effective covariance (s²D + UUᵀ at s = 1; the raw χ² on the
   propagated errors reported beside it, as `chi2_per_night` does), 0.5 ≤ f_raw and f ≤ 3 (a `FLAG_RANGE` warning for 2 < f ≤ 3
   is expected and accepted, §3.2.3), and the peak set
   contains the published position (compared only after the fit).
3. **OzSTAR: the whole gallery, blind.** Success means the published orbits are recovered in
   the sky plane (Thiele–Innes) where the papers solved them, the failing system's top band is
   near its published period (36.5 d for HD 9312), and every miss is explained by flags or an
   absence of calibrators.
4. **Release gate (Gl 229)** (was the release gate through 0.4.1; retired 2026-10-23). The
   criterion was N = 34 first with Δlog Z > 10. It passed at virgil 0696558 (OzSTAR 18849983:
   N = 34, P = 12.132 d, Δlog Z 22.8 over N = 62). An in-field-calibrated run, once a dual-field
   template exists, is optional.
5. **Held-out check of the noise model (optional).** `virgil.crossval` (virgil#440) can score
   leave-one-night-out log likelihoods with and without the gains. A noise model that is
   adequate should not lose held-out likelihood.
6. **Injection–recovery** for p-value calibration (OzSTAR array).

Rules: blind starts only; no noise code in the analysis repository; nothing reimplemented that
virgil or virgil-vlti has; laptop runs ≤ 2 min, `OMP_NUM_THREADS=2`, one process.

## 5. Order of the remaining work

1. Calibrators and calibration (§3.1.1–3), run as one job (in progress).
2. virgil: conditional residuals for tempering and tempered night flags (§3.2.3, §3.2.5), then
   `orbit_search` marginalizing gains (§3.2.4).
3. Gallery reads via `to_oidata`; then the OzSTAR rerun and validation (§4).

## 6. Decisions (Ben, 2026-10-20)

1. R1: numerical marginalization of the white-noise scale.
2. Nights without calibrators: closure phases only.
3. Bin HIGH by 8 (MEDIUM-like sampling).
4. Exclude programme binaries as calibrators.
5. Label the current gallery results "legacy uncalibrated V²" now, before the rerun.
