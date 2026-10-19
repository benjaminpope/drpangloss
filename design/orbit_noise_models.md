# Orbit fits with calibrated data and virgil-vlti noise models

Status: design for review, no code. Scope: virgil `OrbitPipeline`/`fit_orbit_aliases`, the
virgil-vlti data path into it, and the vv orbit gallery (`scripts/gallery_orbits.py`). Written
2026-10-20 after the gallery run (OzSTAR 18821038/18826461/18826486, virgil e3e1f2d, vv
461723b, results in vv#147).

## 1. What went wrong (checked against code and data)

1. **The GRAVITY data were never calibrated.** All 144 GRAVITY files of the 11 GRAVITY gallery
   systems are `SINGLE_SCI_VIS` (ESO PRO CATG; pipeline gravity/1.9.7): reduced, *not*
   calibrated. Only NN Del (PIONIER, `TARGET_OIDATA_CALIBRATED`) is calibrated. vv's own Track B
   rule (`design/trackb_criteria.md` §GRAVITY) says V² from these files must not be used: it
   carries the transfer function T(frame, baseline, λ), a multiplicative bias that a free error
   scale cannot remove. The gallery path put V² back (`reanalyse_orbits._read_with_vis`) with one
   V² amplitude per night (`aliases._epoch_residuals`, `vis_amp`) as the only correction. One
   scalar per night cannot absorb a transfer function that changes per frame, baseline and
   channel. This is the dominant cause of χ²/N 22–104 per night on HD 9312 and of its tempering
   factor 1/854. (Gl 229's data are `DUAL_SCI_VIS` MEDIUM: also uncalibrated.)
2. **HIGH resolution, unbinned.** 9 of the 11 GRAVITY systems are `INS SPEC RES = HIGH` (about
   1,400 channels per baseline inside the Track B ranges), fitted at native resolution. Channel-
   correlated residuals then dominate, and the compiled programs are large. The CPU
   "LLVM … Unable to allocate section memory" aborts (HD 188088, HD 70937, HD 224974) are
   plausibly this; not confirmed.
3. **The noise machinery exists but the orbit engine refuses it.**
   - virgil marginalizes channel-shared calibration gains analytically: `virgil.gains`
     (groups telescope, baseline, chromatic, external `modes`), `OIData.with_gains`,
     `OIData.with_closure_offsets`, with the exact log-determinant in
     `likelihood._whiten_with_log_norm`.
   - But `fit_orbit_aliases` scores every epoch on `epochs._Surface`, whose constructor calls
     `_check_marginal_data` and raises `NotImplementedError` for data with gains or closure
     offsets (and extra observables with a model-dependent covariance, which matters if |V| or
     T3AMP blocks are ever loaded), because the analytic error-scale marginalization assumes
     C = s²D.
   - The `vis_amp` profile in `aliases._epoch_residuals` rebuilds the model as `obs + r·err`,
     which is valid only for diagonal errors.
   - `orbit_search` profiles gains (one dof each) instead of marginalizing them: an
     inconsistency between the two engines.
4. **virgil-vlti already does the data side.**
   - `virgil-vlti-calibrate` calibrates `*_SCI_VIS` with same-night `*_CAL_VIS` and is validated
     against ESO's `gravity_viscal` (V² 0.12%, closure phases 0.06°).
   - It windows (`--window`) and bins (`--bin`).
   - It writes calibration-template modes beside each file (`--template`, with
     `--tf-error propagated` so that nothing is counted twice) and calibrator-diameter modes
     (`--cal`/`--diameter`, the `rel-` keys for `with_gains(modes=…)`).
   - It screens calibrators (`calpca.screen_calibrators`).
   - It ships the GRAVITY single-field MEDIUM template. The HIGH template is virgil-vlti#40
     (open), and calibration fixes are in virgil-vlti#42 (open).
   - The pair test (`pairs`, `templates.gain_template_from_pairs`) gives PIONIER gain widths.
   - The step from files to a virgil `OIData` with gains exists only as tutorial code
     (`docs/templates.md`, `stack_modes`). That is where mode/sample misalignment bugs would
     come from.
5. **Calibrators are partly found.** vv's `~/data/eso_binaries/trackb/gravity_calibrators.json`
   (same-programme candidates, 2.9 GB, never downloaded) covers HD 41255 (7 of 14 science files),
   HD 188088, ο Leo, HD 70937, HD 210763 and κ Vel. It has nothing for AK For, HD 224974,
   HD 9312, LL Aqr or V963 Cen. Several candidates are themselves programme binaries (AK For,
   HD 70937, HD 41255, ο Leo, V963 Cen, HD 224974, HD 210763, HD 188088): unusable as
   calibrators.

## 2. Physics the design rests on

- **Calibration first, nuisances second.** Calibrated V² = V²_true × (1 + residual gain). The
  residual is small (GRAVITY MEDIUM template: PCs of 1.9%, 0.52%, 0.43%, 0.29% per channel in V²,
  41–81% telescope-based; closure phases 0.33°, 0.25°; PIONIER pair test 0.55%/telescope and
  0.45%/baseline on |V|) and shared by a frame's channels. Marginalizing it is what turns 1,400
  channels into the few numbers they really are.
- **What V² still says once gains are marginalized.** Per-frame grey gains absorb the grey part
  of V². A binary's V² signal survives through its chromatic modulation and its pattern between
  baselines. For HD 9312 (separation ≤ 5.5 mas; λ/B ≈ 3.5 mas at 2.2 µm on 130 m) that is about
  0.2 fringe cycles across K, so most position information comes from closure phases. This is
  the honest information content. The evidences will be weaker than white noise claims and far
  stronger than tempering by 1/854.
- **Sidelobes are real ambiguities, not noise.** A single night of AT snapshots has elongated
  (u, v) coverage, so its χ² surface has sidelobes along the poorly sampled direction. HD 9312's
  wrong nights are off in Dec by 3–15 mas: that fits. A right noise model sets the relative
  heights of the peaks correctly but does not remove them. The orbit must choose between them:
  keep several peaks per night and flag implausible ones (§3.2.5).
- **Binning HIGH by 8 channels costs nothing for a continuum binary.** HIGH samples about 2.5
  channels per resolution element, so 8 channels give MEDIUM-like sampling (resolving power
  about 1,200). The phase change across one bin for s = 15 mas, B = 130 m is about 0.004 cycle. Keep the Brγ window
  (2.155–2.175 µm) out, as Track B does.
- **Primary diameter.** Keep the UD primary (virgil 1c). Calibrator-diameter uncertainties enter
  as modes from virgil-vlti, not as inflated errors.

## 3. Design

### 3.1 Data (virgil-vlti, no noise code in vv)

1. **Calibrators.**
   - Extend the manifest to the five uncovered systems with the same archive query (ESO TAP,
     same programme and night; then other programmes on the same night and configuration).
   - Exclude every Track B or gallery science target and any star with known companions.
   - Prefer `STD`/`*_CAL_VIS` products.
   - Download to `~/data/eso_binaries/trackb/<system>/calibrators/`.
2. **Calibrate** each night with `virgil-vlti-calibrate --tf-error propagated --template
   gravity_single_high_combined_… --bin 8 --window 2.05 2.38`, plus calibrator diameters
   (`--cal`/`--diameter`, so diameter modes are written). Screen with
   `calpca.screen_calibrators`. MEDIUM systems (κ Vel, LL Aqr) use the shipped MEDIUM template
   and `--bin 1`.
3. **Nights without a usable calibrator: closure phases only**, with the template's closure-phase
   modes as closure offsets. V² from `SCI_VIS` is never used again.
4. **One loader, in virgil-vlti**, replacing the tutorial `stack_modes`:
   - `virgil_vlti.to_oidata(paths, insname, ranges=…, observables=…) -> OIData` reads the
     calibrated files;
   - it applies virgil's selection, then sets `with_gains(modes=…)` and
     `with_closure_offsets(modes=…)` from the `.template_modes.npz` / `.modes.npz` beside each
     file, aligned sample for sample after selection, zero on flagged samples, closure-phase
     modes per frame;
   - tests assert that the alignment is exact, that the template widths are unchanged, and that
     a zero-mode load equals plain `OIData`.
   (The name is for review.)
5. **PIONIER** (NN Del, and future PIONIER systems): files are already calibrated. Gain widths
   come from the pair test (`gain_template_from_pairs`) through the same loader.

### 3.2 virgil (OrbitPipeline and alias engine; virgil never imports virgil-vlti)

1. **Score data with gains on the marginal surface (R1, decided).** Integrate the white-noise
   scale numerically. Per block, C(s) = s²D + UUᵀ (U from the gains or offsets), integrated over
   ln s with the Jeffreys prior on [−ln s_max, ln s_max]. R2 was rejected (quoted scale s = 1
   with the template's overall width fitted: cheaper, but it trusts the propagated errors).
   - **Quadrature: a fixed Gauss–Legendre rule on a peak-centred window** (±6/√ν around the
     block's peak, inside the Jeffreys range; virgil#437). A rule over the whole range would
     undersample the peak, whose width in ln s is about 1/√(2ν). The node count (16 or 32) is
     kept only if a test shows it matches a dense 1-D integral to better than 0.01 in log L at
     s_max; if both pass, the cheaper.
   - **Optional speed-up: a closed-form integrand.** Whitened by D, C(s) becomes s²I + ŨŨᵀ with
     Ũ = D^{-½}U (`gains._columns`), low-rank per block. One thin SVD per block, Ũ = QΛ^{½}
     (K columns), then for every s, in O(K):
     χ²(s) = (|r|² − |Qᵀr|²)/s² + Σ_k (Qᵀr)_k²/(s² + λ_k), and
     ½ ln det = (N − K) ln s + ½ Σ_k ln(s² + λ_k).
     It is exact, and λ_k = 0 gives back today's integrand. It replaces one whitening per node
     with one SVD per block; worth doing only if profiling shows the per-node whitening
     dominates.
   `_Surface` stops refusing gains. A regression test requires that zero-width gains
   reproduce today's surface to rounding.
2. **`vis_amp` is off for data with gains.** Telescope and baseline gains supersede the
   per-night amplitude (whose profile also assumes diagonal errors). Following the precedent for
   data without V² (`aliases.py`, the `vis_amp` resolution), the default `vis_amp=True` resolves to
   off with a log message, so `test_the_engine_defaults_need_no_per_system_arguments` still holds;
   an explicit amplitude range with gains raises. AGENTS.md's "V² amplitude" row gains a line on
   this. `vis_amp` stays on for the legacy uncalibrated path, which is labelled legacy in
   results and on the gallery.
3. **Tempering after the gains.**
   - **Residuals:** the conditional residuals, r minus the posterior-mean gain contribution,
     divided by the quoted errors. They keep channel order, which `channel_chains` needs (a full
     whitening mixes a frame's channels), and the common mode the gains absorb is gone before
     f is estimated.
   - **Static fits:** the per-night best static fits that `_temper_factors` uses carry the
     gains as well, or f still sees the transfer-function residual.
   - **What `dof` tempers:** the whole per-block log L(s), the log det included (a power
     likelihood, as today). This does not double count: f is measured only on what the gains
     leave.
   - **Acceptance:** block f ≤ 3 on calibrated data. f > 3 is reported as "noise model
     inadequate" and is never silently tempered away. For 2 < f ≤ 3 the existing `FLAG_RANGE`
     (0.5–2) tempering warning still fires, as information: tempering is applied and reported.
4. **One treatment of gains: marginalize.** `orbit_search` profiles gains while the alias
   engine marginalizes them, so with many gain dofs per night the ranking and the evidence can
   disagree about the best orbit. `orbit_search` changes to marginalizing by default. This is
   for the 1e / release-review owners.
5. **Implausible-peak flag and kept modes.**
   - Flag a night whose best peak's separation exceeds the robust maximum of the other nights'
     by a set factor, or that no orbit in the band visits.
   - Carry the top M peaks per night into the band seeding, as the blind-solver design already
     requires ("keep each night's modes").
   - Flags are reported, never silently dropped.
6. **Compile size.** Binning should remove the CPU aborts. If not, map the per-night kernel over
   padded nights (`lax.map`/scan) rather than unrolling. This is the 1e session's call.

### 3.3 vv

`gallery_orbits.py` reads through the loader only. The gallery marks each system's data path
per night ("calibrated + template gains", "closure phases only", or "legacy uncalibrated
V²"); a system whose nights differ (calibrators for only some nights) is labelled "mixed", with
the per-night paths in its table. #147's
current results are labelled legacy until rerun.

## 4. Validation, cheapest first

1. **Unit tests (virgil, virgil-vlti).**
   - Loader alignment.
   - Zero-width gains reproduce today's `_Surface`.
   - R1's integrand matches the dense `GainModes.covariance` (Cholesky χ² and log det) for
     s ≫ widths, s ≪ widths and s ≈ widths, and the fixed rule matches a dense 1-D integral to
     better than 0.01 in log L at s_max (this sets the node count).
2. **One night of HD 9312 on the laptop (≤ 2 min, bin 8).** After calibration and gains:
   χ²/N ≈ 1 on the effective covariance (s²D + UUᵀ at s = 1; the raw χ² on the propagated
   errors is reported beside it, as `chi2_per_night` does today), tempering f ≤ 3, and the peak set contains the published position (compared only
   after the fit).
3. **OzSTAR: all 12 gallery systems, blind.** Success means:
   - the published orbits are recovered in the sky plane (Thiele–Innes) where the papers solved
     them;
   - HD 9312's top band is near 36.5 d;
   - every miss is explained by flags or an absence of calibrators.
4. **Gl 229.** The release gate stays on its current path, unchanged. Once the dual-field
   template exists, add a dual-field in-field-calibrated Gl 229 run (`virgil-vlti --dual-field`).
   It must still rank N = 34 first with Δlog Z > 10.
5. **Injection–recovery** for p-value calibration (OzSTAR array).

Rules: blind starts only; no noise code in vv; nothing reimplemented that virgil or virgil-vlti
has; laptop runs ≤ 2 min, `OMP_NUM_THREADS=2`, one process.

## 5. Order and owners

1. Ben merges virgil-vlti#42 (calibration fixes) and #40 (HIGH template).
2. virgil-vlti PR: the loader (§3.1.4), with tests.
3. Calibrator manifest, download and calibration (§3.1.1–3), run as one job.
4. virgil PR with the 1e/release-review owners: §3.2.1–3, then §3.2.5.
5. vv PR: gallery reads via the loader; then the OzSTAR rerun and validation (§4).

## 6. Decisions (Ben, 2026-10-20)

1. R1: numerical marginalization of the white-noise scale.
2. Nights without calibrators: closure phases only.
3. Bin HIGH by 8 (MEDIUM-like sampling).
4. Exclude programme binaries as calibrators.
5. Label the current gallery results "legacy uncalibrated V²" now, before the rerun.
