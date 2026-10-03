# Spectro-interferometry workflow: lessons from Apep (VLTI/GRAVITY)

Status: **design**, 2026-10-03. Nothing here is implemented, except where it says it is on the local
branch `apep-gravity` (not pushed).

The evidence is in `~/data/apep_gravity`:
- `notes/lessons_for_drpangloss.md`, the source of this note;
- `notebooks/apep_report.html`;
- `scripts/calibrate.py` and `scripts/apep_models.py`.

This note extends Stage 6a ([`imaging_plan.md`](imaging_plan.md), [`pmoired_parity.md`](pmoired_parity.md)). It does not replace 6a: another agent is building 6a, and this note only says how Apep changes or adds to it. The orbit half of the Apep lessons is in [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md).

## 1. The workflow that worked on Apep

| Step | What was done | Where |
|---|---|---|
| 1. Calibrate | The ESO `DUAL_SCI_VIS` products are uncalibrated. The fringe tracker sat on Apep's O supergiant 0.74″ away. The science fibre alternated between the binary (SOBJ ≈ (211, −711) mas) and the O star (SOBJ ≈ 0), so the O star is a calibrator observed minutes apart, in the same mode. V² is divided by its transfer function, interpolated in time per baseline and channel. Closure phases have its offset subtracted, averaged as unit vectors. | `scripts/calibrate.py` |
| 2. Unify times | GRAVITY stamps each baseline of a frame with its own MJD, up to ~12 s apart in 2023. That is beyond `read_oifits`'s 1e-4 d (8.6 s) tolerance, so triangles failed to find their baselines. The fix was to give every frame one MJD, shared by V² and T3. | `calibrate.frame_times` |
| 3. Bin | Continuum 2.21–2.40 µm, binned ×4 to 21 channels. Lines 1.99–2.21 µm, binned ×2 to 50 channels. The binning was chosen by hand to tame correlated channels. | `calibrate.bin_channels` |
| 4. Continuum fit | Grey geometry, chromatic fluxes: `PowerLaw` for the stars, `BlackBody` for the dust, `PowerLaw` for the halo. Fitted to V² and closure phase over three epochs, with an additive error term per epoch (`noise=`). Multi-start L-BFGS, then NUTS with no divergences on 21 parameters. | `apep_models.py`, `fit(..., noise=)` |
| 5. Line fit | The geometry is fixed at the continuum posterior. Each component's flux is free in each line channel (`Tabulated`), and the companion's position is free and shared. Its position came out within 0.15 mas of the continuum fit. C IV identifies the companion as the WC star. | `notebooks/apep_lines.ipynb` |
| 6. Compare models | χ²/N with the **uninflated** errors per epoch, plus scatter plots of data against model, for every model. The Gaussian-process images are compared by Laplace evidence. | `apep_correlation.ipynb` |
| 7. Image | `GaussianField` pixels, with both stars analytic and the halo fixed at its parametric value. | `apep_gravity.ipynb` |

**What already generalises:** grey geometry with spectra (SPARCO), the additive error model, multi-start fits then NUTS, and model ranking on the uninflated χ². Steps 1–3 and 5 are where the gaps are.

## 2. Gaps and proposals

### 2.1 One model across the whole band (6a: sum and node spectra)
**Seen.** The continuum (bin 4) and line (bin 2) fits were separate and **did not join**. Their star–halo split differs by 15–20% at 2.21 µm.

**Proposal.**
1. Fit both bands at once. Each component's flux is a 6a `Sum` of a parametric continuum and a node **excess** that lives only in line windows:
   ```python
   wc = PointSource(
       flux=Sum(PowerLaw(f_wc, idx_wc, W0),
                Nodes(wavel=line_nodes, values=excess_wc, outside=0.0)),
       dra=..., ddec=...)
   ```
   - `Nodes(..., outside=0.0)` is zero outside its node range. The continuum is then fixed by the out-of-window channels, and the excess is identifiable.
   - Without `outside=0.0` (that is, constant extrapolation, as `Tabulated` does now), continuum and excess are degenerate everywhere.
2. **Positivity applies to the total, not to the excess.** An absorption line has a negative excess. So 6a's non-negativity check moves from each spectrum's amplitude to the evaluated `Sum` at the data's wavelengths. That is a likelihood-time check (or a soft barrier), since the values are traced.
3. **Smoothness of node values** is a prior, not a model feature: a GP over wavelength, or a Gaussian-line `Spectrum` when R resolves the line. This matches the "what stays hierarchical lives in the priors" rule in [`chromatic_sources.md`](chromatic_sources.md).
4. **Reconciling `Tabulated`** (on `apep-gravity`). `Tabulated(ratio, wavel)` is exactly a linear node spectrum with constant extrapolation, and a reference flux equal to the mean of the nodes.
   - 6a's `Nodes` should subsume it: `kind="linear" | "cubic"` and `outside="constant" | 0.0`.
   - The `apep-gravity` PR then drops `Tabulated`, or keeps it as a deprecated alias if 6a lands second.
   - **One semantic to settle in 6a:** the reference flux (`wavel=None`, used when rendering). `Tabulated` uses the node mean, but `PowerLaw` uses its value at `wavel0`. Recommendation: every spectrum takes a `wavel0` and is evaluated there, so `render()` means the same thing for all of them.

### 2.2 The total spectrum: OI_FLUX/NFLUX, and the reference star's spectrum (6a, plus a new small item)
**Seen.** The spectra are relative to the reference star, which was assumed to go as λ⁻⁴. The dust temperature then depended on the model (2200 K arc against 2990 K cone) and was degenerate with the halo's index.

**Why it cannot be fixed with V² and closure phase.** V = Σ fᵢ(λ)Vᵢ / Σ fᵢ(λ) is unchanged when every fᵢ(λ) is multiplied by a common g(λ). So the reference star's spectral index is **unidentifiable** from V² and closure phase, and every other fitted index or temperature is measured relative to it. Only a measurement of the total spectrum breaks this.

**Proposal.**
1. 6a reads OI_FLUX (FLUX and NFLUX). The model of the total spectrum is F(λ) = S_ref(λ) Σ fᵢ(λ), where S_ref is the reference component's spectral shape.
   - Add `System.total_spectrum(wavel) -> Σ fᵢ(λ)`, so the same scene object predicts OI_FLUX.
   - **FLUX:** one grey scale nuisance per dataset (absolute calibration and fibre injection), optionally times a low-order polynomial in λ.
   - **NFLUX:** normalised by the same continuum ranges as the data. 6a already plans continuum normalisation for differential phase; use one helper for both.
2. **New:** a prior on the reference spectrum instead of a fixed shape. The reference star's `PowerLaw.index` (or a `BlackBody` T) is freed with a prior from a model atmosphere. Nothing in the library has to change except documentation: today the reference keeps ratio 1 and its index is simply not fitted. The docs must say that, without OI_FLUX, this prior *is* the dust temperature's systematic.
3. **The in-field calibrator gives the total spectrum almost for free.** The target and calibrator frames go through the same fibre minutes apart. So F_target/F_cal(λ), times a model spectrum of the O supergiant, is an NFLUX-quality spectrum of the whole Apep system (§2.7). The fibre-injection ratio is a grey nuisance.

### 2.3 Differential phase (VISPHI) (6a)
**Seen.** VISPHI was not used. Across C IV the WC star's share of the flux rises roughly threefold, and it sits 28 mas from the WN star. So the photocentre moves by several mas along PA 96° within the line.

**Requirements on 6a's differential phase:**
1. **Model arg V(λ) directly.** Apply the *same* continuum normalisation operator as the pipeline: subtract the mean, or the mean and slope, over the continuum channels, per baseline and frame. This is a linear operator on phases, so it fits `OIData`'s existing linear-operator machinery (`phi_mat`).
2. **Do not use the photocentre approximation for resolved systems.** The approximation φ_diff ≈ −2π **u**·Δ**p**(λ) holds only when the line-emitting structure is unresolved (|Δp| ≪ λ/B). On Apep, λ/B ≈ 3.3 mas at 130 m, much smaller than the 28 mas separation. 6a's planned test, "against the analytic photocentre shift", should be joined by a test in the resolved regime: a binary with a line in one star, against the exact arg V.
3. **Join with closure phase.** When VISPHI and T3PHI come from the same frame, VISPHI's closure (the closure of the differential phases) is the closure phase minus its own continuum mean, so it duplicates the chromatic part of T3PHI. 6a's "closure phase with VISPHI in one dataset" must not double-count it. Either drop one triangle's worth of differential phases per frame, or document that the closure phase is the independent part. Recommendation: keep closure phases and use VISPHI only in the line windows, after continuum normalisation. There, both are needed.

### 2.4 Calibration nuisances correlated across channels (new; Stage 8's "spectral correlations", brought forward)
**Seen.**
- Calibrated errors were underestimated by 2–7×.
- Per-epoch additive terms (`noise=`) beat multiplicative ones by ΔlogL ≈ 700.
- Both are diagonal. GRAVITY's systematics are mostly **common to all channels of one frame and baseline**: transfer-function jitter between calibrator frames, and the calibrator scatter that `calibrate.py` adds in quadrature *per channel*. That scatter is in fact one number per frame and baseline.

This is the dataset that Stage 8 said would trigger spectral correlations. Recommendation: do it now, in this form.

**Model.** Per block b, with one block per (frame, baseline) for V² and one per (frame, triangle) for closure phase:
- **V²:** V²_obs = g_b V²_model(λ) + noise. The gains g_b − 1 ~ N(0, τ_V²) are shared by every channel of the block.
- **Closure phase:** φ_obs = φ_model(λ) + δ_b + noise, with offsets δ_b ~ N(0, τ_φ²) shared by every channel of the block.
- **Optionally,** a slope across the band per block, for chromatic transfer-function drift.

**Marginalise analytically.** Each block's covariance is a diagonal matrix plus a rank-one term, C_b = D + τ² m mᵀ:
- for closure phase, m = 1;
- for V², m = V²_model, so C_b depends on the model.

With x = D^{−1/2} r, w = D^{−1/2} m, s = wᵀw and β = (1 + τ²s)^{−1/2}:

  r_w = x − ((1 − β)/s) w (wᵀx),  ‖r_w‖² = rᵀC_b⁻¹r,  log det C_b = Σ log σᵢ² + log(1 + τ²s).

So:
- The correlated model still produces **one whitened residual vector**, preserving the rule in `AGENTS.md`, plus a log-determinant. That log-determinant is the same kind of Σ log σ term that `noise=` already adds.
- LM still works when τ is fixed. With τ fitted, use L-BFGS or NUTS, as `noise=` does now.
- No per-block parameters are sampled. Apep has 10–20 frames × (6 baselines + 4 triangles) blocks per epoch, and all of them are integrated out exactly.
- With unprojected closure phases, the chord residual 2 sin(Δ/2)/σ makes δ enter non-linearly. Applying the rank-one whitening to the chord residuals is then an approximation, good while the offsets are small (δ ≲ 0.3 rad). Document the limit.

**Interface.** The rank-one terms extend `noise=`'s per-dataset vocabulary (§2.6): `vis_gain` (τ_V) and `phi_offset` (τ_φ). They need block labels in `OIData` (§2.5). A *known* τ (from the calibrator scatter) is the same code with τ fixed. OIFITS v2's `OI_CORR` table can carry such correlations from a pipeline. Reading `OI_CORR` into a block structure is a later extension.

**Hand binning.** Once the correlated model exists, binning by 2 or 4 is no longer an error-model decision; only compute cost remains a reason to bin. The tutorial should say so.

### 2.5 Times, frames and epochs in `OIData` (new; also a prerequisite for orbits)
**Seen.** `read_oifits` matches each triangle to the baselines with the nearest MJD, within 1e-4 d. GRAVITY's per-baseline stamps differ by up to 12 s, and `OIData` keeps no time at all after reading.

**Proposal.**
1. **Keep the times.** `OIData` stores `mjd` (float64, shape of `u`) per baseline sample. Closure phases inherit their time through `i_cps*`. It also stores `frame` (int, shape of `u`): which exposure a sample belongs to. The orbit note needs `mjd`; the nuisances in §2.4 need `frame`.
2. **Match by frame, not by exact MJD.** A triangle matches the baselines of the same `INSNAME` whose MJD lies within the row's `INT_TIME` (falling back to the current tolerance when `INT_TIME` is absent). After matching, `frame_mjd="mean"` (the default) gives each frame one time, which is what `calibrate.frame_times` did by hand.
3. **Epoch grouping.** `OIData.epochs(gap_days=0.5)` returns integer labels for per-epoch nuisances. `split_by_epoch()` returns a list of `OIData`, for per-dataset models.
4. **Keep MJD in float64 on the host.** float32 resolves MJD ≈ 60000 to about 0.004 d. Model code receives `mjd − t_ref`, with `t_ref` a static float64 (see the orbit note).

### 2.6 A wavelength-scale nuisance per instrument (new, small)
**Seen.** Instruments and epochs have wavelength scales good to about 0.1%. A 0.1% scale error moves Apep's 28.05 mas separation by 0.028 mas, **twice its statistical error** (0.013 mas). It does not move the position angle.

**Proposal.** `u` and `v` are stored in metres and models divide by `wavel` (`Component.model`). So a scale nuisance s only has to evaluate the model at `wavel·s`. That rescales the spatial frequencies and the spectra together, which is physically right. It is one line in `OIData.model`, or `OIData.with_wavelength_scale(s)`, plus a per-dataset term `wavel_scale` with a prior such as `Normal(1, 1e-3)`. With one dataset it is degenerate with every angular size, so the prior *is* the systematic. Report it.

**One per-dataset nuisance vocabulary.** `noise=` (on `apep-gravity`) takes one dict of priors per dataset. Grow it, under the same name, into the per-dataset nuisance specification:

| Term | Effect | Status |
|---|---|---|
| `vis_scale`, `phi_scale` | multiply σ | on `apep-gravity` |
| `vis_error_rel`, `phi_error` | add in quadrature (relative to the **model** V², and absolute) | on `apep-gravity` |
| `vis_gain`, `phi_offset` | rank-one correlated blocks, marginalised (§2.4) | new |
| `wavel_scale` | evaluate at λ·s | new |
| `flux_scale` (+ `flux_poly`) | OI_FLUX calibration (§2.2) | 6a/new |

**Reconciling with 6a's error floors.** Floors (PMOIRED's `min error`, `min relative error`) are *fixed* changes to the data and belong on `OIData` (`with_error_floor`, like `with_error_scale`). `noise=` is the *fitted* counterpart, in the likelihood. Two differences must be explicit:
- `vis_error_rel` multiplies the **model** V², while PMOIRED's relative floor multiplies the **data**. The model version is the right one for a fitted term, because it does not reward the model for low data points.
- A floor is a `max(σ, floor)`, whereas `noise=` adds in quadrature.

Both should go through one function (`inflated_errors` on `apep-gravity`), with `where="data" | "model"` and `combine="max" | "quadrature"`, so that the two cannot drift apart.

### 2.7 The in-field (dual-field) calibrator recipe (new: example and docs, not library)
GRAVITY-specific tools are "not planned" in [`pmoired_parity.md`](pmoired_parity.md). This is a recipe, not a reader, and it makes archival dual-field data usable without an ESO calibration.
1. Classify frames by `SOBJ` offset: ≈ 0 means the fringe-tracking star (the calibrator); otherwise the target.
2. Per baseline and channel, interpolate the calibrator's V² transfer function in time to each target frame. Average its closure phases as unit vectors and subtract them.
3. Record the calibrator scatter as the **known τ** of §2.4. Today it is added in quadrature per channel, which overcounts it in a binned fit.
4. Unify MJD per frame (§2.5) before writing.
5. Optionally, produce the flux ratio F_target/F_cal(λ) for NFLUX (§2.2).

**Deliverable:** `examples/gravity_dual_field_calibration.py`, generalised from `scripts/calibrate.py`, and a docs page. **Caveat for the docs:** the calibrator must be unresolved, and close enough on the sky that the transfer function is shared. On Apep it is 0.74″ away.

### 2.8 Smearing (6a): Apep as a check
R ≈ 130–240 after binning, with up to 8 fringes across the field on 130 m baselines, so smearing is negligible but only just. Once 6a's smearing exists, refit the Apep cone with smearing on. The change in the separation should be ≲ 0.01 mas. If it is not, the "negligible" claim is wrong for the 2024 epoch.

## 3. Mapping and effort

| # | Item | Goes in | Effort (agent h) | Depends on |
|---|---|---|---|---|
| 2.1 | `Sum` and `Nodes(outside=0.0)`; positivity on the total; `Tabulated` → `Nodes`; one reference-flux rule | 6a (as planned, plus 0.5 h) | +0.5 | — |
| 2.2a | OI_FLUX/NFLUX, `System.total_spectrum`, `flux_scale` | 6a (as planned) | +0.5 for `total_spectrum` | 2.1 |
| 2.2b | A prior on the reference spectrum; docs on the degeneracy | new, small | 1 | 2.2a |
| 2.3 | VISPHI with pipeline-matched normalisation; resolved-regime test; no double counting with T3 | 6a (as planned, plus tests) | +1 | 2.5 |
| 2.4 | Rank-one correlated nuisances (`vis_gain`, `phi_offset`), analytic marginalisation | new "6d", or Stage 8 brought forward | 5–7 | 2.5 (`frame`) |
| 2.5 | `mjd` and `frame` in `OIData`; INT_TIME matching; `epochs()` | new, before 6a's VISPHI | 2–3 | — |
| 2.6 | `wavel_scale`; the `noise=` vocabulary; `with_error_floor` sharing `inflated_errors` | new (small) and 6a | 1–2 | `apep-gravity` PR |
| 2.7 | Dual-field calibration example and docs | new | 3–4 | 2.4 (known τ), 2.5 |
| 2.8 | Smearing check on Apep | 6a's MWE extension | 0.5 | 6a smearing |

**Order.** First 2.5 (small; unblocks 2.3, 2.4 and the orbits). Then the `apep-gravity` PR, reconciled with 6a's `Nodes` and floors. Then 2.4 and 2.6, and 2.7 last. The 6a agent needs only the "+" items folded into its plan.

## 4. Open questions for Ben

1. **Bring 2.4 forward?** Recommendation: yes, as its own stage ("6d") after 6a. The analytic rank-one marginalisation keeps it cheap and keeps one residual vector.
2. **Should `noise=` grow into the general per-dataset nuisance argument** (gains, offsets, wavelength scale, flux scale), or should those get a separate `nuisance=` argument? Recommendation: one argument, keeping the name `noise=` for now.
3. **Which reference flux for node spectra:** the node mean (as `Tabulated` does) or the value at `wavel0` (as `PowerLaw` does)? Recommendation: `wavel0` for all spectra.
4. **VISPHI or closure phases where both exist?** Recommendation: closure phase everywhere, plus continuum-normalised VISPHI in the line windows only.
5. **The dual-field recipe:** an example script (recommended), or a small `drpangloss.gravity` helper module, despite "GRAVITY-specific tools not planned"?
6. **Merge order:** should the `apep-gravity` commits (`0688f34`, `c4ef79d`, `3a6c682`, `37bddad`) go to `imaging` before or after 6a's spectra land? The reconciliation in 2.1 and 2.6 is simpler if 6a lands first.
