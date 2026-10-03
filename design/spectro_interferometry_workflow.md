# Spectro-interferometry workflow (GRAVITY, MATISSE and similar)

Status: **design**, 2026-10-03. Nothing here is implemented, except where it says it is on the local branch `apep-gravity` (not pushed).

This note describes an end-to-end workflow for fitting multi-channel long-baseline data (V², closure phase, differential phase, spectra), the gaps in drpangloss, and proposed APIs. It extends Stage 6a ([`imaging_plan.md`](imaging_plan.md), [`pmoired_parity.md`](pmoired_parity.md)) and does not replace it: another agent is building 6a. Orbits are in [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md).

The colliding-wind binary Apep (VLTI/GRAVITY, three epochs) is the worked example that exposed the gaps. It is a hard case for this workflow:
- a binary with strong emission lines;
- resolved dust brighter than both stars;
- an in-field calibrator;
- calibrated errors underestimated by 2–7×.

The analysis and its scripts are in `~/data/apep_gravity` (`notes/lessons_for_drpangloss.md`, `scripts/calibrate.py`, `scripts/apep_models.py`).

## 1. A reference workflow

| Step | What to do | What drpangloss has | Example (Apep) |
|---|---|---|---|
| 1. Calibrate | Divide V² by a transfer function interpolated in time per baseline and channel. Subtract the calibrator's closure phases, averaged as unit vectors | Nothing (pipeline products are read as given) | The in-field calibrator recipe, §2.7 |
| 2. Times and frames | Give each frame one time, shared by V² and T3. Keep it per datum | Triangles matched to baselines within 1e-4 d; no times kept | GRAVITY stamps baselines up to 12 s apart, so triangles failed to match (§2.5) |
| 3. Channels | Choose the channels; bin only for compute cost | Multi-channel `OIData` | Binned ×2–×4 by hand to tame correlated errors (§2.4) |
| 4. Continuum fit | Grey geometry with chromatic fluxes (SPARCO), fitted to V² and closure phase over all epochs, with fitted error terms. Multi-start, then NUTS | `PowerLaw`, `BlackBody`, `fit`, `numpyro_model`; `noise=` on `apep-gravity` | 21 parameters, three epochs, no divergences |
| 5. Lines | Continuum plus line excess spectra, in **one** model across the band | Free fluxes per channel (`Tabulated`, on `apep-gravity`) | Separate continuum and line fits disagreed by 15–20% (§2.1) |
| 6. Spectra and differential phase | Fit OI_FLUX/NFLUX and VISPHI with the visibilities | Not read | Unused, so the reference star's spectral index was assumed (§2.2, §2.3) |
| 7. Compare models | χ²/N with the **uninflated** errors per dataset, scatter plots of data against model, and evidence where Laplace applies | `fit`'s χ², `imaging` evidence | Five models ranked clearly |
| 8. Image | `GaussianField` pixels, with point-like components analytic | Stage 5 | — |

**What already generalises:** grey geometry with spectra (SPARCO), fitted error terms, multi-start fits then NUTS, and model ranking on the uninflated χ². Steps 1, 2, 3, 5 and 6 are where the gaps are.

## 2. Gaps and proposals

### 2.1 One model across the whole band (6a: sum and node spectra)
**Problem.** Fitting the continuum and the line windows separately gives two descriptions of the scene that need not agree where the windows meet. (Apep: the star–halo split differed by 15–20% at the boundary.)

**Proposal.**
1. Fit the whole band at once. Each component's flux is a 6a `Sum` of a parametric continuum and a node **excess** that lives only in line windows:
   ```python
   star = PointSource(
       flux=Sum(PowerLaw(f, index, W0),
                Nodes(wavel=line_nodes, values=excess, outside=0.0)),
       dra=..., ddec=...)
   ```
   - `Nodes(..., outside=0.0)` is zero outside its node range. The continuum is then fixed by the out-of-window channels, and the excess is identifiable.
   - With constant extrapolation (as `Tabulated` does now), continuum and excess are degenerate everywhere.
2. **Positivity applies to the total, not to the excess.** An absorption line has a negative excess. So 6a's non-negativity check moves from each spectrum's amplitude to the evaluated `Sum` at the data's wavelengths. That is a likelihood-time check (or a soft barrier), since the values are traced.
3. **Smoothness of node values** is a prior, not a model feature: a GP over wavelength, or a Gaussian-line `Spectrum` when R resolves the line. This follows the rule in [`chromatic_sources.md`](chromatic_sources.md) that what stays hierarchical lives in the priors.
4. **Reconciling `Tabulated`** (on `apep-gravity`). `Tabulated(ratio, wavel)` is exactly a linear node spectrum with constant extrapolation, and a reference flux equal to the mean of the nodes.
   - 6a's `Nodes` subsumes it: `kind="linear" | "cubic"` and `outside="constant" | 0.0`.
   - The `apep-gravity` PR, which merges after 6a's spectra, drops `Tabulated`.
   - **One semantic to settle in 6a:** the reference flux (`wavel=None`, used when rendering). `Tabulated` uses the node mean, but `PowerLaw` uses its value at `wavel0`. Default: every spectrum takes a `wavel0` and is evaluated there.

### 2.2 The total spectrum: OI_FLUX/NFLUX, and the reference component's spectrum (6a, plus a small new item)
**Problem.** V = Σ fᵢ(λ)Vᵢ / Σ fᵢ(λ) is unchanged when every fᵢ(λ) is multiplied by a common g(λ). So with V² and closure phase alone:
- the reference component's spectral shape is **unidentifiable**;
- every fitted index or temperature is measured relative to it.

Only a measurement of the total spectrum breaks this. (Apep: with the reference star fixed at λ⁻⁴, the dust temperature depended on the geometric model, 2200 K against 2990 K, and was degenerate with the halo's index.)

**Proposal.**
1. 6a reads OI_FLUX (FLUX and NFLUX). The model of the total spectrum is F(λ) = S_ref(λ) Σ fᵢ(λ), where S_ref is the reference component's spectral shape.
   - Add `System.total_spectrum(wavel) -> Σ fᵢ(λ)`, so the same scene object predicts OI_FLUX.
   - **FLUX:** one grey scale nuisance per dataset (absolute calibration and fibre injection), optionally times a low-order polynomial in λ.
   - **NFLUX:** normalised by the same continuum ranges as the data, with one helper shared with differential phase.
2. **New:** a prior on the reference spectrum instead of a fixed shape. The reference's `PowerLaw.index` (or a `BlackBody` T) is freed with a prior from a model atmosphere. No library change is needed beyond documentation. The docs must say that, without OI_FLUX, this prior *is* the systematic on every other component's colour.
3. **Calibrator spectra.** Any calibrator observed through the same optics gives F_target/F_cal(λ). Times a model spectrum of the calibrator, that is an NFLUX-quality spectrum of the target, with the injection ratio as a grey nuisance (§2.7).

### 2.3 Differential phase (VISPHI) (6a)
**Requirements on 6a's differential phase:**
1. **Model arg V(λ) directly.** Apply the *same* continuum normalisation operator as the pipeline: subtract the mean, or the mean and slope, over the continuum channels, per baseline and frame. This is a linear operator on phases, so it fits `OIData`'s existing linear-operator machinery (`phi_mat`).
2. **Do not use the photocentre approximation for resolved systems.** φ_diff ≈ −2π **u**·Δ**p**(λ) holds only when the line-emitting structure is unresolved (|Δp| ≪ λ/B). 6a's planned test "against the analytic photocentre shift" should be joined by a test in the resolved regime: a binary with a line in one star, against the exact arg V. (Apep: a 28 mas binary against λ/B ≈ 3.3 mas at 130 m, where the WC star's share of the flux triples across C IV.)
3. **Join with closure phase.** When VISPHI and T3PHI come from the same frame, VISPHI's closure (the closure of the differential phases) is the closure phase minus its own continuum mean, so it duplicates the chromatic part of T3PHI. 6a's "closure phase with VISPHI in one dataset" must not double-count it. Default: closure phase everywhere, plus continuum-normalised VISPHI in the line windows only.

### 2.4 Calibration nuisances correlated across channels (Stage 6d, after 6a)
**Problem.** For most spectro-interferometers, the dominant systematics are **common to all channels of one frame and baseline**: transfer-function jitter between calibrator frames, piston and injection drifts. Diagonal error inflation (a scale, or an additive term) does not describe them. Binning channels by hand only hides them. (Apep: errors underestimated by 2–7×. Additive per-epoch terms beat multiplicative ones by ΔlogL ≈ 700, but both are diagonal. The calibrator scatter is one number per frame and baseline, yet it was added to every channel.)

This is the trigger that Stage 8's "spectral correlations" was waiting for. **Decided (2026-10-03): it becomes Stage 6d, after 6a.**

**Model.** Per block b, with one block per (frame, baseline) for V² and one per (frame, triangle) for closure phase:
- **V²:** V²_obs = g_b V²_model(λ) + noise. The gains g_b − 1 ~ N(0, τ_V²) are shared by every channel of the block.
- **Closure phase:** φ_obs = φ_model(λ) + δ_b + noise, with offsets δ_b ~ N(0, τ_φ²) shared by every channel of the block.
- **Optionally,** a slope across the band per block, for chromatic drift.

**Marginalise analytically.** Each block's covariance is a diagonal matrix plus a rank-one term, C_b = D + τ² m mᵀ:
- for closure phase, m = 1;
- for V², m = V²_model, so C_b depends on the model.

With x = D^{−1/2} r, w = D^{−1/2} m, s = wᵀw and β = (1 + τ²s)^{−1/2}:

  r_w = x − ((1 − β)/s) w (wᵀx),  ‖r_w‖² = rᵀC_b⁻¹r,  log det C_b = Σ log σᵢ² + log(1 + τ²s).

So:
- The correlated model still produces **one whitened residual vector**, preserving the rule in `AGENTS.md`, plus a log-determinant. That log-determinant is the same kind of Σ log σ term that `noise=` already adds.
- LM still works when τ is fixed. With τ fitted, use L-BFGS or NUTS.
- No per-block parameters are sampled; every block is integrated out exactly.
- With unprojected closure phases, the chord residual 2 sin(Δ/2)/σ makes δ enter non-linearly. Applying the rank-one whitening to the chord residuals is then an approximation, good while the offsets are small (δ ≲ 0.3 rad). Document the limit.

**Interface.** The rank-one terms extend `noise=`'s per-dataset vocabulary (§2.6): `vis_gain` (τ_V) and `phi_offset` (τ_φ). They need block labels in `OIData` (§2.5). A *known* τ (from calibrator scatter) is the same code with τ fixed. OIFITS v2's `OI_CORR` table can carry such correlations from a pipeline. Reading `OI_CORR` into a block structure is a later extension.

**Binning.** Once the correlated model exists, binning is no longer an error-model decision; only compute cost remains a reason to bin. The tutorial should say so.

### 2.5 Times, frames and epochs in `OIData` (new; also a prerequisite for orbits)
**Problem.** `read_oifits` matches each triangle to the baselines with the nearest MJD, within a fixed 1e-4 d. `OIData` keeps no time after reading. Some pipelines stamp each baseline of a frame with its own MJD, beyond that tolerance. (Apep: GRAVITY, up to 12 s.)

**Proposal.**
1. **Keep the times.** `OIData` stores `mjd` (float64, shape of `u`) per baseline sample. Closure phases inherit their time through `i_cps*`. It also stores `frame` (int, shape of `u`): which exposure a sample belongs to. Orbits need `mjd`; Stage 6d needs `frame`.
2. **Match by frame, not by exact MJD.** A triangle matches the baselines of the same `INSNAME` whose MJD lies within the row's `INT_TIME` (falling back to the current tolerance when `INT_TIME` is absent). After matching, `frame_mjd="mean"` (the default) gives each frame one time.
3. **Epoch grouping.** `OIData.epochs(gap_days=0.5)` returns integer labels for per-epoch nuisances. `split_by_epoch()` returns a list of `OIData`, for per-dataset models.
4. **Keep MJD in float64 on the host.** float32 resolves MJD ≈ 60000 to about 0.004 d. Model code receives `mjd − t_ref`, with `t_ref` a static float64.

### 2.6 A wavelength-scale nuisance per instrument (new, small)
**Problem.** Instruments and epochs have wavelength scales good to about 0.1%. That scales every angular size by the same factor, which can exceed the statistical error of a precise fit. (Apep: 0.028 mas on a 28.05 mas separation, twice its 0.013 mas statistical error.) Position angles are unaffected.

**Proposal.** `u` and `v` are stored in metres and models divide by `wavel` (`Component.model`). So a scale nuisance s only has to evaluate the model at `wavel·s`. That rescales the spatial frequencies and the spectra together, which is physically right. It is one line in `OIData.model`, or `OIData.with_wavelength_scale(s)`, plus a per-dataset term `wavel_scale` with a prior such as `Normal(1, 1e-3)`. With one dataset it is degenerate with every angular size, so the prior *is* the systematic. Report it.

**One per-dataset nuisance vocabulary.** `noise=` (on `apep-gravity`) takes one dict of priors per dataset. It grows, under the same name, into the per-dataset nuisance specification:

| Term | Effect | Status |
|---|---|---|
| `vis_scale`, `phi_scale` | multiply σ | on `apep-gravity` |
| `vis_error_rel`, `phi_error` | add in quadrature (relative to the **model** V², and absolute) | on `apep-gravity` |
| `vis_gain`, `phi_offset` | rank-one correlated blocks, marginalised (§2.4) | Stage 6d |
| `wavel_scale` | evaluate at λ·s | new |
| `flux_scale` (+ `flux_poly`) | OI_FLUX calibration (§2.2) | 6a/new |

**Reconciling with 6a's error floors.** Floors (PMOIRED's `min error`, `min relative error`) are *fixed* changes to the data and belong on `OIData` (`with_error_floor`, like `with_error_scale`). `noise=` is the *fitted* counterpart, in the likelihood. Two differences must be explicit:
- `vis_error_rel` multiplies the **model** V², while PMOIRED's relative floor multiplies the **data**. The model version is the right one for a fitted term, because it does not reward the model for low data points.
- A floor is a `max(σ, floor)`, whereas `noise=` adds in quadrature.

Both go through one function (`inflated_errors` on `apep-gravity`), with `where="data" | "model"` and `combine="max" | "quadrature"`, so that the two cannot drift apart.

### 2.7 The in-field (dual-field) calibrator recipe (new: example and docs, not library)
In dual-field instruments (GRAVITY's dual-field mode), archival data often include frames on the fringe-tracking star, interleaved with the target. When that star is unresolved, it is a calibrator observed minutes apart, in the same mode, with no separate calibration needed. GRAVITY-specific tools are "not planned" in [`pmoired_parity.md`](pmoired_parity.md), so this is a recipe, not a reader:
1. Classify frames by `SOBJ` offset: ≈ 0 means the fringe-tracking star (the calibrator); otherwise the target.
2. Per baseline and channel, interpolate the calibrator's V² transfer function in time to each target frame. Average its closure phases as unit vectors and subtract them.
3. Record the calibrator scatter as the **known τ** of §2.4, not as a per-channel error.
4. Unify MJD per frame (§2.5) before writing.
5. Optionally, produce F_target/F_cal(λ) for NFLUX (§2.2).

**Deliverable:** `examples/gravity_dual_field_calibration.py`, generalised from Apep's `scripts/calibrate.py`, and a docs page. **Caveat for the docs:** the calibrator must be unresolved, and close enough on the sky that the transfer function is shared.

### 2.8 Smearing (6a)
Once 6a's smearing exists, its docs should give the rule for when it matters: the number of fringes across the scene's extent, B·θ/λ, compared with the resolving power R. They should also include a check on real data near the limit. (Apep: R ≈ 130–240 after binning, with up to 8 fringes across the scene on 130 m baselines. Refitting with smearing on should move the separation by ≲ 0.01 mas.)

## 3. Mapping and effort

| # | Item | Goes in | Effort (agent h) | Depends on |
|---|---|---|---|---|
| 2.1 | `Sum` and `Nodes(outside=0.0)`; positivity on the total; `Tabulated` → `Nodes`; one reference-flux rule | 6a (as planned) | +0.5 | — |
| 2.2a | OI_FLUX/NFLUX, `System.total_spectrum`, `flux_scale` | 6a (as planned) | +0.5 for `total_spectrum` | 2.1 |
| 2.2b | A prior on the reference spectrum; docs on the degeneracy | new, small | 1 | 2.2a |
| 2.3 | VISPHI with pipeline-matched normalisation; resolved-regime test; no double counting with T3 | 6a (as planned, plus tests) | +1 | 2.5 |
| 2.4 | Rank-one correlated nuisances (`vis_gain`, `phi_offset`), analytic marginalisation | **Stage 6d** (decided), after 6a | 5–7 | 6a; 2.5 (`frame`) |
| 2.5 | `mjd` and `frame` in `OIData`; INT_TIME matching; `epochs()` | new, before 6a's VISPHI | 2–3 | — |
| 2.6 | `wavel_scale`; the `noise=` vocabulary; `with_error_floor` sharing `inflated_errors` | new (small) and 6a | 1–2 | the `apep-gravity` PR |
| 2.7 | Dual-field calibration example and docs | new | 3–4 | 2.4 (known τ), 2.5 |
| 2.8 | Smearing rule and a real-data check | 6a's docs | 0.5 | 6a smearing |

**Order.**
1. 2.5 (small; it unblocks 2.3, 2.4 and the orbits).
2. 6a's spectra and error floors.
3. **Then** the `apep-gravity` PR (decided: it merges *after* 6a's spectra), reconciled with 6a's `Nodes` and floors.
4. Stage 6d (2.4) and 2.6.
5. 2.7 last.

The 6a agent needs only the "+" items folded into its plan.

## 4. Decisions and open questions

### Decided (Ben, 2026-10-03)
1. **Correlated channel nuisances (2.4) are their own stage, 6d, after 6a.**
2. **The `apep-gravity` library commits** merge into `imaging` **after 6a's spectra land.** Those are `0688f34` (`EllipticalGaussian`, `Tabulated`, `noise=`), `c4ef79d` (anisotropic `GaussianField`), `3a6c682` (`GaussianArc`) and `37bddad` (finite rim gradients at u = v = 0). That PR replaces `Tabulated` with 6a's `Nodes`, and routes `noise=` through the same function as 6a's error floors (2.1, 2.6).

### Still open (defaults in force until Ben says otherwise)
1. **`noise=` grows into the general per-dataset nuisance argument** (gains, offsets, wavelength scale, flux scale), keeping its name (default), rather than a separate `nuisance=` argument.
2. **Reference flux for every spectrum:** the value at `wavel0` (default), not the node mean.
3. **Closure phase everywhere,** plus continuum-normalised VISPHI in the line windows only (default).
4. **The dual-field recipe:** an example script and docs (default), not a `drpangloss.gravity` module.
5. **Real anchor binaries** for the position-angle round trips are not chosen yet; see the reminder in [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md) §7.
