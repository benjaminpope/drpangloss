# GRAVITY calibration: a literature review of the spectro-interferometry design

Status: **review**, 2026-10-03. Footguns 1, 2, 3 and 9 are fixed in Stage 6.0 (PR #120); the rest is not implemented. Ben's decisions on the open questions are in §12, the closure-offset question is expanded in §13, and the questions for the GRAVITY team are in [`questions_for_gravity_team.md`](questions_for_gravity_team.md). This note checks the plans in [`spectro_interferometry_workflow.md`](spectro_interferometry_workflow.md) (S) §2.2–2.8, Stage 6d of [`imaging_plan.md`](imaging_plan.md), and the GRAVITY layout in [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md) (O) §5.3. It compares them with what the GRAVITY pipeline does and with how the GRAVITY Collaboration and others calibrate and fit GRAVITY data. It changes no other note: each recommendation below is for the owner of that note to adopt or reject.

For each design element: **(a)** what the literature does, **(b)** whether our design agrees, is incomplete, or is wrong or risky, and **(c)** the change I recommend. Citation keys are in §11, together with how much of each source I actually read. **[unverified]** marks a claim I could not check against a primary source.

## 0. Footguns to avoid, in priority order

Each item is something the GRAVITY literature, the pipeline manual or our own reader shows to be wrong or misleading. The first four affect drpangloss **today**, before any new stage.

1. **Mixing fringe-tracker and science-channel tables.** A GRAVITY product holds `OI_VIS`, `OI_VIS2`, `OI_T3` and `OI_FLUX` for `INSNAME = GRAVITY_FT` (6 channels, R ≈ 20) *and* `GRAVITY_SC` (e.g. 233 channels at MEDIUM), and in split-polarisation mode `_P1` and `_P2` versions of each ([PM] §8.2; checked on an archival product, §11). `read_oifits` reads every table regardless of `INSNAME` and concatenates them. FT and SC samples then share one error model, one smearing kernel and one set of nuisances. **Fix:** an `insname=` selector that is required when a file holds more than one `INSNAME`. FT data, if used, form a separate dataset (§9). **Fixed in Stage 6.0:** `read_oifits(insname=...)`; a file with GRAVITY FT and SC tables raises unless one is chosen; FT data are skipped (§12).
2. **Reading VISPHI without its `PHITYP`.** In dual-field products the SC `VISPHI` is `PHITYP = absolute`: phase-referenced to the FT through the metrology. In the same file the FT `VISPHI` is `differential`, with `PHIORDER = 1` (both checked on the archival product). In single-field products the pipeline's default removes "the mean group-delay and mean phases" over **all** channels, lines included ([PM] §6 `--output-phase-sc`, §9.16; [GC18b] Methods). `read_oifits` reads any `VISPHI` as an absolute phase. **Fix:** read `PHITYP`, `AMPTYP`, `PHIORDER` and `VISREFMAP` ([D17] §5), and refuse to treat differential phases as absolute. **Partly fixed in Stage 6.0:** VISPHI whose `PHITYP` is not `absolute` is refused; reading the other keywords, and fitting differential phases, are Stage 6a.
3. **Keeping only the diagonal after a projection.** `OIData._propagate_uncertainty` keeps only the diagonal of A D Aᵀ for a general `phi_mat`. Both projections that S §2.3 plans (continuum normalisation, and the closure-free part of VISPHI) produce correlated, rank-deficient outputs. Using √diag as independent errors counts information twice: six projected baseline phases per channel carry only three independent numbers. **Fix:** every projection returns a *whitened* basis (§3c), or `OIData` carries a block covariance. **Fixed in Stage 6.0:** an operator whose outputs correlate is rotated onto the eigenvectors of A D Aᵀ, keeping the non-zero ones, so the outputs are independent.
4. **Trusting pipeline error bars.** They are statistical only: bootstrapped over frames, with no calibration term ([PM] §5.2, §9.19). With fewer than 5 frames they are padded with theoretical photon noise and are "less realistic" (*ibid.*). Calibrator diameter errors are not propagated ([PM] §9.20). The pipeline stores no channel correlations, although it interpolates every spectrum linearly onto a common grid ([PM] §9.5) and samples about 2 pixels per resolution element ([GC18b] Methods; [Me22] code: `wl kernel` 1.4 px). In practice every team replaces or rescales them: errors from the r.m.s. of line-free channels [GC18b, GC20b, GC21b]; V² errors ×10 for imaging [GC21b §4.1]; error scale tuned by hand to χ²_r ≈ 1 [GC22a §2.4]; empirical covariances from the individual DITs [N20a App. A.1].
5. **Achromatic gains that are independent between baselines** (S §2.4). The dominant V² calibration error is coherence loss from piston jitter and anisoplanatism, which goes as exp(−a/λ²) ([PM] §9.10 eq. 58; [GW22] §4.1 eq. 4). Piston jitter is telescope-based: Kammerer et al. measure a correlation of x ≈ 0.32 between channels of one baseline and x/2 between baselines that share a telescope [K20 §2.2]. A rank-one achromatic gain per (frame, baseline) has neither the colour nor the coupling between baselines (§4).
6. **Telescope-based chromatic phase errors in VISPHI.** Delay-line air dispersion adds a chromatic phase of fixed shape and variable amplitude to every exposure. A stable instrumental phase adds another, and a cubic or even a 7th-order polynomial does not remove it below 0.1° ([GC20b] §2.3 and App. A). These errors are telescope-based. So they live **entirely** in the closure-free part of VISPHI, which S §2.3 keeps (§3).
7. **Calibrating off-axis targets with the on-axis star.** In dual-field mode the target sees anisoplanatic coherence loss (0.2–0.4 at tens of arcseconds with GRAVITY Wide [GW22 §4.1]). It also sees field-dependent injection and aberrations ("phase maps", [GC20] App. A.3; [GC21]) that the on-axis fringe-tracking star does not. The in-field recipe of S §2.7 transfers V² from star frames to target frames as if the transfer function were shared (§7).
8. **Ignoring the fibre's field of view.** Each telescope injects into a single-mode fibre with a field of view of about 60 mas at K on the UTs [L19 §2], and about 4.5 times larger on the ATs (scaled by the aperture ratio) **[unverified]**. Components away from the fibre centre are attenuated and phase-shifted, by a different amount at each telescope. The GC binary fits use a telescope-dependent injection ratio per component ([GC18a] App. A.4; [GC20] eqs. A.2, A.6). The GC imaging response includes the coupling efficiency [GC22b App. A.2]. Our models assume a flat primary beam, and so does the total spectrum of S §2.2 (§1).
9. **Over-counting closure phases.** GRAVITY writes all four triangles, of which three are independent. With diagonal errors and equal S/N this over-weights closure phases by 4/3 ([GC22a] §2.4, after Blackburn et al. 2020). The exact correlation between triangles that share a baseline is ±1/3 [K20 §2.2]. `read_oifits` keeps all four **[I found no redundancy handling in `oifits.py`]**. **Fixed in Stage 6.0:** triangles sharing baselines are grouped per frame and channel; their covariance is T diag(s) Tᵀ, with baseline-phase variances s fitted to the reported errors (the ±1/3 structure for equal errors); only the independent combinations are kept, whitened by its Cholesky factor (`OIData.cp_noise`, `n_independent`). On PIONIER data this keeps 390 of IRAS 08544's 504 closure phases, 558 of IW Car's 744 and 342 of HR 4049's 456.
10. **A wavelength-scale nuisance that is only a scale, with the wrong prior.** The pipeline quotes an absolute spectral calibration of 0.5 nm (≈2.3×10⁻⁴ at 2.2 µm, more like an offset than a scale) and 0.1 nm between baselines. The baseline-to-baseline error biases closure phases by "≈3 deg when observing at a group-delay of 40 µm" ([PM] §5.1). S §2.6 assumes a 10⁻³ scale (§6).
11. **Covariances built from the data for multiplicative errors.** This produces Peelle's Pertinent Puzzle: fits biased low, sometimes outside the range of the data. The remedy is to build the covariance from the model [La21, abstract]. S §2.4's choice m = V²_model is the right one. Keep it, including for the "known τ" of §2.7.
12. **Unmatched times and frames** (S §2.5, agreed). The pipeline rejects frames per baseline, so baselines of one exposure get different effective times. Triangles use only frames where all three baselines pass ([PM] §9.12, §9.18). On the archival product, baselines of one exposure differ by up to 10 s, with `INT_TIME` = 180 s.

## 1. The total spectrum: OI_FLUX/NFLUX and the reference spectrum (S §2.2)

**(a) Literature.**
- GRAVITY `OI_FLUX` is per telescope (NEXP×4 rows), in electrons, `CALSTAT = 'U'`: uncalibrated, with tellurics, instrument transmission and fibre injection in it ([PM] §6.13, §8.2, §9.15). `gravity_viscal --calib-flux` defaults to FALSE ([PM] §6.13). It is the flux *coupled into each fibre*, not the sky flux.
- PMOIRED divides FLUX by the P2VM internal-lamp spectrum when the pipeline did not flat-field it. It can fit a telluric model to GRAVITY spectra whose parameters include a cubic wavelength correction (`tellcorr.gravity`: `dl0`…`dl3`, `pwv`, kernel width). NFLUX is normalised by a polynomial over continuum ranges, of order `ptp(λ_cont)/0.15 µm` by default, i.e. 2 across K ([Me22] code: `oimodels.computeNormFluxOI`).
- The BLR papers build the line profile from the summed telescope fluxes, divided by a calibrator spectrum whose own lines are removed with templates. They add the r.m.s. between epochs in quadrature, with a floor of 0.002 in normalised flux ([GC18b] Methods; [GC20b] §2.2).
- The exoplanet work extracts a **contrast spectrum** C(λ) = S_p/S_* and multiplies it by a stellar model atmosphere ([N20a] §2.3; [L19] eq. 2; [N24] §3.1). This is S §2.2 item 3, and it is how the reference-spectrum degeneracy is broken in practice.
- The ratio G(t)/G(t*) between the planet exposure and the star exposure (atmospheric throughput) is corrected with the FT flux ratio, which is grey only ([N20b] App. B). Fibre mis-pointing changes the flux by < 3% ([N20b] App. B; [W21] eq. 1, γ).

**(b) Verdict.** The degeneracy argument and the single representation (each component's own fᵢ(λ), with one grey scale) **agree** with the literature. Two things are **incomplete**:
- *Fibre coupling.* For a scene comparable to the fibre's field of view, or a target offset from the fibre centre, F(λ) = k Σ fᵢ(λ) ηᵢ(λ), where ηᵢ is the coupling at component i's position. ηᵢ is chromatic (the beam scales with λ/D) and differs between telescopes. The same weights enter the denominator √(F_k F_l) of every visibility ([GC20] eq. A.6).
- *A per-dataset grey scale is too coarse.* Injection varies per telescope and per exposure: that is the γ of [W21] and the G(t)/G(t*) of [N20b].

**(c) Changes.**
1. `flux_scale` becomes one grey factor per (frame, telescope) under a log-normal prior. Its width comes from the scatter of the FT fluxes, which the reader can supply. It is marginalised like the gains of §4, since FLUX is linear in it. `flux_poly` stays one low-order polynomial per dataset.
2. Add an optional instrument-level primary beam (fibre coupling) to `OIData`: a Gaussian of FWHM ≈ λ/D, optionally smoothed by tip-tilt jitter ([GC20] App. A.3 uses ≈15 mas rms per axis at the GC). It is applied to component fluxes before both `total_spectrum` and the visibility. Default off. Turning it on is required when any component lies beyond about ⅓ of the fibre FWHM **[threshold is my estimate]**.
3. Document that `OI_FLUX` is uncalibrated in GRAVITY products. Recommend dividing by a calibrator spectrum (or a telluric model, as PMOIRED does) before using FLUX, and fitting NFLUX rather than FLUX unless the user has flux-calibrated the data.
4. Read OIFITS2 `FLUXDATA`, falling back to v1 `FLUX` ([PM] §6, `--oifits2`; [D17] §7).

## 2. Differential phase: normalisation (S §2.3, items 1–2)

**(a) Literature.**
- **The pipeline, single field.** Each SC frame is referenced to the FT phase during that frame. That 6-channel phase is interpolated to the SC wavelengths "with a polynomial fit of order 2" ([PM] §9.13), or with a self-reference polynomial of degree up to 3 (`--phase-ref-sc SELF_REF`, `--phase-ref-sc-maxdeg 3`). [GC20b] App. A describes the self-reference as a 3rd-order polynomial fitted to each DIT. After the frames are averaged, "the mean spectral slope (stored in the GDELAY quantity) and mean spectral value (stored in the PHASE quantity) are removed from the VISPHI" ([PM] §9.16). GDELAY is in metres, so the slope is linear in wavenumber 1/λ. [GC18b] Methods: "removes a mean and slope … calculated using all wavelength channels."
- **The pipeline, dual field.** SC `VISPHI` is absolute (`PHITYP = absolute` on the archival product). It is referenced to the FT plus the metrology and the separation vector ([PM] §9.13 eq. 62).
- **OIFITS2.** φ_diff = φ − φ_ref − η/λ − ζ/λ² − …: a polynomial in **1/λ**, with the order in `PHIORDER` and the reference channels in `VISREFMAP` ([D17] §3.2 eq. 3; §5).
- **BLR work.** The reference for each channel is recomputed with the AMBER method, from all other channels, excluding the "work channel"; this improves phase errors by 10–20% ([GC18b] Methods; [GC20b] §2.1).
  - Broad instrumental shapes are removed: by a 24-pixel Gaussian high-pass in [GC18b]; later by a PCA template of air dispersion in the delay lines (one component, C1, with a fitted amplitude per exposure) plus a stable cryostat profile, then a local 1st- or 2nd-order polynomial around the line ([GC20b] §2.3, App. A; [GC21b] §2).
  - Errors come from the r.m.s. of line-free channels, and the systematic floor is 0.05° ([GC20b] App. A).
  - The pipeline's whole-band fit "would create a phase dip around the blue wing of the Brγ line because it derives the phase reference without considering the scientific signal" ([GC20b] §2.3).
- **PMOIRED.** DPHI = φ − polyfit(λ, φ) over the continuum ranges, of order `ptp(λ_cont)/0.2 µm` (at least 2) by default. Applied identically to data and model ([Me22] code: `oimodels.computeDiffPhiOI`).
- **Photocentre approximation.** The BLR papers use φ = −2π f/(1+f) **u**·**x** only because the BLR (< 100 µas) is far below the resolution (≈ 3 mas) ([GC18b] Methods).

**(b) Verdict.**
- Item 2 (no photocentre approximation for resolved systems, plus a resolved-regime test) **agrees** with the literature.
- Item 1, "apply the *same* continuum normalisation operator as the pipeline", is **wrong as worded**. The pipeline's operator is not a continuum normalisation: it fits all channels, lines included. Before that, it subtracts a per-frame polynomial built from the FT's own (object-dependent) phase. For a chromatically resolved scene, such as a binary whose FT phase changes across the band, that interpolated FT phase carries object signal **[my inference from [PM] §9.13; not tested]**. We cannot reproduce these steps exactly from the product alone.
- The basis is also wrong: S says "mean, or mean and slope", without the variable. The pipeline and OIFITS2 use 1/λ; PMOIRED uses λ.

**(c) Changes.**
1. Replace "the same operator as the pipeline" with: **apply one user-chosen normalisation operator N to data and model alike**. N subtracts a fit in a basis B, polynomial in 1/λ, with B containing at least the span the pipeline removed (degree ≥ 1, i.e. mean and group delay). Because the pipeline's subtraction lies in span(B), N annihilates it. The result is then independent of pipeline version and settings. Fit the coefficients on the continuum channels only, optionally excluding the work channel as AMBER does. N is then an oblique projector, I − B (B_cᵀ W B_c)⁻¹ B_cᵀ W S_c, and still linear.
2. Default degree 2 in 1/λ across K, matching PMOIRED's default order. Document that a higher degree is needed when the data show instrumental structure, and that it can absorb science signal ([GC20b] App. A).
3. Offer an optional extra basis vector per (frame, baseline): a dispersion template. It can be estimated empirically (a PCA of calibrator phases, as in [GC20b] App. A), or computed from the header air paths (`ESO DEL DLT* OPL`, temperature, pressure, humidity), which PMOIRED reads (`oifits.py`; its display code using them is switched off). [GC20b] App. A found that models of the refractive index of air were not accurate enough in their functional form, so the empirical template is the safer default.
4. The normalised data's covariance is N D Nᵀ, singular with rank n − dim(B). Whiten in a basis of the complement, never with √diag (footgun 3).
5. Read `PHITYP`. Dual-field absolute phases go through N too, unless the user fits astrometry (the phase-centre offset) explicitly, as [GC22b] App. A.1.2 does.
6. Keep the resolved-regime test. Add a test that N applied to the output of a simulated pipeline (whole-band mean + GD removed) equals N applied to the raw phase.

## 3. VISPHI with T3PHI: the closure-free projection (S §2.3, item 3)

**(a) Literature.**
- GRAVITY-RESOLVE fits closure phases and amplitudes only, with no visibility phases ([GC22a] §2.3).
- The GC binary fits "omit the visibility phases … because they mostly contain information about the location of the phase centre" ([GC20] App. A.3).
- The GC dual-beam astrometry fits closure phases and visibility phases "with equal weights" to reduce telescope-based errors. That is a weighting heuristic, not a decomposition ([GC22b] App. A.1.2).
- PMOIRED allows `T3PHI` and `DPHI` in one fit with no projection **[from reading its fit setup; I found no projection code]**.
- I found no published GRAVITY analysis that removes the closure part of VISPHI. S §2.3's projection is new: sound in principle, but untested by others.

**(b) Verdict.** The idea is **right**, but **two details are risky**.
1. *The metric.* "Orthogonal to the closures" must hold in the **noise metric**. With per-baseline variances D (unequal in practice), the Euclidean projector P_A = A(AᵀA)⁻¹Aᵀ onto telescope differences (A is the 6×4 incidence matrix) leaves the projected VISPHI correlated with T3PHI: Cov(Cφ, P_A φ) = C D P_Aᵀ ≠ 0. The generalised-least-squares projector P_A^D = A(AᵀD⁻¹A)⁻¹AᵀD⁻¹ satisfies C D (P_A^D)ᵀ = C A(…) = 0, because closures annihilate A (C A = 0). With it, the closure-free VISPHI and T3PHI are uncorrelated to first order.
2. *What lives in the kept part.* The closure-free subspace is exactly where telescope-based errors live: air dispersion, piston and group-delay residuals, FT-to-SC referencing. Its continuum normalisation (§2) and nuisance basis therefore do all the work, which S §2.3 does not say. The approximation is also only first-order: pipeline T3PHI comes from a per-frame bispectrum over frames where all three baselines pass, while VISPHI is the phase of a coherent average over each baseline's own frames ([PM] §9.16, §9.18).

**(c) Changes.**
1. Express the closure-free part as three telescope phases per (frame, channel), â = (AᵀD⁻¹A)⁻¹AᵀD⁻¹(Nφ), with one telescope fixed. Their covariance is (AᵀD⁻¹A)⁻¹. Whiten with its Cholesky factor, giving three residuals per (frame, channel), each independent of T3PHI.
2. Apply N (§2) before the projection, since N acts per baseline. N then commutes with the telescope decomposition when B is the same on every baseline.
3. Test: on simulated data with unequal per-baseline errors, the joint χ² from (T3PHI, projected VISPHI) equals the dense generalised-least-squares χ² of the six VISPHI and four T3PHI with their exact joint covariance.
4. Keep "closure phase everywhere, projected VISPHI in the line windows" as the default (S §4, open question 3). Note in the docs that the GC teams chose either closures only or both with heuristic weights, so this default has no published precedent.

## 4. Correlated calibration nuisances (S §2.4, Stage 6d)

**(a) Literature.**
- **GRAVITY-RESOLVE** infers "an independent scaling factor, C(t, b), for each exposure and baseline", shared by all channels, with a Gaussian prior of unit mean and standard deviation 0.1, on the visibility **amplitude** ([GC22a] §2.3, App. B). The factors are sampled (MGVI), not marginalised. No closure-phase offsets are used; closure phases enter through their phasors on the unit circle (App. B eq. B.9), as in our chord likelihood.
- **Kammerer et al.** measured correlations within GRAVITY exposures from the per-DIT data [K20 §2.1–2.2]:
  - for V²: a constant correlation x ≈ 0.32 between all channels of one baseline; x/2 between baselines sharing a telescope; 0 otherwise;
  - for T3PHI: y ≈ 0.074 within a triangle, ±1/3 between triangles in the same channel, ±y/3 otherwise.
  The correlated part is proportional to the errors σ, not to the model. Accounting for it improves detection limits by up to 2 and removes false detections.
- **PMOIRED** `correlations=True` estimates, per spectrum (tag = file : baseline/triangle ; MJD), ρ = 1 − var(residuals of a smooth polynomial)/median(σ)². It then uses a constant-correlation covariance with ρ off the diagonal ([Me22] code: `oicorr.varVsErr`, `corrSpectra`, `dpfit.invCOVconstRho`). That is a rank-one block with m ∝ σ, estimated from the data. PMOIRED also has explicit transfer-function parameters per baseline or triangle (and optionally per MJD): a polynomial in (λ − λ̄) for each, multiplicative for |V| and V², additive for T3PHI (`oifits._applyTF`, `#TF_…`).
- **Nowak et al.** estimate a full complex covariance and pseudo-covariance per exposure from the DIT sequence (6 × 235 channels). They marginalise polynomial nuisances by projecting onto the orthogonal complement of the polynomial span ([N20a] App. A.1, A.4–A.5): flat priors, exactly.
- **Lachaume** shows that multiplicative correlated errors with covariance built from the *data* bias the fit (Peelle's Pertinent Puzzle). Building it from the *model* removes the bias [La21, abstract]. The same calibration-induced correlations were the subject of [P03] (abstract only read).
- **The pipeline** smooths the transfer function spectrally with a degree-5 polynomial by default (`--maxdeg-tfvis-sc 5`, [PM] §6.13). The vFactor correction extrapolates coherence loss as exp(−ln(v) λ₀²/λ²) ([PM] §9.10 eq. 58).

**(b) Verdict.** The core design **agrees** with the literature:
- per-(frame, baseline) multiplicative V² gains shared across channels, as in [GC22a] and the PMOIRED blocks;
- covariance from the model, which avoids PPP [La21];
- exact marginalisation, where the exposure-level gain is linear in V², so the Gaussian marginal is exact, not an approximation;
- one whitened residual vector plus a log-determinant.

I checked the stable rank-one whitening algebra: with c = τ²/(q(q+1)), 2c − c²s = τ²/q² as required. The determinant follows from the matrix determinant lemma.

The model is **incomplete** in four ways:
1. *Colour.* Coherence loss from piston jitter and anisoplanatism goes as exp(−a/λ²) ([PM] eq. 58; [GW22] eq. 4), and the transfer function is smooth but not flat ([PM] §6.13). An achromatic gain leaves a coloured residual exactly where chromatic science (dust temperatures, spectral indices) lives. The "optional slope" in S §2.4 is in the wrong variable.
2. *Coupling between baselines.* Telescope-based jitter and injection give gains of the form g_kl = g_k g_l. That is the x/2 correlation between baselines in [K20]. Independent per-baseline gains miss it.
3. *Closure offsets.* Four independent offsets per (frame, triangle) are not what baseline-based non-closing errors produce. A per-baseline phase error e_b ~ N(0, τ²) gives triangle offsets δ = C e, with covariance τ² C Cᵀ: rank 3, correlated between triangles. The one documented closure systematic, a baseline-to-baseline wavelength mismatch times group delay ([PM] §5.1), is of this form, and chromatic. The measured within-triangle correlation is small (y ≈ 0.07 [K20]), and no GC paper fits closure offsets. A default-on `phi_offset` is unsupported by the literature.
4. *Noise correlations as well as gains.* [K20]'s within-exposure correlation is σ-proportional (m ∝ σ). It is a different block from a calibration gain (m ∝ V²_model), and both are present.

**(c) Changes.** Generalise "rank-one" to **low-rank per frame** (Woodbury). The cost is still linear in the number of channels.
1. **V² modes per frame** (blocks over all baselines and channels of one frame), as a linearised log-gain, δV²/V² = Σ over modes:
   - per telescope k: a_k on the three baselines containing k (m = V²_model there), width τ_T;
   - per baseline: e_b (m = V²_model), width τ_B. This is S §2.4's term;
   - per telescope: a chromatic mode, m = V²_model·((λ₀/λ)² − 1), width τ_χ, from the exp(−a/λ²) shape;
   - optional per baseline: a noise-correlation mode with m = σ (PMOIRED/[K20]), width ρ.
   The rank per frame is at most 4 + 6 + 4 (+ 6). With V = [τ m₁, …], the whitened residual is x − Q diag(σⱼ²/(qⱼ(qⱼ + 1))) Qᵀx, where Q Σ Pᵀ is the thin SVD of D^{−1/2}V and qⱼ = √(1 + σⱼ²). This reduces exactly to S §2.4's stable formula at rank 1, and log det adds Σ log(1 + σⱼ²).
2. **Closure-phase modes per frame:** a baseline-based non-closing error mapped through the closure matrix, m = (C e_b) ⊗ 1, rank ≤ 3. Optionally a 1/λ-slope version (group-delay-like). **Default off**, with τ_φ ≲ 1° when on, since GRAVITY's closure accuracy is better than 0.5° [G17, abstract].
3. **Closure redundancy:** within each frame use the exact ±1/3 triangle structure, or keep three independent triangles (footgun 9). That is a projection, so it must be whitened properly (footgun 3).
4. **Gain on V or on V²:** state the convention. [GC22a]'s prior N(1, 0.1) on |V| is about N(1, 0.2) on V². Use the log-gain linearisation and document its validity (τ ≲ 0.2).
5. **Seed the widths from the data**, as PMOIRED does: estimate ρ from the scatter about a smooth fit, compared with σ. Report it as a diagnostic even when the widths are fitted.
6. **Tests**, beyond those in Stage 6d: Woodbury against a dense covariance for rank > 1; a PPP test showing that a data-built covariance biases the fit low and the model-built one does not [La21]; recovery of injected telescope-based chromatic gains, where per-baseline achromatic gains would leave a coloured residual.

## 5. Times and frames (S §2.5, Stage 6a.0)

**(a) Literature.**
- Frame selection is per baseline, "thus will have a different effective time after the averaging process" ([PM] §9.12).
- Closure phases use only frames where all three baselines are accepted ([PM] §9.18).
- `gravity_vis --force-same-time` (default FALSE) forces one TIME and MJD for all baselines and quantities ([PM] §6).
- Tables are in fixed order: NEXP×6 rows for OI_VIS/OI_VIS2 with baselines in beam pairs 12, 13, 14, 23, 24, 34; NEXP×4 rows for OI_FLUX ([PM] §8.2).
- On the archival product, the six baselines of one exposure carried MJDs 0, 0, 0, 7.9, 7.9 and 10.1 s apart, with `INT_TIME` = 180 s. Triangle MJDs followed their own pattern.
- PMOIRED keys each spectrum by file, baseline and MJD ([Me22] code: `oimodels` residual tags).

**(b) Verdict.** **Agrees,** and the pipeline manual explains the cause. INT_TIME-window matching is right.

**(c) Changes.**
1. For GRAVITY products, the frame label can be exact: (file, row // 6) for baseline tables and (file, row // 4) for triangle and flux tables. Use it once the reader has checked the STA_INDEX pattern repeats. Fall back to INT_TIME windows otherwise.
2. A frame label must include the file and `INSNAME`, so that split polarisations (P1, P2) of one exposure share a frame (they share atmosphere and fringe tracking). Stage 6d may then let P1 and P2 share telescope modes (§4).
3. Keep each baseline's own u, v as stored: the pipeline computed them for that baseline's frames. `frame_mjd = "mean"` is for orbit time only.

## 6. Wavelength scale (S §2.6)

**(a) Literature.**
- The wavelength scale is derived on the calibration unit with the metrology laser as the fiducial reference ([G17] §2.4; [PM] §9.4).
- The pipeline quotes 0.1 nm accuracy between baselines (half a pixel at HIGH resolution) and 0.5 nm absolute (one resolution element at HIGH) ([PM] §5.1). A colour-dependent effective-wavelength correction exists but is off by default (`--color-wave-correction`, [PM] §6).
- PMOIRED fits a cubic wavelength polynomial in its telluric model and offers `wlOffset` and a telluric-calibrated wavelength (`useTelluricsWl`) ([Me22] code: `tellcorr`, `__init__`).
- Lacour et al. state that "the plate scale and true north error is negligible" at the 50 µas level for a 390 mas separation, "as the spatial frequencies are defined by the physical position of the telescopes" ([L19] §3). That implies a scale error well below 10⁻⁴ for that mode **[their claim; not independently checked]**.

**(b) Verdict.** **Partly agrees.** Applying λ' to both spatial frequencies and spectra is right. But:
- the error model is too narrow: the documented error is mostly an absolute offset (0.5 nm) plus a per-baseline mismatch (0.1 nm), not a pure scale;
- the default prior N(1, 10⁻³) is about 4× wider than the pipeline's absolute figure at K;
- the per-baseline mismatch is a closure-phase systematic, not an angular-scale one.

**(c) Changes.**
1. Use λ' = λ(1 + s) + δ per dataset, with instrument-mode priors. For GRAVITY MEDIUM and HIGH: s ~ N(0, 2×10⁻⁴) **[my reading of [PM] §5.1; Ben to confirm]** and δ ~ N(0, 0.5 nm), with δ fixed to zero unless lines constrain it.
2. LOW mode, and any binned data, get their own entry, since effective channel wavelengths then depend on the source colour.
3. Do not model the 0.1 nm per-baseline mismatch as a wavelength nuisance. It enters as the group-delay-proportional closure mode of §4(c)2.
4. Warn that s is degenerate with a fitted line velocity when lines are fitted.

## 7. The dual-field calibrator recipe (S §2.7)

**(a) Literature.**
- **Exoplanets.** On-star exposures are interleaved with each planet exposure (before and after). The planet's coherent flux is referenced to the star's phase, averaged over the two neighbouring star exposures ([N20a] App. A.2). The amplitude reference is the on-star coherent flux. The throughput ratio between the two is corrected with the FT flux ratio (grey only) [N20b App. B].
- **Off-axis exoplanet work.** The phase reference comes from a binary calibrator instead [N24 §3.1].
- **GC dual-beam.** S2 serves as phase calibrator. Each of the N target frames is calibrated with each of the M S2 frames, and the results are averaged. The calibration adds "a systematic uncertainty of 60 µas, divided by the square root of the number of available calibrations" ([GC22b] App. A.1.2).
- **The pipeline.** The TF is interpolated with weights exp(−2(T − T_c)²/Δ²)/median(σ²), Δ = 3600 s by default; phases are averaged as phasors ([PM] §9.20, eqs. 78–80).
- **GRAVITY Wide.** Coherence loss from anisoplanatism is 0.2–0.4 at separations of tens of arcseconds and follows exp(−2π²σ_p²(θ)/λ²) [GW22 §4.1].
- **Field-dependent effects.** Aberrations within the fibre's field shift astrometry by up to about 0.5 mas [GC21, abstract]; static aberrations change amplitude and phase with field position [GC20 App. A.3].

**(b) Verdict.**
- Steps 1, 2 (phasor averaging of calibrator closure phases), 4 and 5 **agree**.
- Step 3, "the calibrator scatter as the known τ", is **incomplete**. The relevant τ is the error in *predicting* the transfer function at the target's time, not the calibrator's scatter.
- Transferring V² from the star frames to the target frames is **risky**. The star is on the fibre axis and on-axis for the FT; the target is not. Anisoplanatism (separation-dependent and chromatic), off-centre injection and field-dependent aberrations all differ. The exoplanet teams avoid absolute V² calibration for exactly this reason: they work with contrast spectra and phase-referenced coherent flux.
- The caveat "close enough on the sky" needs numbers. There are also uncorrelated DIT differences (bright star at short DIT, faint target at long DIT), which change the vFactor correction.

**(c) Changes.**
1. Estimate the known τ by **leave-one-out**: predict each calibrator frame's TF from the others with the same interpolator, and use the r.m.s. prediction error per baseline. Report the predicted chromatic mode too (§4).
2. Say in the docs that, for off-axis targets, V² calibrated from the on-axis star carries an anisoplanatic loss that the recipe does not remove. Either fit it as the chromatic telescope mode of §4 with a free width, or restrict the recipe to closure phases, phases and contrast spectra. Give a separation threshold from [GW22]'s eq. 4 for typical τ₀ and θ₀ **[to compute; not in this note]**.
3. Propagate the calibrator's diameter uncertainty ([PM] §9.20 does not): one nuisance per calibrator and night, entering every target frame through ∂V²_cal/∂θ. This is the multiplicative, all-frame correlation of [P03] and [La21].
4. Use the pipeline's TF weighting as the default interpolator, so that results match `gravity_viscal`. Optionally add a GP in time.
5. Document that the recipe needs the products to have been reduced with the same `--vis-correction-sc` (VFACTOR, FORCE or NONE) for star and target, since that choice changes the chromatic shape of V² ([PM] §6, §9.10).

## 8. Smearing (S §2.8)

**(a) Literature.**
- The GC binary fits integrate the complex visibility over a top-hat bandpass of the measured spectral resolution, weighted by each component's spectrum λ^(−1−α). They form numerator and denominator separately ([GC20] App. A.1, eqs. A.2–A.3). They also include "bandwidth smearing" ([GC18a] App. A.4; [GC22b] App. A.1.1) and average over finite channels in imaging [GC22b App. A.2].
- PMOIRED uses a Gaussian spectral kernel of 1.4 pixels for GRAVITY ([Me22] code: `wl kernel`).
- GRAVITY MEDIUM samples about 2 channels per resolution element, so neighbouring channels are correlated ([GC18b] Methods: significance re-checked with half the channels).

**(b) Verdict.** The B·θ/λ-versus-R rule **agrees** in spirit. It is **incomplete** in three ways:
- *Order of operations.* What the instrument averages over a channel is the coherent flux and the photometric flux, **separately**; V or V² is formed afterwards ([PM] §9.16–9.17). Smearing V² directly is wrong for chromatic scenes.
- *Time smearing is missing.* The phase-referenced coherent flux is summed over all accepted frames of an exposure ([PM] §9.16 eq. 70, as I read it), and u, v rotate during it. Over 180 s on a 130 m baseline, Δu ≤ Bω_⊕T ≈ 1.7 m. A component 30 mas from the phase centre then sweeps about 0.7 rad, a V² loss of about 4% **[my estimate, not from the literature]**. For V² the pipeline's estimator may average |C|² per frame (incoherently between DITs), which would limit time smearing to one DIT **[eq. 72 is ambiguous in the extracted text; unverified]**.
- *Binning.* Binning raises the effective bandwidth; the rule must use the binned channel width.

**(c) Changes.**
1. 6a's smearing integrates the complex coherent flux and the flux separately over the channel's response (Gaussian or top-hat, configurable; GRAVITY default Gaussian of 1.4 px FWHM), then divides.
2. Add time smearing for long coherent integrations: the number of fringes swept, Bω_⊕T θ/λ, beside B θ/(λR). Check it on the same real dataset.
3. In dual-field mode, measure θ from the **phase centre** (the SC fibre position after metrology referencing, [PM] eq. 62), not from the FT star.
4. Recommend binning MEDIUM data by 2 (one resolution element) before fitting when channel correlations are not modelled, and say why (§0 item 4).

## 9. Stage 6a and the order of work (`imaging_plan.md`)

**(a)** PMOIRED treats GRAVITY-specific steps (P2VM flat for FLUX, tellurics, per-baseline TF polynomials, correlations) as reader or fit options ([Me22] code). The GC and BLR teams treat the FT data as a separate low-resolution dataset (FT V² selected by group delay, errors rescaled) ([GC21b] §2, §4.1).

**(b)** 6a.0 (times and frames) first is **right**. 6a's reader items are **incomplete**: `INSNAME` selection, `PHITYP`/`PHIORDER`/`VISREFMAP`, `FLUXDATA` and `CALSTAT` are missing. [`pmoired_parity.md`](pmoired_parity.md)'s "GRAVITY-specific tools: not planned" is **risky** for the fibre beam and for FLUX: without them, OI_FLUX (S §2.2) cannot be used as intended.

**(c)**
1. Add to 6a.0: the `insname=` selector (footgun 1), and reading of `PHITYP`, `AMPTYP`, `PHIORDER` and `VISREFMAP`.
2. Add to 6a:
   - whitened projections in place of √diag propagation (footgun 3);
   - the normalisation operator of §2;
   - the D⁻¹-weighted closure-free projection of §3;
   - FLUXDATA/CALSTAT handling;
   - the optional primary beam of §1.
3. Revisit "GRAVITY-specific tools: not planned". Keep telluric fitting and polarisation averaging out. Bring in the primary beam and the FT-as-separate-dataset convention.
4. 6d as planned, but low-rank (§4).

## 10. OIFITS conventions and the GRAVITY layout (O §5.3)

**(a) Literature and inspection.**
- In the archival GRAVITY product:
  - `OI_VIS2` STA_INDEX rows run (K0,J2), (K0,G1), (K0,A0), (J2,G1), (J2,A0), (G1,A0), i.e. beam pairs 12, 13, 14, 23, 24, 34, with beam 1 on the highest station index;
  - `OI_T3` triangles are beams (1,2,3), (1,2,4), (1,3,4), (2,3,4);
  - each triangle's (a,b), (b,c) and (a,c) baselines appear in `OI_VIS2` in that orientation, which is what `read_oifits` assumes.
- The exoplanet papers model GRAVITY's phase-referenced visibility as S(λ) exp(−i 2π/λ (Δα U + Δδ V)) ([N20a] eq. A.16), with astrometry that agrees with independent orbits. drpangloss's `offset_phase` uses the same sign, exp(−2πi(u·dra + v·ddec)).
- OIFITS2 defines the differential phase as a polynomial in 1/λ ([D17] §3.2). I did not find an explicit statement of the complex-visibility sign convention in [D17]; it defers to OIFITS1 (Pauls et al. 2005), which I did not read **[unverified]**.

**(b) Verdict.** The plan **agrees**. The synthetic "GRAVITY-layout" writer should copy the beam ordering above. The sign agreement with [N20a] is encouraging, but it is not a substitute for the real anchor binary that O §5.3 item 2 requires.

**(c) Changes.**
1. Build the synthetic GRAVITY file from the beam ordering above, including split-polarisation `INSNAME`s and an FT table that the reader must skip.
2. Add a test that a differential `PHITYP` is rejected as an absolute phase.
3. When the real anchor is chosen, prefer a dual-field observation, so that the absolute (phase-referenced) `VISPHI` sign is anchored as well as T3PHI.

## 11. Sources and how far each was checked

Read in full text (arXiv PDF, converted to text, and the relevant sections read):
- **[PM]** ESO, *GRAVITY Pipeline User Manual*, issue 1.11.0 (2026-07-28): §5.1–5.3, §6 (`gravity_vis`, `gravity_viscal` parameters and pseudo-code), §8.2, §9.4–9.20.
- **[G17]** GRAVITY Collaboration (Abuter et al.) 2017, A&A 602, A94, arXiv:1705.02345: abstract, §2.4–2.5.
- **[GC18a]** GRAVITY Collaboration 2018, A&A 615, L15, arXiv:1807.09409: App. A.3–A.6.
- **[GC20]** GRAVITY Collaboration 2020, A&A 636, L5, arXiv:2004.07187: App. A.1–A.3.
- **[GC21]** GRAVITY Collaboration 2021, A&A 647, A59, arXiv:2101.12098: abstract and §1 only.
- **[GC22a]** GRAVITY Collaboration 2022, A&A 657, A82 (*Deep images of the Galactic Center*), arXiv:2112.07477: §2.3–2.5, App. B, §6 (Conclusions).
- **[GC22b]** GRAVITY Collaboration 2022, A&A 657, L12 (*Mass distribution*), arXiv:2112.07478: App. A.1–A.2.
- **[L19]** GRAVITY Collaboration (Lacour et al.) 2019, A&A 623, L11, arXiv:1903.11903: §2–3.
- **[N20a]** GRAVITY Collaboration (Nowak et al.) 2020, A&A 633, A110 (β Pic b), arXiv:1912.04651: §2, App. A.1–A.5.
- **[N20b]** Nowak et al. 2020, A&A 642, L2 (β Pic c), arXiv:2010.04442: App. B.
- **[W21]** Wang et al. 2021, AJ 161, 148 (PDS 70), arXiv:2101.04187: §2.2–2.3, §4 (spectral fits).
- **[N24]** Nasedkin et al. 2024, A&A 687, A298 (HR 8799), arXiv:2404.03776: §3.1. The journal reference is from memory **[unverified]**.
- **[K20]** Kammerer, Mérand, Ireland & Lacour 2020, A&A 644, A110, arXiv:2011.01209: §2.1–2.2.
- **[GC18b]** GRAVITY Collaboration (Sturm et al.) 2018, Nature 563, 657 (3C 273), arXiv:1811.11195: Methods.
- **[GC20b]** GRAVITY Collaboration 2020, A&A 643, A154 (IRAS 09149−6206), arXiv:2009.08463: §2.1–2.3, App. A.
- **[GC21b]** GRAVITY Collaboration 2021, A&A 648, A117 (NGC 3783), arXiv:2102.00068: §2, §4.1. The journal reference is from memory **[unverified]**; the arXiv text is what I read.
- **[GW22]** GRAVITY+ Collaboration 2022, A&A 665, A75 (GRAVITY Wide), arXiv:2206.00684: §4.1, App. B.
- **[D17]** Duvert, Young & Hummel 2017, A&A 597, A8 (OIFITS 2), arXiv:1510.04556: §3, §5 keywords, §7.2 (OI_CORR).
- **[La21]** Lachaume 2021, PASA, arXiv:2104.07082: abstract and introduction only.
- **[Me22]** PMOIRED (Mérand 2022, SPIE 12183): source code at github.com/amerand/PMOIRED, read 2026-10-03 (`oicorr.py`, `dpfit.py`, `oimodels.py`, `oifits.py`, `tellcorr.py`, `__init__.py`). The SPIE paper itself was not read.
- **Archival product.** One ESO Phase 3 GRAVITY `DUAL_SCI_VIS` file (ATs, MEDIUM, combined polarisation, pipeline parameters in its header), inspected for `INSNAME`s, `PHITYP`, STA_INDEX ordering and per-baseline MJDs. It is not cited for any science.

Abstract or secondary only:
- **[P03]** Perrin 2003, A&A 400, 1173, arXiv:astro-ph/0301140: abstract via search; calibration-induced correlations between transfer functions sharing a calibrator.
- Lapeyrère et al. 2014, SPIE 9146 (the pipeline paper): cited through [PM] and [N20a]; not read. [PM] supersedes it for current behaviour.
- GRAVITY+ Collaboration 2025 (GPAO first light), arXiv:2509.21431: abstract only. I found nothing in it on calibration of the observables.
- Lachaume et al. 2019, MNRAS 484, 2656; Blackburn et al. 2020, ApJ 894, 31; Millour et al. 2006, 2008 (AMBER differential phase): cited through [La21], [GC22a] and [GC18b]; not read.

Not covered: Kervella's work on correlated calibration errors (asked for in the brief; I did not get to it), MATISSE, and any GRAVITY+ pipeline changes after manual issue 1.11.0.

## 12. Decisions (Ben, 2026-10-03)

The open questions as asked, and the answers.

1. **Per-DIT products.** *Yes:* estimate empirical covariances from the individual DITs (`P2VMRED`/`ASTROREDUCED`), as [N20a] and [K20] do. Instrument-specific reduction like this belongs to a separate GRAVITY project, started once the core of the package (renamed virgil) works; until then, the fitted low-rank nuisances of §4 are the fallback.
2. **The fibre beam.** *Yes:* include the optional primary beam (§1). GRAVITY-specific tools are now in scope, reversing "not planned" in [`pmoired_parity.md`](pmoired_parity.md). The generic primary beam goes in Stage 6a; the GRAVITY-specific parts go in the GRAVITY project.
3. **Closure-phase offsets.** Undecided; expanded in §13. Recommendation: leave them out, and add the baseline-based form only if calibrators show non-closing errors.
4. **VISPHI nuisance basis.** *Yes, eventually:* an empirical template from a PCA of archival calibrators, planned in [`gravity_calibrator_pca.md`](gravity_calibrator_pca.md), as part of the GRAVITY project. Until then, polynomials in 1/λ.
5. **Wavelength-scale prior.** Unknown; asked of the GRAVITY team ([`questions_for_gravity_team.md`](questions_for_gravity_team.md)). Until then N(0, 2×10⁻⁴) as a scale, with the offset form of §6.
6. **FT data.** *Skip* them. `read_oifits` refuses to merge them with the SC tables.
7. **Gain convention.** Gains are on **absolute visibilities**, |V|, as in [GC22a] (a prior N(1, 0.1)), stated in log space.

## 13. Closure-phase offsets, expanded

**Why closure phases normally need no offset.** A closure phase sums the baseline phases around a triangle. Any error attached to one *telescope* (atmospheric piston, a delay-line drift, a telescope's instrumental phase) enters two of the triangle's baselines with opposite signs and cancels exactly. That is what closure phases are for.

**What survives.** Only *non-closing* errors, attached to a baseline rather than a telescope, reach the closure phase:
- a baseline-to-baseline mismatch of the wavelength scale times the group delay: about 3° at 40 µm of delay for a 0.1 nm mismatch ([PM] §5.1), chromatic, and larger off the fringe-tracking reference;
- baseline-dependent instrumental phases, such as differential birefringence between the polarisations, if the polarisations are combined;
- bispectrum bias at low signal-to-noise.

**What S §2.4 proposed, and why it is not right.** A free offset per (frame, triangle), shared by all channels:
- *Wrong structure.* Baseline errors e_b give triangle offsets δ = T e: correlated between triangles, rank 3 for four telescopes, not four independent numbers.
- *No precedent.* No Galactic Centre or GRAVITY-RESOLVE paper fits closure offsets ([GC22a] uses phasors of the closure phases with no offset), and GRAVITY's closure accuracy is quoted as better than 0.5° [G17].
- *It absorbs signal.* An offset shared by all channels is nearly degenerate with any real closure phase that changes little across the band, such as an asymmetric disc's, so marginalising it would weaken exactly the asymmetry measurements we want.

**Recommendation.**
1. Leave closure offsets out by default.
2. Test the need empirically: the closure phases of unresolved calibrators should be zero. Compare their scatter, per frame, with the reported errors, over many calibrators (this fits naturally into the calibrator PCA).
3. If calibrators do show a non-closing excess, add the baseline-based form δ = T e with e_b ~ N(0, τ²), τ ≲ 1°, optionally with a 1/λ (group-delay-like) shape. It fits the low-rank machinery of §4(c), marginalised exactly.



