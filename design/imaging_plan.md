# drpangloss imaging: staged execution plan (revision 4)

## Context
We are adding regularised maximum-likelihood image reconstruction, and optional Bayesian sampling, to drpangloss.
- **Data:** AMIGO DISCO data from JWST AMI, closure phases from long-baseline interferometers (VLTI, CHARA), and aperture-masking data.
- **Composition:** analytic components (stars, companions) stay analytic, and the image is one more component of a `System`.
- **Build strategy:** incremental, with a **minimal working example (MWE) that does something scientifically useful at every stage**, and a feedback checkpoint on each before the next begins.
  - It starts by reproducing dorito's AMI DISCO results.
  - Then grey long-baseline imaging with closure phases.
  - Then Gaussian-process priors and sampling.
  - Then polychromatic imaging.
- **Priorities:** robustness, minimal technical debt, and readability for new PhD students and sceptical astronomers.

The design rationale was established in the earlier research: the gauge survey, IFT versus Gaussian processes, DFT against NUFFT accuracy, optimisers, and zodiax 0.6. Stage 0 writes it to `design/image_reconstruction.md`, and this plan does not repeat it.

## Standing design decisions
1. **Numerical precision.**
   - Every Fourier transform and matmul uses `precision=lax.Precision.HIGHEST`, which defeats TF32 on A100/H100 GPUs.
   - Fitting entry points (`fit`) take `dtype="float64"` by default and run inside a local `jax.enable_x64(True)` context, casting inputs. `dtype="float32"` is opt-in.
   - Forward-model code is dtype-agnostic and still passes the float32 test suite; x64 is never set globally.
   - `AGENTS.md` is updated to say that library code works in both precisions and that imaging entry points default to local x64.
2. **One likelihood.** `whitened_residuals(model, data)` returns:
   - (model − data)/σ, and Δ/σ for projected (kernel/DISCO) phases;
   - **2 sin(Δ/2)/σ** for unprojected closure phases, which is exactly a von Mises likelihood and is smooth at ±π.

   `model_loglike` is redefined on top of it, so grids, limits, `numpyro_model` and fitting all share it.
3. **One specification for fitting and sampling.** `fit(model, priors, data, regularisers)` and `numpyro_model(model, priors, data, regularisers)` take the same arguments:
   - Free parameters are exactly the keys of `priors`, a dict of NumPyro *distributions*; numpyro is used only as a distributions library.
   - Support-respecting transforms come from `biject_to`.
   - `fit` finds the MAP; `numpyro_model` is the posterior for samplers, and accepts only genuine prior regularisers (it rejects MEM, TV and TSV).
4. **Optimisers:** `fit(..., method=...)` with:
   - `"lm"`: optimistix Levenberg–Marquardt with a matrix-free, fixed-length `lx.Normal(lx.CG)` or `lx.LSMR` inner solve. It never materialises a Jacobian and is the default when residuals exist.
   - `"lbfgs"`: optimistix LBFGS, for MEM, TV and free error inflation.
   - `"adam"`: optax, now a declared runtime dependency.
5. **Image parameterisation.**
   - `Image(log_brightness, pixel_scale_mas, support=None, flux, dra, ddec, rotation_deg=0)` is a `Component`, and `brightness = softmax(η, where=support)`.
   - `log_brightness` is either a plain array (free log-pixels, MAP-only) or an object with `evaluate(pixel_scale_mas=...)`, e.g. `GaussianField` in Stage 5. That is the zodiax 0.6 `Expression` protocol; migrate when it is released.
6. **Fourier backends.**
   - The separable DFT is the default and the oracle.
   - On uv lattices (AMI/AMIGO DISCOs) an `Image` whose `rotation_deg` matches the lattice uses the exact two-sided MFT (Stage 2b).
   - A NUFFT backend (jax-finufft) was built in Stage 2 and **removed** after Stage 3 (issue #75): on an A100 it cost a flat ~7 ms per call because jax-finufft re-plans every time (jax-finufft #157). Its code is in the git history of PR #71; revisit only when jax-finufft can reuse plans.
   - Padded FFT plus interpolation is never used.
7. **Rules for users:**
   - The default field of view is the smaller of 500 mas and the interferometric field of view, λ/B_min (`imaging.field_of_view`); pixels are finer than Nyquist (`imaging.nyquist_pixel_scale`).
   - Reconstructed images are shown with the beam (`imaging.beam`, the FWHM of the dirty beam's core) shaded in the lower left, and next to a residual map on a symmetric diverging scale: signed residuals for MAP images, z-scores whenever there are uncertainties.
   - Unresolved things are analytic; resolved emission goes in pixels.
   - Something must fix the origin: an analytic star, a `Centroid` prior, or a centred mean.
   - Initialise from a parametric fit.
8. **Dependencies:** optax (and lineax) become required. `blackjax` (`[sampling]`) is an optional extra. zodiax stays at `>=0.4`.

## Branching and workflow
- **Branch.** Create `imaging` in `/Users/benpope/code/drpangloss`, branched from `chromatic-scenes`. It needs the chromatic work (`Spectrum`, `Resolved`, multi-channel `OIData`), and `chromatic-scenes` is 3 commits ahead of `main`. Once `chromatic-scenes` merges, rebase `imaging` onto `main`.
- **Each stage is a PR** from `imaging-sN-<name>` (git cannot hold both `imaging` and `imaging/…` branches), stacked on the previous stage's branch and retargeted to `imaging` as earlier stages merge. Each PR has:
  - the code;
  - its tests;
  - an MWE notebook (or a section of one) in `notebooks/`, synced to the docs by `scripts/sync_tutorial_docs.py`;
  - a short entry in `design/image_reconstruction.md`.
- **Merge `imaging` into `main`** at two milestones: after Stage 4 (the AMI and long-baseline MAP imaging are usable) and after Stage 6.
- **Feedback checkpoint** at the end of every stage: I show you the MWE figures and numbers, and the next stage starts only after your comments. That is where stage scope can change.
- **Every stage must pass:** `uv run pytest` (float32 and x64 paths), `ruff check` (0.11.0), the tutorial sync test, and the docs build.

**Time estimates** are Claude-agent working time (writing code, tests and MWEs, running them, iterating). They exclude waiting for your feedback, data transfers, and OzSTAR GPU queue time. They assume no major surprises; the flagged risks say where the surprises could come from.

---

## Stage 0: groundwork and likelihood unification (about 2–3 h)
**Build:**
- The `imaging` branch.
- `design/image_reconstruction.md`: the decisions above, the research summary, and a deferred list with triggers.
- `AGENTS.md`: the precision policy; new modules (`fitting.py`, `imaging.py`, `fields.py`) and the import DAG.
- `likelihood.whitened_residuals`, and `model_loglike` redefined through it (Gaussian-limit normaliser).

**Tests:**
- The new phase term equals the old for small Δ and is smooth across ±π.
- **Regression:** existing grid, Absil and Ruffio outputs and the tutorial fits on the in-repo data are unchanged within tolerance; the PR lists any shifts.

**MWE ("nothing broke, one thing got better"):** re-run the existing binary-search and contrast-limit tutorials with identical results. Add a two-panel figure: the old and new closure-phase χ² along a companion-position slice through a phase-wrap region, showing the kink gone.

**Checkpoint:** do you accept the likelihood change?

## Stage 1: `Image` with the exact DFT; forward simulation of AMI DISCOs (about 3–4 h)
**Build:**
- `_geometry.image_visibilities(brightness, uu, vv, pixel_scale_mas, backend="dft")`: separable, HIGHEST precision, frequencies in cycles/mas.
- `models.Image`: array `log_brightness`, circular `support`, `Image.from_model(model, npix, pixel_scale_mas)` built through `render()`, the `brightness` property, and `_centred_image` (exact native pixels, bilinear resampling for display).
- Exports.

**Tests:**
- DFT against a float64 reference, and against the `render()`-FT pattern.
- A delta pixel equals `PointSource`; a pixelised Gaussian equals `GaussianDisk`; `V(0) = 1`; masked pixels are exactly 0.
- The orientation test required by AGENTS.md.
- Finite gradients for every observable kind (V², amplitude, log-amplitude, CP, kernel `phi_mat`, `mixed_log_complex`).
- `System` mixing; float32 and x64 at extreme baselines.

**MWE ("simulate dorito-style data"):**
- Take the in-repo AMIGO record (`data/calibrated_visibility.npy`: real AMI uv coverage and DISCO operators).
- Put a WR 137-like dusty spiral image (a parametric spiral rendered to pixels) next to an analytic star.
- Compute the DISCO observables, add noise with `OIData.with_model`, and plot data against model.
- Show that the image evaluated at a single pixel reproduces a `PointSource` binary's DISCOs exactly.

**Checkpoint:** do the image API and the sky conventions read naturally?

## Stage 2: NUFFT backend and benchmark (about 3–4 h, plus your GPU run)
**Status:** done (PR #71), including the OzSTAR A100 run. Outcome: the NUFFT is accurate but too slow on GPUs (issue #75), and was removed after Stage 3; the DFT and MFT are the supported paths.

**Build:**
- `backend="nufft"` in `image_visibilities`, via jax-finufft `nufft2`:
  - `iflag=+1`; the image rows pair with v; even N gets the half-pixel phase factor;
  - `eps` defaults to 1e-7 under x64, and must be ≥ 1e-5 under float32 (below that it raises).
- The `drpangloss[nufft]` extra, with a lazy import and a clear error if it is missing.
- `scripts/bench_ft.py` (DFT against jax-finufft, value+grad, npix 64–512, M 10³–10⁵, both dtypes).

**Tests** (skipped without the extra): agreement with the DFT per point, |ΔV| ≤ 3·eps·V(0), including phases on low-|V| baselines; odd and even N; orientation; `check_grads`; vmap; inside `System` with CP and DISCO data. A CI job with the extra installed.

**MWE ("which transform, when"):**
- The Stage 1 simulation evaluated with both backends, with an accuracy table of max |ΔV| and phase error on low-|V| baselines, in float32 and float64.
- Laptop timing curves.
- Slurm commands for you to run the same benchmark on an OzSTAR GPU, including a DFT accuracy assertion that confirms HIGHEST beats TF32.
- A documented rule of thumb for the crossover.

**Risk:** jax-finufft's use of private jax internals and the fact that it re-plans every call (#157). The DFT fallback makes this non-blocking.

**Checkpoint:** do you accept the default-backend rule?

## Stage 2b: exact MFT on uv lattices (added after Stage 2)
**Status:** done (PR #72). AMIGO DISCO data lie exactly on a detector-frame uv lattice rotated by the parallactic angle. `OIData.uv_grid` records it, `SourceModel.model_on_grid` defaults to `model`, and `Image(rotation_deg=...)` matching the lattice uses the two-sided MFT (Soummer et al. 2007): exact, and 3–8× faster on the ν Hor coverage. `amigo.simulated_disco_record` gives a small AMI-like record with diagonal errors, which replaces large fixtures in tests and tutorials; `disco_covariance` is optional.

## Stage 3: `fit`, regularisers; dorito-style AMI imaging on simulated truth (about 6–9 h)
**Status:** implemented on `imaging-s3-fitting` (see the design note's Stage 3 log). Differences from the plan below: L-BFGS uses optax (gradient stopping) rather than optimistix; `l_curve` has `corner` and `discrepancy` criteria; `diagnose` has no field-of-view check (on a uv lattice the shortest spacing bounds the useful field from above); MWE-B uses `amigo.simulated_disco_record` with ν Hor-like errors rather than the real ν Hor operators (large files); the dorito recipe (Adam then BFGS, MEM weight 1e5) is not reproduced literally, since MEM with L-BFGS at an L-curve weight does the same job.
**Build:**
- `fitting.py`: `fit` with `lm`, `lbfgs` and `adam` (§3–4), with loss scaling, unscaled χ² reporting, and `info` (converged flag, steps, χ² per block).
- A small private helper, `_precision.py`: a `run_in(dtype)` context and a `cast_tree(tree, dtype)` function used by the entry points (moved here from Stage 0, where nothing would use it yet).
- `imaging.py`:
  - regularisers `MaxEntropy(prior=None)`, `TSV`, `TV` (ε-smoothed) and `Centroid(sigma_mas)`. Each has `value`, an optional `residuals`, and a `probabilistic` flag.
  - `image_priors(scene)`;
  - `nyquist_pixel_scale(data)`;
  - `diagnose(model, data)`: χ²_red per block, Nyquist, edge flux, centroid and anchor, flip Δχ², the **projected-phase regime check** (|arg V| > 0.8π or |V| < 0.05, where unwrapped DISCO/kernel phases become unreliable), and the backend oracle check.

**Tests:**
- `fit(lm)` recovers a synthetic binary, consistent with the grid fit.
- LM and L-BFGS agree on a TSV problem.
- Gauge: rolling the image leaves the CP loss invariant; a `Centroid` fit stays centred; a star-anchored offset is recovered.
- Positivity transforms work.
- `numpyro_model` rejects MEM, TV and TSV.
- Every `diagnose` warning fires on a constructed bad case.
- float32 and x64 paths agree to tolerance.

**MWE-A ("recover a known image from AMI DISCOs"):** fit the Stage 1 synthetic WR 137-like data. Compare MEM (L-BFGS), TSV (LM) and TV (L-BFGS); include a weight-sweep L-curve loop, the `diagnose` report, and recovery metrics against the truth after centroid alignment.

**MWE-B ("dorito-style AMI imaging against ground truth"):** simulate, rather than use real WR 137 or NGC 1068 data, because real data have no ground truth to test against.
- **Coverage and noise:** reuse the real ν Hor AMI baselines and DISCO operators from the in-repo AMIGO record (`data/calibrated_visibility.npy`, `NuHor_F480M.oifits`). Set per-observable noise to the ν Hor measured σ, scaled to a few SNR levels.
- **Truth scenes:**
  - (i) a WR 137-like dusty spiral;
  - (ii) an inclined ring or disk;
  - (iii) star + faint companion + extended emission.
- **Fit:** dorito's recipe (MaxEntropy plus a centroid prior, grid and weights as in `wr137_disco.py`), alongside TSV/LM and TV.
- **Report:** recovery metrics against the truth (normalised cross-correlation after centroid alignment, flux fraction, and companion astrometry where present) as a function of SNR, plus the regime check.

**Risk:** the regularisation weight depends on the scene. The SNR sweep shows where each regulariser breaks down.

**Checkpoint:** is the fitting API right, and are the recovery metrics good enough? **Merge milestone candidate.**

## Stage 3c: real AMI data, PDS 70 (once DISCO deconvolution is mature)
**Data:** the AMIGO DISCO products for PDS 70 in `/Users/benpope/code/nuHor/data/PDS70/` (local only; never committed). Read only the fields needed (operators, coefficients, σ, uv, wavelength, rotation), and avoid listing or printing large files.
**Scope:** drpangloss supplies a fast JAX library with the features interferometrists expect; synthetic truths for calibrating PDS 70 reconstructions are being built separately, so this stage does not do that.
**Build:** an agent that deconvolves PDS 70 in each filter, separately and jointly (Stage 6's joint multi-filter machinery when available), with a wide range of options: regularisers (maximum entropy expected best, then TSV, then TV), weights from L-curves (discrepancy and corner), fields of view and pixel scales, starts (flat, parametric fit), analytic star or not, supports, and centroid priors. "Beat it to death": the aim is a general picture of what is robust across choices.
**Compute:** demo locally first on a reduced set. If the full grid would take hours or exceed the laptop's RAM, hand the user an OzSTAR GPU script (`ozstar` skill) rather than running it here.
**Report:** a notebook (not in the docs) comparing the reconstructions across options and filters, with beams, residual maps and `diagnose` output.

## Stage 4: grey long-baseline imaging with closure phases (about 4–6 h)
**Build:**
- Only glue and documentation should be needed; this stage tests generality.
- `Image` inside `System` with an analytic star (SPARCO-style, grey, then a `PowerLaw` spectral index for star and envelope).
- Multi-file joint fits (a list of `OIData`), the support-hole option under the star, and `from_model` initialisation from a parametric fit.
- The ν Hor MATISSE coverage fixture and the synthetic-coverage generator `tests/_coverage.py` (see "Coverage fixtures" below; moved here from Stage 0).

**Tests:**
- Joint multi-file fits.
- Recovery of a star + inclined disk + companion on synthetic VLTI-like coverage (4 telescopes, many channels).
- The star-anchored gauge (no centroid prior needed).
- The support hole prevents flux piling up under the star.

**MWE-A ("grey VLTI imaging"):** synthetic star + ring + companion on **real MATISSE ν Hor uv coverage**, with noise copied from the per-point σ of the nuHor OIFITS. The workflow is parametric fit, then image with TSV/MEM, then `diagnose`.

**MWE-B ("SNR and coverage stress test"):** the same simulated scenes with ν Hor-derived noise scaled up and down. Drop telescopes or epochs to thin the uv coverage, and report where reconstructions fail and what `diagnose` flags. There is no real-data imaging in this plan.

**Risk:** bandwidth smearing in MATISSE LOW-resolution data is not modelled (an existing limitation), so the field of view has to be kept ≲ Rλ/B. The MWE documents this.

**Checkpoint:** is closure-phase imaging robust enough? **Merge `imaging` into `main`** (milestone 1: MAP imaging for AMI and long-baseline data).

**Log (2026-10-01):**
- No new fitting code was needed: `fit` and `l_curve` already take a list of datasets, and the worst-fitted one sets the discrepancy weight.
- Coverage is synthetic. `coverage.vlti_oidata` gives four-UT Earth-rotation tracks over many channels, and `coverage.nrm_oidata` gives the 21 V² and 35 closure phases of the NIRISS mask. This is the fallback in "Coverage fixtures", so no ν Hor fixture is committed. Stage 4c uses real PIONIER data instead.
- `models.circular_support(..., inner_radius_mas)` makes the support hole, and `starting_image(..., hole_mas=...)` applies it. Both MWEs use it, at half the beam's minor axis.
- Fitting the envelope flux (a prior on `"env.flux"`) turned out to be what removes the spurious spot next to the star. With the flux fixed at `starting_image`'s estimate, the excess piles up at the centre; with the flux fitted, NCC rises from 0.86 to 0.93 in the SAM MWE. A hole of half a beam under the star (`starting_image(..., hole_mas=...)`) then clears the remaining core flux, which is degenerate with the star's. In the VLTI MWE it removes the bias in the SPARCO ratio, which goes from 0.545 to 0.509 against a truth of 0.5.
- MWEs:
  - `mwe_sam_v2_cp`: two NRM rolls, compared with AMIGO-style modes (NCC 0.93 against 0.98).
  - `mwe_vlti`: two nights, eleven channels, with a SPARCO `PowerLaw` ratio and index fitted together with the pixels. NCC 0.89, with ratio 0.509 against 0.5 and index 1.87 against 2.
- The field is limited to λ/B_min, so VLTI scenes are only 2–3 beams across.
- MWE-B (the SNR and coverage stress test) is still to do.

## Stage 4b: a rotating scene over two epochs
A spiral that turns by 60° between two epochs, fitted jointly with the rotation known and then unknown. For an unknown rotation:
- the angle must be traceable;
- the likelihood can be multimodal in angle, so a coarse scan comes before a joint refinement;
- an Archimedean spiral has a near-degeneracy between rotation and expansion.

**Log (2026-10-01):**
- `fit` (and so `l_curve`) accepts a model function that returns one model per dataset, sharing parameters; regularisers act on the first.
- `models.Rotated(source, rotation_deg)` wraps any model with a traceable angle. It rotates the uv coordinates, so the MFT is not used for that epoch.
- In `mwe_rotating_epochs`, two AMI epochs with the known 60° give NCC 0.92, against 0.90 for one epoch at the same weight.
- With the angle unknown, a 15° scan of fixed-angle fits (warm-started, with a 300-step limit) has a single sharp minimum at 60°. Reconstructing the image again with the angle free (an L-curve started from the scan's best angle) gives 59.89° and the same NCC, 0.92, as with the angle known.
- No secondary minimum appeared for this spiral: its fading ends break the rotation–expansion degeneracy.

## Stage 4c: real PIONIER data (SPARCO), and the FlaredDisk merge
Real-data notebooks live in `nuHor/notebooks/`, next to the data, and are not committed here:
- `pionier_iras08544`: one star;
- `pionier_binary`: Hillen et al.'s model;
- `pionier_iwcar`, for Toon.

They compare results with the papers' text, not their images.

**Library additions:**
- `spectra.BlackBody`, a Planck F_λ spectrum normalised at `wavel0` with a fittable `temperature`.
- SPARCO verification tests: the published mixing formula with blackbody components, per-channel weighting of multichannel data, temperature gradients, and the Rayleigh–Jeans limit.
- PR #80's `FlaredDisk` components (Blakely et al. 2024), merged into this stack. Edge-on inclinations are now rejected, and g < 0 is documented.

**IRAS 08544-4431** (Hillen et al. 2016; 27 files, 828 V² and 504 closure phases):
- **Parametric model:** a 7250 K primary; a secondary, a ring with m = 1, 2 modulations and a resolved background, all with blackbody spectra; and the ring anchored to the binary's centre of mass (κ = q/(1+q) = 0.75). It converges to one solution with χ² per point 2.50.
  - Fractions: 57.0 / 6.1 / 20.7 / 16.2%, against the paper's 59.7 / 3.9 / 20.9 / 15.5%.
  - Temperatures: secondary 3430 K, ring 1098 K, background 2620 K, against 4000, 1120 and 2400 K.
  - Binary separation 0.72 mas (paper 0.81 mas).
  - Ring 14.33 mas across, FWHM 3.03 mas, i = 20.1° (paper 14.15 mas, 3.2 mas, 19°).
- **Images:** a λ⁻⁴ primary and a power-law environment, as in the paper. With the primary alone, the primary has 61.4% (paper 61%). With the binary subtracted, the knot next to the primary is gone.
- **Open:** the companion's PA is 219°, against the paper's 56°, though the ring's bright side agrees. To raise with Toon.

**IW Car** (De Prins et al. 2026; the 23 files within the paper's 81-day window):
- No single-ring model fits, as the paper found.
- The parametric secondary lands at (1.06, −1.69) mas with 2.3% of the flux (paper: (1.12, −1.90) mas, about 2%), with the primary at 62% (paper 63.6%).
- The SPARCO image recovers the inner arcs at about 5 mas.

**SPARCO convention (for Toon):** a spectral index of the environment is only defined relative to the star's assumed spectrum, since the ratio goes as λ^(d_env − d_star). Both apparent mismatches with the papers came from this, not from the code:
- IRAS 08544: a blackbody primary in place of λ⁻⁴ shifts d_env by about 0.7.
- IW Car: the paper fixes d_prim = −3.17, so their d_rim of 0.89 is 4.06 relative to the star, against our 4.0 with a λ⁻⁴ star.

A background split off as a `Resolved` component also changes what the image's index means, and a smooth image is degenerate with a resolved background.

## Status after Stage 4, and next steps
**Done:**
- AMI and long-baseline MAP imaging: per-dataset models (`Rotated` epochs), SPARCO with power-law and blackbody spectra, the support hole, and synthetic VLTI and NRM coverage.
- Synthetic MWEs: SAM V² + CP, two-night VLTI, and rotating epochs.

**Next:**
1. Merge this stack into `imaging`, then `imaging` into `main` (milestone 1).
2. MWE-B: the SNR and coverage stress test (deferred from Stage 4).
3. Stage 5: Gaussian-process pixels, sampling (posteriors for the companion's flux and separation, which are correlated), and evidence-based weights.
4. Stage 3c: PDS 70 (local demo first, then an OzSTAR script if needed).
5. Cache MFT matrices (Stage 7).

## Stage 5: Gaussian-process pixels and sampling (about 6–8 h)
**Build:**
- `fields.GaussianField(latent, sigma, length_mas, order=2, mean=None, mean_floor=1e-3)`:
  - η = log(μ/max μ + ε) + IDCT₂[√S ⊙ latent] (orthonormal);
  - S ∝ (κ² + λ_jk)^(−order) on reflecting-boundary Laplacian eigenvalues, with S₀₀ = 0, scaled to a mean variance of σ².
- `image_priors` gives N(0,1) latents.
- `fit` warns if you MAP the hyperparameters.
- **Regularisation weights from the evidence**, building the matrix-free curvature machinery once:
  - for quadratic and GP priors (TSV, `GaussianField`), the Laplace-approximated evidence as a function of the weight (or σ, length), with the log-determinant of the Gauss–Newton Hessian by stochastic Lanczos quadrature or Hutchinson probes; maximise it, or marginalise when sampling;
  - for maximum entropy, Gull & Skilling's "classic MaxEnt" choice of the weight (−2αS equal to the number of well-measured directions, from the eigenvalues of the same curvature);
  - both as helpers alongside `LCurve.corner` / `discrepancy`, returning a weight; `fit` never chooses it silently.
- `imaging.gauss_newton_diagonal` (a Gauss–Newton–Bartlett estimate, for mass-matrix initialisation).
- The `[sampling]` extra (blackjax).
- There is no sampler wrapper: tutorials sample `numpyro_model` with numpyro's NUTS, and show BlackJAX on the potential from `numpyro.infer.util.initialize_model` as the alternative.

**Tests:**
- Exact 8×8 covariance against the inverse of (κ²I + L)^order; `order=1` equals TSV + L2 on η.
- Prior σ and length calibration.
- `latent = 0` reproduces the template, and template-parameter gradients are finite.
- 16² NUTS coverage of the injected hyperparameters, under x64.

**MWE-A:** a SPARCO star + GP envelope with a centred `GaussianDisk` mean, on the Stage 4 synthetic data. Compare the MAP (LM) with TSV/MEM from Stage 4.

**MWE-B:** posterior sampling at 32² (BlackJAX NUTS and MCLMC/MAMS), with marginalised σ and length. Show the posterior mean and standard-deviation maps, hyperparameter posteriors, and chains against wall-clock time on laptop CPU. Optionally repeat on the simulated AMI scene.

**Risk:** sampling cost and multimodality at larger grids. Scope stays at 32–64², with the DFT/MFT (on an A100 the DFT is ~1 ms at 256² × 10⁴ points, so GPU sampling does not need the NUFFT).

**Checkpoint:** is the GP prior worth it compared with classical regularisers, and does sampling go into the docs as supported?

## Stage 6: polychromatic imaging (about 8–12 h; design is finalised at the Stage 5 checkpoint)
**Build**, in increasing order of complexity; stop where the science needs stop:
1. **Grey image with a spectral index.** This already works through `flux=PowerLaw(...)`. Add documentation and an MWE only.
2. **Joint multi-filter AMI.** Several filters (F380M/F430M/F480M) share one image, with per-filter flux ratios (the analogue of dorito PR #32's `JointResolvedDiscoModel`). This needs per-observation models in `fit`, i.e. a `model_fn(values, index)`, like `joint_loglike`.
3. **`ImageCube`.** Per-channel log-brightness with a GP along wavelength: a separable DCT field over (λ, y, x). The translation gauge is fixed per channel (one centroid per channel), unless an analytic star anchors it.

**Tests:**
- Cube visibilities per channel equal grey `Image` visibilities when all channels are equal.
- Joint AMI recovers the per-filter fluxes.
- Chromatic recovery of a synthetic disk whose colour changes with radius.

**MWE-A:** joint three-filter AMI imaging (simulated, on ν Hor-like coverage and noise in each filter).

**MWE-B:** a MATISSE chromatic disk across the L band, with the star analytic.

**Risk:** this is the least-specified stage, and per-channel transforms use the DFT (or one MFT per channel on lattices); the NUFFT is shelved (issue #75).

**Checkpoint:** **merge milestone 2**.

## Stage 7: hardening and release (about 3–4 h)
- API review for consistency and naming, with docstrings (units and examples) for every public object.
- A mkdocs API page and a "choosing a regulariser and prior" guide.
- Update `design/chromatic_sources.md` to mark `Image` as done.
- Remove any leftovers, and do a final full-suite run under float32 and x64.
- A version bump and a changelog entry.

---

## Totals
- **Stages 0–4** (MAP imaging for AMI and long-baseline data, the transform benchmark, reproduction of the dorito result): about 18–26 h of agent time.
- **Stages 5–7:** about 17–24 h more.
- **Overall:** about 35–50 h of agent time, spread over 8 feedback checkpoints.

External waits: only the OzSTAR GPU benchmark run, which you launch. All test data are simulated using ν Hor baselines and noise.

**Coverage fixtures.** The MATISSE ν Hor OIFITS files live in `/Users/benpope/code/nuHor/data/`, where they are git-ignored. In Stage 4, a script (`scripts/make_coverage_fixture.py`) extracts **only the geometry and noise**: u, v, wavelengths, closure-triangle indices, flags and per-point σ. These go into a small `data/coverage_nuhor_matisse.npz`, with no measured visibilities or phases, so tests and notebooks are self-contained. The AMI record already in the repo covers AMI. If you'd rather not commit even the derived coverage, the fallback is synthetic 4-telescope coverage with σ statistics matched to ν Hor.

## Deferred (triggers recorded in the design note)
| Item | Revisit when |
|---|---|
| Forward-model centroid recentring | Sampling drifts in position despite the centroid prior |
| Twin-folding helpers | We fit V²-only data routinely |
| Basis and decoder parameterisations | zodiax 0.6 `Map`/`Mask`/decoders are released |
| MGVI/geoVI | Not planned |
| Cross-validation for the weight (e.g. held-out DISCO modes) | Not now; discrepancy, L-curve and (Stage 5) evidence suffice |
| Deep Probabilistic Imaging and learned priors | Not planned; covered by other work in the group |
| Top-Set surveys over synthetic truths | Not planned here; sophisticated PDS 70 truths are being built separately |
| Bandwidth-smearing forward model | A real field-of-view need |
| Reinstating a NUFFT backend (issue #75; code in PR #71) | jax-finufft reuses plans (jax-finufft #157) and we have datasets of ≳10⁵ irregular points with ≳256² images |

## Critical files
- **New:**
  - `src/drpangloss/fitting.py`, `src/drpangloss/imaging.py`, `src/drpangloss/fields.py`, `src/drpangloss/_precision.py`
  - `design/image_reconstruction.md`
  - `tests/test_image_model.py`, `tests/test_fitting.py`, `tests/test_imaging.py`, `tests/test_fields.py`
  - Notebooks: `imaging_ami.ipynb` (Stages 1–3), `imaging_long_baseline.ipynb` (Stage 4), `imaging_gp_sampling.ipynb` (Stage 5), `imaging_chromatic.ipynb` (Stage 6)
- **Modified:**
  - `models.py`: `Image`, later `ImageCube`.
  - `_geometry.py`: `image_visibilities`.
  - `likelihood.py`: `whitened_residuals`, `model_loglike`.
  - `__init__.py`, `pyproject.toml` (optax, lineax; `[sampling]` extra), `AGENTS.md`, `mkdocs.yml`, `scripts/sync_tutorial_docs.py`, `design/chromatic_sources.md`.
- **Reused unchanged:** `OIData` (including `with_model`), `amigo.load_oi_data`, `build_model`, `System`, `spectra.PowerLaw`, `plot_model`, `inference.gaussian_fisher` (for parametric uncertainties), `_grid.warn_unconverged`.

## Verification
- Each stage has its tests plus a green full suite (float32 and x64), `ruff`, the tutorial sync test, and the docs build.
- **Stage-specific quantitative gates:**
  - Stage 0: regression tolerances.
  - Stage 3: synthetic recovery metrics and recovery metrics against simulated truth (AMI).
  - Stage 4: star-anchored recovery and the SNR/coverage sweep on ν Hor coverage.
  - Stage 5: exact GP covariance and sampling coverage.
  - Stage 6: per-channel consistency.
- The MWE notebooks are re-executed locally at every later stage, so earlier examples keep working. The docs-sync test keeps them matched to the docs. Heavy steps (real-data fits, sampling) are cached or reduced so re-running stays quick.

**Scope changes since revision 4:**
- The OzSTAR GPU benchmark was run on 2026-09-30 (A100; results in `design/image_reconstruction.md`). It confirmed the TF32 fix and led to shelving the NUFFT (issue #75).
- Coverage comes from (a) a geometry-and-noise fixture extracted from ν Hor and (b) a simple synthetic-coverage generator (`tests/_coverage.py`: N telescopes, Earth-rotation tracks, channels, σ drawn from ν Hor statistics).

### OzSTAR GPU test (done 2026-09-30)
Run by the user (Claude never connects to OzSTAR) on an A100. It confirmed that `Precision.HIGHEST` gives float32-accurate DFTs on the GPU (7.9e-7, against 2.7e-5 with TF32), and that the jax-finufft NUFFT was too slow there, which led to its removal. Results and figures are in `design/image_reconstruction.md`; the scripts are in the git history of PR #71.

---

## Implementation orchestration

### Dependency graph (chunks)
```
C0a design note ─────────────────────────────────────────────┐
C0b whitened_residuals + model_loglike + regression ──┐       │
C4a coverage fixture + synthetic coverage generator ─────────┼──▶ C4 long-baseline MWEs
C1a image_visibilities (DFT) ──┬─▶ C1b Image ──┬─▶ C1c tests/MWE (AMI simulation)
                               │               ├─▶ C1d scene library (spiral/ring/star+comp)
                               │               ├─▶ C5a GaussianField (math; can start here)
                               └─▶ C2 NUFFT backend + CPU bench
C0b + C1b ─▶ C3a fit + _precision ─┬─▶ C3c diagnose ─▶ C3d AMI MWEs (needs C1d)
C1b ─▶ C3b regularisers ────────────┘
C3a + C5a ─▶ C5b image_priors/GN diagonal/sampling MWE
C3 + C5 ─▶ C6 polychromatic (design first) ─▶ C7 hardening
```
- **Critical path:** C1a → C1b → C3a → C3c/C3d → C4.
- **Parallel lanes:**
  - C0a runs alongside C0b.
  - C2 runs alongside C1b–C1d.
  - C3b runs alongside C3a.
  - C5a starts as soon as C1b exists, alongside Stage 3.
- **Shared files** (`models.py`, `__init__.py`, `AGENTS.md`, `pyproject.toml`) have a single owner per stage, which is the orchestrator. Sub-agents return diffs to those files and do not edit them directly, which avoids merge conflicts between parallel worktrees.

### Model allocation
| Chunk | Model | Why |
|---|---|---|
| C0b likelihood unification | **Opus** | Changes existing semantics; the regression judgement is subtle |
| C1a DFT conventions and precision | **Opus** | Sign and orientation errors are the historical bug source |
| C2 NUFFT convention mapping | **Opus** | Half-pixel factors, axis pairing, eps floors |
| C3a `fit` (transforms, LM pitfalls, dtype policy) | **Opus** | The core API; numerical subtleties |
| C5a `GaussianField` (DCT spectrum, normalisation, exact-covariance test) | **Opus** | Mathematical correctness |
| C6 polychromatic design | **Opus** (orchestrator) | Open design questions |
| C1b `Image`, C3b regularisers, C3c `diagnose`, C5b helpers | **Sonnet** | Well specified by this plan; Opus reviews the diff |
| All tests except the exactness tests, C1d scenes, MWE notebooks, tutorials | **Sonnet** | Clear specs; iterate against running code |
| C0a design note, C7 docstrings, mkdocs pages | **Sonnet** | Writing from existing material |
| C4a fixture extraction, running the suite/ruff/docs build, tutorial sync, benchmark runs and reporting | **Haiku** | Mechanical |
| Every PR's final review before a checkpoint | **Opus** (orchestrator) | Consistency across chunks |

Fable 5.1 is also available in this environment, but I have no evidence about how it compares to Opus on this kind of numerical work. Try it on one Opus-tier chunk, e.g. C5a with its exact tests as the arbiter, before relying on it.

### Wall-clock estimates with this orchestration
These are agent time only, excluding your checkpoint reviews. The serial estimate was 35–50 h.

| Stage | Serial | Orchestrated | Notes |
|---|---|---|---|
| 0 | 2–3 h | **~1.5 h** | Four parallel chunks; C0b is the long pole |
| 1 + 2 | 6–8 h | **~3.5–4.5 h** | C2 runs in parallel with C1b–d |
| 3 | 6–9 h | **~4–5 h** | C3a and C3b in parallel, then C3c/C3d |
| 4 | 4–6 h | **~2.5–3 h** | Mostly Sonnet MWE work plus the fixture |
| 5 | 6–8 h | **~3–4 h** | C5a is largely done during Stage 3 |
| 6 | 8–12 h | **~5–7 h** | Design by Opus first, then parallel implementation |
| 7 | 3–4 h | **~1.5–2 h** | Sonnet and Haiku |
| **Total** | 35–50 h | **~21–27 h** | Plus the 8 checkpoint waits |

Orchestration overhead (reviews, integration, fixing cross-chunk mismatches) is included, at about 20% per stage. The main risk to these numbers is iteration on numerical issues in C1a, C3a and C5a, which could add 1–3 h each.

### Recommended interface
- **Run the orchestrator in the Claude Code CLI or the desktop app, one session per stage,** in `/Users/benpope/code/drpangloss`:
  - They handle long autonomous runs, background sub-agents and `isolation: "worktree"` (each parallel chunk gets its own git worktree and branch) better than an editor-bound session.
  - A session per stage keeps context clean. The plan file and `design/image_reconstruction.md` are the handoff between sessions.
- **Use VS Code (this extension) for checkpoints:** reviewing diffs, running the MWE notebooks and looking at the figures. It is good for that, less so for multi-hour orchestration tied to an open editor window.
- **For the parallel stages** (0, 1+2, 3), you can say "use a workflow" to have the orchestrator run a scripted multi-agent workflow (a fan-out of chunks, each followed by an Opus review) instead of hand-launching agents. The size guideline is under 10 agents per workflow, which fits these stages.
- **Checkpoints as GitHub PRs** (`imaging-sN-*`, stacked, into `imaging`), so you can comment asynchronously; the next session starts by reading those comments.
