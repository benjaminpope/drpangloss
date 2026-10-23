# Design note: marginalized antenna gains for complex visibilities

Status: Stage 1 approved for implementation 2026-10-23. Stage 2 is a plan.

## Purpose

Fit images and models to complex visibilities (`virgil.radio`) with
antenna-based complex gains **marginalized**, not solved for and applied
(CASA self-calibration) or projected out (closure-only imaging). The image
posterior then carries the calibration uncertainty, and the gain priors
encode what is known about the calibration.

## Background and prior art

- **Closure equivalence** (Blackburn et al. 2020, ApJ 894, 31,
  arXiv:1910.02062).
  - With Gaussian errors and unconstrained gains, inference from closure
    quantities equals inference from visibilities with the gains
    marginalized (§IV).
  - With Gaussian gain priors of finite width, the posterior moves smoothly
    from perfect calibration to closure-only. It approaches the perfectly
    calibrated case once the gains are known about 10 times better than the
    data (§IV.4).
  - It counts N−1 phase and N amplitude degrees of freedom lost per
    snapshot (§III.2).
  - It treats single snapshots only. It does not study gains shared across
    integrations, nor priors correlated in time (§II.3). At low
    signal-to-noise it suggests forward-modelling the complex gains and
    marginalizing them (§II.4), which is what this note does.
- **THEMIS** (Broderick et al. 2020, ApJ 897, 139) marginalizes one complex
  gain per station per timestamp with a Laplace approximation at every
  likelihood call. Its amplitude priors are Gaussian or log-normal. This is
  the closest existing method; virgil's version adds shared solution
  intervals and, in Stage 2, priors correlated in time.
- **resolve** (Arras et al. 2019, A&A 627, A134).
  - Joint calibration and imaging with MGVI, which approximates the posterior
    with a Gaussian.
  - The log-amplitudes and phases of each antenna are Gaussian processes in
    time.
  - They use a non-parametric power spectrum (a power law plus smooth
    deviations, learned from the data, on an FFT grid). This is not a
    quasiseparable kernel.
- **VLTI in virgil** (`virgil.gains`, Lachaume 2021) uses the same idea for
  log|V| with linearized gains.

## Model

For visibility k on baseline (i, j) in solution interval s:

    V_obs,k = g_i,s · conj(g_j,s) · M_k(θ) + n_k,    g = exp(λ + iφ)

- M_k(θ) is the existing model, including `CalibrationNuisance`.
- n_k is the existing complex Gaussian noise, with weights scaled by the
  error scales `cal.sigma`.
- **Priors:** λ ~ N(0, τ_a²) and φ ~ N(0, τ_φ²), independent between
  antennas and intervals in Stage 1.
- **The phase prior must be finite.** With τ_φ = ∞, the phase of each
  interval is a pure gauge, and the absolute position is lost (as with
  closure phases). A finite τ_φ fixes both.
- **Amplitude gauge:** the mean λ is degenerate with `cal.flux` and
  `cal.gain`. The finite τ_a identifies it; document this.
- τ_a and τ_φ are sites (`cal.gain_amp_width`, `cal.gain_phase_width`)
  that can be fixed or fitted, with log-uniform (Jeffreys) priors.
- **ALMA defaults (a starting point).** The documented calibration accuracy
  is on the absolute flux scale: about 5% in Bands 3–5, 10% in Bands 6–8 and
  20% in Bands 9–10 (Technical Handbook, quoted secondhand; check the
  current Cycle handbook). No residual phase specification exists. Default
  τ_φ to about 0.5 rad, and say in the docs that it is a choice, not a
  measurement.

## Marginalization: Laplace, centred on the gain solution

Phases enter non-linearly, so the integral has no closed form. With many
more baselines than antennas (≈ N/2 times as many) and many data per
interval, the gain posterior is narrow. A Laplace approximation is then
accurate **provided it is centred on the gain solution, not on g = 1**.
Linearizing around g = 1 fails once phase errors exceed about 0.3 rad;
centring on the solution does not have that limit.

Per interval s, at each likelihood evaluation:

1. Compute model visibilities M_k(θ) **once**. This is the expensive step.
2. **Gain solve.** Run a fixed number K of Gauss–Newton steps for
   (λ, φ)_s, starting from zero, and vmap over intervals. Start with K
   about 5 and test it. Each step needs only M_k, the data and segment sums,
   at O(N_vis) cost, which is small next to step 1.
3. **Hessian.** At the solution, the Gauss–Newton Hessian **splits into an
   amplitude block and a phase block**, because the λ–φ cross terms cancel
   for these Jacobians:
   - H_λ[i,i] = Σ w|V|² + τ_a⁻², and H_λ[i,j] = +Σ_ij w|V|²;
   - H_φ is the graph Laplacian with edge weights Σ_ij w|V|², plus τ_φ⁻²·1.
   It is singular without the phase prior.

   Here V = ĝ_i conj(ĝ_j) M_k, and the sums run over the interval's
   visibilities, through `segment_sum` over (s, i, j).
4. **Marginal.** The negative log marginal likelihood is

       ½ χ²(ĝ) + ½ Σ (λ̂²/τ_a² + φ̂²/τ_φ²)
         + ½ log det H_λ + ½ log det H_φ
         + N_s N (log τ_a + log τ_φ)

   where N_s is the number of solution intervals. The batched N × N
   Cholesky factorizations are cheap (N ≈ 50).
5. **Gradients.** Differentiate straight through the K unrolled Gauss–Newton
   steps. This is exact for the computed approximation and cheap, because
   the steps do not re-evaluate M.

Fit: the log-determinant depends on the model, so, as with the VLTI gains,
there is no least-squares form. `fit` should raise `NoLeastSquaresForm` for
LM; `lbfgs` and `adam` work, and so does `numpyro_model`. The term reports
½χ² + prior through `residuals` and the rest through `log_norm`, following
the existing `has_log_norm` convention.

**Why not log space.** Correlator data record each antenna pair separately,
so the gains are *exactly* linear in log V: a_i + a_j + i(φ_i − φ_j). There
they could be marginalized in closed form with no solve and no Laplace
step. This is Blackburn's setting, and DISCO's projection, K·J = 0, is its
limit as τ → ∞. The cost is that log V has non-Gaussian noise unless the
signal-to-noise of *each visibility* is well above 1 (Blackburn §II.4). For
ALMA that fails on most long baselines. The Laplace approach in complex
space only needs the *interval* (all its visibilities together) to
determine the gains well, so it gathers signal-to-noise before any
non-linearity matters. A log-space path is an option for later, for bright
or averaged data.

The "Woodbury per interval" route in `design/ordered_subsets.md` is the
same computation with the gain solution held at g = 1. The Laplace version
replaces it.

## Stage 1 (this PR): independent gains per solution interval

- `CalibrationNuisance(..., antenna_gains=True, solint=...)`:
  - `solint` is `"int"`, `"scan"` or a time in seconds;
  - intervals come from `time` and `scan` and do not cross scans;
  - it needs `time`, `ant1` and `ant2` and raises if they are missing;
  - optionally one interval per spectral window.
- Data are sorted, or indexed, by interval so that segment sums are
  contiguous. The chunked layout of `_RadioTerm` stays. Padding takes zero
  weight and an interval index that is ignored.
- `jit` must not recompile when values change. Interval and antenna counts
  are static.
- Raise a clear error if this is combined with the compressed
  (dirty-beam) likelihood being built in parallel: they cannot be combined,
  because the gains break the compression.
- **Tests.** Simulate small arrays, for example N = 7 or 10 antennas, a
  Gaussian or binary source and known gains. Fast cases go in CI; slower
  ones in the slow set.
  1. **Calibrated limit.** As τ → 0, the marginal equals the existing
     likelihood up to a constant.
  2. **Closure invariance.** With large τ, high signal-to-noise and one
     interval per integration, the θ-dependence of the marginal is unchanged
     when arbitrary gains are applied to the data.
  3. **Closure count.** On one snapshot with large τ, the curvature of the
     marginal with respect to a source shift tends to 0. Closure phases do
     not know position.
  4. **Recovery.** With phase errors up to about 1 rad, a fit recovers the
     source parameters within errors.
  5. **Gain solve.** The solved gains match the injected ones.
  6. **Gradient.** `jax.grad` agrees with finite differences.
  7. **LM refused.** `fit(method="lm")` raises `NoLeastSquaresForm`.
- **Docs.** Add a section to the Radio interferometry guide, with terse
  docstrings. Add Blackburn 2020, Broderick 2020 (THEMIS) and Arras 2019 to
  `references/provenance.toml` and `docs/references.md`.

## Stage 1b (later): EM with a Toeplitz M-step

The direct marginal of Stage 1 is O(N_vis) at every evaluation. The gain
solution and the Hessian need the model at every visibility, and the
log-det needs per-baseline sums of the model's power within each interval.
It therefore cannot be added to the Toeplitz (dirty-beam) term in one
objective. EM recovers the compression:

- **E-step** (O(N_vis), one forward pass): solve the gains per interval and
  take the Laplace posterior q(λ, φ). For each visibility, with
  c_k = g_i conj(g_j), form E[c_k] and E|c_k|². Both are log-normal moments
  of the Gaussian q, in closed form.
- **M-step.** E_q χ² = Σ w_k E|c_k|² · |M_k − d_k conj(E[c_k]) / E|c_k|²|²
  + const. This is a χ² with effective data
  d_eff = d · conj(E[c]) / E|c|² and effective weights w_eff = w · E|c|²,
  so it is exactly the Toeplitz form. Grid d_eff and w_eff once, giving a
  dirty image and a dirty beam, then optimize the image and regularizer
  with no dependence on N_vis.
- **Convergence.** EM increases the marginal likelihood at each iteration
  (exactly with an exact E-step, and as variational EM with the Laplace q).
  Its rate is set by the fraction of missing information. Gains that are
  well determined per interval mean a small fraction and few outer
  iterations. When the gains are known exactly, it reduces to alternating
  plain self-calibration.
- **Cost per outer iteration:** one forward pass for the model, plus one
  gridding pass for d_eff and w_eff.
- **Limits.**
  - Each per-observation nuisance that is not linear in the image (the
    offsets) needs its own dirty image and beam, or an update in the
    E-step.
  - NUTS over the image cannot use EM. Sampling uses the Stage 1 term
    directly at O(N_vis), or fixes the gains at their EM values (which
    understates the uncertainty).
- **Requirement on the Toeplitz term:** it must take effective data and
  weights from outside, and rebuild its dirty image and beam cheaply.

## Stage 2 (later): gains correlated in time with celerite-type kernels

The aim is to replace independent intervals with a Gaussian process in time
per antenna for λ and for φ, with kernels that have a state-space form:
Matérn-1/2, Matérn-3/2 and Matérn-5/2, or a SHO or CARMA. Atmospheric phase
noise has a temporal spectrum of about f^(−8/3), between Matérn-1/2 (f⁻²)
and Matérn-3/2 (f⁻⁴). A Matérn-3/2, or a sum of a few terms, should
approximate it.

- **Structure.** The gain prior is then Markov in an augmented state of
  dimension J per gain (J = 2 for Matérn-3/2). The Laplace Hessian becomes
  block tridiagonal in time: the prior precision plus the per-interval data
  blocks of Stage 1. Its log-determinant and solves come from a
  Kalman/RTS-smoother pass in `lax.scan` (cost O(N_s (2NJ)³)). The parallel
  form uses `lax.associative_scan` (Särkkä & García-Fernández 2021). The gain
  solve becomes a few Gauss–Newton steps, each one smoother pass.
- **Kalman over intervals, tinygp to check it.** The covariance is still
  quasiseparable even though antennas are coupled at each time: it is a sum
  of diag(h) K diag(h) terms of total rank R, and tinygp handles it with a
  custom `observation_model`. But a scan over data is latency-bound on a
  GPU. A Kalman filter with one measurement update per interval scales
  better, so use it for production and keep tinygp as the independent
  check. The engine is shared with the VLTI transfer-function GP
  (virgil-vlti `design/gp_transfer_function.md`). VLTI's problem is linear
  and Gaussian and much smaller, so build and test the engine there first.
- **Hyperparameters.** The kernel amplitudes and timescales per gain type
  become sites. They are learned like resolve's spectrum, but in a
  parametric family.

## Not in scope

- Polarization leakage (D-terms).
- Direction-dependent gains.
- Bandpass terms beyond one gain per spectral window.
- Closure-only likelihoods.
- The ordered-subsets ideas in `design/ordered_subsets.md`.
