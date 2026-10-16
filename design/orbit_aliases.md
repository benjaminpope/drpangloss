# Period aliases of undersampled orbits

> **Superseded by [orbit_pipeline.md](orbit_pipeline.md) (2026-10-16).** `fit_orbit_aliases` stays as the engine under `OrbitPipeline`, so the problem statement and method below are still valid; the defaults it lists as options (`sign_free`, `flip_flux`, `vis_amp`, `primary_diam="auto"`, a data-driven `p_range`, night flags, tempering) are now on by default.

Status: implemented in `virgil.aliases` (`fit_orbit_aliases`).

## Problem

When epochs are separated by many periods, the number of orbital cycles
between them is ambiguous. The likelihood has one narrow ridge per cycle
count, and a fit started in the wrong ridge cannot find the right one. A
position per epoch is a lossy summary (at λ/B resolution, positions are
biased, errors are non-Gaussian, and several fringe peaks per night are
common), so evidences computed from positions are unreliable. Closure
phases of every epoch discriminate the ridges far better.

## Method

**Alias bands.** Let T be the first-to-last epoch span. Band N holds the
periods with `round(T/P) = N`: `T/(N+½) ≤ P ≤ T/(N−½)`, clipped to the prior
range. Bands are one cycle wide in frequency and tile the period prior.
The spectral window of the epoch times, `|⟨exp 2πi f t⟩|²`, shows which
frequency offsets the timing cannot tell apart; a cheap position-level
Thiele–Innes profile (`position_profile`: best χ² per band from
`period_grid` and `starting_orbits`) cross-checks that the bands are viable.
Positions are used for initialization and diagnostics only.

**Per-band fit.** The method is the joint squared-visibility and
closure-phase likelihood of all epochs, evaluated exposure by exposure,
with one flux ratio shared by every epoch. Closure phases alone are not
enough for near-equal binaries: for flux ratios of 0.9–1 the closure phases
are almost zero, the point reflection of the sky is almost exactly
degenerate in every epoch separately, and only the visibilities fix the
separation. The V² block has its own error scale in every epoch, apart
from the closure phases' (see below). Uncalibrated V² (ESO GRAVITY Phase 3
`SINGLE_SCI_VIS`) carry a transfer function that a scale cannot undo, so the
V² scales come out large and V² then weighs little; report raw χ²/N for
V² and closure phases separately. It is hierarchical: each epoch's observable blocks have an error scale
`s_b` with a Jeffreys prior `1/s_b`, integrated out analytically
(`marginal_loglike`, `m = −Σ ν_b/2 ln χ²_b`), so `m` is unchanged by a
constant rescaling of an epoch's quoted errors and no epoch dominates
through an optimistic error bar. Candidate orbits come from the position
profile (both senses of motion, a flux scan) and are ranked by `m`. With
`sign_free=True` (flux ratios 0.9–1) each position is known only up to a
sign: the seeds solve the Thiele–Innes matrix from epoch pairs, set every
epoch's sign against it, and scan fluxes near the top of the range;
the best distinct few are refined by damped Newton steps (Levenberg–Marquardt on the iteratively reweighted least-squares curvature of the marginal likelihood, accepted only if the log likelihood rises; L-BFGS-B is the fallback for a start that does not converge), in the flat coordinates of the
Jeffreys priors (log P, fractional time of periastron, e, cos i, ω, Ω,
log a, log flux). The (Ω+180, ω+180) mirror is exactly degenerate for
closure phases and is folded by restricting Ω to [0, 180). Fits are
deduplicated by their tracks at the epochs.

**Evidence.** Per mode, Laplace at the optimum (Hessian of `m` in the flat
coordinates; the priors are uniform there, so they add only a constant),
and an importance-sampling estimate from a Student-t proposal with the
Laplace covariance. The band evidence is the sum over its distinct modes.
The IS estimate uses the deterministic mixture of the modes' proposals, so
overlapping tails are not counted twice. A mode within 3σ of a hard prior
bound (a period edge between bands, or e ≈ 0) has a cut-off Gaussian, so
its IS value (which truncates the target) replaces its Laplace value. Report
ESS and a `flags` list: `near-prior-bound`, `not-converged`,
`hessian-not-positive-definite`, `hessian-singular`, `band-failed` (an
exception in the band, kept in `error`; the other bands are unaffected),
`laplace-is-differ` (more than 1 nat),
`low-ess` (below 50), `no-valid-mode`. The band probability is
`p_N ∝ Z_N` under the log-uniform prior on P over the whole range.

**Posterior samples.** Importance resampling with the mixture weights, mirror folded, in
the winning band and in every band with `p_N > 0.01`.

**Reporting.** Lead with the raw χ²/ν on the quoted errors (before any
error scale), then the fitted per-epoch scales ŝ, `log Z_N`, `p_N`, ESS.
A rescaled χ² near 1 is tautological, and large scales mean a failed fit.

## Extensions

* **Resolved primary.** `primary_diam` swaps the point primary for a uniform
  disk; a fitted diameter is one more coordinate with a log-uniform prior.
* **Per-epoch V² amplitude.** A contrast loss multiplies an epoch's V² by one
  number on every baseline and wavelength, which a resolved primary cannot
  mimic. The model is linear in it, so the best amplitude (clipped to its
  Jeffreys range) is profiled analytically, keeping the epoch's error scale.
* **Flux flip.** The flux prior (lo, 1] is not symmetric under swapping the
  stars. A mode pinned at the top is refit from the swapped labels
  (ω + 180°, flux → 1/flux) and the better mode kept; closure phases and V²
  are the same under the swap, so the two are one sky orbit.
* **Warm start.** `init=` joins every band's pool and is always refined in
  its own band, so a reference orbit cannot be missed by the seeding.

## Limits

* The sum over modes misses modes the multistart does not find.
* The sum over modes uses a Gaussian local approximation: strongly
  non-Gaussian modes (very eccentric orbits) are what the IS check catches.
* The scale marginal assumes Gaussian small-σ closure-phase noise; for weak
  closure phases pass `s_max`.
* Errors correlated within an epoch carry fewer degrees of freedom than ν;
  see `dof` in `marginal_loglike`.

## Worked example

Gl 229 Ba–Bb, 7 GRAVITY epochs over 414 d, P ≈ 12.1 d: seven bands
N = 30–41. A prototype of this method fitting closure phases picked N = 34
(the published orbit) over N = 37 and 41 by more than 2000 in marginal
log likelihood, whereas evidences from single-night positions were
unstable (Laplace vs importance sampling differed by 2 nats, and
2–4 mas jitters were needed). Positions are a start, not the evidence.
