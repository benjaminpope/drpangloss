# Design note: composite scenes

## Purpose and scope

A **composite scene** is a [`System`][virgil.models.System] of several
components, each with its own prior. Some are parametric (a point star, a
uniform disk) and several are [`Image`][virgil.models.Image]s, each with its
own grid, pixel scale, support, offset (``dra``, ``ddec``) and regulariser. A
regulariser is a penalty such as MaxEnt, TSV, TV or starlet L1, or a
[`GaussianField`][virgil.fields.GaussianField] with its own (ℓ, σ).
Examples:
- a star, a disk with a hole at the star and a planet's postage stamp;
- a compact core beside an extended envelope;
- a resolved stellar surface with a companion.

virgil already has every piece. A `System` mixes components as
``Σ fᵢ Vᵢ / Σ fᵢ``, an `Image` has an offset and a support, regularisers
attach by path, and [`log_evidence`][virgil.imaging.log_evidence],
[`error_scale`][virgil.imaging.error_scale] and
[`laplace_samples`][virgil.imaging.laplace_samples] take a tuple of image
paths (Stage 5e of `design/imaging_plan.md`). What is missing is knowing
which scenes the data can tell apart, how to place and keep a second
component, and how to choose several weights at once. This note covers
those, as checked in a linear-Gaussian toy
(`design/sketches/composite_identifiability.py`, NumPy only).

Two GPs on one grid add to a single GP whose spectrum is the sum of theirs.
They are one prior, not two components, and their flux split is set by the
prior alone (the toy, below). A composite earns its keep only when its
components differ in something the data or the prior can see: their
support, position, pixel scale or kind of regulariser.

## The toy

The toy uses random six-station coverage with 600 complex visibilities at
σ = 0.01. A zero-spacing total-flux row stands in for normalised data, so
only flux ratios are identifiable. Every image is linear in its pixels,
with a squared-exponential GP prior and a flux mode. The evidence
N(d; A m, A K Aᵀ + N) is then exact. Hyperparameters (ℓ, amplitude) are
chosen by evidence on a grid; lengths are in beams.

Case A is a star (0.5), a ring with a gap (0.47) and a planet (0.03),
inclined 45°. The planet sits either on the inner ring (1.3 beams) or in
the gap (2.6 beams). Case B is a core (0.4, unresolved) inside a Gaussian
envelope (0.6) of radius R = 0.5, 1, 2 or 4 beams (FWHM 2R).

Fluxes and pixels are marginalised exactly; hyperparameters (ℓ, amplitude)
and stamp offsets are maximised (empirical Bayes). Δ log Z for an added
stamp therefore lacks its offset's Occam factor, roughly ln(search area /
posterior position area), a few nats: comparable with the ln 100 ≈ 4.6
threshold for keeping a stamp, so the second stamps' +24 to +26 are upper
bounds. The toy's stamp is ½ beam across, with 0.1-beam pixels and
ℓ = 0.1 beam.

| Scene | Case A, planet on ring | Case A, planet in gap |
|---|---|---|
| one GP image | log Z 2753 | 2755 |
| star + GP | 3150 (+397) | 3193 (+438) |
| star + GP with a ½-beam hole | 3148 | 3192 |
| + 1 stamp (search, evidence-refined) | 3251 (+103); 0.10 beam from the planet | 3240 (+48); 1.4 beams from it, on the rim |
| + 2nd stamp | 3277 (+26); on the ring | 3264 (+24); on the ring |
| stamp at the true planet (reference) | +100.5 | +10.8 |

| Case B | star + GP: star flux (truth 0.4) | with hole | hole minus no hole vs sd | compact GP + extended GP, Δ log Z against star + GP |
|---|---|---|---|---|
| R = 0.5 | 0.618 ± 0.011 | 0.691 | 0.072 vs 0.011 (6.5 sd) | wins (+207), split 1.02 / −0.02 |
| R = 1 | 0.409 ± 0.011 | 0.474 | 0.064 vs 0.011 (5.8 sd) | wins (+196), split 0.83 / 0.17 |
| R = 2 | 0.398 ± 0.003 | 0.423 | 0.025 vs 0.003 | wins (+21), split 0.53 / 0.47 |
| R = 4 | 0.398 ± 0.001 | 0.400 | 0.002 vs 0.001 | loses (−69), split 0.42 / 0.58 |

Two GPs on one grid (short and long ℓ) give a split of 0.4/0.6 ± 0.72 for
every R: the split comes from the prior.

## Which combinations are identifiable

With normalised data, any two components have flux correlation ρ ≈ −1,
because their fluxes sum to the total. Correlation therefore does not
measure identifiability. Two tests do:
- **Model sensitivity:** how far the split moves between reasonable
  modelling choices (here hole against no hole), compared with its
  posterior sd. A split that moves by many sd is set by the model, not by
  the data.
- **Bias** against truth, in simulations.

By those tests:

1. **An analytic star with an image is identifiable once the image is
   resolved.** Star + GP beats one GP image by Δ log Z ≈ 400 in case A, and
   by 500–1000 in case B. The star flux is right to 1% when the envelope's
   radius is at least 2 beams (4 beams across); at R = 1 it is 2% high. At
   R ≤ 1 beam the split is set by the model: a hole of half a beam moves
   the star flux by about 6 sd (6.5 at R = 0.5, 5.8 at R = 1).
   The star then absorbs the compact envelope, or a hole gives it all the
   central flux. Only something outside the image prior can fix that:
   - a parametric envelope, such as a Gaussian or a ring;
   - photometry giving the star's flux;
   - chromatic leverage: SPARCO's different spectral indices
     (`design/chromatic_sources.md`).
2. **Compact plus extended images are identifiable only at large R, and
   then add nothing over a point star.** Below that, the two-image scene
   wins the evidence with a meaningless split (core 1.02 at R = 0.5). It
   is a better total image but not a decomposition. Report the split only
   when the core's support is smaller than a beam and its grid is
   evidence-chosen.
3. **Two GPs on one grid are never identifiable**: their split is the
   prior's. Use different supports, positions or pixel scales, or one GP
   and one parametric component.
4. **A stamp beside a GP is identifiable when its component is more
   compact than the GP prefers.** That holds for a planet on a bright ring
   (correct position and Δ log Z = +103). It fails for a faint planet in a
   gap the GP covers: the GP absorbs it, and a stamp placed at the truth is
   worth only +10.8. Stamps also take whatever the GP under-fits, such as
   the bright rim of a ring. The second stamps in case A are ring
   structure (+24 to +26), and the first stamp's flux is biased upward,
   because it takes ring flux (0.086 against 0.03; ρ(disk, stamp) = −0.95).
   The posterior sd does not show this bias. A stamp is evidence of
   compact flux, not of a planet. Calling it a companion needs more:
   - the stamp's flux stays the same as the GP's grid changes;
   - the stamp is outside the GP's support (a gap mask from pre-submission
     information, badged as informed);
   - the stamp is consistent across epochs or wavelengths.
5. **Different regulariser kinds** (a TSV disk with a GP stamp) are
   identifiable for the same reasons as in item 4. Their weights interact,
   though (below).

## Placing a second component

A stamp is a small `Image` centred at an offset. The virgil-validation
implementation's default is 2 beams across, with ⅛-beam pixels and a GP of
ℓ = ¼ beam. The toy's stamp was smaller (½ beam, ℓ = 0.1 beam); a 2-beam
stamp has 16 times the area and can take more of a ring's flux, so the
toy's bias (0.086 against 0.03) and its Δ log Z values may not carry over
to the default. Benchmark both sizes.

**Search.** The noise-only linearised flux map
([`linear_flux_grid`][virgil.grid_fit.linear_flux_grid]) on the fitted
residual finds nothing: the GP has already taken the flux. That gave
SNR 1.1 against 13.9 in the toy. In an exploratory check on a simulated
binary in virgil-validation (vv below), a GP free to absorb a 10%
companion left a residual flat to the noise. The search must
instead be the matched filter against the GP's prior predictive:
- set the latents z to 0, so each image sits at its mean template, keeping
  the fitted fluxes and star (the vv recipe);
- the whitened residuals are then r(z) ≈ r₀ + J z with z ~ N(0, I), so
  their covariance is C = I + J Jᵀ;
- a point with tangent g = ∂r/∂f has

      f = −gᵀC⁻¹r₀ / gᵀC⁻¹g,  SNR = f (gᵀC⁻¹g)^½.

C⁻¹ comes from Woodbury on J's thin SVD, and g from forward mode at each
position, as in `linear_flux_grid`. This is exact for a linear model, where
r* = C⁻¹r₀, so it equals the search on the posterior-mean residual with
the GP covariance. With no GP image it reduces to `linear_flux_grid`, to
rounding (vv's `tests/test_contest_composite.py::
test_marginal_search_finds_a_companion_beside_a_gp`, which also checks
that the map peaks at a 10% companion beside a GP).

The toy's search is broader than this recipe: its covariance
A K Aᵀ + N marginalises every component's flux mode and the star as well
as the pixels, about their prior means. Holding the fitted fluxes, as the
recipe does, leaves any flux error of the star or the image in r₀ without
down-weighting it in C. So the toy's SNR of 13.9 is evidence for the
marginalised search, not a direct test of the recipe. Adding the flux
columns to J would close the gap.

The local alternative linearises about the fit and profiles δz out,
giving f = −gᵀr*/gᵀC⁻¹g. For a log-normal field that has already made a
bump at the companion, it finds nothing: the likelihood has nothing left
to gain at first order.

**Choosing.** Take the k = 3 highest-SNR positive peaks, at least a beam
apart and outside the star's half-beam hole. Fit each as a stamp with
``dra`` and ``ddec`` free, under a uniform prior (a location) within half a
beam of the peak. Keep the one with the largest evidence if it raises
log Z by ln 100. Then search again for the next stamp. In the toy, refining
the offset by evidence over a ±0.1-beam grid put the on-ring planet
0.10 beam from the truth.

**Flux priors.** Each image's flux is relative to the star (or to the
first image, fixed at 1, when there is no star). Its prior is log-uniform
(Jeffreys, a scale) between fixed bounds.

## Joint weight selection

Each GP image's (ℓ, σ) is a hyperparameter, chosen by the evidence. Each
penalty's weight is a regularisation strength, chosen by the discrepancy
principle (`design/regulariser_weight_selection.md`). A composite mixes
the two, so:

1. **Several penalties: fixed ratios, one sweep.** Wrap the penalties as
   one `Joint` penalty, weight × Σ members. The members keep their own
   weights, so their ratios are fixed, defaulting to each family's
   per-data-point middle weight × w. [`l_curve`][virgil.imaging.l_curve]
   then sweeps the common factor, and `LCurve.discrepancy()` picks it on
   the joint fit's χ². If the sweep never reaches χ²/N = 1, take the corner.
   This makes a 2-D problem 1-D and keeps the discrepancy principle's
   meaning: one χ² target, one degree of freedom. The ratio w is a choice
   to vary in a benchmark, not to tune on one dataset. When every member
   has least-squares residuals, the Joint's are their concatenation times
   √weight (virgil's ½ Σ r² = value convention), so Levenberg–Marquardt
   still applies.
2. **The 2-D L-surface** (each weight swept) is the general case. It is
   affordable only for two penalties, and has no unique corner: the
   discrepancy principle gives a curve of solutions, not a point. Use it
   as a diagnostic (does the chosen ratio sit in a flat valley?), not as
   the selector.
3. **GP images by evidence, penalties by discrepancy.** Do coordinate
   ascent: each GP's (ℓ, σ) grid in turn by the evidence, with the other
   images held, then the penalties' factor by discrepancy, holding the GPs.
   One pass is the default. Whether a second pass changes anything is a
   benchmark question.

No virgil change is needed for this. `Joint` is a duck-typed regulariser,
with `value` and optional `residuals`, that `fit` and `l_curve` already
accept. If it proves out in the benchmark, it belongs in
`virgil.imaging` beside the penalties, with a test that its
`l_curve` equals a sweep of the members at fixed ratios.

## Evidence for mixed scenes

- **GP-only composites:** `log_evidence(path=<every GP image>)` integrates
  all their latents together, using the Jacobian of all their columns.
  Add the Occam factor of the other parameters: fluxes, offsets and the
  star's. Integrate each in the coordinates where its Jeffreys prior is
  flat (log for a flux, linear for an offset), with the curvature
  Jᵀ(I + J_z J_zᵀ)⁻¹J marginalised over the latents. A flat prior of
  width R is approximated by a Gaussian of the same variance, R²/12, so
  12/R² is added to the curvature (a precision). The Occam term is then
  −Σ ln Rᵢ + ½ ln det(2π H⁻¹), with H the curvature plus diag(12/R²). The
  added precision keeps an unconstrained parameter's term finite, at
  ½ ln(π/6) ≈ −0.3, instead of a divergent log determinant. Each R is in
  the same units as its parameter, so the term does not depend on units. virgil's `log_evidence` holds them at their MAP. Without the
  Occam factor, a stamp's three extra parameters cost nothing, which
  favours keeping stamps.
- **With penalties**, the evidence is conditional: it is the evidence of
  the GP images and parameters with the penalised images held at their
  fit. It compares stamps and GP grids within one penalised fit, but not
  scenes with different penalties or weights. MaxEnt and TSV have no
  normalised prior in virgil.
- **Comparability:** compare log Z only between scenes fitted to the same
  data with the same error scale. Refit with MacKay's error scale over all
  GP images when it is outside 1/1.3–1.3, then recompute. Report the raw
  χ²/N on the quoted errors first. Evidence with fixed hyperparameters
  (the toy's two-GPs-on-one-grid line) is not comparable with
  evidence-chosen ones.

## Starts

CLEAN ([`clean`][virgil.imaging.clean]) gives a start for each component,
split by scale or by position:
- **By scale:** multi-scale CLEAN (``scales_mas``) assigns each component
  a scale. Restore the point-scale components with a short beam as the
  compact image's mean, and the broad ones as the extended image's mean.
- **By position:** components inside the star's hole start the star's
  flux. Compact clusters outside the main source (the connected region
  above a fraction of the peak, grown by half a beam) start stamps and
  seed the search's candidates. The rest is the main image's mean.
- **The first image's mean** is CLEAN's restored image; the others start
  flat. With a GP, a planet already in the first image's mean is not in
  r₀, so the search cannot find it. Restore the mean without the clusters
  that seed stamps.

## Implementation in virgil-validation

`scripts/contest_composite.py` implements this for the contest pipeline:
- a ``--scene`` grammar, ``name=kind[key=value,...] + ...``, with kinds
  point, ud, image and stamp;
- per-image prior, (ℓ, σ) grids, w, field or fov, pixel, hole, rmax, and
  ``at`` = search or DRA:DDEC;
- the coordinate ascent above;
- outputs in the benchmark's format.

Opt-in benchmark arms in `scripts/contest_bench.py` cover:
- a star, a GP disk and a stamp;
- a compact GP plus an extended GP;
- a GP surface plus a stamp;
- a star, a TSV disk and a GP stamp (mixed regularisers).

## Expectations for the benchmark families

Expected from the toy, before the benchmark:
- **Star + disk + planet** (the weakest family for single images): the
  star + GP-with-hole + stamp scene should win. The star term carries most
  of the gain (Δ log Z ≈ 400). Stamps help most for planets on or near
  bright structure, and least for faint planets in gaps.
- **Envelopes:** star + GP is already near the ceiling. Compact + extended
  should help the total image only where the radius R ≲ 2 beams, and its split should
  not be reported.
- **Spotted stars:** a stamp should find the companion in the draws that
  have one, provided the surface's mean does not already contain it.

## Open items

- A gap or ring support for the main image from pre-submission
  information, so that a planet in the gap is not absorbed.
- The stamp's own flux stability across the main GP's grid, as a
  companion test.
- A parametric envelope (Gaussian or ring) as a component kind, for
  envelopes of 1–2 beams.
- Restoring the first image's mean without the CLEAN clusters that seed
  stamps (vv cuts CLEAN's compact clusters from the mean, but not flux
  CLEAN leaves in the main source).
- Moving `Joint` and the Occam-corrected multi-image evidence into virgil,
  if the benchmark supports them.
