# Radio interferometry

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

This module is not final and claims no parity with MPoL or with published
DSHARP analyses; a comparison with MPoL on ALMA IM Lup data is in progress in
virgil-validation.

[`virgil.radio`](api/radio.md) fits calibrated complex visibilities at
arbitrary (u, v) points, the data of any radio or millimetre array whose
visibilities are in a CASA measurement set. Any [source model](source_models.md) works, including a pixel
[`Image`](api/models/classes/image.md) whose Fourier transform runs through
the NUFFT backend. Each observation (observation) may have its own flux scale,
phase-centre offset and error scale, fitted together with the sky.

## Install

Reading a CASA measurement set needs `casatools`:

```bash
pip install "virgil-astro[radio]"
```

`casatools` publishes wheels for Python 3.12 and 3.13 in its newest releases
(older releases for 3.11) and none for 3.14, so the extra is skipped on
Python 3.14. `ComplexVisData.from_arrays` needs nothing extra.
`casatools` also needs CASA's measures data (see CASA's `casaconfig`) on
first use.

The extra brings `casatools>=6.6`. On some systems casatools 6.7.6 fails to
import with a missing `OPENSSL_3.0.1` symbol; casatools 6.7.0.31 works
(`pip install "casatools==6.7.0.31"`). virgil does not pin the version.

## Conventions

* The visibility is `V(u, v) = Σ I(x, y) exp(−2πi (u x + v y))`, with `x`
  towards East, `y` towards North and `u`, `v` in wavelengths: virgil's
  convention everywhere.
* The CASA `DATA` column is the **conjugate** of this `V`, with
  `u = UVW[0] ν / c` and `v = UVW[1] ν / c`. `read_ms` conjugates it for you.
  This was decided with a point source predicted by CASA's simulator
  (`tests/test_radio.py::test_read_ms_convention`, casatools 6.7.6.14): the
  data match virgil's model to 5×10⁻⁷ with the conjugation and differ by
  order 1 without it. (The equivalent statement without conjugation is
  `u = −UVW[0] ν / c`, `v = −UVW[1] ν / c`.)
* Only cross-correlations are read. A sample is one (row, channel), kept
  when the row and both parallel-hand polarizations (XX and YY, or RR and
  LL) are unflagged and both weights positive. The two polarizations are
  averaged with their weights, `V = (w₁V₁ + w₂V₂)/(w₁ + w₂)`, with weight
  `w₁ + w₂`.
* A weight is `1/σ²` for each of the real and imaginary parts (CASA's
  meaning). `WEIGHT_SPECTRUM` is used if present, otherwise `WEIGHT`.
* `obs` (`OBSERVATION_ID`) is relabelled 0..n−1 in sorted order, the
  originals kept in `obs_ids`. `ddid` numbers the sorted
  (`DATA_DESC_ID`, observation) pairs, so a concatenated set in which one
  `DATA_DESC_ID` spans several observations gets a label, and an error
  scale, for each; `ddid_ids` holds each label's original `DATA_DESC_ID`.

## Reading data

```python
from virgil import ComplexVisData

data = ComplexVisData.read_ms("target.ms", stride=100)  # every 100th row
```

`read_ms` raises if the selection holds more than one field (a mosaic); pass `field=` to pick one. It handles one ddid at a time, so memory is that of the largest ddid.
`select(ddids=..., obs=...)` and `subset(mask)` keep the labels, so a
calibration model built for the full data still fits.

## A model with calibration nuisances

The model at a point `k` of observation `e` is

`M_k = F g_e (ν_k/ν₀)^α exp(−2πi(δα_e u_k + δδ_e v_k)) T(u_k, v_k) V_scene(u_k, v_k)`

with the total flux `F` (Jy), relative gains `g_e`, phase-centre offsets
`(δα_e, δδ_e)` in mas, a spectral index `α` and an optional Gaussian taper
`T` (a convolution of the image). The reference observation has `g = 1` and no
offset. Each ddid label `d` has an error scale `s_d` multiplying its errors, fitted
with the Jeffreys prior `1/s` (or profiled out, below). The spectral index `α` has the prior
`Uniform(0, 5)` by default, which excludes the falling spectra of
synchrotron or free-free sources: pass `priors(alpha=(-3, 5))` for those, or
`alpha=False` to fix it at 0. `CalibrationNuisance` holds this configuration
and the default priors:

```python
import numpyro.distributions as dist
from virgil import CalibrationNuisance, Image, fit

nuis = CalibrationNuisance(data, ref_obs=0)  # nu0_hz: the data's mean frequency
priors = nuis.priors()  # cal.flux, cal.gain, cal.offset, cal.sigma, cal.alpha
init = nuis.defaults()

def scene(values):
    return Image(values["log_brightness"], pixel_scale_mas=2.0)

def model(**values):  # the likelihood term holds the data
    return scene(values)

priors["log_brightness"] = dist.Normal(0.0, 3.0)  # per pixel, shape (64, 64)
init["log_brightness"] = ...                      # a starting image
term = data.term(scene, nuis, chunk_size=1_000_000)
result = fit(model, priors, data=(), likelihoods=[term], init=init, method="lbfgs")
```

Vector-valued sites (`cal.gain` has one entry per non-reference observation) are
ordinary array-valued parameters. Because the error scales are fitted, the
term reports a log normalization and `fit` uses L-BFGS.

**Natural-unit coordinates.** `fit` optimizes in unconstrained coordinates, and
a poorly scaled coordinate makes L-BFGS crawl. The gains and error scales have
the improper Jeffreys prior `1/x` by default (flat in `ln x`, no box), so
the coordinate is `ln x`. The offset of observation `k` is uniform over
three beams (`offset_beams=3`; one beam is `2π b_k`) and is optimized in units of
`b_k = 1/(2π q_max,k)`, `q_max,k` the longest baseline of that observation, so a
unit step moves the phase at the longest baseline by about a turn. An offset known
to a microarcsecond from long baselines and one known to a tenth of a
milliarcsecond from short ones then have curvatures of the same order. The site
values stay physical (`cal.gain`, `cal.offset` in mas, `cal.sigma`). Explicit
bounds, such as `priors(offset=(-lo, hi))` in mas, `priors(gain=(0.5, 2))`
or `sigma=(0.1, 10)`, restore a box (the old defaults used `±1/(2 q_min)`,
about 10⁴ mas, for the offset).

**Profiled error scales.** `CalibrationNuisance(data, sigmas="profile")` has no
`cal.sigma` site. At every evaluation `s_d² = χ²_d(s=1) / (2 N_d)`, `N_d` the number
of unmasked complex points of ddid `d`. The loss of a point is
`½|r|²/s² + 2 ln s`, and for the Jeffreys prior this `s_d` is exactly the
maximum over `ln s_d`, so the profiled optimum of the other parameters is the
joint optimum, found with fewer coordinates and without the degeneracy
between `s` and the amplitude. The gradient is the partial derivative at `ŝ`
(the envelope theorem; `ŝ` is held by `stop_gradient`). `fit` returns `ŝ` in
`result.info["derived"]["cal.sigma"]` and
[`data.sigma_hat`][virgil.radio.ComplexVisData.sigma_hat] computes it for any
values. The reported χ² is at `ŝ` (so it is `2 N_d` per ddid by construction): quote the
χ² at the quoted errors, `Σ_d 2 N_d ŝ_d²`, instead.

The term evaluates the points in chunks of `chunk_size` with `jax.lax.map`,
padded with zero weight, so memory is bounded by the chunk rather than the
number of visibilities.

The data live in the term as array leaves, so `fit` passes them to its jitted
functions as arguments: they are never compile-time constants, and fits that
differ only in the mask (`mask=fold != i`) or in a regularizer weight reuse one
compiled loss. Keep it that way in your own jitted code by passing the term as
an argument:

```python
import equinox as eqx
import jax.numpy as jnp

@eqx.filter_jit
def loss_and_grad(term, values):
    return eqx.filter_value_and_grad(
        lambda v: 0.5 * jnp.sum(term(v) ** 2) + term.log_norm(v)
    )(values)

loss, grad = loss_and_grad(term, values)
```

Do not write `jax.jit(lambda values: ... term(values) ...)`: a closure over the
term embeds every visibility in the compiled program (gigabytes at ALMA scale,
with JAX's "large amount of constants were captured" warning, slow compiles and
executables too big for the persistent cache). The same holds for a
`jax.jit(jax.value_and_grad(f))` where `f` closes over `term`.

For a fit with the scene on the model side only, `data.model(scene, values,
nuis)` returns `M_k` and `data.chi2_by_ddid(scene, values, nuis)` the χ² and
number of points of each ddid.

## Smoothing: `Image(smooth_fwhm_mas=...)` or the nuisance taper

Two Gaussian blurs are available, for different jobs.

- `CalibrationNuisance(taper_fwhm_mas=...)` multiplies the visibilities of
  **any** scene (a parametric model, an `Image`, a `System`) by the Gaussian
  `exp(-π² θ² q² / (4 ln 2))`. It is a uv-plane factor, so a regulariser,
  which acts on pixels, never sees it.
- `Image(..., smooth_fwhm_mas=...)` declares that the sky the data see is the
  image's pixels convolved with a fixed Gaussian, as in MPoL's
  `GaussConvFourier`. Its visibilities are the same factor times the pixel
  transform (exact at any uv point for every `ft`), and
  `Image.smoothed()` returns the convolved pixels by an FFT on a
  zero-padded grid. Regularisers then penalize either the raw pixels (the
  default) or the smoothed ones, with their `smoothed=True` option:

```python
env = Image(eta, 5.0, smooth_fwhm_mas=30.0)
reg = MaxEntropy(w, smoothed=True)  # entropy of the blurred image
```

The regulariser option, rather than a model-level switch, keeps the default
(raw pixels) untouched and lets you mix: entropy on the smoothed image with
total variation on the raw one. `smoothed=True` clips FFT round-off at zero;
`MaxEntropy` and `LogSum` then use a flat default over all pixels, since
smoothing spreads flux beyond the `support`.

MPoL's chain `BaseCube` (softplus) → `GaussConvFourier` (FWHM 30 mas) →
`ImageCube` → NuFFT, with entropy on the smoothed cube, is
`Image(eta, scale, smooth_fwhm_mas=30.0)` with `MaxEntropy(w, smoothed=True)`.

Use the nuisance taper for a scene that is not an `Image`, or to blur a
parametric model; use `smooth_fwhm_mas` when the blur belongs to the image
model and regularisers should see it. Setting both blurs the model twice, so `fit` and
`data.term` warn (`UserWarning`) when they meet.

**Accuracy.** `smoothed()` crops its result to the image, so flux spreading
past the edges is lost (keep the emission clear of them). For a kernel of at
least 3 pixels FWHM the transform of the smoothed pixels equals the analytic
Gaussian times the raw transform to about `exp(-π² θ² (1/p - q)² / (4 ln 2))`
of the flux (an upper bound, times the raw transform at the aliased
frequency), p the pixel size: about 1.5e-8 at FWHM 3 px and `q = 0.25 / p`.
Measured on a compact image in float64 with `|q| <= 0.25 / p`, the error is
3e-11 of the flux at 3 px. A narrower kernel is not band-limited on the pixel grid, and
the discretisation error grows quickly; the visibilities, which use the
analytic factor, stay exact.

## Primary beam and cross-validation

`primary_beam(npix, pixel_scale_mas, freq_hz, dish_m, factor)` is the log
attenuation `−4 ln 2 r²/FWHM²` of a Gaussian primary beam with
`FWHM = factor × λ/dish_m`. Nothing is assumed about the telescope: take the
dish diameter and factor from its documentation. For example, ALMA's 12 m
antennas have `factor = 1.13` (ALMA Technical Handbook), and ATCA's antennas
are 22 m in diameter (ATNF). Add the result to an `Image`'s log-brightness so
that the image is the sky times the beam.

`dartboard_folds(data, k=5)` splits the data into `k` folds for
cross-validation by cells of the uv plane (annuli in q, sectors in the
azimuth mod π), so that a point and its Hermitian partner never straddle a
train/test split and most neighbouring points share a fold (points either
side of a cell edge can still fall in different folds, so scores can be
slightly optimistic). Fit with
`term(..., mask=fold != i)` and score the held-out points with
`chi2_by_ddid(..., mask=fold == i)`.

## Diagnostics

Both diagnostics are model-free displays or checks: neither is part of the
likelihood.

### Error scale from time-adjacent samples

`data.sigma_scale(by="ddid")` estimates the error scale `s` (the factor on
the quoted errors `1/√w`) without a sky model. Samples of one baseline,
channel and scan are sorted by time and paired (1st with 2nd, 3rd with
4th, …); a pair has `Var(ΔRe) = s²(1/wᵢ + 1/wⱼ)`, so

`ŝ² = mean over pairs of ΔRe² / (1/wᵢ + 1/wⱼ)`,

the same for Im, and `ŝ` combines the two. Standard errors are the sample
errors of those means.

```python
res = data.sigma_scale()            # per ddid label; by="obs" or None also work
for label, s, se in zip(res.labels, res.s, res.s_se):
    print(f"ddid {label}: s = {s:.3f} ± {se:.3f}")
```

Caveats:

* **It is an upper bound.** A change of the sky, gain or phase between the
  two integrations adds to `|ΔV|²`, so `ŝ` is biased high where the data vary
  quickly (atmospheric phase, long integrations). Compare it with the fitted
  `cal.sigma`: a fit well below `ŝ` is expected, a fit above it is not.
* It needs `time`, `ant1`, `ant2` and `scan`, which `read_ms` fills (otherwise
  `from_arrays(..., time=, ant1=, ant2=, scan=)`); it raises if they are
  missing. With `read_ms(stride=k>1)` the rows are no longer adjacent.
* `lag_ratio` compares the mean-square difference of far pairs (half a run
  apart) to adjacent pairs. A ratio well above 1 means the pairing does what
  it should; near 1 means constant data or a time order that does not follow
  the data (for example a scrambled time column), so check `ŝ`.
* It sorts one ddid at a time with NumPy and no Python loop over points:
  memory is a few arrays of the largest ddid's length (see the PR notes for
  numbers).

### Dirty and residual images

```python
beam = data.dirty_beam(256, 0.5, weighting="briggs", robust=0.5)
dirty = data.dirty_image(256, 0.5, weighting="briggs", robust=0.5)  # Jy/beam
resid = data.dirty_image(
    256, 0.5, weighting="briggs", scene=scene, values=values, nuisance=nuis,
    use_sigma=True,
)  # image of data - model, in Jy/beam
```

The image is `Σ wₖ Re[Vₖ e^{+2πi(uₖx + vₖy)}] / Σ wₖ`, computed with the
adjoint (type-1) NUFFT of nufftax at the raw (u, v) (nothing is gridded onto
cell centres; the accuracy is that of
[`set_ft_defaults`][virgil._geometry.set_ft_defaults]'s tolerance); the real
part adds the Hermitian conjugates. Dividing by `Σ w` makes the dirty beam
`Σ w cos(2π(ux + vy)) / Σ w` peak at 1 at the phase centre, so the image is in
Jy/beam and a point source of flux `F` at the centre peaks at `F`.

* **Orientation** is that of `Image`: East to the left, North up, the phase
  centre at pixel `((npix − 1)/2, (npix − 1)/2)` (between pixels for an even
  `npix`, where the sampled beam peak is a little below 1: use an odd `npix` for a
  centred pixel), pixel `(row, col)` at `dra = (c − col)s`, `ddec = (c − row)s`.
* **Weighting**: `"natural"` is `w`; `"uniform"` is `w/W_c` with `W_c` the
  weight in the point's uv cell; `"briggs"` is `w/(1 + W_c f²)` with
  `f² = (5·10^−R)² / (Σ_c W_c² / Σ_k w_k)`, natural as `R → +∞` and uniform as
  `R → −∞`. The density is counted on a uv grid (cell `1/FoV` by default;
  Hermitian partners share a cell) and only reweights the points.
* **Residuals**: with `scene`, `values` and `nuisance` the data minus the
  model (with the calibration nuisances applied, as in `term`) is imaged;
  `use_sigma=True` divides the weights by the fitted `s_d²`.
* The beam is not deconvolved: this is a display and a check on a fit, never
  a likelihood. Large data are transformed in chunks of `chunk_size` points.
