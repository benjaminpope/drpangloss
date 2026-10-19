# Radio interferometry

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

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
with a log-uniform prior. The spectral index `α` has the prior
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
