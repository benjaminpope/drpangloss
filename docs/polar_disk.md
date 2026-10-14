# `PolarDisk` reference

`PolarDisk` is a component for disks, rings and haloes described by a radial
brightness profile $I(r)$ in the plane of the disk. It is placed and weighted
like any other component of a [`System`](composition.md), and its visibility is
the Hankel transform of the profile (the maths, and a worked fit, are in the
tutorial [A planet in a gap between rings](polar_disk_rings.md)). This page
lists the API: the constructor, the profiles, harmonics, height, Gaussian-process
priors, spectral index, frozen geometry and `PolarField`. The ideas follow
[frank](https://github.com/discsim/frank) (Jennings et al. 2020) and
[vis-r](https://github.com/drgmk/vis-r) (Kennedy 2025); virgil reimplements them,
with no shared code, for V² and closure-phase data and for composition in a scene.

## The constructor

`PolarDisk(profile, harmonics=None, height=None, inc=0.0, pa=0.0, blur_fwhm=None,
flux=1.0, dra=0.0, ddec=0.0, n_r=64, spectral_index=None, wavel0=None)`

| Argument | Meaning |
| --- | --- |
| `profile` | the radial profile ([below](#profiles-and-quadrature)); it owns its quadrature rule |
| `harmonics` | azimuthal structure: [`Harmonics`](#harmonics) or `VonMises`; none gives a symmetric disk |
| `height` | `Cone` or `PowerLawHeight`: [lifts each ring above the plane](#height-flared-disks-and-cones) |
| `inc`, `pa` | inclination (degrees, 0 is face-on) and position angle of the projected major axis (North to East) |
| `blur_fwhm` | Gaussian smoothing in the plane of the disk (mas); an exact Fourier envelope |
| `flux` | weight in a `System`, a number or a spectrum |
| `dra`, `ddec` | offset of the centre (mas) |
| `n_r` | quadrature nodes per profile; static (changing it recompiles) |
| `spectral_index`, `wavel0` | [chromatic profile](#spectral-index): $I_0(r)(\lambda/\lambda_0)^{\alpha(r)}$ |

The visibility is normalized to one at zero baseline, so `flux` is the only
weight. Without harmonics or a height the disk is azimuthally symmetric: a
centred disk has real visibilities (changing sign past their nulls) and
closure phases of 0° or 180°, and other values appear with an offset, odd
harmonics, a height, or other components. Parameters are reached by path, so
they can be fitted like any other: `"disk.inc"`, `"disk.profile.r0"`,
`"disk.profile.base.sigma"`.

## Profiles and quadrature

Each profile owns its quadrature rule, so the transform is a sum over
`n_r` nodes that converges quickly:

| Profile | Brightness | Rule |
| --- | --- | --- |
| `ThinRing(r0)` | $\delta(r - r_0)$ | exact: $J_0(2\pi q r_0)$ |
| `GaussianRing(r0, sigma)` | Gaussian in $r$ | Gauss–Legendre over $r_0 \pm 6\sigma$ |
| `RadialPowerLaw(r_in, r_out, index, edge)` | $r^{\rm index}$ between the radii; sharp edges (`edge=None`) or error-function edges | Gauss–Legendre |
| `ExpTail(r0, scale)` | $\exp[-(r - r_0)/L]$ for $r \ge r_0$ | Gauss–Laguerre, to infinity |
| `Gap(base, r0, width, depth)` | a Gaussian gap cut in `base` | that of `base` |
| `RadialSum(a=..., b=...)` | named profiles, each normalized and mixed by its `weight` | each part's |
| `RadialGP(r_in, r_out, sigma, ell, nu)` | $\exp g(r)$, $g$ a Gaussian process ([below](#gaussian-process-profiles)) | its own nodes: Gauss–Legendre on $[r_{\rm in}, r_{\rm out}]$ |

Because the rules converge spectrally, a narrow ring needs tens of nodes, not
the hundreds of a midpoint stack. `n_r` is a static integer: changing it
recompiles, and changing any parameter does not. Set it with
`PolarDisk(profile, n_r=...)`; `disk.suggest_n_r(data)` returns a starting
value from the longest baseline and the profile's extent, which you should
check by doubling it.

## Example

A star, and a ring with a gap in it, inclined at 50° and blurred:

```python
import jax.numpy as np
from virgil import (
    GaussianRing, Gap, PointSource, PolarDisk, RadialSum, System, ThinRing,
)

ring = Gap(GaussianRing(r0=10.0, sigma=3.0), r0=10.0, width=0.8)
disk = PolarDisk(ring, inc=50.0, pa=30.0, blur_fwhm=1.0, n_r=48)
scene = System(star=PointSource(), disk=disk.set("flux", 0.3))

u = np.linspace(0.0, 120.0, 64)  # baselines in metres
cvis = scene.model(u, 0.3 * u, 2.2e-6)
image = scene.render(npix=256, fov_mas=60.0)
```

With a `RadialSum`, name the parts and tie their weights:

```python
profile = RadialSum(
    inner=GaussianRing(3.0, 0.5),
    outer=GaussianRing(8.0, 1.0, weight=0.5),
)
disk = PolarDisk(profile, inc=40.0, pa=20.0, n_r=32)  # "disk.profile.outer.weight"
```

## Harmonics

`harmonics=` makes the disk lopsided: the brightness at radius $r$ and azimuth
$\theta$ (in the plane of the disk, from the projected major axis, at position
angle `pa`, towards `pa + 90`) is the profile times an azimuthal factor of mean
one and **exact degree $M$**, with the harmonics $0 \dots M$ and none beyond.
Each coefficient is a constant or a function of the radius (`RadialLinear`,
`RadialPower`), and the degree is static. The visibility is
$2\pi\sum_m (-i)^m e^{im\phi}\sum_k w_k I_m(r_k) J_m(2\pi q r_k)$ with the
*image harmonics* $I_m$ of the factor ($I_{-m} = \bar I_m$, $I_0 = 1$), from
one call to `bessel_jn`. Odd orders give closure phases for a centred disk;
even orders are seen by $V^2$ alone and are partly degenerate with `inc` and
`pa`.

**Positive form** (the default when no `c` or `s` is given):
$|h(e^{i\theta})|^2 / \sum_j |h_j|^2$ with
$h(z) = 1 + h_1 z + \dots + h_M z^M$. Every non-negative image of degree $M$
has this form (the Fejér–Riesz theorem), so the brightness is positive for
all parameter values, with no constraint to enforce. Reflecting a root of $h$
through the unit circle leaves the image unchanged, so $h$ itself has up to
$2^M$ equivalent values; the parametrization keeps only the minimum-phase
one, built from complex reflection coefficients $k_j$ with $|k_j| < 1$ by the
Levinson step-up recursion, and each $k_j = w_j \tanh|w_j| / |w_j|$ comes
from an unconstrained Cartesian latent $w_j = $ `re[j-1]` $+ i\,$`im[j-1]`
(finite gradient at $w = 0$, which is a uniform ring). Optimizers and samplers
then see one smooth mode. The latents are an internal coordinate: report the
image harmonics, `disk.harmonics.image_harmonics()`.

**Linear form** (`c=` and `s=`): $1 + \sum_m c_m \cos m\theta + s_m \sin m\theta$,
exactly linear in the Cartesian coefficients (no amplitude and phase, whose
funnel and wrap make fits harder). It does not keep the image positive:
`disk.min_intensity()` gives the least brightness relative to the azimuthal
mean (negative is a violation, and `is_physical` is then false).

```python
from virgil import GaussianRing, Harmonics, PolarDisk

# A positive lopsided ring with a second harmonic: latents w_1, w_2.
lopsided = PolarDisk(
    GaussianRing(6.0, 1.0),
    harmonics=Harmonics(w=(0.4 + 0.1j, 0.2j)),
    inc=30.0, pa=40.0,
)

# The linear form, with c_m and s_m given directly.
linear = PolarDisk(
    GaussianRing(6.0, 1.0),
    harmonics=Harmonics(c=(0.3, 0.1), s=(-0.2,)),
    inc=30.0, pa=40.0,
)
```

### Arcs: `VonMises`

`VonMises(kappa, theta0, degree=M)` is the arc $\exp[\kappa\cos(\theta -
\theta_0)]$ (width about $1/\sqrt\kappa$ radians, centred on the azimuth
`theta0` in degrees; `kappa=0` is a uniform ring). Its harmonics are
analytic, $\propto I_m(\kappa)/I_0(\kappa)\,e^{-im\theta_0}$ with $I_m$ the
modified Bessel function, and are computed by a small FFT, so they are
differentiable in both parameters. The series is truncated at `degree`, which
must grow with $\kappa$ (about $5.7\sqrt\kappa$ for harmonics below
$10^{-6}$; the default picks it for a constant `kappa`);
`disk.harmonics.truncation_power()` reports the fraction of the power the
truncation drops.

```python
from virgil import PolarDisk, ThinRing, VonMises

arc = PolarDisk(ThinRing(20.0), harmonics=VonMises(kappa=30.0, theta0=250.0),
                blur_fwhm=2.0)
```

[`GaussianArc`](api/models/classes/composition.md) is exactly this, with
$\kappa = (2\sqrt{2\ln 2}\,\mathrm{radius}/\mathrm{length})^2$: the arc's
`length` is the FWHM of the weight along it, as before, but the weight is the
von Mises function of the position angle where it used to be a Gaussian in
arc length cut off at the antipode. The two agree for arcs short against the
radius (to about 1e-3 in $V$ for `length` below 0.1 of the semicircle, 1e-2 at
0.2) and differ for long arcs. It keeps its name and parameters.

### Converting between forms

The module [`virgil.harmonics`](api/harmonics.md) maps every form to the image
harmonics $I_m$ and back, to report any model in one convention, to warm-start
a positive-form fit from a linear one, and to check models against each other:

```python
from virgil import Harmonics, harmonics

linear = Harmonics(c=(0.5, 0.1), s=(0.2, -0.1))
image = harmonics.to_image_harmonics(linear)        # I_0 .. I_2, complex
positive = linear.convert("positive")               # same image, latents w
again = harmonics.from_image_harmonics(image, form="positive")
```

Linear and image harmonics are exact both ways ($c_m = 2\,\mathrm{Re}\,I_m$,
$s_m = -2\,\mathrm{Im}\,I_m$), and so is positive to image. Image to positive
is a spectral factorization in numpy (not differentiated): it takes the roots
of $z^M \sum_m I_m z^m$ outside the unit circle, then the step-down recursion
and the inverse tanh. It raises an error if the image is negative somewhere,
and an image that touches zero (a root on the circle) is clipped to
$|k| = 1 - 10^{-6}$ with a warning. `from_legacy` and `to_legacy` convert the
amplitude and azimuth pairs `az_amps`/`az_pas` of `ModulatedGaussianRim` and
`TruncatedCone`, with each class's azimuth origin and sense;
`ModulatedGaussianRim` is a `ThinRing`, linear `Harmonics` and `blur_fwhm`.

## Height: flared disks and cones

`height=` lifts each ring above the plane of the disk. `Cone(alpha, tip)` puts
the ring of radius $r$ at $r\cot\alpha - \mathrm{tip}$, and
`PowerLawHeight(h0, r0, psi)` at $h_0 (r/r_0)^\psi$. On the sky the ring moves
by $h\sin i$ along the projected minor axis, towards position angle
`pa + 90`, a phase factor per node after the stretch. A positive height
shifts the ring towards `pa + 90`. A top surface lifted towards the observer
shifts the apparent ring towards the *far* side, so when `pa + 90` is the near
side of the disk (the side tilted towards the observer), the top surface of a
flared disk needs a negative `h0`. The emitting surface is
thin and nothing is occulted.

[`TruncatedCone`](api/models/classes/composition.md) is exactly this: an
`ExpTail(s0 sin(alpha), length sin(alpha), cutoff=5)` profile, a `Cone` height
and harmonics, with `inc = 90 - |tilt|` and `pa` the cone's `pa - 90`. It keeps
its name and parameters, and now converges spectrally in `n_rings`.

## Gaussian-process profiles

[`RadialGP`](api/models/classes/polar_disk.md) is a nonparametric profile: the
log-intensity at the quadrature nodes is $g = \mu + \sigma L z$, with $L$ the
Cholesky factor of a Matérn kernel of length $\ell$ and $z \sim N(0, I)$ the
free parameters ("whitened" latents, which NUTS samples well). The intensity
is $e^{g}$, so it is positive for any $z$. The nodes are the Gauss-Legendre
nodes of the profile's own range, so `n_r` is the number of latents (keep it
at 100 or below: the factor is dense). Linear interpolation between nodes is
used only for rendering.

```python
from virgil import PolarDisk, RadialGP, fit, gp_priors

start = PolarDisk(RadialGP(0.5, 10.0, sigma=1.5, ell=1.5), n_r=24)
result = fit(start, gp_priors(start), data)
```

`gp_priors(model)` gives `Normal(0, 1)` priors on every latent vector and,
when bounds are passed (`sigma=(0.3, 3.0)`, `ell=(1.0, 6.0)`), log-uniform
priors on the scales. The scales are fixed unless given priors. A MAP fit of
`sigma` or `ell` is biased towards the prior edge (`fit` warns), so sample
them with NUTS (`numpyro_model(..., flat_coordinates=True)`) or leave them
fixed and compare `imaging.log_evidence` at several values: the Laplace
evidence is over the latents, conditional on the hyperparameters.

**What `ell` means.** The Matérn length $\ell$ is a correlation length of
$\ln I_0$ in milliarcseconds, not the width of a feature. A squared-exponential
bump with the same curvature as the default Matérn-3/2 kernel has width
$\ell/\sqrt3$ (so $\ell = 1.5$ mas corresponds to a Gaussian $\sigma$ of about
0.9 mas), and the prior's rms slope in $\ln I_0$ is $\sqrt3\,\sigma/\ell$ per
mas, so a large `sigma` with a short `ell` is a very rough prior (many
maxima). Visibilities cannot constrain structure finer than the half-fringe
$\lambda/2B$ of the longest baseline, so below that scale, and where the data
have no leverage (the inner edge, the outer tail), the posterior is the prior.
A modest evidence difference between two `ell` values means the data do not
choose between them: report `ell` as a prior choice.

### Priors over the harmonics

`Harmonics(degree=M, prior=PolarKernel(ell_r, ell_theta, sigma))` replaces the
coefficients with a prior: each order $m$ is a `RadialGP` of length `ell_r`
(across radius), independent between orders. The azimuthal length
`ell_theta` (on the same scale as `ell_r`, in mas along the ring at
$\bar r$) sets how fast the prior falls off with $m$: large values suppress
high orders. The variance of the order-$m$ coefficients is proportional to
$(1 + (m \ell_\theta / \bar r)^2 / 2\nu)^{-\nu - 1/2}$ (the Matérn
spectral density; a squared exponential for $\nu = \infty$). This law is
exact for the coefficients of the linear form (`form="linear"`). In the
default positive form the latents drive the reflection-coefficient inputs
rather than the brightness harmonics, so it holds only loosely there.
$\bar r$ is `r_ref`, by default the middle of the profile's radial extent, so
for a `GaussianRing` (whose range runs from 0 to a few widths beyond the ring)
it is not the ring radius: pass `r_ref` to set it. The latents are the field
`harmonics.z` of shape `(2, M, n)`.

## Spectral index

`PolarDisk(..., spectral_index=alpha, wavel0=lambda0)` makes the intensity
$I(r, \lambda) = I_0(r)(\lambda/\lambda_0)^{\alpha(r)}$. `alpha` is a number,
a radial function from `virgil.models` (for example `RadialLinear`) or a
`RadialGP` (here $\alpha = g$ directly, not exponentiated). The visibility is
normalized at every wavelength, and the share of the disk's light in a
[`System`](api/models/classes/composition.md) follows
$\int I(r, \lambda)\, r\, dr$ relative to $\lambda_0$, with `flux` the
flux at `wavel0` (a scalar: a spectrum from `virgil.spectra` is refused).
Harmonics are grey.

## Frozen geometry

`disk.freeze_geometry(data)` returns the disk with the Bessel matrices for
those baselines cached. With `inc`, `pa`, the height and the nodes held fixed,
a refit of the profile, the harmonics or the index is then a matrix-vector
product per order. It is never automatic. If a frozen quantity or the
baselines change the model falls back to the full computation (same answer,
no speed-up), checked at every call. Freeze after the data are in the
precision you will fit in if you need full accuracy: a cache built in float32
(virgil's default) is still recognized when `fit` or `log_evidence` evaluates
in float64, but its matrices keep float32 accuracy (about $10^{-7}$).

## Residuals aligned with the disk: `PolarField`

A parametric disk misses some of the light. Rather than a Cartesian
[`GaussianField`](api/fields.md), which knows nothing about the disk, a
`PolarField` is a Gaussian-process residual on a polar grid in the disk's
frame, with a Matérn kernel in radius (length `length_r`) and a periodic one
in azimuth (`length_theta`, radians). A long `length_theta` favours rings and
arcs without imposing them; a short `length_r` allows gaps. It is evaluated
on the grid, deprojected with `inc`, `pa` and the centre, and interpolated
onto the pixels of an `Image`, whose brightness stays positive (a softmax).
The latents are whitened, so `image_priors`, `fit`, `log_evidence` and the
numpyro model work as for a `GaussianField`:

```python
from virgil import Image, PolarField
from virgil.fields import polar_field_priors
from virgil.imaging import image_priors

field = PolarField.from_disk(
    disk, np.zeros((24, 32)), image_shape=(64, 64), sigma=1.0,
    length_r=4.0, length_theta=1.0,
)
scene = System(star=PointSource(), disk=disk, resid=Image(field, 1.5, flux=0.1))
priors = image_priors(scene) | polar_field_priors("resid.log_brightness")
```

`polar_field_priors` gives `sigma` and the two lengths bounded log-uniform
priors, so the evidence is proper; fit them by sampling or from the evidence,
not by MAP (`fit` warns). The geometry is copied from the disk, not tied to
it. The construction follows the whitened latents of
[FRAP](https://github.com/tomyoshida/frap) (Yoshida et al. 2026) and the polar
representation of [frank](https://github.com/discsim/frank) (Jennings et al.
2020); no code is shared.

## Worked example

[A planet in a gap between rings](polar_disk_rings.md) simulates three inclined
rings and a planet, fits them parametrically, then recovers the same structure
with one Gaussian-process disk (profile, spectral index and lopsidedness) and
compares the two by their Laplace evidence.

## Notes

- **Normalization.** The disk has unit flux, and the transform is divided by
  its own total so that $V(0) = 1$. A weight is the `flux` of the component.
- **Over-resolved light.** Emission on scales much larger than the shortest
  baseline resolves only as a loss of visibility, which is also what
  `Resolved` models. The data constrain the sum of the two, so report the
  total, and let the priors split them.
- **The blur** is a Gaussian envelope in the Fourier plane, applied in the
  plane of the disk: exact and free.
- **Exponential tails** take more nodes than finite ranges when the tail is
  well resolved (the rule of thumb $16 \cdot 2\pi q L$ is optimistic: about
  $2 \times 10^{-4}$ in $V$ at 128 nodes in a widely resolved benchmark, and
  $4 \times 10^{-5}$ at the 150-node cap); for that, use a power law with a
  large outer radius and a soft edge, or a `cutoff`.
- **Precision.** A profile's reference nodes are built in the default
  precision when the disk is made, so to evaluate in float64 construct the
  disk inside `jax.enable_x64(True)`.
