# Radial profiles and `PolarDisk`

`PolarDisk` is a component for disks, rings and haloes that is described by a
radial brightness profile $I(r)$ in the plane of the disk. It is inclined by
`inc`, rotated by `pa` (the projected major axis, North to East), and placed
by `dra`/`ddec` and weighted by `flux` like any other component of a
[`System`](composition.md). Its visibility is the Hankel transform

$$
V(q) = \frac{\int I(r)\, J_0(2\pi q' r)\, r\,\mathrm{d}r}{\int I(r)\, r\,\mathrm{d}r},
$$

where $q'$ is the baseline's spatial frequency in the plane of the disk, after
undoing the inclination and rotation. Without harmonics or a height the disk
is azimuthally symmetric, so a centred disk has real visibilities (which
change sign past their nulls) and closure phases of 0° or 180°; other values
appear with an offset, odd harmonics, a height, or alongside other
components.

The ideas follow [frank](https://github.com/discsim/frank) (Jennings et al.
2020), which writes a radial profile as a Hankel transform on a set of nodes,
and [vis-r](https://github.com/drgmk/vis-r) (Kennedy 2025), which fits
parametric radial profiles to visibilities. virgil reimplements them, with no
shared code, for V² and closure-phase data and for composition in a scene.

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

Parameters are reached by their path, so they can be fitted like any other:
`"disk.inc"`, `"disk.profile.r0"`, `"disk.profile.base.sigma"`. With a
`RadialSum`, name the parts and tie their weights:

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
`pa + 90`, a phase factor per node after the stretch. The emitting surface is
thin and nothing is occulted.

[`TruncatedCone`](api/models/classes/composition.md) is exactly this: an
`ExpTail(s0 sin(alpha), length sin(alpha), cutoff=5)` profile, a `Cone` height
and harmonics, with `inc = 90 - |tilt|` and `pa` the cone's `pa - 90`. It keeps
its name and parameters, and now converges spectrally in `n_rings`.

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
  well resolved (about $16 \cdot 2\pi q L$); for that, use a power law with a
  large outer radius and a soft edge.
- **Precision.** A profile's reference nodes are built in the default
  precision when the disk is made, so to evaluate in float64 construct the
  disk inside `jax.enable_x64(True)`.
