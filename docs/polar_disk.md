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
is azimuthally symmetric, so a centred disk has real visibilities and zero
closure phases; they appear with an offset, odd harmonics, a height, or
alongside other components.

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

`Harmonics(c=..., s=...)` multiplies the profile by
$1 + \sum_m c_m(r)\cos m\theta + s_m(r)\sin m\theta$, with the azimuth
$\theta$ in the plane of the disk, from the projected major axis (position
angle `pa`) towards `pa + 90`. The coefficients are Cartesian (no amplitude
and phase, whose funnel and wrap make fits harder), each a constant or a
function of the radius (`RadialLinear`, `RadialPower`); the degree is the
number of orders, or set with `degree=`. The visibility is
$2\pi\sum_m (-i)^m e^{im\phi}\sum_k w_k I_m(r_k) J_m(2\pi q r_k)$ with
$I_{\pm m} = (c_m \mp i s_m)/2$, from one call to `bessel_jn`. The model is
linear in the coefficients but does not keep the image positive:
`disk.min_intensity()` gives the least brightness relative to the azimuthal
mean (negative is a violation, and `is_physical` is then false).

```python
from virgil import GaussianRing, Harmonics, PolarDisk

lopsided = PolarDisk(
    GaussianRing(6.0, 1.0),
    harmonics=Harmonics(c=(0.3, 0.1), s=(-0.2,)),
    inc=30.0, pa=40.0,
)
```

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
