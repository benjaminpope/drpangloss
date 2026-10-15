# `PolarDisk`: radial profiles by quadrature

A disk, ring or halo described by its radial brightness profile $I(r)$ in
the plane of the disk, as one [component](composition.md) of a `System`. It is
placed and weighted like any other component, and its visibility is the
Hankel transform of the profile. The maths, and a worked fit, are in the
tutorial [A planet in a gap between rings](../../../polar_disk_rings.md). The
ideas follow [frank](https://github.com/discsim/frank) (Jennings et al. 2020)
and [vis-r](https://github.com/drgmk/vis-r) (Kennedy 2025); virgil
reimplements them, with no shared code, for V² and closure-phase data and for
composition in a scene.

Each profile owns its quadrature rule, so the transform is a sum over `n_r`
nodes that converges spectrally: a narrow ring needs tens of nodes, not the
hundreds of a midpoint stack.

| Profile | Brightness | Rule |
| --- | --- | --- |
| `ThinRing(r0)` | $\delta(r - r_0)$ | exact: $J_0(2\pi q r_0)$ |
| `GaussianRing(r0, sigma)` | Gaussian in $r$ | Gauss–Legendre over $r_0 \pm 6\sigma$ |
| `RadialPowerLaw(r_in, r_out, index, edge)` | $r^{\rm index}$ between the radii, sharp or soft edges | Gauss–Legendre |
| `ExpTail(r0, scale)` | $\exp[-(r - r_0)/L]$ for $r \ge r_0$ | Gauss–Laguerre, to infinity |
| `Gap(base, r0, width, depth)` | a Gaussian gap cut in `base` | that of `base` |
| `RadialSum(a=..., b=...)` | named profiles, each normalized and mixed by its `weight` | each part's |
| `RadialGP(r_in, r_out, sigma, ell, nu)` | $\exp g(r)$, $g$ a Gaussian process | its own: Gauss–Legendre on $[r_{\rm in}, r_{\rm out}]$ |

The options each have their own entry below: azimuthal structure
([`Harmonics`](#virgil.models.Harmonics), [`VonMises`](#virgil.models.VonMises),
and the priors of [`PolarKernel`](#virgil.models.PolarKernel), with
conversions between the forms in [`virgil.harmonics`](../../harmonics.md)),
height ([`Cone`](#virgil.models.Cone), [`PowerLawHeight`](#virgil.models.PowerLawHeight)),
Gaussian-process priors ([`gp_priors`](#virgil.models.gp_priors)), and a
residual aligned with the disk ([`PolarField`](../../fields.md)).

::: virgil.models.PolarDisk
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - suggest_n_r
        - freeze_geometry
        - min_intensity
        - gp_sites

## Profiles

::: virgil.models.ThinRing
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.GaussianRing
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.RadialPowerLaw
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.ExpTail
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.Gap
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.RadialSum
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.Harmonics
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - image_harmonics
        - convert
        - coefficients
        - polynomial

::: virgil.models.VonMises
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - image_harmonics
        - truncation_power

::: virgil.models.RadialLinear
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.RadialPower
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.Cone
    options:
      show_root_heading: true
      heading_level: 3
      members: false

::: virgil.models.PowerLawHeight
    options:
      show_root_heading: true
      heading_level: 3
      members: false

## Gaussian-process priors

::: virgil.models.RadialGP
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - draws
        - nodes

::: virgil.models.PolarKernel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - spectrum
        - draws

::: virgil.models.gp_priors
    options:
      show_root_heading: true
      heading_level: 3
