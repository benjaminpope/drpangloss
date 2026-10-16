# `MultipoleDisk`: radial profiles by quadrature

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

A disk, ring or halo described by its radial brightness profile $I(r)$ in
the plane of the disk, as one [component](composition.md) of a `System`. It is
placed and weighted like any other component, and its visibility is the
Hankel transform of the profile. The maths, and a worked fit, are in the
tutorial [A planet in a gap between rings](../../../multipole_disk_rings.md). The
ideas follow [frank](https://github.com/discsim/frank) (Jennings et al. 2020)
and [vis-r](https://github.com/drgmk/vis-r) (Kennedy 2025); virgil
reimplements them, with no shared code, for V² and closure-phase data and for
composition in a scene.

Each profile owns its quadrature rule, so the transform is a sum over `n_r`
nodes that converges spectrally: a narrow ring needs tens of nodes, not the
hundreds of a midpoint stack.

| Profile | Brightness | Rule |
| --- | --- | --- |
| `ThinRing(radius)` | $\delta(r - r_0)$ | exact: $J_0(2\pi q r_0)$ |
| `GaussianRing(radius, sigma)` | Gaussian in $r$ | Gauss–Legendre over $r_0 \pm 6\sigma$ |
| `RadialPowerLaw(r_in, r_out, index, edge_sigma)` | $r^{\rm index}$ between the radii, sharp or soft edges | Gauss–Legendre |
| `ExpTail(r_in, scale_length, *, cutoff)` | $\exp[-(r - r_{\rm in})/L]$ for $r \ge r_{\rm in}$ | Gauss–Laguerre, to infinity |
| `Gap(base, radius, sigma, depth)` | a Gaussian gap cut in `base` | that of `base` |
| `RadialSum(a=..., b=(profile, weight))` | named profiles, each normalized and mixed by its weight | each part's |
| `RadialGP(r_in, r_out, sigma, ell, nu)` | $\exp g(r)$, $g$ a Gaussian process | its own: Gauss–Legendre on $[r_{\rm in}, r_{\rm out}]$ |

A profile has no weight of its own: only the parts of a `RadialSum` do
(`disk.profile.outer.weight`).

## Names and units

One name for one concept across the family. Lengths are in milliarcseconds
in the plane of the disk, angles in degrees, and a name carries a unit
suffix only where the quantity would otherwise be ambiguous
(`pixel_scale_mas`, `rotation_deg`), as in the older components.

| Concept | Name | Where |
| --- | --- | --- |
| radius of a feature's centre | `radius` | `ThinRing`, `GaussianRing`, `Gap` (as `GaussianArc`, `FlaredDisk`) |
| edges of the support | `r_in`, `r_out` | `RadialPowerLaw`, `ExpTail` (`r_in`), `RadialGP`, `RadialGPFunction`, `MultipoleKernel` |
| reference radius of a function | `r_ref` | `RadialLinear`, `RadialPower`, `PowerLawHeight`, `MultipoleKernel` |
| Gaussian standard deviation | `sigma` (FWHM = 2.355 `sigma`) | `GaussianRing`, `Gap`, as `GaussianDisk`; `edge_sigma` for a soft edge |
| FWHM of a blur | `blur_fwhm` | `MultipoleDisk` |
| e-folding length | `scale_length` | `ExpTail` |
| inclination, position angle of the major axis | `inc`, `pa` (degrees; default 0, face-on, North) | `MultipoleDisk`, `MultipoleField` |
| height | `h_ref`, `psi`; `alpha`, `tip` | `PowerLawHeight`, `Cone` (as `TruncatedCone`) |

The older components keep their names: `ModulatedGaussianRim` takes a
diameter `diam` where a profile takes `radius` (`radius = diam / 2`), and
`EllipticalGaussian` and `TruncatedCone` take an axis ratio `ratio` where the
disk takes `inc` (`ratio = cos(inc)`).

The options each have their own entry below: azimuthal structure
([`Harmonics`](#virgil.models.Harmonics), [`VonMises`](#virgil.models.VonMises),
and the priors of [`MultipoleKernel`](#virgil.models.MultipoleKernel), with
conversions between the forms in [`virgil.harmonics`](../../harmonics.md)),
height ([`Cone`](#virgil.models.Cone), [`PowerLawHeight`](#virgil.models.PowerLawHeight)),
Gaussian-process priors ([`gp_priors`](#virgil.models.gp_priors)), and a
residual aligned with the disk ([`MultipoleField`](../../fields.md)).

::: virgil.models.MultipoleDisk
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
      inherited_members: true
      members:
        - draws
        - nodes

::: virgil.models.RadialGPFunction
    options:
      show_root_heading: true
      heading_level: 3
      inherited_members: true
      members:
        - draws
        - nodes

::: virgil.models.MultipoleKernel
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
