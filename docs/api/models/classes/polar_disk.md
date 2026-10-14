# `PolarDisk`: radial profiles by quadrature

A disk, ring or halo described by its radial profile $I(r)$, as one
[component](composition.md) of a `System`. See [Radial profiles and
PolarDisk](../../../polar_disk.md) for usage.

::: virgil.models.PolarDisk
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - suggest_n_r

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
