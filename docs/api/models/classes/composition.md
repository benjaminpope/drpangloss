# Composition: `System` and building blocks

See the [Composing Models](../../../composition.md) tutorial for usage.

::: virgil.models.System
    options:
      show_root_heading: true
      heading_level: 2
      show_attributes: false
      members:
        - model
        - render

::: virgil.models.Component
    options:
      show_root_heading: true
      heading_level: 2
      show_attributes: false
      members: false

::: virgil.models.PointSource
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.GaussianDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.EllipticalGaussian
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.GaussianArc
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.UniformDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.GravityDarkenedStar
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - plot_surface

::: virgil.models.ModulatedGaussianRim
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDiskHG
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDiskGaussian
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDiskPowerLaw
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.Resolved
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.Rotated
    options:
      show_root_heading: true
      heading_level: 2
      show_attributes: false

Models are fitted through [`virgil.likelihood`](../../likelihood.md)
(`build_model`, `loglike`, `numpyro_model`).
