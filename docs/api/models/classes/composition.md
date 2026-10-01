# Composition: `System` and building blocks

See the [Composing Models](../../../composition.md) tutorial for usage.

::: drpangloss.models.System
    options:
      show_root_heading: true
      heading_level: 2
      show_attributes: false
      members:
        - model
        - render

::: drpangloss.models.Component
    options:
      show_root_heading: true
      heading_level: 2
      show_attributes: false
      members: false

::: drpangloss.models.PointSource
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.GaussianDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.EllipticalGaussian
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.UniformDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.ModulatedGaussianRim
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.FlaredDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.FlaredDiskHG
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.FlaredDiskGaussian
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.FlaredDiskPowerLaw
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.Resolved
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: drpangloss.models.Rotated
    options:
      show_root_heading: true
      heading_level: 2
      show_attributes: false

Models are fitted through [`drpangloss.likelihood`](../../likelihood.md)
(`build_model`, `loglike`, `numpyro_model`).
