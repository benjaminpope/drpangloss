# `drpangloss.oidata`

Data containers and observable helpers for interferometric observations.

`OIData` maps each supported data product into a likelihood comparison vector.
For ordinary OIFITS-style data, this standardized vector concatenates the
visibility and phase observables. For AMIGO mixed-DISCO products, it is the
single mixed log-complex vector defined by the stored log-amplitude and phase
projection operators. In this context, "standardize" means "put data and model
predictions into the same comparison basis", not z-score normalization.

Every (baseline, wavelength) sample is one entry of `u`, `v` and `wavel`, so
data with several wavelength channels need no special handling in models.
Flagged samples are left out of the observables (see `vis_index` and
`phi_index`). Phase residuals are wrapped into `[-π, π)` by `residuals`,
which every likelihood in drpangloss uses.

The bundled `data/calibrated_visibility.npy` fixture is synthetic; see the
AMIGO DISCO tutorial and [`drpangloss.amigo`](amigo.md) for loading it.

## Classes

::: drpangloss.oidata.OIData
    options:
      show_root_heading: false
      heading_level: 3
      show_attributes: false
      members:
        - __init__
        - flatten_data
        - standardize_model
        - to_vis
        - to_phases
        - model
        - residuals
        - with_model

## uv grids

AMIGO DISCO data are sampled on a regular uv lattice; `OIData.uv_grid`
records it, so that a matching `Image` can use an exact matrix Fourier
transform.

::: drpangloss.oidata.UVGrid
    options:
      show_root_heading: true
      heading_level: 3
      show_attributes: false

::: drpangloss.oidata.find_uv_grid
    options:
      show_root_heading: true
      heading_level: 3

## Functions

::: drpangloss.oidata
    options:
      show_root_heading: false
      members:
        - closure_phases
        - cp_indices