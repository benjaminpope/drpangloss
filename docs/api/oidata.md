# `drpangloss.oidata`

Data containers and observable helpers for interferometric observations.

`OIData` maps each supported data product into a likelihood comparison vector.
For ordinary OIFITS-style data, this standardized vector concatenates the
visibility and phase observables. For AMIGO mixed-DISCO products, it is the
single mixed log-complex vector defined by the stored log-amplitude and phase
projection operators. In this context, "standardize" means "put data and model
predictions into the same comparison basis", not z-score normalization.

The bundled `data/calibrated_visibility.npy` fixture is synthetic; see the
AMIGO DISCO tutorial for a schema-compatible loading example.

## Classes

::: drpangloss.oidata.OIData
    options:
      show_root_heading: false
      heading_level: 3
      show_attributes: false
      members:
        - __init__
        - standardize_data
        - standardize_errors
        - standardize_model
        - flatten_data
        - unpack_all
        - flatten_model
        - to_vis
        - to_phases
        - model

## Functions

::: drpangloss.oidata
    options:
      show_root_heading: false
      members:
        - load_oi_data
        - closure_phases
        - cp_indices