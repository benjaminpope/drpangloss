# `drpangloss.coverage`

Synthetic uv coverage and noise for simulations, computed on the fly: AMI as
AMIGO represents it (a uv grid with a splodge-weighted mode basis) and as a
classical masking observation (V² and closure phases at the splodge centres),
and a long-baseline observation with Earth-rotation tracks and spectral
channels (VLTI/MATISSE-like).

::: drpangloss.coverage
    options:
      members:
        - ami_grid_record
        - nrm_oidata
        - vlti_oidata
        - mask_transfer
