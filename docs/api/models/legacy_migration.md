# Legacy migration from `drpangloss.models_old`

`drpangloss.models_old` is deprecated and scheduled for removal.

## Use direct replacements where available

- `OIData` -> `drpangloss.models.OIData`
- `BinaryModelAngular` -> `drpangloss.models.BinaryModelAngular`
- `BinaryModelCartesian` -> `drpangloss.models.BinaryModelCartesian`
- `cvis_binary_angular` -> `drpangloss.models.cvis_binary_angular`
- `cvis_binary` -> `drpangloss.models.cvis_binary`
- `closure_phases` -> `drpangloss.models.closure_phases`
- `cp_indices` -> `drpangloss.models.cp_indices`
- `nsigma` -> `drpangloss.models.nsigma`

## Legacy-only APIs (no strict 1:1 replacement)

These legacy helpers are retained temporarily for compatibility but should be migrated to modern workflows in `drpangloss.models` and `drpangloss.grid_fit`:

- `vis_binary2`
- `log_like_binary`
- `chi2_binary`
- `log_like_star`
- `log_like_wrap`
- `optimize_log_like`
- `sigma`
- `nsigma_wrap`
- `optimize_nsigma`
- `chi2all`
- `chi2_suball`
- `lim_absil`

For contrast-limit and grid-search workflows, prefer:

- `drpangloss.grid_fit.optimized_contrast_grid`
- `drpangloss.grid_fit.laplace_contrast_uncertainty_grid`
- `drpangloss.grid_fit.absil_limits`
