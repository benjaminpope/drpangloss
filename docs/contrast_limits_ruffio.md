# Contrast limits with the Ruffio method

This page is adapted from the upper-limit notebook workflow and rewritten to be lightweight and reproducible with synthetic data.

The procedure is:
1. Build synthetic binary-like observables.
2. Compute likelihood and optimized-contrast maps.
3. Estimate local Laplace uncertainties.
4. Convert to Ruffio upper limits.

## Imports

```python
import jax.numpy as jnp
import numpy as onp
import jax.scipy as jsp

from drpangloss.models import OIData, BinaryModelCartesian
from drpangloss.grid_fit import (
    likelihood_grid,
    optimized_contrast_grid,
    laplace_contrast_uncertainty_grid,
    ruffio_upperlimit,
)
```

## Synthetic observables (small and self-contained)

```python
rng = onp.random.default_rng(7)

n_bl = 18
u = jnp.linspace(-22.0, 22.0, n_bl)
v = jnp.linspace(18.0, -18.0, n_bl)
wavel = jnp.array([4.8e-6])

truth = {"dra": 100.0, "ddec": 60.0, "flux": 2.5e-3}
model_true = BinaryModelCartesian(**truth)
cvis_true = model_true.model(u, v, wavel)

vis_true = jnp.abs(cvis_true) ** 2
phi_true = jnp.rad2deg(jnp.angle(cvis_true))

d_vis = 0.02 * jnp.ones_like(vis_true)
d_phi = 1.0 * jnp.ones_like(phi_true)

vis_obs = vis_true + d_vis * jnp.array(rng.normal(size=vis_true.shape))
phi_obs = phi_true + d_phi * jnp.array(rng.normal(size=phi_true.shape))

oidata = OIData(
    {
        "u": u,
        "v": v,
        "wavel": wavel,
        "vis": vis_obs,
        "d_vis": d_vis,
        "phi": phi_obs,
        "d_phi": d_phi,
        "i_cps1": None,
        "i_cps2": None,
        "i_cps3": None,
        "v2_flag": True,
        "cp_flag": False,
    }
)
```

## Grid products and Ruffio limits

```python
samples = {
    "dra": jnp.linspace(-250.0, 250.0, 61),
    "ddec": jnp.linspace(-250.0, 250.0, 61),
    "flux": 10 ** jnp.linspace(-5.0, -1.5, 50),
}

# 1) Coarse likelihood cube
ll_cube = likelihood_grid(oidata, BinaryModelCartesian, samples)

# 2) Local optimization over flux at each (dra, ddec)
opt_flux = optimized_contrast_grid(oidata, BinaryModelCartesian, samples)

# 3) Laplace uncertainty map around local optimum
best_idx = jnp.argmax(ll_cube, axis=2)
sigma_flux = laplace_contrast_uncertainty_grid(best_idx, oidata, BinaryModelCartesian, samples)

# 4) Ruffio upper limit at 2-sigma confidence
perc = jnp.array([jsp.stats.norm.cdf(2.0)])
limit_flat = ruffio_upperlimit(opt_flux.flatten(), sigma_flux.flatten(), perc)
limit_map = limit_flat.reshape(*opt_flux.shape, perc.shape[0])[:, :, 0]

{
    "limit_min": float(jnp.nanmin(limit_map)),
    "limit_median": float(jnp.nanmedian(limit_map)),
    "limit_max": float(jnp.nanmax(limit_map)),
}
```

## Notes

- This method combines a local contrast estimate with uncertainty to report upper limits at a chosen confidence.
- In production analyses, use denser grids and observationally calibrated noise models.
- You can convert flux-ratio limits to magnitudes with `-2.5 * log10(limit_map)` when needed.
