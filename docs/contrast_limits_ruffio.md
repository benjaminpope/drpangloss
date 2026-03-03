<!-- AUTO-GENERATED FROM /Users/benpope/code/drpangloss/notebooks/contrast_limits_ruffio.ipynb by scripts/sync_tutorial_docs.py. -->
<!-- Edit the notebook, then re-run the sync script. -->

# Contrast limits with Ruffio method

Notebook version of docs/contrast_limits_ruffio.md.

```python
import jax.numpy as jnp
import numpy as onp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import pyoifits as oifits

from drpangloss.models import OIData, BinaryModelCartesian
from drpangloss.grid_fit import (
    likelihood_grid,
    optimized_contrast_grid,
    laplace_contrast_uncertainty_grid,
    ruffio_upperlimit,
)
from drpangloss.plotting import (
    plot_contrast_limit_map,
    radial_limit_summary,
    plot_radial_limit_summary,
)
```

```python
rng = onp.random.default_rng(7)

fname = "NuHor_F480M.oifits"
ddir = "../data/"
data = oifits.open(ddir + fname)
try:
    data.verify("silentfix")
except AttributeError:
    pass

oidata = OIData(data)

truth = {"dra": 100.0, "ddec": 60.0, "flux": 2.5e-3}
model_true = BinaryModelCartesian(**truth)
cvis_true = model_true.model(oidata.u, oidata.v, oidata.wavel)

sim_data = {
    "u": oidata.u,
    "v": oidata.v,
    "wavel": oidata.wavel,
    "vis": oidata.to_vis(cvis_true)
    + jnp.array(rng.normal(size=oidata.vis.shape)) * oidata.d_vis,
    "d_vis": oidata.d_vis,
    "phi": oidata.to_phases(cvis_true)
    + jnp.array(rng.normal(size=oidata.phi.shape)) * oidata.d_phi,
    "d_phi": oidata.d_phi,
    "i_cps1": oidata.i_cps1,
    "i_cps2": oidata.i_cps2,
    "i_cps3": oidata.i_cps3,
    "v2_flag": oidata.v2_flag,
    "cp_flag": oidata.cp_flag,
}

oidata_sim = OIData(sim_data)
```

```python
samples = {
    "dra": jnp.linspace(-250.0, 250.0, 61),
    "ddec": jnp.linspace(-250.0, 250.0, 61),
    "flux": 10 ** jnp.linspace(-5.0, -1.5, 50),
}

ll_cube = likelihood_grid(oidata_sim, BinaryModelCartesian, samples)
opt_flux = optimized_contrast_grid(oidata_sim, BinaryModelCartesian, samples)
best_idx = jnp.argmax(ll_cube, axis=2)
sigma_flux = laplace_contrast_uncertainty_grid(
    best_idx, oidata_sim, BinaryModelCartesian, samples
)

perc = jnp.array([jsp.stats.norm.cdf(2.0)])
limit_flat = ruffio_upperlimit(opt_flux.flatten(), sigma_flux.flatten(), perc)
limit_map = limit_flat.reshape(*opt_flux.shape, perc.shape[0])[:, :, 0]

{
    "opt_flux_finite_frac": float(jnp.mean(jnp.isfinite(opt_flux))),
    "sigma_flux_finite_frac": float(jnp.mean(jnp.isfinite(sigma_flux))),
    "limit_finite_frac": float(jnp.mean(jnp.isfinite(limit_map))),
    "limit_min": float(jnp.nanmin(limit_map)),
    "limit_median": float(jnp.nanmedian(limit_map)),
    "limit_max": float(jnp.nanmax(limit_map)),
}
```

```python
# 2D contrast-limit map (Δmag)

dra_axis = onp.array(samples["dra"])
ddec_axis = onp.array(samples["ddec"])
limit_np = onp.array(limit_map)

plot_contrast_limit_map(
    limit_np,
    dra_axis,
    ddec_axis,
    truth=truth,
    unit_mode="delta_mag",
    title="Ruffio 2σ Upper-Limit Map (Δmag)",
    cmap="inferno",
)
plt.show()
```

```python
# Radial summary: median and spread of limits vs separation
radial_summary = radial_limit_summary(
    limit_np,
    dra_axis,
    ddec_axis,
    center=(0.0, 0.0),
    r_max=350.0,
    n_bins=20,
)
plot_radial_limit_summary(
    radial_summary,
    unit_mode="flux_ratio",
    title="Radial Ruffio limit summary",
)
plt.xlabel("Separation from origin (mas)")
plt.show()
```
