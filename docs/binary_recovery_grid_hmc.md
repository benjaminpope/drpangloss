# Binary recovery with grid search and HMC

This page is adapted from the existing exploratory notebooks and rewritten as a lightweight, reproducible tutorial that does **not** require large external datasets.

The workflow is:
1. Generate synthetic interferometric observables from a binary model.
2. Recover the companion with a coarse likelihood grid.
3. Refine inference with HMC (NUTS) over `(dra, ddec, flux)`.
4. Reparameterize the latent space with a local Fisher projection and run HMC in near-unit coordinates.

## Imports

```python
import jax
import jax.numpy as jnp
import numpy as onp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from drpangloss.models import OIData, BinaryModelCartesian, loglike
from drpangloss.grid_fit import likelihood_grid
```

## Build a compact synthetic dataset

```python
# Fixed PRNG for reproducibility
rng = onp.random.default_rng(42)

# Baseline geometry (small synthetic layout)
n_bl = 18
u = jnp.linspace(-22.0, 22.0, n_bl)
v = jnp.linspace(18.0, -18.0, n_bl)
wavel = jnp.array([4.8e-6])

# True binary parameters (mas, mas, flux ratio)
truth = {"dra": 120.0, "ddec": -80.0, "flux": 4e-3}
model_true = BinaryModelCartesian(**truth)

# Create noiseless complex visibilities
cvis_true = model_true.model(u, v, wavel)

# Data model: use V^2 + absolute phase (degrees)
vis_true = jnp.abs(cvis_true) ** 2
phi_true = jnp.rad2deg(jnp.angle(cvis_true))

# Modest observational noise
d_vis = 0.02 * jnp.ones_like(vis_true)
d_phi = 1.0 * jnp.ones_like(phi_true)

vis_obs = vis_true + d_vis * jnp.array(rng.normal(size=vis_true.shape))
phi_obs = phi_true + d_phi * jnp.array(rng.normal(size=phi_true.shape))

# OIData from dict (no large files required)
data = OIData(
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

## Step 1: coarse recovery with a grid search

```python
samples = {
    "dra": jnp.linspace(-250.0, 250.0, 81),
    "ddec": jnp.linspace(-250.0, 250.0, 81),
    "flux": 10 ** jnp.linspace(-4.5, -1.5, 60),
}

ll_cube = likelihood_grid(data, BinaryModelCartesian, samples)
max_idx = jnp.unravel_index(jnp.argmax(ll_cube), ll_cube.shape)

grid_est = {
    "dra": float(samples["dra"][max_idx[0]]),
    "ddec": float(samples["ddec"][max_idx[1]]),
    "flux": float(samples["flux"][max_idx[2]]),
}

grid_est
```

## Step 2: posterior refinement with HMC (NUTS)

```python
params = ["dra", "ddec", "flux"]


def model_hmc(oidata):
    dra = numpyro.sample("dra", dist.Uniform(-300.0, 300.0))
    ddec = numpyro.sample("ddec", dist.Uniform(-300.0, 300.0))
    log10_flux = numpyro.sample("log10_flux", dist.Uniform(-6.0, -1.0))
    flux = 10.0 ** log10_flux

    ll = loglike([dra, ddec, flux], params, oidata, BinaryModelCartesian)
    numpyro.factor("loglike", ll)


kernel = NUTS(model_hmc)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1, progress_bar=False)

mcmc.run(jax.random.PRNGKey(123), oidata=data)
posterior = mcmc.get_samples()

summary = {
    "dra_median": float(jnp.median(posterior["dra"])),
    "ddec_median": float(jnp.median(posterior["ddec"])),
    "flux_median": float(jnp.median(10.0 ** posterior["log10_flux"])),
}
summary
```

## Step 3: Fisher-reparameterized HMC (faster, better-conditioned)

The standard HMC above samples directly in physical coordinates. A lightweight conditioning trick is to:

1. Build a local Fisher matrix near a good initial point (for example the grid estimate).
2. Compute a projection matrix $P$ such that $x = x_0 + P u$ with $u \sim \mathcal{N}(0, I)$.
3. Sample in latent coordinates `u` and transform to physical parameters inside the model.

In the executable docs workflow this is implemented in `examples/synthetic_binary_workflow.py` via:

- `drpangloss.inference.fisher_matrix`
- `drpangloss.inference.fisher_projection`
- `_recover_hmc_fisher(...)`

This keeps priors simple in latent space and often improves sampler geometry for strongly correlated parameters.

## Notes

- Grid search provides robust initialization and a quick sanity check of global structure.
- HMC adds uncertainty quantification and posterior correlations.
- Fisher reparameterization is a practical conditioning layer for HMC and black-box optimizers.
- This tutorial keeps everything small and synthetic for docs portability.

For larger science runs, switch the `OIData` dict to real OIFITS-derived arrays and increase chain/grid settings.
