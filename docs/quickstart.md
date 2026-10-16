# Quickstart

Install virgil (the PyPI name is `virgil-astro`; `pip install virgil` is a different project), load an OIFITS file, and fit a binary to it. It runs in seconds on a laptop.

```bash
pip install virgil-astro
```

The file is `data/NuHor_F480M.oifits` in the [repository](https://github.com/benjaminpope/virgil): JWST aperture-masking observations of a single star, so the data contain no companion. To show what a detection looks like, the example injects one into the file's own sampling and errors with [`simulate`][virgil.simulate.simulate]. To fit your own file, load it and skip that line. After `pip install`, the file is not on your disk: download it first and point `OIData` at the copy.

```bash
curl -LO https://raw.githubusercontent.com/benjaminpope/virgil/main/data/NuHor_F480M.oifits
```

```python
import jax
import jax.numpy as np
import numpyro.distributions as dist
import virgil as vg
from virgil.grid_fit import best_grid_point, likelihood_grid
from virgil.models import BinaryModelCartesian
from virgil.simulate import simulate

template = vg.OIData("data/NuHor_F480M.oifits")  # your file here
truth = BinaryModelCartesian(dra=120.0, ddec=-80.0, flux=0.02)
data = simulate(truth, template, key=jax.random.PRNGKey(0))  # inject a companion

# 1. Search a grid of positions (mas) and fluxes (companion/primary).
grid = {
    "dra": np.linspace(-250.0, 250.0, 50),
    "ddec": np.linspace(-250.0, 250.0, 50),
    "flux": 10 ** np.linspace(-3.0, -1.0, 9),
}
log_like = likelihood_grid(BinaryModelCartesian, data=data, grid=grid)
start = best_grid_point(log_like, grid)

# 2. Refine the best grid point with a maximum a posteriori fit.
priors = {
    "dra": dist.Uniform(-250.0, 250.0),
    "ddec": dist.Uniform(-250.0, 250.0),
    "flux": dist.LogUniform(1e-4, 0.1),
}
result = vg.fit(BinaryModelCartesian(**start), priors, data)

print({name: float(value) for name, value in result.values.items()})
print(result.info["chi2_red"], result.info["converged"])
```

The result recovers the injected companion: `dra` near 120 mas, `ddec` near -80 mas and `flux` near 0.02, with a reduced χ² near 1 (the exact numbers vary a little between JAX versions).

- `dra` and `ddec` are offsets in milliarcseconds, with East positive in `dra` and North positive in `ddec`; `flux` is the companion/primary flux ratio. See [conventions](conventions.md).
- The grid search is the first step in any companion search, because the likelihood of visibilities has many local maxima. [Binary search](binary_search.md) shows how to plot the grid and sample the posterior.
- `result.model` is the fitted model, and `result.values` its parameters. [`fit`][virgil.fitting.fit] accepts any model and prior, not only binaries.

## Where next

The [landing page](index.md#where-to-start) lists a page for each task. The [pipelines](pipeline.md) run the same search, limits, fit and sampling in one call and leave a run folder.
