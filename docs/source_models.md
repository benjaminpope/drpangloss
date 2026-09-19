<!-- AUTO-GENERATED FROM notebooks/source_models.ipynb by scripts/sync_tutorial_docs.py. -->
# Extended source models

This tutorial mirrors the binary-model walkthrough style, but focuses on the first non-binary source extensions in `drpangloss`: `GaussianDiskModel` and `HarmonixModel`.

We'll build synthetic interferometric observables from a resolved Gaussian disk, pass them through `OIData`, and then wrap a harmonix-style external source object to show how the visibility and rendering interfaces fit together.

```python
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = repo_root.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from drpangloss.models import OIData, GaussianDiskModel, HarmonixModel
```

## Simulate resolved-disk observables

As with the binary examples, we start from a simple baseline geometry, instantiate a source model, and evaluate complex visibilities on those baselines.

```python
rng = np.random.default_rng(7)
n_bl = 32
u = jnp.array(rng.uniform(-24.0, 24.0, size=n_bl))
v = jnp.array(rng.uniform(-24.0, 24.0, size=n_bl))
wavel = jnp.full((n_bl,), 1.65e-6)

truth = {"sigma": 18.0, "dra": 12.0, "ddec": -7.5}
disk = GaussianDiskModel(**truth)
cvis_true = disk.model(u, v, wavel)

vis_true = jnp.abs(cvis_true) ** 2
phi_true = jnp.rad2deg(jnp.angle(cvis_true))

vis_scale = jnp.maximum(jnp.median(vis_true), 1e-6)
phi_scale = jnp.maximum(jnp.median(jnp.abs(phi_true)), 5.0)
d_vis = 0.01 * vis_scale * jnp.ones_like(vis_true)
d_phi = 0.02 * phi_scale * jnp.ones_like(phi_true)

vis_obs = vis_true + d_vis * jnp.array(rng.normal(size=vis_true.shape))
phi_obs = phi_true + d_phi * jnp.array(rng.normal(size=phi_true.shape))

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

{
    "n_baselines": int(n_bl),
    "sigma_mas": truth["sigma"],
    "centroid_mas": (truth["dra"], truth["ddec"]),
    "vis_range": (float(jnp.min(vis_true)), float(jnp.max(vis_true))),
}
```

```text
{'n_baselines': 32,
 'sigma_mas': 18.0,
 'centroid_mas': (12.0, -7.5),
 'vis_range': (0.0, 0.31383904814720154)}
```

## Use `OIData` to flatten observables

`OIData.model(...)` evaluates the source model, converts complex visibilities to the configured observables, and flattens the result into a vector that can be compared directly to the measured data.

```python
model_vec = data.model(disk)
data_vec, err_vec = data.flatten_data()

{
    "model_len": int(model_vec.shape[0]),
    "data_len": int(data_vec.shape[0]),
    "error_len": int(err_vec.shape[0]),
    "vector_alignment": bool(
        model_vec.shape == data_vec.shape == err_vec.shape
    ),
}
```

```text
{'model_len': 64, 'data_len': 64, 'error_len': 64, 'vector_alignment': True}
```

## Visualize the Gaussian disk in Fourier and image space

A resolved Gaussian disk has visibility amplitudes that fall smoothly with baseline length. The new `render(...)` method gives a matching image-plane representation for plotting and sanity checks.

```python
baseline = np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2)
image = np.asarray(disk.render(npix=128, fov_mas=120.0))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(baseline, np.asarray(vis_obs), alpha=0.8, label="noisy V$^2$")
axes[0].scatter(baseline, np.asarray(vis_true), alpha=0.8, label="true V$^2$")
axes[0].set_xlabel("Baseline length (m)")
axes[0].set_ylabel("Visibility squared")
axes[0].set_title("Resolved Gaussian disk")
axes[0].legend(loc="best")

axes[1].scatter(baseline, np.asarray(phi_obs), alpha=0.8, label="noisy phase")
axes[1].scatter(baseline, np.asarray(phi_true), alpha=0.8, label="true phase")
axes[1].set_xlabel("Baseline length (m)")
axes[1].set_ylabel("Phase (deg)")
axes[1].set_title("Phase response")

im = axes[2].imshow(
    image,
    origin="lower",
    extent=[-60.0, 60.0, -60.0, 60.0],
    cmap="magma",
)
axes[2].set_xlabel(r"$\Delta$RA (mas)")
axes[2].set_ylabel(r"$\Delta$Dec (mas)")
axes[2].set_title("`GaussianDiskModel.render(...)`")
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
```

![source_models output 8.1](generated/source_models_cell008_out01.png)

## Wrap a harmonix-style external source

`HarmonixModel` is the bridge between `drpangloss` and external source objects that already know how to evaluate visibilities. In the real package this can wrap a harmonix source with a `model(u, v, t)` method. Here we use a compact toy object with the same calling convention so the data flow is completely explicit.

```python
class ToySurface:
    def render(self, res, theta):
        coords = jnp.linspace(-1.0, 1.0, res)
        xx, yy = jnp.meshgrid(coords, coords, indexing="xy")
        c, s = jnp.cos(theta), jnp.sin(theta)
        xr = c * xx + s * yy
        yr = -s * xx + c * yy
        image = jnp.exp(-0.5 * ((xr / 0.35) ** 2 + (yr / 0.18) ** 2))
        return image / jnp.sum(image)


class ToyHarmonixSource:
    def __init__(self):
        self.surface = ToySurface()

    def rotational_phase(self, time):
        return 2.0 * jnp.pi * time

    def model(self, uu, vv, time):
        rho = jnp.sqrt((uu / 8.0e7) ** 2 + (vv / 5.0e7) ** 2)
        envelope = jnp.exp(-(rho**2))
        phase = jnp.exp(-2j * jnp.pi * time * uu / 2.0e8)
        return envelope * phase


wrapped = HarmonixModel(ToyHarmonixSource(), observation_time=0.2)
cvis_ext = wrapped.model(u, v, wavel)
image_ext = np.asarray(wrapped.render(npix=128, fov_mas=40.0))

{
    "cvis_shape": tuple(cvis_ext.shape),
    "all_finite": bool(jnp.all(jnp.isfinite(cvis_ext))),
    "render_sum": float(np.sum(image_ext)),
}
```

```text
{'cvis_shape': (32,), 'all_finite': True, 'render_sum': 1.0}
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(baseline, np.asarray(jnp.abs(cvis_ext)), color="tab:green")
axes[0].set_xlabel("Baseline length (m)")
axes[0].set_ylabel("|V|")
axes[0].set_title("`HarmonixModel.model(...)`")

im = axes[1].imshow(
    image_ext,
    origin="lower",
    extent=[-20.0, 20.0, -20.0, 20.0],
    cmap="viridis",
)
axes[1].set_xlabel(r"$\Delta$RA (mas)")
axes[1].set_ylabel(r"$\Delta$Dec (mas)")
axes[1].set_title("Wrapped surface render")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
```

![source_models output 11.1](generated/source_models_cell011_out01.png)

This is the same pattern you would use with a real harmonix object: instantiate the external source, wrap it in `HarmonixModel`, and then call `model(...)` or `render(...)` through the common `SourceModel` interface.
