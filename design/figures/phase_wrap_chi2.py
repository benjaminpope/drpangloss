"""Stage 0 figure: closure-phase χ² with wrapped Δ² against the chord term.

Run from the repository root:
    python design/figures/phase_wrap_chi2.py
"""

import warnings

import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from drpangloss.likelihood import whitened_residuals
from drpangloss.models import BinaryModelCartesian
from drpangloss.oidata import OIData

warnings.filterwarnings("ignore")
oidata = OIData("data/NuHor_F480M.oifits")

# Bright companion, so model closure phases reach ±π.
truth = BinaryModelCartesian(120.0, 80.0, 0.8)
data = oidata.with_model(truth, key=jax.random.PRNGKey(0))
n_vis = data.vis.size
_, errors = data.flatten_data()
sigma = errors[n_vis:]


def chi2_old(dra):
    model = BinaryModelCartesian(dra, 80.0, 0.8)
    wrapped = data.residuals(data.model(model))[n_vis:]
    return np.sum((wrapped / sigma) ** 2)


def chi2_new(dra):
    model = BinaryModelCartesian(dra, 80.0, 0.8)
    return np.sum(whitened_residuals(model, data)[n_vis:] ** 2)


dra = np.linspace(200.0, 340.0, 1401)
old = jax.vmap(chi2_old)(dra)
new = jax.vmap(chi2_new)(dra)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8))
delta = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 801)
wrapped = np.mod(delta + np.pi, 2 * np.pi) - np.pi
ax0.plot(delta, wrapped**2, label=r"old: wrapped $\Delta^2$")
ax0.plot(delta, 2 * (1 - np.cos(delta)), label=r"new: $4\sin^2(\Delta/2)$")
ax0.set_xlabel(r"phase residual $\Delta$ (rad)")
ax0.set_ylabel(r"$\sigma^2\,\chi^2$ per closure phase")
ax0.legend(frameon=False)
ax1.plot(dra, old, label="old")
ax1.plot(dra, new, label="new")
ax1.set_xlabel(r"companion $\Delta$RA (mas), $\Delta$Dec = 80 mas (truth: 120 mas)")
ax1.set_ylabel(r"closure-phase $\chi^2$")
ax1.legend(frameon=False)
fig.tight_layout()
fig.savefig("design/figures/phase_wrap_chi2.png", dpi=130)
