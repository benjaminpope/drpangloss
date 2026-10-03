"""Shared helpers for the PAVO robustness scripts (multistart, profile,
injection, summarise).

Everything here uses the conventions of ``../fit_pavo.py``: the data
loader and cuts, ``GravityDarkenedStar`` with ``n_lat=32``, float64, his
prior box, and his posterior (``../reference_posteriors.json``) in our
conventions. The jitter is held fixed at his posterior median and added in
quadrature to the V^2 errors, so the likelihood is the same Gaussian as his
and ``chi2`` below is the sum of squared whitened V^2 residuals.

Conventions
-----------
* ``pa`` is the position angle (degrees, East of North) of the visible
  pole; V^2 cannot tell ``pa`` from ``pa + 180``, so PAs are compared mod
  180 (``circ_diff``).
* ``virgil`` offsets use ``u`` East and ``v`` North (``offset_phase``), so a
  baseline's position angle is ``atan2(u, v)`` (``baseline_pa``). Note that
  ``fit_pavo.py``'s colour-by-angle plot uses ``atan2(v, u)``, which is the
  angle from the u axis, 90 degrees away from a position angle.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAVO = HERE.parent
sys.path.insert(
    0, str(PAVO)
)  # for fit_pavo (an example script, not a package)

import jax  # noqa: E402

# float64 for the whole process, as in fit_pavo.py
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from fit_pavo import MAX_DIAM, N_LAT, STARS, load_pavo  # noqa: E402,F401
from scipy.stats import qmc  # noqa: E402

from virgil import GravityDarkenedStar, OIData, UniformDisk  # noqa: E402
from virgil.fitting import fit  # noqa: E402

OUT = HERE / "output"
PARAMS = ("diam_eq", "omega", "inc", "pa")
REF_KEYS = dict(diam_eq="diam_eq", omega="omega", inc="inc_deg", pa="pa_deg")
# Prior box for the fits. ``inc`` must lie in [0, 90] for the model. ``pa``
# is allowed to run past [0, 180) so that no optimum sits on a boundary; it
# is wrapped mod 180 afterwards.
PRIORS = dict(
    diam_eq=dist.Uniform(1e-4, MAX_DIAM),
    omega=dist.Uniform(0.0, 0.99),
    inc=dist.Uniform(0.0, 90.0),
    pa=dist.Uniform(-90.0, 270.0),
)
# fits on a hard-to-fit surface may not converge: record, don't raise
MAX_STEPS = 400


# ----------------------------------------------------------------- angles
def wrap180(pa):
    """Wrap a position angle (deg) into [0, 180)."""
    return np.mod(pa, 180.0)


def circ_diff(a, b):
    """Signed difference a - b of position angles mod 180, in [-90, 90)."""
    return np.mod(np.asarray(a) - np.asarray(b) + 90.0, 180.0) - 90.0


def baseline_pa(u, v):
    """Baseline position angle (deg, East of North, mod 180) = atan2(u, v)."""
    return wrap180(np.degrees(np.arctan2(u, v)))


def coverage_density(pa_grid, bl_pa, kappa=8.0):
    """Density of baseline PAs (von Mises kernel on 2*PA), peak-normalised
    to mean 1 over a uniform grid, evaluated at ``pa_grid`` (deg)."""
    th = np.radians(2.0 * np.asarray(pa_grid, float))[:, None]
    ph = np.radians(2.0 * np.asarray(bl_pa, float))[None, :]
    dens = np.exp(kappa * (np.cos(th - ph) - 1.0)).sum(axis=1)
    grid = np.radians(2.0 * np.linspace(0, 180, 360, endpoint=False))
    ref = np.exp(kappa * (np.cos(grid[:, None] - ph) - 1.0)).sum(1).mean()
    return dens / ref


# ------------------------------------------------------------------ data
class Dataset:
    """A star's real V^2 coverage with his median jitter."""

    def __init__(self, star):
        self.star = star
        self.raw = load_pavo(star)
        self.ref = reference(star)
        self.jitter = self.ref["jitter"]["median"]
        self.sigma = np.hypot(self.raw["v2_err"], self.jitter)
        self.u, self.v = self.raw["u"], self.raw["v"]
        self.wavel, self.v2 = self.raw["wavel"], self.raw["v2"]
        self.n = len(self.v2)
        self.bl_pa = baseline_pa(self.u, self.v)

    def oidata(self, v2):
        """V^2-only OIData (no phases) with the fixed total error."""
        n = self.n
        return OIData(
            dict(
                u=self.u,
                v=self.v,
                wavel=self.wavel,
                vis=np.asarray(v2),
                d_vis=self.sigma,
                v2_flag=True,
                phi=np.zeros(n),
                d_phi=np.ones(n),
                phi_flag=np.ones(n, bool),  # drop the (absent) phases
                cp_flag=False,
            )
        )

    def v2_model(self, **p):
        star = GravityDarkenedStar(**p, n_lat=N_LAT)
        return np.asarray(jnp.abs(star.model(self.u, self.v, self.wavel)) ** 2)

    def v2_disk(self, diam):
        """Exact uniform-disk V^2: an orientation-free null with no mesh."""
        disk = UniformDisk(diam)
        return np.asarray(jnp.abs(disk.model(self.u, self.v, self.wavel)) ** 2)

    def chi2(self, v2, **p):
        """chi2 of ``v2`` against the model at ``p`` (all four parameters)."""
        return float((((v2 - self.v2_model(**p)) / self.sigma) ** 2).sum())


def reference(star):
    """His posterior (our conventions): {param: {median, sd, ...}}."""
    ref = json.loads((PAVO / "reference_posteriors.json").read_text())
    r = ref["stars"][STARS[star][0]]["virgil"]
    out = {p: r[k] for p, k in REF_KEYS.items()}
    out["jitter"] = r["jitter_v2"]
    return out


def ref_median(star):
    return {p: reference(star)[p]["median"] for p in PARAMS}


# ------------------------------------------------------------------ fits
def fit_ml(data, v2, init, free=PARAMS, max_steps=MAX_STEPS):
    """Maximum-likelihood fit with ``virgil.fitting.fit`` (Levenberg-
    Marquardt, bounded through the priors' sigmoid bijections).

    ``init`` holds all four parameters; those not in ``free`` are held
    fixed at their ``init`` values. Returns a dict with the four fitted
    parameters (``pa`` wrapped to [0, 180)), ``chi2``, ``converged``,
    ``steps`` and ``seconds``.
    """
    t0 = time.time()
    star = GravityDarkenedStar(
        **{k: float(init[k]) for k in PARAMS}, n_lat=N_LAT
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-convergence is recorded below
        # cg_steps: the LM inner solve is exact after n_free CG steps, so a
        # handful suffices (the default 50 is ~4x slower for no gain here)
        res = fit(
            star,
            {k: PRIORS[k] for k in free},
            data.oidata(v2),
            init={k: float(init[k]) for k in free},
            max_steps=max_steps,
            cg_steps=len(free) + 2,
        )
    out = {
        k: float(res.values[k]) if k in free else float(init[k])
        for k in PARAMS
    }
    out["pa"] = float(wrap180(out["pa"]))
    out.update(
        chi2=float(res.info["chi2"][0]),
        converged=bool(res.info["converged"]),
        steps=int(res.info["steps"]),
        seconds=time.time() - t0,
    )
    return out


def start_design(n, seed):
    """Latin-hypercube starts over the prior box: diam in (0.3, 2.0) mas,
    omega (0, 0.99), cos i (0, 1), pa (0, 180). Returns an (n, 4) array of
    (diam_eq, omega, inc, pa)."""
    x = qmc.LatinHypercube(d=4, seed=seed).random(n)
    return np.column_stack(
        [
            0.3 + 1.7 * x[:, 0],
            0.99 * x[:, 1],
            np.degrees(np.arccos(x[:, 2])),
            180.0 * x[:, 3],
        ]
    )


def as_init(row):
    return dict(zip(PARAMS, map(float, row)))


# ----------------------------------------------------------------- tasks
def add_task_args(p):
    p.add_argument("--star", required=True, choices=list(STARS))
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--n-tasks", type=int, default=1)
    p.add_argument("--out", default=str(OUT))


def my_items(n, args):
    """Indices 0..n-1 handled by this array task (interleaved, so that
    slow and fast items are spread evenly)."""
    return np.arange(n)[args.task_index :: args.n_tasks]


def save(out, stem, records, meta):
    """Write ``<stem>.npz`` (arrays) and ``<stem>.json`` (metadata)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{stem}.npz", **records)
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / stem}.npz")


def collect(out, pattern):
    """Concatenate the per-task npz files matching ``pattern`` (sorted)."""
    files = sorted(Path(out).glob(pattern))
    if not files:
        return None
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
