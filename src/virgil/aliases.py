"""Orbits whose epochs undersample the period: alias bands and their evidences.

When epochs are far apart compared with the period, the number of orbital
cycles between them is ambiguous, and the likelihood has one narrow ridge
for each candidate count. [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases]
labels the ridges by ``N = round(T/P)``, with ``T`` the span from the first
to the last epoch, fits each *band* of periods directly to the closure phases
(or visibilities) of every epoch, and weighs the bands by their marginal
likelihoods. Positions are used only to start the fits and as a
diagnostic: a position is a lossy summary of an epoch.

The likelihood is hierarchical across epochs: each epoch's observable
blocks have an error scale with a Jeffreys (log-uniform) prior, integrated
out ([`marginal_loglike`][virgil.epochs.marginal_loglike]). The evidence of
each band is the sum over its distinct modes of a Laplace approximation,
checked by importance sampling from the Laplace proposal.

The aliasing of an irregular time sampling is described by its spectral
window, as in the Lomb–Scargle periodogram (VanderPlas 2018,
https://arxiv.org/abs/1703.09824). The linear solve at fixed (P, e, T) that
starts each band follows Lucy (2014, A&A 563, A126,
https://arxiv.org/abs/1309.2868) and The Joker's period sampling
(Price-Whelan et al. 2017, https://arxiv.org/abs/1610.07602). The error
scales are integrated out under a Jeffreys prior (Jeffreys 1946,
https://doi.org/10.1098/rspa.1946.0056) as in the evidence framework of
MacKay (1992, https://doi.org/10.1162/neco.1992.4.3.415).

See ``fit_orbit_aliases`` for the entry point.
"""

import dataclasses
import functools
import json
import logging
import math
import pathlib
import pickle
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from scipy import optimize
from scipy.special import logsumexp

from . import _gauss_newton
from ._deprecate import external_stacklevel
from ._precision import cast_tree, in_float64, run_in
from ._utils import clean_json
from .detection import _fingerprint
from .epochs import EpochPositions, _Surface
from .likelihood import whitened_residuals
from .models import OrbitalBinary, PointSource, System, UniformDisk
from .orbits import KeplerOrbit, period_grid, starting_orbits

_LOG = logging.getLogger(__name__)

__all__ = [
    "AliasBand",
    "AliasBandResult",
    "AliasResult",
    "UNRECORDED",
    "alias_bands",
    "fit_orbit_aliases",
    "position_profile",
    "spectral_window",
]

# Sampled coordinates (flat under the Jeffreys prior): log P, the time of
# periastron as a fraction of the period, e, cos i, omega, Omega (degrees),
# log a, log flux, and, with ``primary_diam="fit"``, log of the primary's
# uniform-disk diameter (mas).
_NAMES = (
    "log_period",
    "tau",
    "ecc",
    "cos_inc",
    "omega",
    "Omega",
    "log_a_mas",
    "log_flux",
    "log_diam",
)
_PERIODIC = {
    1: 1.0,
    4: 360.0,
    5: 180.0,
}  # periodic coordinates and their periods
_N_FLUX = 4  # fluxes tried for each position-level candidate
# Marks a setting a loaded result does not record (a file written before
# ``primary_diam`` and ``vis_amp`` were saved). It is not None, which means
# "point primary" or "no V² amplitude": the accessors refuse to guess.
UNRECORDED = "unrecorded"
_ELEMENTS = ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")


# ---------------------------------------------------------------------------
# Bands and diagnostics
# ---------------------------------------------------------------------------


def spectral_window(times, frequencies):
    """The spectral window ``|mean exp(2πi f t)|²`` of the epochs' times.

    It is 1 at ``f = 0`` and returns to near 1 at the aliases of the
    sampling: a frequency offset at which the window is high cannot be told
    from zero offset by the timing alone. See VanderPlas (2018,
    https://arxiv.org/abs/1703.09824) for the spectral window of a
    periodogram.
    """
    t = onp.asarray(times, float)
    f = onp.atleast_1d(onp.asarray(frequencies, float))
    phase = onp.exp(2j * onp.pi * f[:, None] * (t - t.mean())[None, :])
    return onp.abs(phase.mean(axis=1)) ** 2


@dataclasses.dataclass(frozen=True)
class AliasBand:
    """One period band, ``N = round(T/P)``: ``p_lo <= P <= p_hi`` (days)."""

    n: int
    p_lo: float
    p_hi: float


def alias_bands(times, p_range):
    """The alias bands of the epochs within ``p_range``.

    Band ``N`` holds the periods ``T/(N + 1/2) <= P <= T/(N - 1/2)`` that
    fit ``N`` cycles, rounded, between the first and last epoch (span T),
    clipped to ``p_range``. The bands are one cycle wide in frequency, so
    they tile the period prior without gaps. Periods above ``2T`` form band
    ``N = 0``: not an alias, but the part of the prior with less than one
    cycle across the span.

    Parameters
    ----------
    times : array-like
        The epochs' times (days).
    p_range : (float, float)
        The shortest and longest period (days) to consider.

    Returns
    -------
    list of AliasBand
        Ordered by ``N`` (longest period last).
    """
    times = onp.asarray(times, float)
    span = float(onp.ptp(times))
    p_min, p_max = map(float, p_range)
    if span <= 0 or not 0 < p_min < p_max:
        raise ValueError(
            "Need epochs spanning a positive time and 0 < p_min < p_max."
        )
    first = math.floor(span / p_max + 0.5)
    last = math.floor(span / p_min + 0.5)
    bands = []
    for n in range(first, last + 1):
        lo = span / (n + 0.5)
        hi = span / (n - 0.5) if n > 0 else p_max  # N = 0: P > 2T
        lo, hi = max(lo, p_min), min(hi, p_max)
        if hi > lo:
            bands.append(AliasBand(n, lo, hi))
    return bands


def position_profile(
    positions, bands, k=9.0, n_best=3, eccs=None, sign_free=False
):
    """A cheap diagnostic: the best position-level χ² in each band.

    For each band, the best Thiele–Innes fits
    ([`starting_orbits`][virgil.orbits.starting_orbits]) of ``positions``
    on a coherent period grid. Positions are lossy: use this to see
    which bands are viable and to seed the fits, not to weigh them.

    Returns
    -------
    list of dict
        Per band: ``n``, ``chi2`` (best), ``period`` of the best orbit and
        the ``orbits`` ``[(KeplerOrbit, χ²), ...]``.
    """
    rows = []
    for band in bands:
        periods = period_grid(
            positions.dt + positions.t_ref, band.p_lo, band.p_hi, k=k
        )
        found = starting_orbits(
            positions,
            periods,
            eccs=eccs,
            n_best=n_best,
            sign_free=sign_free,
        )
        rows.append(
            dict(
                n=band.n,
                chi2=found[0][1],
                period=float(found[0][0].period),
                orbits=found,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# The marginal likelihood and its parameters
# ---------------------------------------------------------------------------


def _elements(x, t_ref):
    """KeplerOrbit and flux from the sampled vector ``x`` (traceable)."""
    period = np.exp(x[0])
    inc = np.degrees(np.arccos(np.clip(x[3], -1.0 + 1e-9, 1.0 - 1e-9)))
    orbit = KeplerOrbit(
        period, x[1] * period, x[2], inc, x[4], x[5], np.exp(x[6]), t_ref=t_ref
    )
    return orbit, np.exp(x[7])


class _UDBinary(OrbitalBinary):
    """An [`OrbitalBinary`][virgil.models.OrbitalBinary] whose primary is a
    uniform disk of diameter ``diam`` (mas)."""

    diam: jax.Array

    def __init__(self, orbit, flux, diam):
        super().__init__(orbit, flux)
        self.diam = np.asarray(diam, dtype=float)

    def at(self, mjd, t_ref=0.0):
        snap = super().at(mjd, t_ref)
        return System(
            primary=UniformDisk(self.diam),
            companion=PointSource(snap.flux, snap.dra, snap.ddec),
        )


def _model(x, cfg):
    """The scene at the sampled vector ``x`` (traceable); ``cfg`` is
    ``_Settings.cfg``: ``(t_ref, fixed diameter or None, fit diameter,
    amplitude range or None)``."""
    t_ref, diam, fit_diam, _ = cfg
    orbit, flux = _elements(x, t_ref)
    if fit_diam:
        return _UDBinary(orbit, flux, np.exp(x[8]))
    if diam:
        return _UDBinary(orbit, flux, diam)
    return OrbitalBinary(orbit, flux)


def _epoch_residuals(surface, model, d, amp_range):
    """The whitened residuals of one epoch and its V² amplitude.

    With ``amp_range`` (lo, hi), the squared visibilities are multiplied by
    one amplitude ``A``, a per-epoch contrast loss that does not depend on
    baseline or wavelength. Its profile (the weighted least-squares ``A``,
    clipped to the range, which is the bound of its Jeffreys prior) is
    analytic because the model is linear in ``A``, so the visibility
    residuals are those of ``A`` times the model, with ``A`` live (variable
    projection). ``A`` is 1 without it.
    """
    r = whitened_residuals(model, d)
    if amp_range is None or surface.names[0] != "vis":
        return r, np.ones((), r.dtype)
    if d.vis_mode != "v2" or d.observable_kind != "split":
        raise NotImplementedError(
            "vis_amp needs squared visibilities as the visibility channel "
            f"(vis_mode='v2', unprojected), got vis_mode={d.vis_mode!r}."
        )
    n = surface.blocks[0][2]
    obs = np.reshape(d.vis, -1)
    err = np.reshape(d.d_vis, -1)
    mod = obs + r[:n] * err
    w = 1.0 / err**2
    tiny = np.finfo(r.dtype).tiny
    amp = np.sum(w * obs * mod) / np.maximum(np.sum(w * mod**2), tiny)
    amp = np.clip(amp, *amp_range)
    return _scale_vis(r, d, n, amp), amp


def _scale_vis(r, d, n, amp):
    """The whitened residuals ``r`` with the first ``n`` (the squared
    visibilities) recomputed for the model scaled by the V² amplitude
    ``amp``: ``(amp * model - obs) / err``, the model being recovered from
    the residual as ``obs + r * err``. Shared by the fit
    (``_epoch_residuals``, ``amp`` profiled) and the accessors of
    [`AliasBandResult`][virgil.aliases.AliasBandResult] (``amp`` as fitted),
    so that both score the same residuals."""
    obs = np.reshape(d.vis, -1)
    err = np.reshape(d.d_vis, -1)
    mod = obs + r[:n] * err
    return r.at[:n].set((amp * mod - obs) / err)


def _epoch_chi2(surface, model, d, amp_range):
    """Raw χ² per block of one epoch and its V² amplitude."""
    r, amp = _epoch_residuals(surface, model, d, amp_range)
    chi2 = np.stack([np.sum(r[a:b] ** 2) for _, a, b, _, _ in surface.blocks])
    return chi2, amp


# The kernels take the data as traced arguments, so one compilation serves
# every band, mode and draw set of the same shapes. The surfaces, which hold
# only the layout of the data (see ``epochs._Surface``), and ``t_ref``, one
# float per run, are static.


def _loglike(surfaces, x, data, cfg):
    """The scale-marginalized log likelihood of all epochs at ``x``."""
    model = _model(x, cfg)
    return sum(
        s.score(_epoch_chi2(s, model, d, cfg[3])[0], 1.0, d)
        for s, d in zip(surfaces, data)
    )


class _KernelProblem(eqx.Module):
    """The negative scale-marginalized log likelihood as a kernel problem
    ([`_gauss_newton`][virgil._gauss_newton]), in the sampled coordinates.

    The curvature rows are each epoch's block residuals (the V² amplitude
    live, as in the likelihood) multiplied by √(stop_gradient ``w_b``), with
    ``w_b`` the IRLS weights of the surface
    ([`_Surface.weights`][virgil.epochs._Surface.weights]): iteratively
    reweighted least squares. The priors are flat in these coordinates, so
    there is no diagonal. ``data`` is traced, so one compilation serves
    every band and start of the same shapes.
    """

    data: tuple
    surfaces: tuple = eqx.field(static=True)
    cfg: tuple = eqx.field(static=True)

    def loss(self, x):
        return -_loglike(self.surfaces, x, self.data, self.cfg)

    def rows(self, x):
        model = _model(x, self.cfg)
        return np.concatenate(
            [
                s.weighted_rows(
                    _epoch_residuals(s, model, d, self.cfg[3])[0], 1.0, d
                )
                for s, d in zip(self.surfaces, self.data)
            ]
        )

    def diag(self, x):
        return np.zeros_like(x)


_value_and_grad = functools.partial(jax.jit, static_argnums=(0, 3))(
    jax.value_and_grad(_loglike, argnums=1)
)
_hessian = functools.partial(jax.jit, static_argnums=(0, 3))(
    jax.hessian(_loglike, argnums=1)
)


@functools.partial(jax.jit, static_argnums=(0, 3))
def _batch(surfaces, xs, data, cfg):
    """``_loglike`` at each row of ``xs``, 64 rows at a time."""
    return jax.lax.map(
        lambda x: _loglike(surfaces, x, data, cfg), xs, batch_size=64
    )


class _Problem:
    """The scale-marginalized log likelihood of all epochs and the priors."""

    def __init__(self, data, settings):
        self.data, self.t_ref = list(data), settings.t_ref
        self.cfg = settings.cfg
        self.surfaces = tuple(_Surface(d, settings.s_max) for d in self.data)
        self.kernel = _KernelProblem(tuple(self.data), self.surfaces, self.cfg)
        # Proper priors: log P log-uniform over the whole p_range, tau
        # uniform over one cycle, e uniform, cos i uniform, omega and Omega
        # uniform with the (Omega + 180, omega + 180) mirror folded away
        # (Omega in [0, 180)), log a and log flux uniform.
        s = settings
        self.log_prior = float(
            -onp.log(onp.log(s.p_range[1] / s.p_range[0]))
            - onp.log(s.ecc_max)
            - onp.log(2.0)
            - onp.log(360.0)
            - onp.log(180.0)
            - onp.log(onp.log(s.a_range[1] / s.a_range[0]))
            - onp.log(onp.log(s.flux_range[1] / s.flux_range[0]))
            - (
                onp.log(onp.log(s.diam_range[1] / s.diam_range[0]))
                if s.primary_diam == "fit"
                else 0.0
            )
        )

    def value_and_grad(self, x):
        return _value_and_grad(self.surfaces, x, self.data, self.cfg)

    def hessian(self, x):
        return _hessian(self.surfaces, x, self.data, self.cfg)

    def batch(self, xs):
        return _batch(self.surfaces, xs, self.data, self.cfg)

    def per_epoch(self, x):
        """Raw χ²/ν per epoch and block, and the profile scales ŝ."""
        model = _model(np.asarray(x), self.cfg)
        rows = []
        for s, d in zip(self.surfaces, self.data):
            chi2, amp = _epoch_chi2(s, model, d, self.cfg[3])
            chi2 = onp.asarray(chi2)
            rows.append(
                dict(
                    blocks=list(s.names),
                    chi2_red=(chi2 / s.nu).tolist(),
                    scale=s.scales(chi2).tolist(),
                    vis_amp=float(amp),
                )
            )
        return rows


@dataclasses.dataclass(frozen=True)
class _Settings:
    """Everything but the data that shapes a fit (and so a checkpoint)."""

    p_range: tuple
    a_range: tuple
    flux_range: tuple
    ecc_max: float
    s_max: float | None
    t_ref: float
    n_candidates: int
    n_refine: int
    n_random: int
    pool_size: int  # rows of the candidate batch: one compilation per run
    k: float
    eccs: tuple | None
    min_distance_mas: float
    max_steps: int
    n_is: int
    flag_nats: float
    ess_min: float
    seed: int
    sign_free: bool = False
    primary_diam: float | str | None = None  # None, mas, or "fit"
    diam_range: tuple = (0.05, 5.0)  # mas: log-uniform prior of a fitted one
    vis_amp: tuple | None = None  # per-epoch V² amplitude range, or None
    flip_flux: bool = True
    init: tuple | None = (
        None  # period, dt_peri, ecc, inc, omega, Omega, a, flux
    )

    @property
    def fit_diam(self):
        return self.primary_diam == "fit"

    @property
    def cfg(self):
        """What the compiled kernels need besides the data (static)."""
        fixed = None if self.fit_diam else self.primary_diam
        return (self.t_ref, fixed, self.fit_diam, self.vis_amp)

    @property
    def flux_scan(self):
        """The fluxes tried for each position-level candidate: log-spaced
        inside the range, and, for near-equal fluxes (``sign_free``), also
        0.7 and 0.95 of its upper end."""
        scan = onp.geomspace(*self.flux_range, _N_FLUX + 2)[1:-1]
        if self.sign_free:
            top = onp.array([0.7, 0.95]) * self.flux_range[1]
            scan = onp.append(scan, top[top >= self.flux_range[0]])
        return scan


@dataclasses.dataclass(frozen=True, eq=False)
class _Bounds:
    """The hard bounds of the prior in the sampled coordinates, for one band
    (the angles are periodic, with none)."""

    lower: onp.ndarray
    upper: onp.ndarray

    @classmethod
    def of(cls, band, settings):
        inf = onp.inf
        a, f = settings.a_range, settings.flux_range
        d = settings.diam_range
        extra = [onp.log(d)] if settings.fit_diam else []
        return cls(
            onp.array(
                [onp.log(band.p_lo), -inf, 0.0, -1.0, -inf, -inf]
                + [onp.log(a[0]), onp.log(f[0])]
                + [e[0] for e in extra]
            ),
            onp.array(
                [onp.log(band.p_hi), inf, settings.ecc_max, 1.0, inf, inf]
                + [onp.log(a[1]), onp.log(f[1])]
                + [e[1] for e in extra]
            ),
        )

    def inside(self, xs):
        """Whether each row of ``xs`` is inside the bounds."""
        return onp.all((xs >= self.lower) & (xs <= self.upper), axis=-1)

    def clip(self, xs):
        """``xs`` moved onto the bounds, where the likelihood is defined."""
        return onp.clip(xs, self.lower, self.upper)


def _fold(x):
    """Fold the (Omega + 180, omega + 180) mirror: Omega in [0, 180), omega
    in [0, 360)."""
    x = onp.array(x, float)
    flip = onp.floor(x[5] / 180.0)
    x[5] -= 180.0 * flip
    x[4] += 180.0 * flip
    x[4] = onp.mod(x[4], 360.0)
    return x


def _start_vector(orbit, flux, t_ref, band, bounds, diam=None):
    period = float(onp.clip(float(orbit.period), band.p_lo, band.p_hi))
    dt = float(orbit.dt_peri) + float(orbit.t_ref) - t_ref
    tau = (dt / period + 0.5) % 1.0 - 0.5
    x = onp.array(
        [
            onp.log(period),
            tau,
            float(orbit.ecc),
            onp.cos(onp.radians(float(orbit.inc))),
            float(orbit.omega),
            float(orbit.Omega),
            onp.log(float(orbit.a_mas)),
            onp.log(flux),
        ]
        + ([onp.log(diam)] if len(bounds.lower) > 8 else [])
    )
    x = _fold(x)
    eps = 1e-3
    lo, hi = bounds.lower, bounds.upper
    return onp.clip(
        x,
        onp.where(onp.isfinite(lo), lo + eps, lo),
        onp.where(onp.isfinite(hi), hi - eps, hi),
    )


def _senses(x):
    """The seed and its other sense of motion (i -> 180 - i)."""
    other = onp.array(x)
    other[3] = -other[3]
    return [x, other]


# The likelihood is a narrow, curved ridge for an orbit of many cycles, where
# L-BFGS-B stops 50-250 nats short of the optimum. Damped Newton on the IRLS
# curvature (``_KernelProblem``) follows the ridge; L-BFGS-B is only the
# fallback for a start where the kernel stalls.
_DTOL = 1e-9  # Newton decrement, nats, at which a mode is converged


def _optimize(problem, bounds, x0, max_steps):
    """Maximize the marginal likelihood from ``x0``, in the sampled
    coordinates and within ``bounds``.

    Damped Newton on the IRLS curvature
    ([`_gauss_newton`][virgil._gauss_newton], at most ``max_steps``
    iterations). If it does not converge, L-BFGS-B from its point, then
    the kernel again from there. Returns the folded point and its log
    likelihood (``-inf`` if the likelihood or its gradient is not finite
    at ``x0``).
    """

    def kernel(start):
        return _gauss_newton.solve(
            problem.kernel,
            start,
            lower=bounds.lower,
            upper=bounds.upper,
            dtol=_DTOL,
            max_steps=max_steps,
        )

    state = kernel(onp.asarray(x0, float))
    if int(state.stop) == _gauss_newton.NON_FINITE:
        return _fold(x0), -onp.inf
    if int(state.stop) != _gauss_newton.CONVERGED:

        def fun(x):
            value, grad = problem.value_and_grad(np.asarray(x))
            return -float(value), -onp.asarray(grad, float)

        res = optimize.minimize(
            fun,
            onp.asarray(state.z, float),
            jac=True,
            method="L-BFGS-B",
            bounds=list(
                zip(
                    *(
                        [None if not onp.isfinite(v) else float(v) for v in b]
                        for b in (bounds.lower, bounds.upper)
                    )
                )
            ),
            options=dict(maxiter=max_steps, ftol=1e-13, gtol=1e-9),
        )
        if res.fun < float(state.value):  # also false for a NaN
            state = kernel(onp.asarray(res.x, float))
    return _fold(state.z), -float(state.value)


# ---------------------------------------------------------------------------
# Evidence: Laplace, checked by importance sampling
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Proposal:
    """A mode's multivariate Student-t proposal (``df`` degrees of freedom,
    the Laplace covariance ``chol chol^T``, ``l_inv = chol^-1``)."""

    x: onp.ndarray
    chol: onp.ndarray
    l_inv: onp.ndarray
    logdet: float
    df: float

    def draw(self, n, rng):
        """``n`` draws and their squared Mahalanobis distances."""
        z = rng.standard_normal((n, len(self.x)))
        g = rng.chisquare(self.df, n) / self.df
        return self.x + (z @ self.chol.T) / onp.sqrt(g)[:, None], (
            onp.sum(z**2, axis=1) / g
        )

    def log_pdf(self, xs, mahal=None):
        """Log density at ``xs`` (``mahal`` if already known)."""
        if mahal is None:
            mahal = onp.sum(((xs - self.x) @ self.l_inv.T) ** 2, axis=1)
        d, df = len(self.x), self.df
        return (
            -0.5 * (df + d) * onp.log1p(mahal / df)
            - 0.5 * self.logdet
            + math.lgamma(0.5 * (df + d))
            - math.lgamma(0.5 * df)
            - 0.5 * d * onp.log(df * onp.pi)
        )


@dataclasses.dataclass(frozen=True, eq=False)
class _Draws:
    """A mode's importance draws: points, log target (likelihood times
    prior, -inf outside the support), log weights and proposal."""

    xs: onp.ndarray
    log_t: onp.ndarray
    log_w: onp.ndarray
    proposal: _Proposal


def _laplace_proposal(h, x, df):
    """The Student-t proposal from the negative Hessian ``h`` at ``x``, or
    the flag that says why there is none."""
    finite = bool(onp.all(onp.isfinite(h)))
    eig = onp.linalg.eigvalsh(h) if finite else onp.array([-onp.inf])
    if not (finite and eig.min() > 0):
        return None, "hessian-not-positive-definite"
    if eig.min() <= 1e-13 * eig.max():
        # Positive, but too ill-conditioned to invert: a flat direction.
        return None, "hessian-singular"
    try:
        cov = onp.linalg.inv(h)
        chol = onp.linalg.cholesky(cov)
        l_inv = onp.linalg.inv(chol)
    except onp.linalg.LinAlgError:
        return None, "hessian-singular"
    logdet = float(onp.linalg.slogdet(cov)[1])
    return _Proposal(x, chol, l_inv, logdet, df), None


def _mode_evidence(
    problem, bounds, x, value, n_is, rng, df=5.0, bound_sigma=3.0
):
    """Laplace and importance-sampling log evidences of one mode at ``x``.

    The importance proposal is a multivariate Student-t (``df``) with the
    Laplace covariance. The hard prior bounds truncate the target; angles
    are periodic and are kept within half a turn of the mode. A mode whose
    Hessian is not positive definite, or is singular, has no proposal: both
    evidences are ``-inf`` and it is dropped.

    All ``n_is`` draws are evaluated (one compiled shape), those outside the
    support at the nearest point of the bounds, and then set to ``-inf``.
    """
    h = -onp.asarray(problem.hessian(np.asarray(x)), float)
    proposal, flag = _laplace_proposal(0.5 * (h + h.T), x, df)
    out = dict(
        x=x.tolist(),
        loglike=value,
        log_z_laplace=-onp.inf,
        log_z_is=-onp.inf,
        ess=0.0,
        flags=[] if flag is None else [flag],
    )
    if proposal is None:
        return out, None
    d = len(x)
    sd = onp.sqrt(onp.sum(proposal.chol**2, axis=1))
    out["log_z_laplace"] = float(
        value
        + problem.log_prior
        + 0.5 * d * onp.log(2 * onp.pi)
        + 0.5 * proposal.logdet
    )
    distance = onp.minimum(x - bounds.lower, bounds.upper - x) / sd
    if onp.any(distance < bound_sigma):
        out["flags"].append("near-prior-bound")
    out["sigma"] = sd.tolist()
    # Newton step in units of sigma, over the coordinates not on a bound:
    # large means L-BFGS stopped short of a stationary point.
    grad = onp.asarray(problem.value_and_grad(np.asarray(x))[1], float)
    free = (x - bounds.lower > 1e-6) & (bounds.upper - x > 1e-6)
    if onp.any(onp.abs(grad[free]) * sd[free] > 0.1):
        out["flags"].append("not-converged")
    xs, mahal = proposal.draw(n_is, rng)
    log_q = proposal.log_pdf(xs, mahal)
    # Angles live within half a turn of the mode.
    ok = bounds.inside(xs)
    for k, width in _PERIODIC.items():
        ok &= onp.abs(xs[:, k] - x[k]) < width / 2.0
    values = onp.asarray(problem.batch(np.asarray(bounds.clip(xs))), float)
    log_t = onp.where(ok, values, -onp.inf) + problem.log_prior
    log_w = log_t - log_q
    log_w[~onp.isfinite(log_w)] = -onp.inf
    out["log_z_is"] = float(logsumexp(log_w)) - onp.log(n_is)
    w = (
        onp.exp(log_w - onp.max(log_w[onp.isfinite(log_w)]))
        if onp.any(onp.isfinite(log_w))
        else onp.zeros(n_is)
    )
    out["ess"] = float(w.sum() ** 2 / max((w**2).sum(), 1e-300))
    return out, _Draws(xs, log_t, log_w, proposal)


def _mixture_weights(draws):
    """Log weights of the pooled draws of a band's modes under their
    deterministic mixture proposal, ``p / mean_k q_k``: the modes' tails
    overlap, and per-mode weights would count the overlap twice."""
    xs = onp.concatenate([d.xs for d in draws])
    log_t = onp.concatenate([d.log_t for d in draws])
    log_q = onp.stack([d.proposal.log_pdf(xs) for d in draws])
    log_mix = logsumexp(log_q, axis=0) - onp.log(len(draws))
    log_w = log_t - log_mix
    log_w[~onp.isfinite(log_w)] = -onp.inf
    return xs, log_w


# ---------------------------------------------------------------------------
# Fitting one band
# ---------------------------------------------------------------------------


def _candidates(settings, band, bounds, rng, orbits):
    """The band's seed pool: position-level ``orbits`` (``[(KeplerOrbit,
    χ²), ...]`` or None) at several fluxes and both senses of motion, then
    ``n_random`` random orbits."""
    s = settings
    diam = _DIAM_SEED if s.fit_diam else None
    pool = [
        x
        for orbit, _ in orbits or ()
        for flux in s.flux_scan
        for x in _senses(
            _start_vector(orbit, flux, s.t_ref, band, bounds, diam)
        )
    ]
    pool.extend(_init_starts(s, band, bounds))
    for _ in range(s.n_random):
        pool.append(
            _fold(
                [
                    rng.uniform(onp.log(band.p_lo), onp.log(band.p_hi)),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(0.0, 0.6 * s.ecc_max),
                    rng.uniform(-0.95, 0.95),
                    rng.uniform(0, 360),
                    rng.uniform(0, 180),
                    rng.uniform(*onp.log(s.a_range)),
                    rng.uniform(*onp.log(s.flux_range)),
                ]
                + ([rng.uniform(*onp.log(s.diam_range))] if s.fit_diam else [])
            )
        )
    if not pool:
        raise ValueError(
            "No candidate orbits: pass positions or a positive n_random."
        )
    return onp.asarray(pool)


_DIAM_SEED = 0.5  # mas: where a fitted diameter starts


def _init_orbit(settings):
    """The warm start's ``(KeplerOrbit, flux)``, or None."""
    if settings.init is None:
        return None
    period, dt_peri, ecc, inc, omega, big_omega, a_mas, flux = settings.init
    orbit = KeplerOrbit(
        period,
        dt_peri,
        ecc,
        inc,
        omega,
        big_omega,
        a_mas,
        t_ref=settings.t_ref,
    )
    return orbit, flux


def _init_starts(settings, band, bounds):
    """The warm start (``init=``) as a start vector in ``band``."""
    start = _init_orbit(settings)
    if start is None:
        return []
    diam = _DIAM_SEED if settings.fit_diam else None
    return [_start_vector(*start, settings.t_ref, band, bounds, diam)]


def _flipped(x, settings, bounds):
    """``x`` with the stars' labels swapped: the same sky orbit seen from
    the other star (ω + 180°, the same elements otherwise) and the flux
    ratio inverted, kept inside the range (a ratio above the range's top
    starts at 0.9 of it)."""
    y = onp.array(x, float)
    y[4] = y[4] + 180.0
    top = settings.flux_range[1]
    y[7] = onp.log(min(1.0 / onp.exp(x[7]), 0.9 * top))
    y = _fold(y)
    return onp.clip(
        y,
        onp.where(onp.isfinite(bounds.lower), bounds.lower + 1e-3, -onp.inf),
        onp.where(onp.isfinite(bounds.upper), bounds.upper - 1e-3, onp.inf),
    )


def _score_pool(problem, settings, pool):
    """The marginal log likelihood of each candidate (-inf if not finite),
    evaluated in a batch of ``pool_size`` rows, padded with the first."""
    pad = max(settings.pool_size - len(pool), 0)
    padded = onp.concatenate([pool, onp.repeat(pool[:1], pad, axis=0)])
    score = onp.array(problem.batch(np.asarray(padded)), float)[: len(pool)]
    score[~onp.isfinite(score)] = -onp.inf
    return score


def _distinct(modes, times, t_ref, min_distance_mas):
    """Keep the best of the modes whose tracks at ``times`` agree within
    ``min_distance_mas``."""
    modes = sorted(modes, key=lambda m: -m[1])
    kept, tracks = [], []
    for x, value in modes:
        orbit, _ = _elements(np.asarray(x), t_ref)
        dra, ddec, _ = orbit.relative(np.asarray(times))
        track = onp.stack([onp.asarray(dra), onp.asarray(ddec)], -1)
        if all(
            onp.max(onp.hypot(*(track - t).T)) > min_distance_mas
            for t in tracks
        ):
            kept.append((x, value))
            tracks.append(track)
    return kept


def _refine(problem, settings, bounds, pool, times, forced=()):
    """Rank the candidates by the observables and refine the best distinct
    few into modes.

    Returns the distinct modes ``[(x, loglike), ...]`` and the number of
    starts that failed. Only a ``TypeError`` (a bug or a misuse) propagates;
    any other exception, a ``ValueError`` or ``LinAlgError`` from a start
    say, counts it as failed. If every start fails, a ``RuntimeError``
    chained to the last error is raised.
    """
    s = settings
    score = _score_pool(problem, s, pool)
    keep = onp.argsort(-score, kind="stable")[: 4 * s.n_refine]
    ranked = _distinct(
        [(pool[i], score[i]) for i in keep],
        times,
        s.t_ref,
        s.min_distance_mas,
    )
    seeds = [x for x, _ in ranked[: s.n_refine]] + [
        onp.asarray(x, float) for x in forced
    ]
    fitted, n_failed, error = [], 0, None
    for x0 in seeds:
        try:
            x, value = _optimize(problem, bounds, x0, s.max_steps)
        except TypeError:
            raise
        except Exception as caught:  # noqa: BLE001 - one bad start
            n_failed, error = n_failed + 1, caught
            continue
        if onp.isfinite(value):
            fitted.append((x, value))
    if seeds and n_failed == len(seeds):
        raise RuntimeError(
            f"Every one of the {n_failed} optimizer starts failed."
        ) from error
    return _distinct(fitted, times, s.t_ref, s.min_distance_mas), n_failed


def _band_evidence(problem, settings, bounds, modes_xy, rng):
    """Every mode's evidence, and the band's: the Laplace sum ``z_lap``
    and the mixture importance-sampling estimate ``z_is``.

    Where a mode sits on a prior bound (a period edge between bands, say)
    the Gaussian is cut off, so that mode's ``log_z`` is its
    importance-sampling value, which truncates the target; elsewhere it is
    the Laplace value.

    Returns ``(modes, z_lap, z_is, pool)``, with ``pool`` the pooled draws
    and their mixture log weights (``(None, None)`` without any).
    """
    modes, draws = [], []
    for x, value in modes_xy:
        record, draw = _mode_evidence(
            problem, bounds, x, value, settings.n_is, rng
        )
        modes.append(record)
        if draw is not None:
            draws.append(draw)
    for m in modes:
        edge = (
            "near-prior-bound" in m["flags"] and m["ess"] >= settings.ess_min
        )
        m["log_z"] = m["log_z_is"] if edge else m["log_z_laplace"]
    z_lap = (
        float(logsumexp([m["log_z"] for m in modes])) if modes else -onp.inf
    )
    if not draws:
        return modes, z_lap, -onp.inf, (None, None)
    xs, log_w = _mixture_weights(draws)
    z_is = float(logsumexp(log_w)) - onp.log(len(xs))
    return modes, z_lap, z_is, (xs, log_w)


def _band_result(problem, settings, band, modes, z_lap, z_is, n_failed):
    """The band's flags, best mode, χ²/ν and per-epoch scales."""
    live = [m for m in modes if onp.isfinite(m["log_z"])]
    best_mode = max(live, key=lambda m: m["log_z"]) if live else None
    best, epochs, chi2_red = {}, [], float("nan")
    if best_mode is not None:
        best = {
            k: float(v[0])
            for k, v in _physical(
                onp.array(best_mode["x"]), settings.t_ref
            ).items()
        }
        epochs = problem.per_epoch(best_mode["x"])
        nu = onp.concatenate([s.nu for s in problem.surfaces])
        chi2 = onp.concatenate([onp.array(e["chi2_red"]) for e in epochs])
        chi2_red = float(onp.sum(chi2 * nu) / onp.sum(nu))
    flags = sorted({f for m in modes for f in m["flags"]})
    if (
        onp.isfinite(z_lap)
        and onp.isfinite(z_is)
        and abs(z_lap - z_is) > settings.flag_nats
    ):
        flags.append("laplace-is-differ")
    if best_mode is not None and best_mode["ess"] < settings.ess_min:
        flags.append("low-ess")
    if best_mode is None:
        flags.append("no-valid-mode")
    if n_failed:
        flags.append("starts-failed")
    return AliasBandResult(
        n=band.n,
        p_lo=band.p_lo,
        p_hi=band.p_hi,
        modes=modes,
        log_z=z_lap,
        log_z_is=z_is,
        ess=best_mode["ess"] if best_mode else 0.0,
        flagged=bool(flags),
        flags=flags,
        best=best,
        chi2_red=chi2_red,
        epochs=epochs,
        n_starts_failed=n_failed,
    )


def _pinned(x, bounds, tol=0.02):
    """Whether the flux ratio sits at the top of its range (within ``tol`` in
    log flux): a fit that wants a ratio above the range."""
    return bool(x[7] >= bounds.upper[7] - tol)


def _flip_modes(problem, settings, bounds, modes_xy, times):
    """Refit each mode whose flux ratio sits at the top of ``flux_range``
    from the same orbit with the stars' labels swapped (the ratio is then
    below the top: the log-uniform prior on (lo, top] is not symmetric under
    the swap, and a fit pinned at the top is the signature of a ratio above
    it). Returns the modes with the refits (distinct ones), the refits on
    tracks no mode had, and the number that failed."""
    flips, refits, n_failed = [], [], 0
    for x, _ in modes_xy:
        if not _pinned(x, bounds):
            continue
        try:
            xf, vf = _optimize(
                problem,
                bounds,
                _flipped(x, settings, bounds),
                settings.max_steps,
            )
        except TypeError:
            raise
        except Exception:  # noqa: BLE001 - one bad start
            n_failed += 1
            continue
        if onp.isfinite(vf):
            refits.append((xf, vf))
            new = _track(xf, settings.t_ref, times)
            seen = (_track(m, settings.t_ref, times) for m, _ in modes_xy)
            if all(  # a refit that lands on a known mode relabels nothing
                onp.max(onp.hypot(*(new - t).T)) > settings.min_distance_mas
                for t in seen
            ):
                flips.append((xf, vf))
    merged = _distinct(
        list(modes_xy) + refits,
        times,
        settings.t_ref,
        settings.min_distance_mas,
    )
    return merged, flips, n_failed


def _flip_is_relabelling(problem, settings):
    """The swap is a relabelling of one sky orbit only for two point
    sources and no differential phases (a resolved primary stays on the
    first star; differential phases are chromatic), so it is skipped
    otherwise."""
    return not settings.primary_diam and not any(
        d.extras for d in problem.data
    )


def _is_flipped(result, flips, settings, times):
    """Whether the band's best mode is one of the relabelled refits."""
    if not flips or not result.modes:
        return False
    live = [m for m in result.modes if onp.isfinite(m["log_z"])]
    if not live:
        return False
    best = onp.array(max(live, key=lambda m: m["log_z"])["x"])
    tracks = [
        _track(x, settings.t_ref, times)
        for x in [best] + [f[0] for f in flips]
    ]
    return any(
        onp.max(onp.hypot(*(tracks[0] - t).T)) <= settings.min_distance_mas
        for t in tracks[1:]
    )


def _track(x, t_ref, times):
    orbit, _ = _elements(np.asarray(x), t_ref)
    dra, ddec, _ = orbit.relative(np.asarray(times))
    return onp.stack([onp.asarray(dra), onp.asarray(ddec)], -1)


def _fit_band(problem, settings, band, orbits, times):
    """Fit one band: ``(AliasBandResult, (draws, log weights))``.

    It has its own random stream, so a band does not depend on which bands
    were fitted or loaded before it.
    """
    rng = onp.random.default_rng([settings.seed, band.n])
    bounds = _Bounds.of(band, settings)
    pool = _candidates(settings, band, bounds, rng, orbits)
    forced = []
    if (
        settings.init is not None
        and band.p_lo <= settings.init[0] <= band.p_hi
    ):
        forced = _init_starts(settings, band, bounds)  # a fair band
    modes_xy, n_failed = _refine(
        problem, settings, bounds, pool, times, forced
    )
    flips = []
    if settings.flip_flux and _flip_is_relabelling(problem, settings):
        modes_xy, flips, n_flip_failed = _flip_modes(
            problem, settings, bounds, modes_xy, times
        )
        n_failed += n_flip_failed
    modes, z_lap, z_is, draws = _band_evidence(
        problem, settings, bounds, modes_xy, rng
    )
    result = _band_result(
        problem, settings, band, modes, z_lap, z_is, n_failed
    )
    result.flipped = _is_flipped(result, flips, settings, times)
    if result.flipped:
        _LOG.info(
            "band N=%d: the best mode has the stars' labels swapped "
            "(flux ratio was pinned at the top of flux_range)",
            band.n,
        )
    return result, draws


def _preflight(problem, settings, band):
    """Evaluate the likelihood and its gradient once, at the centre of the
    first band's bounds, outside any ``try``: misconfiguration (bad data,
    options or shapes) surfaces here, before any band is fitted, and what
    raises later is a failure of one start or band."""
    bounds = _Bounds.of(band, settings)
    finite = onp.isfinite(bounds.lower) & onp.isfinite(bounds.upper)
    centre = onp.zeros(len(finite))
    centre[finite] = (bounds.lower[finite] + bounds.upper[finite]) / 2
    problem.value_and_grad(np.asarray(centre))


def _failed_band(band, error):
    """The row of a band that raised (flag ``band-failed``)."""
    return AliasBandResult(
        n=band.n,
        p_lo=band.p_lo,
        p_hi=band.p_hi,
        modes=[],
        log_z=-onp.inf,
        log_z_is=-onp.inf,
        ess=0.0,
        flagged=True,
        flags=["band-failed"],
        error=f"{type(error).__name__}: {error}",
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

_CHECKPOINT_FORMAT = 2  # bump when a checkpoint's contents change


def _checkpoint_key(data, positions, bands, times, settings):
    """What a band's checkpoint depends on: the data (every field, static
    ones included), the positions, the bands and all settings.

    Raises
    ------
    ValueError
        If the data or positions cannot be hashed.
    """
    prints = dict(data=_fingerprint(data), positions=_fingerprint(positions))
    if None in prints.values():
        raise ValueError(
            "checkpoint_dir needs data and positions that can be hashed, to "
            "tell a checkpoint's data from new data; one holds something "
            "unhashable (a function or an arbitrary object)."
        )
    return dict(
        format=_CHECKPOINT_FORMAT,
        edges=[[bd.p_lo, bd.p_hi] for bd in bands],
        times=onp.asarray(times, float).tolist(),
        settings=dataclasses.asdict(settings),
        **prints,
    )


def _band_files(directory, band):
    d = pathlib.Path(directory)
    return d / f"band_{band.n}.pkl", d / f"band_{band.n}.json"


def _load_band(directory, band, key):
    """A band's checkpoint, or None if there is none or it is for other
    settings."""
    if directory is None:
        return None
    pkl, _ = _band_files(directory, band)
    try:
        with open(pkl, "rb") as f:
            saved_key, done = pickle.load(f)
    except Exception:  # noqa: BLE001 - missing or truncated: refit
        return None
    return done if saved_key == clean_json(key) else None


def _save_band(directory, band, key, done):
    """Write a band's checkpoint atomically (pickle to reload, JSON to read)."""
    if directory is None:
        return
    pkl, js = _band_files(directory, band)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    tmp = pkl.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump((clean_json(key), done), f)
    tmp.replace(pkl)
    with open(js, "w") as f:
        json.dump(
            clean_json(dataclasses.asdict(done[0])), f, indent=1, default=str
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AliasBandResult:
    """The fit of one alias band (see [`AliasResult`][virgil.aliases.AliasResult]).

    Besides the fields below, the band records what it needs to rebuild
    the scene it was fitted with: ``t_ref``, ``primary_diam`` and
    ``vis_amp``, which
    [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases] (through
    [`AliasResult`][virgil.aliases.AliasResult]) fills in. They back the
    accessors [`scene`][virgil.aliases.AliasBandResult.scene],
    [`residuals`][virgil.aliases.AliasBandResult.residuals] and
    [`chi2_per_night`][virgil.aliases.AliasBandResult.chi2_per_night].

    Attributes
    ----------
    t_ref : float or None
        Reference time (MJD) of ``best["dt_peri"]``.
    primary_diam : float, ``"fit"`` or None
        The fit's primary: a point (None), a uniform disk of this diameter
        (mas), or one whose diameter was fitted (``best["diam"]``).
    vis_amp : tuple or None
        The ``(lo, hi)`` range of the per-epoch V² amplitude if it was
        fitted (the amplitudes themselves are in ``epochs``), else None.

    Both are [`UNRECORDED`][virgil.aliases.UNRECORDED] on a band loaded
    from a file that did not save them; the accessors then raise unless
    they were given to
    [`AliasResult.load`][virgil.aliases.AliasResult.load].
    """

    n: int
    p_lo: float
    p_hi: float
    modes: list
    log_z: float  # sum over modes of the Laplace (or, at a prior bound, IS) evidences
    log_z_is: float  # importance sampling from the modes' mixture proposal
    ess: float  # of the best mode's importance sample
    flagged: bool  # any entry in ``flags``
    flags: list = dataclasses.field(
        default_factory=list
    )  # why a band is not to be trusted
    p: float = float("nan")  # posterior probability of the band
    best: dict = dataclasses.field(default_factory=dict)
    chi2_red: float = float("nan")  # raw χ²/ν on the quoted errors, all epochs
    epochs: list = dataclasses.field(
        default_factory=list
    )  # per-epoch raw χ²/ν and scales
    flipped: bool = False  # best mode has the stars' labels swapped
    error: str = ""  # the exception, if the band failed (flag ``band-failed``)
    n_starts_failed: int = (
        0  # optimizer starts that raised (``starts-failed``)
    )
    t_ref: float | None = None
    primary_diam: float | str | None = None
    vis_amp: tuple | None = None

    def scene(self, params=None):
        """The scene the band was fitted with, at its best orbit.

        A point or uniform-disk primary and one orbiting point companion,
        built from ``best`` (or from ``params``) exactly as the fit's
        likelihood builds it, so that it reproduces the fit's model.

        If the stars' labels were swapped in the fit (``flipped``), the
        scene is the swapped one: ``best["flux"]`` is already the ratio of
        the refit, with the labels as the fit used them, so nothing is
        undone here.

        Parameters
        ----------
        params : dict, optional
            Orbital elements (``period``, ``dt_peri``, ``ecc``, ``inc``,
            ``omega``, ``Omega``, ``a_mas`` and ``flux``, and ``diam`` for
            a fitted primary) to build the scene for instead of ``best``,
            for example a published orbit, to score it with the band's own
            model.

        Returns
        -------
        virgil.models.OrbitalBinary
            An [`OrbitalBinary`][virgil.models.OrbitalBinary], or, with a
            resolved primary, the same with a uniform-disk primary (private
            ``_UDBinary``). Evaluate it with ``.at(mjd, t_ref)`` or pass it
            to [`whitened_residuals`][virgil.likelihood.whitened_residuals].

        Raises
        ------
        ValueError
            If the band has no best orbit (it failed), no ``t_ref``, or
            ``primary_diam="fit"`` but no diameter is available, or the
            band's ``primary_diam`` is unrecorded (see
            [`UNRECORDED`][virgil.aliases.UNRECORDED]).
        """
        b = self.best if params is None else params
        if not b:
            raise ValueError(
                f"band N={self.n} has no best orbit (flags: {self.flags})."
            )
        if self.t_ref is None:
            raise ValueError("this band does not record its t_ref.")
        if _unrecorded(self.primary_diam):
            raise ValueError(_UNRECORDED_MESSAGE.format("primary_diam"))
        orbit = KeplerOrbit(
            b["period"],
            b["dt_peri"],
            b["ecc"],
            b["inc"],
            b["omega"],
            b["Omega"],
            b["a_mas"],
            t_ref=self.t_ref,
        )
        diam = b.get("diam")
        if diam is None:  # ``params`` without a diameter: use the fitted one
            diam = (
                self.best.get("diam")
                if self.primary_diam == "fit"
                else self.primary_diam
            )
        if self.primary_diam == "fit" and diam is None:
            raise ValueError(
                "primary_diam is 'fit' but no fitted 'diam' is available: "
                "cannot rebuild the fit's scene."
            )
        if diam:  # 0 or None: a point primary, as in ``_model``
            return _UDBinary(orbit, b["flux"], diam)
        return OrbitalBinary(orbit, b["flux"])

    def _whitened(self, data, params=None):
        """Per epoch, ``([(block name, ν, whitened residuals)], amp)`` at
        the best orbit (or ``params``).

        Like the fit, it runs on float64 copies of the data, in float64.
        The V² amplitude is re-profiled at the orbit scored (the weighted
        least-squares step of ``_epoch_residuals``) rather than read from
        ``epochs``, so that the residuals at ``params`` are those of the
        fit's likelihood there; at ``best`` it returns the fitted one.
        """
        if _unrecorded(self.vis_amp):
            raise ValueError(_UNRECORDED_MESSAGE.format("vis_amp"))
        if len(data) != len(self.epochs):
            raise ValueError(
                f"{len(data)} datasets for a fit of {len(self.epochs)} epochs."
            )
        out = []
        with run_in("float64"):
            model = self.scene(params)  # inside: built in float64
            for d in data:
                d = cast_tree(d, "float64")
                surface = _Surface(d)
                r, amp = _epoch_residuals(surface, model, d, self.vis_amp)
                r = onp.asarray(r, dtype=float)
                out.append(
                    (
                        [
                            (name, a, z, nu, r[a:z])
                            for name, a, z, nu, _ in surface.blocks
                        ],
                        float(amp),
                    )
                )
        return out

    def residuals(self, data, params=None):
        """Whitened residuals of ``data`` at the best orbit, block by block.

        These are the residuals the fit's likelihood is built from
        ([`whitened_residuals`][virgil.likelihood.whitened_residuals] of
        [`scene`][virgil.aliases.AliasBandResult.scene], on the quoted
        errors), evaluated in float64 as the fit is. If the fit had a V²
        amplitude, each epoch's squared visibilities are scaled by the
        amplitude re-profiled at the orbit scored, which is the fitted
        one at ``best``. They are virgil's own model, so they check the
        bookkeeping, not the model: for that, compare with an independent
        forward model.

        Parameters
        ----------
        data : sequence of OIData
            The epochs the band was fitted to, in the same order.
        params : dict, optional
            Elements to use instead of ``best``; see
            [`scene`][virgil.aliases.AliasBandResult.scene].

        Returns
        -------
        list of dict
            One ``{block name: residuals}`` per epoch. The blocks
            (``"vis"``, ``"phi"``, ...) are in the order of the whitened
            residual vector, so ``numpy.concatenate(list(blocks.values()))``
            is that vector. The penalty rows of correlated closure phases
            are the last rows of ``"phi"``, as in the fit's χ².

        Raises
        ------
        ValueError
            If ``primary_diam`` or ``vis_amp`` is unrecorded (see
            [`UNRECORDED`][virgil.aliases.UNRECORDED]), or ``data`` does
            not match the epochs.
        """
        rows = self._whitened(list(data), params)
        return [{name: r for name, _, _, _, r in row} for row, _ in rows]

    def chi2_per_night(self, data, params=None):
        """Raw χ² and its count of independent observables, per night.

        On the quoted errors (no error scales), at the best orbit with the
        V² amplitude re-profiled there (the fitted one at ``best``), in
        float64: the quantity to quote against the papers' χ²/N. Two
        sums are given. ``chi2`` is the data χ²: the ``n`` independent
        observables only. ``chi2_fit`` is the χ² the fit used: for
        correlated closure phases (four or more telescopes) it adds the
        periodic penalty rows, ``2 sin²(Δ/2)/σ``, which have no
        independent degree of freedom of their own (see
        [`whitened_residuals`][virgil.likelihood.whitened_residuals]).
        ``chi2_fit / n`` is the ``chi2_red`` the fit recorded in
        ``epochs``; without correlated closure phases the two agree.

        Parameters
        ----------
        data : sequence of OIData
            The epochs the band was fitted to, in the same order.
        params : dict, optional
            Elements to use instead of ``best``; see
            [`scene`][virgil.aliases.AliasBandResult.scene].

        Returns
        -------
        list of dict
            One dict per epoch with ``chi2``, ``chi2_fit``, ``n``
            (``data.n_independent``), ``chi2_per_n``, ``chi2_per_n_fit``
            and ``vis_amp`` (the amplitude used, 1 without ``vis_amp``),
            and ``blocks``: the same sums for each observable block. Sum
            ``chi2`` and ``n`` over nights for a pooled value.

        Raises
        ------
        ValueError
            As for [`residuals`][virgil.aliases.AliasBandResult.residuals].
        """
        out = []
        for row, amp in self._whitened(list(data), params):
            blocks = {}
            for name, a, z, nu, r in row:
                # The first ν rows are the independent observables; any
                # rows after them are the closure-phase penalty.
                blocks[name] = _chi2_entry(
                    float(onp.sum(r[:nu] ** 2)), float(onp.sum(r**2)), nu
                )
            total = _chi2_entry(
                sum(b["chi2"] for b in blocks.values()),
                sum(b["chi2_fit"] for b in blocks.values()),
                sum(b["n"] for b in blocks.values()),
            )
            out.append(dict(total, vis_amp=amp, blocks=blocks))
        return out


def _unrecorded(value):
    """Whether ``value`` is the [`UNRECORDED`][virgil.aliases.UNRECORDED]
    marker."""
    return isinstance(value, str) and value == UNRECORDED


_UNRECORDED_MESSAGE = (
    "this result does not record {0} (it was saved before it was kept): "
    "pass the fit's value to AliasResult.load(path, {0}=...)."
)


def _chi2_entry(chi2, chi2_fit, n):
    """The χ² of the data and of the fit, ``n`` and their ratios to it
    (None without data)."""
    return dict(
        chi2=chi2,
        chi2_fit=chi2_fit,
        n=int(n),
        chi2_per_n=chi2 / n if n else None,
        chi2_per_n_fit=chi2_fit / n if n else None,
    )


@dataclasses.dataclass
class AliasResult:
    """Result of [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases].

    Attributes
    ----------
    t_ref : float
        Reference time (MJD) of ``dt_peri``.
    times : numpy.ndarray
        The epochs' mean times (MJD).
    span : float
        First-to-last epoch span T (days).
    bands : list of AliasBandResult
        One per alias band, in order of decreasing ``log_z``.
    samples : dict
        ``{N: {name: array}}`` posterior draws (importance resampling from
        the modes' Laplace proposals) in the winning band and in every band
        with ``p > p_min_samples``. Keys: ``period``, ``dt_peri``, ``ecc``,
        ``inc``, ``omega``, ``Omega``, ``a_mas``, ``flux``.
    positions : list of dict or None
        The position-level diagnostic per band, if positions were given.
        Not saved by [`save`][virgil.aliases.AliasResult.save]: None after
        a [`load`][virgil.aliases.AliasResult.load].
    primary_diam : float, ``"fit"`` or None
        The primary of the fit's scene, as ``fit_orbit_aliases`` took it.
    vis_amp : tuple or None
        The range of the per-epoch V² amplitude, if it was fitted.
    unseeded : tuple of dict
        The nights that were flagged by
        [`epoch_positions`][virgil.epochs.epoch_positions] and so seeded
        nothing: ``{"name", "mjd", "flags": [{"code", "reason"}, ...]}``.
        Their data still count in every band's likelihood. Empty when
        nothing was skipped (also for ``positions`` that were not an
        ``EpochPositions``). Kept by ``save`` and ``load``.

    Notes
    -----
    ``primary_diam`` and ``vis_amp`` (and ``t_ref``) are copied into every
    band that lacks them, so the band accessors
    ([`scene`][virgil.aliases.AliasResult.scene],
    [`residuals`][virgil.aliases.AliasResult.residuals] and
    [`chi2_per_night`][virgil.aliases.AliasResult.chi2_per_night], which
    delegate to ``best``) can rebuild the scene without the fit's
    settings. A result written by [`save`][virgil.aliases.AliasResult.save]
    and read back by [`load`][virgil.aliases.AliasResult.load] keeps them.
    """

    t_ref: float
    times: onp.ndarray
    span: float
    bands: list
    samples: dict
    positions: list = None
    primary_diam: float | str | None = None
    vis_amp: tuple | None = None
    unseeded: tuple = ()

    def __post_init__(self):
        for b in self.bands:
            if b.t_ref is None:
                b.t_ref = self.t_ref
            if b.primary_diam is None:
                b.primary_diam = self.primary_diam
            if b.vis_amp is None:
                b.vis_amp = self.vis_amp

    @property
    def best(self):
        """The band with the highest evidence."""
        return self.bands[0]

    def table(self):
        """One row per band: raw χ²/ν, error scales (``scales`` flat over
        epochs and blocks; ``epoch_scales`` one list per epoch), log Z
        (Laplace and IS), p, ESS and the flags."""
        return [
            dict(
                n=b.n,
                p_lo=b.p_lo,
                p_hi=b.p_hi,
                period=b.best.get("period"),
                chi2_red=b.chi2_red,
                scales=[
                    v
                    for e in b.epochs
                    for v in onp.atleast_1d(e["scale"]).tolist()
                ],
                epoch_scales=[e["scale"] for e in b.epochs],
                log_z=b.log_z,
                log_z_is=b.log_z_is,
                p=b.p,
                ess=b.ess,
                flagged=b.flagged,
                flags=b.flags,
                n_modes=len(b.modes),
            )
            for b in self.bands
        ]

    def scene(self, params=None):
        """The best band's fitted scene; see
        [`AliasBandResult.scene`][virgil.aliases.AliasBandResult.scene]."""
        return self.best.scene(params)

    def residuals(self, data, params=None):
        """Whitened residuals at the best band's best orbit; see
        [`AliasBandResult.residuals`][virgil.aliases.AliasBandResult.residuals]."""
        return self.best.residuals(data, params)

    def chi2_per_night(self, data, params=None):
        """Raw χ² and N per night at the best band's best orbit; see
        [`AliasBandResult.chi2_per_night`][virgil.aliases.AliasBandResult.chi2_per_night]."""
        return self.best.chi2_per_night(data, params)

    def _payload(self, n_samples):
        """The JSON-ready dict of ``to_json`` and ``save``: ``n_samples``
        draws per band (all if None)."""
        return dict(
            t_ref=self.t_ref,
            times=self.times.tolist(),
            span=self.span,
            primary_diam=self.primary_diam,
            vis_amp=self.vis_amp,
            unseeded=list(self.unseeded),
            bands=self.table(),
            band_details=[dataclasses.asdict(b) for b in self.bands],
            samples={
                str(n): {k: v[:n_samples].tolist() for k, v in s.items()}
                for n, s in self.samples.items()
            }
            if n_samples is None or n_samples
            else {},
        )

    def to_json(self, path, n_samples=0):
        """Write the band table and each band's modes to ``path`` as JSON.

        Parameters
        ----------
        path : str or os.PathLike
            The file to write.
        n_samples : int or None, optional
            Posterior draws written per band: none by default, the first
            ``n_samples``, or, with ``None``, all of them (what
            [`save`][virgil.aliases.AliasResult.save] does).

        Notes
        -----
        [`load`][virgil.aliases.AliasResult.load] reads the file back.
        ``positions`` is not written.
        """
        with open(path, "w") as f:
            json.dump(
                clean_json(self._payload(n_samples)),
                f,
                indent=1,
                default=str,
            )

    def save(self, path):
        """Write the whole result to ``path`` as JSON, every posterior draw
        included; [`load`][virgil.aliases.AliasResult.load] is its inverse.

        Non-finite numbers are written as null (strict JSON) and restored
        on loading. The ``positions`` diagnostic is not written.
        """
        self.to_json(path, n_samples=None)

    @classmethod
    def load(cls, path, primary_diam=UNRECORDED, vis_amp=UNRECORDED):
        """Read a result written by
        [`save`][virgil.aliases.AliasResult.save] (or
        [`to_json`][virgil.aliases.AliasResult.to_json]).

        Parameters
        ----------
        path : str or os.PathLike
            The JSON file.
        primary_diam, vis_amp : optional
            The fit's settings, for a file written before they were saved
            (they would otherwise stay
            [`UNRECORDED`][virgil.aliases.UNRECORDED], and the accessors
            would raise instead of guessing a point primary or no V²
            amplitude). Given for a file that does record them, they
            override it.

        Returns
        -------
        AliasResult
            With every band's modes, evidences, flags, best orbit, per-epoch
            ``epochs`` and ``flipped``, the resampled ``samples`` (those
            written; none after ``to_json`` with the default
            ``n_samples=0``), and the settings the accessors need, so
            ``scene``, ``residuals`` and ``chi2_per_night`` work on it.
            ``positions`` is None.
        """
        with open(path) as f:
            p = json.load(f)
        saved = p.get("vis_amp", UNRECORDED)
        recorded = dict(
            primary_diam=p.get("primary_diam", UNRECORDED),
            vis_amp=saved
            if _unrecorded(saved) or saved is None
            else tuple(saved),
        )
        given = dict(primary_diam=primary_diam, vis_amp=vis_amp)
        if not _unrecorded(vis_amp) and vis_amp is not None:
            given["vis_amp"] = tuple(vis_amp)
        settings = {
            k: recorded[k] if _unrecorded(given[k]) else given[k]
            for k in recorded
        }
        bands = [_band_from_dict(d) for d in p["band_details"]]
        for b in bands:
            # A band that did not record a setting, or one that was
            # overridden, takes the result's.
            for k in settings:
                if _unrecorded(getattr(b, k)) or not _unrecorded(given[k]):
                    setattr(b, k, settings[k])
        return cls(
            t_ref=p["t_ref"],
            times=onp.asarray(p["times"], dtype=float),
            span=p["span"],
            bands=bands,
            samples={
                int(n): {k: onp.asarray(v, dtype=float) for k, v in s.items()}
                for n, s in p["samples"].items()
            },
            unseeded=tuple(p.get("unseeded", ())),
            **settings,
        )


def _nan_for_none(value):
    """``value`` with every None (how ``clean_json`` writes NaN and
    infinities) turned back into NaN, through lists and dicts."""
    if value is None:
        return float("nan")
    if isinstance(value, list):
        return [_nan_for_none(v) for v in value]
    return value


def _band_from_dict(d):
    """An [`AliasBandResult`][virgil.aliases.AliasBandResult] from its
    ``dataclasses.asdict`` JSON, restoring what ``clean_json`` made null.
    The evidences (``log_z`` fields) of failed modes and bands are ``-inf``;
    every other non-finite number (``p``, ``chi2_red``, a mode's
    ``loglike``, ``sigma`` and ``x``, the per-epoch numbers) is NaN, so
    that the two stay distinct. Settings the file did not record are
    [`UNRECORDED`][virgil.aliases.UNRECORDED]."""
    names = {f.name for f in dataclasses.fields(AliasBandResult)}
    d = {k: v for k, v in d.items() if k in names}
    for k in ("primary_diam", "vis_amp"):
        if k not in d:
            d[k] = UNRECORDED
    for k in ("log_z", "log_z_is"):
        if d.get(k) is None:
            d[k] = -onp.inf
    for k in ("p", "chi2_red"):
        if k in d and d[k] is None:
            d[k] = float("nan")
    d["modes"] = [
        {
            k: (-onp.inf if v is None else v)
            if k.startswith("log_z")
            else _nan_for_none(v)
            for k, v in m.items()
        }
        for m in d.get("modes", [])
    ]
    d["epochs"] = [
        {k: _nan_for_none(v) for k, v in e.items()}
        for e in d.get("epochs", [])
    ]
    if isinstance(d.get("vis_amp"), list):
        d["vis_amp"] = tuple(d["vis_amp"])
    return AliasBandResult(**d)


def _physical(x, t_ref):
    x = onp.stack([_fold(row) for row in onp.atleast_2d(x)])
    period = onp.exp(x[:, 0])
    return dict(
        period=period,
        dt_peri=x[:, 1] * period,
        ecc=x[:, 2],
        inc=onp.degrees(onp.arccos(onp.clip(x[:, 3], -1, 1))),
        omega=onp.mod(x[:, 4], 360.0),
        Omega=onp.mod(x[:, 5], 360.0),
        a_mas=onp.exp(x[:, 6]),
        flux=onp.exp(x[:, 7]),
        **({"diam": onp.exp(x[:, 8])} if x.shape[1] > 8 else {}),
    )


def _band_probabilities(results):
    """Set each band's ``p`` from its ``log_z`` and return the bands in order
    of decreasing ``log_z``."""
    log_z = onp.array([b.log_z for b in results])
    finite = onp.isfinite(log_z)
    weights = (
        onp.where(finite, onp.exp(log_z - onp.max(log_z[finite])), 0.0)
        if finite.any()
        else onp.zeros_like(log_z)
    )
    for b, w in zip(results, weights / max(weights.sum(), 1e-300)):
        b.p = float(w)
    order = onp.argsort(-onp.where(finite, log_z, -onp.inf), kind="stable")
    return [results[i] for i in order]


# Raw χ²/ν on the quoted errors above which the best band is flagged. The
# per-epoch error scales absorb the routine underestimates of interferometric
# errors (χ²/ν of 1-2), so the bar sits well above them: a missed
# likelihood ridge gives χ²/ν of 65-800, and errors underestimated by more
# than 3x deserve a look anyway.
_POOR_FIT_CHI2_RED = 10.0


def _check_fit_quality(results, k):
    """Flag, and warn about, a best band that does not fit the data."""
    best = results[0]
    if not best.chi2_red > _POOR_FIT_CHI2_RED:  # False for NaN too
        return
    best.flags.append("poor-fit")
    best.flagged = True
    warnings.warn(
        f"fit_orbit_aliases(): the best band (N={best.n}) has chi2_red = "
        f"{best.chi2_red:.3g} on the quoted errors (above "
        f"{_POOR_FIT_CHI2_RED:g}). Either the optimizer missed the narrow "
        "likelihood ridge, so the band probabilities are not trustworthy "
        f"(try a larger n_refine or k, now {k:g}), or the errors are "
        "underestimated by more than 3x.",
        UserWarning,
        stacklevel=external_stacklevel(),
    )


def _seeding_positions(positions, t_ref, skip_flagged, n_data):
    """The ``PositionData`` that seeds the bands, and the nights it skipped.

    ``positions`` may be ``PositionData`` (used as given) or the
    ``EpochPositions`` of ``epoch_positions``. With the latter and
    ``skip_flagged``, every flagged night seeds nothing. Returns the
    seeding positions (None if no night is left) and a record of each
    skipped night for ``AliasResult.unseeded``. Skipping drops a night's
    *seed* only: ``data`` is untouched, so its observables still enter
    the likelihood.
    """
    if not isinstance(positions, EpochPositions):
        return positions, ()
    if len(positions.names) != n_data:
        raise ValueError(
            f"positions has {len(positions.names)} datasets but data has "
            f"{n_data}: pass the epoch_positions of this data."
        )
    if not skip_flagged:
        return positions.positions(t_ref=t_ref), ()
    skipped = tuple(
        dict(
            name=name,
            mjd=float(mjd),
            flags=[dict(code=f.code, reason=f.reason) for f in fl],
        )
        for name, mjd, fl in zip(
            positions.names, positions.mjd, positions.flags
        )
        if fl
    )
    if positions.flagged.all():
        warnings.warn(
            "fit_orbit_aliases(): every night is flagged "
            f"({positions.flag_codes()}), so none seeds the bands; "
            "falling back to random starts.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
        return None, skipped
    return positions.positions(t_ref=t_ref, skip_flagged=True), skipped


def _resample(results, store, t_ref, n_samples, p_min, seed):
    """Posterior draws by importance resampling, for the best band and every
    band with ``p > p_min``."""
    samples = {}
    for rank, b in enumerate(results):
        if rank > 0 and b.p <= p_min:
            continue
        xs, lw = store[b.n]
        if xs is None or not onp.any(onp.isfinite(lw)):
            continue
        w = onp.exp(lw - onp.max(lw))
        pick = onp.random.default_rng([seed, 2**31, b.n]).choice(
            len(xs), size=n_samples, p=w / w.sum()
        )
        samples[b.n] = _physical(xs[pick], t_ref)
    return samples


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def _flatten_init(init, t_ref):
    """``init`` as ``(period, dt_peri, ecc, inc, omega, Omega, a_mas, flux)``
    (``dt_peri`` relative to ``t_ref``), or None."""
    if init is None:
        return None
    if isinstance(init, dict):
        keys = _ELEMENTS + ("flux",)
        missing = [k for k in keys if k not in init]
        if missing:
            raise ValueError(f"init is missing {missing}.")
        return tuple(float(init[k]) for k in keys)
    orbit, flux = init
    dt = float(orbit.dt_peri) + float(orbit.t_ref) - t_ref
    return (
        float(orbit.period),
        dt,
        float(orbit.ecc),
        float(orbit.inc),
        float(orbit.omega),
        float(orbit.Omega),
        float(orbit.a_mas),
        float(flux),
    )


@in_float64
def fit_orbit_aliases(
    data,
    p_range,
    *,
    positions=None,
    times=None,
    t_ref=None,
    a_range=(1.0, 30.0),
    flux_range=(0.01, 1.0),
    ecc_max=0.9,
    s_max=None,
    n_candidates=100,
    n_refine=6,
    n_random=None,
    k=9.0,
    eccs=None,
    sign_free=False,
    min_distance_mas=0.5,
    max_steps=200,
    n_is=2000,
    n_samples=1000,
    p_min_samples=0.01,
    flag_nats=1.0,
    ess_min=50.0,
    seed=0,
    checkpoint_dir=None,
    on_band=None,
    primary_diam=None,
    diam_range=(0.05, 5.0),
    vis_amp=None,
    flip_flux=True,
    init=None,
    skip_flagged=True,
):
    """Weigh the period aliases of an undersampled orbit with the observables.

    Parameters
    ----------
    data : sequence of OIData
        One dataset per epoch, with per-exposure times. The scene is a
        primary and a point companion on a Keplerian orbit
        ([`OrbitalBinary`][virgil.models.OrbitalBinary]).
    p_range : (float, float)
        The shortest and longest period (days): the log-uniform prior on P.
    positions : PositionData or EpochPositions, optional
        Per-epoch positions: ``PositionData``, or the
        [`EpochPositions`][virgil.epochs.EpochPositions] of
        [`epoch_positions`][virgil.epochs.epoch_positions] for the same
        ``data`` (then its night flags apply, see ``skip_flagged``). They
        seed each band's fits through Thiele–Innes grids
        ([`position_profile`][virgil.aliases.position_profile]); without them
        the candidates are random orbits.
    times : array-like, optional
        Epoch times (MJD) used for the bands and distinctness; by default
        each dataset's mean time.
    t_ref : float, optional
        Reference MJD of ``dt_peri``; by default the mean epoch.
    a_range, flux_range : (float, float), optional
        Log-uniform prior ranges of the semimajor axis (mas) and the
        companion/primary flux.
    ecc_max : float, optional
        Eccentricity is uniform on ``[0, ecc_max]``.
    sign_free : bool, optional
        The companion is nearly as bright as the primary, so each epoch's
        position is known only up to a sign (the point reflection): seed
        with sign-free Thiele–Innes fits and flux seeds near ``flux_range[1]``.
    s_max : float, optional
        Bound on the error scales, as in
        [`marginal_loglike`][virgil.epochs.marginal_loglike].
    n_candidates : int, optional
        Position-level candidate orbits per band (each in both senses of
        motion and at several fluxes), ranked by the observables.
    n_refine : int, optional
        The best distinct candidates per band that are refined into modes.
    n_random : int, optional
        Extra random candidates per band; by default 1000 without
        ``positions`` and none with them.
    min_distance_mas : float, optional
        Fitted modes whose tracks at the epochs differ by less are one mode.
    max_steps : int, optional
        Iterations of the optimizer for each start (default 200); see the
        Notes.
    n_is : int, optional
        Importance draws per mode.
    n_samples : int, optional
        Posterior draws per reported band.
    p_min_samples : float, optional
        Bands with at least this probability get posterior samples (and the
        winner always does).
    flag_nats : float, optional
        A band gets the flag ``laplace-is-differ`` when its Laplace and
        importance-sampling log evidences differ by more than this.
    checkpoint_dir : path-like, optional
        Each finished band is written here (``band_<N>.pkl`` for reuse and
        ``band_<N>.json`` to read) before the next starts. A rerun with the
        same data (every field), positions, ranges and settings loads the
        finished bands instead of refitting them; a band that raised is not
        saved, so it is retried.
    on_band : callable, optional
        Called with each band's
        [`AliasBandResult`][virgil.aliases.AliasBandResult] as soon as it is
        fitted or loaded (``p`` is not yet set: it needs all bands).

    primary_diam : float or "fit", optional
        The primary's uniform-disk diameter (mas). ``None`` (default) or 0 is
        a point source (the fast path); a float fixes it; ``"fit"`` samples
        one shared diameter, log-uniform over ``diam_range``, as an extra
        coordinate (reported as ``diam`` in the band's ``best`` and the
        samples).
    diam_range : (float, float), optional
        Prior range (mas) of a fitted diameter.
    vis_amp : (float, float), optional
        Give every epoch's squared visibilities their own multiplicative
        amplitude ``A`` (a contrast loss that is the same on every baseline
        and wavelength, unlike a resolved primary), profiled analytically:
        the best ``A``, clipped to this range (which only bounds it: no
        prior or Occam factor for ``A`` enters ``log_z``, which is then a
        profile in ``A``; every band has the same number of them). It is
        reported per epoch as ``vis_amp``. It needs squared-visibility data
        (``vis_mode="v2"``) and raises otherwise. The error scale of the
        epoch is kept. ``None`` (default) leaves the visibilities as they are.
    flip_flux : bool, optional
        When a band's mode has its flux ratio at the top of ``flux_range``
        refit it with the stars' labels swapped
        (ω + 180°, flux → 1/flux), since a ratio above 1 sits outside the
        prior; the better result is kept and ``flipped`` is set on the band
        if it won. It applies to a point primary without differential
        phases only (otherwise the swap is not a relabelling) and acts on a
        mode within 0.02 of the top in log flux.
    skip_flagged : bool, optional
        With ``EpochPositions``, a night flagged by ``epoch_positions``
        (a single-night alias peak, a low gap, a χ²/N far above the median
        night, an unresolved separation, a flux at its bound) **seeds
        nothing**, but its data still count in the likelihood of every
        band (default True). The skipped nights are recorded in
        ``AliasResult.unseeded``. False seeds with every night. If every
        night is flagged, the bands start from random orbits, with a
        warning.
    init : (KeplerOrbit, float) or dict, optional
        A warm start: an orbit and flux ratio, or a dict with ``period``,
        ``dt_peri`` (relative to ``t_ref``), ``ecc``, ``inc``, ``omega``,
        ``Omega``, ``a_mas`` and ``flux``. It joins every band's candidate
        pool, and is refined in the band that contains its period whatever
        its rank.

    Returns
    -------
    AliasResult

    Raises
    ------
    RuntimeError
        If every band fails (chained to the last error).
    ValueError
        If no band lies within ``p_range``, or ``checkpoint_dir`` is given
        and the data or positions cannot be hashed.
        Also whatever the preflight likelihood evaluation raises
        (misconfiguration), before any band is fitted.

    Notes
    -----
    A band that raises is kept as a ``band-failed`` row (``error`` holds the
    exception) with ``log_z = -inf``; the other bands are unaffected, unless
    all fail. Misconfiguration surfaces first: the likelihood and its
    gradient are evaluated once at the centre of the first band before any
    band is fitted, so an error there propagates unchanged, and so does a
    ``TypeError`` anywhere. Any other exception in a band is isolated. An
    optimizer start that raises is dropped and counted (flag
    ``starts-failed``, ``n_starts_failed``); if every start of a band
    fails the band fails.

    The band evidence ``log_z`` is the sum, over the band's distinct fitted
    modes (both senses of motion; the (Ω+180, ω+180) mirror is folded
    away by ``Ω < 180``), of the Laplace approximation to the marginal
    likelihood of the scale-marginalized closure-phase (and visibility)
    likelihood, times the proper prior. A mode on a prior bound (a period
    edge between bands) uses its importance-sampling value instead, which
    truncates the target at the bound. It misses modes the multistart does
    not find. ``log_z_is`` is the importance-sampling estimate from the
    deterministic mixture of the modes' Student-t proposals (so overlapping
    tails are not counted twice), and the posterior samples are resampled
    with the same weights, with the mirror folded.

    The mixture is an approximation: each mode's Student-t proposal is
    evaluated without periodic images of the angles, and each mode's draws
    are restricted to its own window (within half a period of the mode).
    When modes straddle the Ω fold, ``log_z_is`` and the resampled draws can
    be slightly biased. ``log_z``, which uses the Laplace value except near
    prior bounds, is unaffected.

    Each start is optimized by damped Newton (Levenberg–Marquardt) steps
    on the scale-marginalized log likelihood. Its curvature is that of
    iteratively reweighted least squares: each observable block's
    residuals weighted by the slope of the log likelihood in its χ² (for
    ``s_max=None``, 1/ŝ² with ŝ² = χ²/ν the error scale those residuals
    imply), and the profiled V² amplitude differentiated through.
    A step is kept only if the log likelihood rises, so the curvature
    sets the speed and not the optimum. If a start does not converge in
    ``max_steps`` (or stalls), L-BFGS-B continues from where it stopped,
    and the damped Newton steps polish its result.

    ``flags`` lists why a band should not be trusted:
    ``near-prior-bound``, ``not-converged`` (the optimizer stopped more
    than 0.1σ from a stationary point), ``hessian-not-positive-definite``,
    ``hessian-singular``, ``band-failed`` (an exception in the band; the
    others are unaffected), ``starts-failed`` (some optimizer start raised),
    ``laplace-is-differ``, ``low-ess`` (best mode's ESS below ``ess_min``),
    ``no-valid-mode`` and ``poor-fit``. ``chi2_red`` is the raw χ²/ν on the
    quoted errors at the best mode, before any error scale.

    ``poor-fit`` is set on the winning band only, with a ``UserWarning``,
    when its ``chi2_red`` exceeds 10: the fits missed the orbit, or the
    errors are underestimated by more than 3x (the per-epoch error scales
    absorb smaller underestimates without a flag).

    Without ``s_max`` the Jeffreys prior on each error scale is improper, so
    the absolute ``log_z`` is defined only up to a constant shared by all
    bands (``p`` and differences are meaningful).
    """
    data = cast_tree(list(data), "float64")  # a generator, too
    mean_times = (
        onp.array([onp.mean(onp.asarray(d.mjd, float)) for d in data])
        if times is None
        else onp.asarray(times, float)
    )
    t_ref = (
        float(onp.round(onp.mean(mean_times), 1))
        if t_ref is None
        else float(t_ref)
    )
    bands = alias_bands(mean_times, p_range)
    if not bands:
        raise ValueError("No alias band lies within p_range.")
    # Single-night alias peaks: the true position is often not the best
    # peak (Gl 229 2024-02-27 and 2024-03-28), so a flagged night must not
    # seed the bands. Only its seed is dropped: ``data`` keeps the night,
    # so the likelihood still sees it and can reject a wrong band.
    positions, unseeded = _seeding_positions(
        positions, t_ref, skip_flagged, len(data)
    )
    profile = None
    if positions is not None:
        profile = position_profile(
            positions,
            bands,
            k=k,
            n_best=n_candidates,
            eccs=eccs,
            sign_free=sign_free,
        )
    if n_random is None:
        n_random = 1000 if positions is None else 0
    if primary_diam is not None and primary_diam != "fit":
        primary_diam = float(primary_diam)
        if not primary_diam >= 0:
            raise ValueError("primary_diam must be None, >= 0, or 'fit'.")
    if vis_amp is not None:
        vis_amp = tuple(map(float, vis_amp))
        if not 0 < vis_amp[0] < vis_amp[1]:
            raise ValueError("vis_amp must be (lo, hi) with 0 < lo < hi.")
    init_flat = _flatten_init(init, t_ref)
    n_scan = _N_FLUX + (2 if sign_free else 0)
    n_seeded = 0 if positions is None else 2 * n_scan * n_candidates
    n_pool = n_random + n_seeded + (1 if init else 0)  # senses, fluxes, init
    settings = _Settings(
        p_range=tuple(map(float, p_range)),
        a_range=tuple(map(float, a_range)),
        flux_range=tuple(map(float, flux_range)),
        ecc_max=ecc_max,
        s_max=s_max,
        t_ref=t_ref,
        n_candidates=n_candidates,
        n_refine=n_refine,
        n_random=n_random,
        pool_size=-(-n_pool // 64) * 64,
        k=k,
        eccs=None if eccs is None else tuple(map(float, eccs)),
        min_distance_mas=min_distance_mas,
        max_steps=max_steps,
        n_is=n_is,
        flag_nats=flag_nats,
        ess_min=ess_min,
        seed=seed,
        sign_free=sign_free,
        primary_diam=primary_diam,
        diam_range=tuple(map(float, diam_range)),
        vis_amp=vis_amp,
        flip_flux=bool(flip_flux),
        init=init_flat,
    )
    problem = _Problem(data, settings)
    _preflight(problem, settings, bands[0])
    key = (
        _checkpoint_key(data, positions, bands, mean_times, settings)
        if checkpoint_dir is not None
        else None
    )

    results, store, error = [], {}, None
    for index, band in enumerate(bands):
        done = _load_band(checkpoint_dir, band, key)
        if done is None:
            try:
                done = _fit_band(
                    problem,
                    settings,
                    band,
                    None if profile is None else profile[index]["orbits"],
                    mean_times,
                )
            except TypeError:
                raise
            except Exception as caught:  # noqa: BLE001 - keep the other bands
                error = caught
                done = (_failed_band(band, caught), (None, None))
            else:
                _save_band(checkpoint_dir, band, key, done)
        result, store[band.n] = done
        results.append(result)
        if on_band is not None:
            on_band(result)
    if all("band-failed" in b.flags for b in results):
        raise RuntimeError(
            f"Every band failed; the last error was: {results[-1].error}"
        ) from error

    results = _band_probabilities(results)
    _check_fit_quality(results, k)
    samples = _resample(results, store, t_ref, n_samples, p_min_samples, seed)
    return AliasResult(
        t_ref,
        mean_times,
        float(onp.ptp(mean_times)),
        results,
        samples,
        profile,
        primary_diam=primary_diam,
        vis_amp=vis_amp,
        unseeded=unseeded,
    )
