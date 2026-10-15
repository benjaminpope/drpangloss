"""Tempering an evidence for correlated spectral channels.

**The solved problem: untempered evidence (band p = 1.000 at χ²/N 12).**
Spectral channels of one baseline (and of one closure triangle) are not
independent measurements: the same atmosphere, the same calibrator and the
same detector rows sit under all of them. A likelihood that treats ``N``
channels as ``N`` independent points is therefore too sharp by the factor
``f = N / N_eff``. On real data (χ²/N ≈ 12 in virgil-validation) the
alias band with the best evidence got p = 1.000, and every posterior was
too narrow (sky-plane offset/spread 3.7 to 61).

The cure is a *tempered* likelihood, ``L^(1/f)``: the log likelihood, and
with it every evidence difference and posterior curvature, is divided by
``f`` (so the posterior is wider by √f). It is applied per observable
block, ``dof_b = 1 / f_b``, through the ``dof`` argument of the
scale-marginalized score
(``_Surface.score`` of ``virgil.epochs``); with ``f = 1`` nothing
changes.

**The estimator.** ``f`` comes from the residuals of the best static fit of
each night, not from χ²: the error scale already absorbs a global
underestimate of the errors (χ²/N = 12 alone says the errors are 3.5 times
too small, which the per-block scale integrates out, not that the channels
are correlated), whereas correlation across channels is what the scale
cannot absorb. For the whitened residuals ``z`` of neighbouring channels
``i, i+1`` of the same baseline (or triangle) and frame, the lag-1
correlation is

    ρ = Σ z_i z_(i+1) / (½ Σ (z_i² + z_(i+1)²)),

which is scale free (the same inflation of the errors cancels). For an
AR(1) chain of correlation ρ the effective number of points is
``N (1 - ρ)/(1 + ρ)``, so

    f = (1 + ρ) / (1 - ρ).

White noise has ρ = 0, so ``f = 1`` (to 1/√m for m pairs; a ρ below 2/√m,
two standard errors, is taken as zero, so white noise gives exactly 1 about
95% of the time and never far from it). A model error that is smooth in
wavelength (a resolved primary fitted as a point, say) is also correlated
between channels and raises ``f``, which is the intent: it is an error the
channels share. Blocks with fewer than ``MIN_PAIRS`` neighbouring pairs, or
without a channel axis (extra observables), take the pooled ``f`` of all
pairs. Data with a single wavelength have no neighbours and ``f = 1``: the
tempering is then off, and a fixed ``f`` is the way to temper them.
``f`` is limited to ``[F_MIN, F_MAX]`` and the result is flagged
(``AliasResult.tempering``) outside ``FLAG_RANGE``, [0.5, 2].

This module imports nothing from virgil.
"""

import dataclasses

import numpy as onp

__all__ = [
    "ChannelFactors",
    "FLAG_RANGE",
    "MIN_PAIRS",
    "adjacent_channels",
    "estimate_factors",
    "lag1_correlation",
    "factor_from_correlation",
]

# Outside this range of ``f`` the result is flagged: well below 1 means the
# residuals are anticorrelated (differenced or over-fitted data), well above
# 2 that fewer than half the channels carry independent information.
FLAG_RANGE = (0.5, 2.0)
# A block with fewer neighbouring pairs than this takes the pooled f: its
# own correlation is uncertain by 1/√m, 0.18 at 30 pairs.
MIN_PAIRS = 30
# The limits of f: ρ is kept below 0.98 (f = 99, one independent point in
# a hundred) and f above 1/3 (ρ = -0.5).
F_MIN = 1.0 / 3.0
F_MAX = 99.0
# Neighbouring samples share a baseline when their (u, v) agree to this many
# metres.
_UV_TOL = 1e-3
# A correlation below this many standard errors (1/√m) is zero.
_SIGNIFICANCE = 2.0


@dataclasses.dataclass(frozen=True)
class ChannelFactors:
    """The effective-channel factors of one run.

    Attributes
    ----------
    f : float
        The pooled factor ``N / N_eff`` of all neighbouring pairs; 1 when
        the data have none.
    blocks : dict
        ``{block name: f_b}`` for every block of the data (``"vis"``,
        ``"phi"``, ``"extra[k]"``), each the pooled value unless it has
        ``MIN_PAIRS`` pairs of its own.
    rho : dict
        ``{block name: ρ}`` of the blocks with enough pairs of their own,
        plus ``"pooled"`` (NaN without pairs).
    n_pairs : dict
        ``{block name: pairs}``, plus ``"pooled"``.
    """

    f: float
    blocks: dict
    rho: dict
    n_pairs: dict

    def dof(self):
        """``{block name: 1/f_b}``: the ``dof`` of the tempered score."""
        return {k: 1.0 / v for k, v in self.blocks.items()}


def adjacent_channels(data):
    """The neighbouring-channel pairs of each block of ``data``.

    Two samples are neighbours when they share a baseline (the same
    ``u``, ``v`` in metres) and frame and have adjacent distinct
    wavelengths; a closure phase is a neighbour of the same triangle in the
    next channel. Samples are laid out baseline-major with channels
    fastest, which is how OIData stores several channels, but the pairs are
    found from the ``u``, ``v``, ``wavel`` and ``frame`` of each sample, not
    from that order.

    Parameters
    ----------
    data : OIData
        The data (duck-typed: ``u``, ``v``, ``wavel``, ``frame``,
        ``vis``, ``phi``, ``vis_index``, ``phi_index``, ``i_cps1``,
        ``i_cps2``).

    Returns
    -------
    dict
        ``{"vis": (a, b), "phi": (a, b)}``: integer arrays indexing the
        residual vector of each block, empty when the block has no
        neighbours or its layout cannot be read.
    """
    u = onp.asarray(data.u, float)
    v = onp.asarray(data.v, float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, float), u.shape)
    frame = (
        onp.zeros(u.shape, int)
        if data.frame is None
        else onp.asarray(data.frame)
    )
    n_vis = int(onp.size(data.vis))
    n_phi = int(onp.size(data.phi))
    vis_index = (
        onp.arange(u.size)
        if data.vis_index is None
        else onp.asarray(data.vis_index)
    )
    out = {
        "vis": _pairs(_sample_keys(u, v, frame, vis_index), wavel[vis_index])
    }
    if len(vis_index) != n_vis:
        out["vis"] = _EMPTY
    if not n_phi:
        out["phi"] = _EMPTY
    elif data.i_cps1 is not None and onp.size(data.i_cps1) == n_phi:
        # A closure phase is keyed by two of its baselines, at the
        # wavelength of the first.
        i1 = onp.asarray(data.i_cps1)
        i2 = onp.asarray(data.i_cps2)
        keys = list(
            zip(_sample_keys(u, v, frame, i1), _sample_keys(u, v, frame, i2))
        )
        out["phi"] = _pairs(keys, wavel[i1])
    else:
        index = (
            onp.arange(u.size)
            if data.phi_index is None
            else onp.asarray(data.phi_index)
        )
        out["phi"] = (
            _pairs(_sample_keys(u, v, frame, index), wavel[index])
            if len(index) == n_phi
            else _EMPTY
        )
    return out


_EMPTY = (onp.zeros(0, int), onp.zeros(0, int))


def _sample_keys(u, v, frame, index):
    """One hashable key per sample of ``index``: its baseline and frame."""
    q = 1.0 / _UV_TOL
    return [
        (int(round(u[i] * q)), int(round(v[i] * q)), int(frame[i]))
        for i in index
    ]


def _pairs(keys, wavel):
    """Index pairs of consecutive wavelengths within each key's group."""
    groups = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    a, b = [], []
    for members in groups.values():
        members = sorted(members, key=lambda p: wavel[p])
        for p, q in zip(members[:-1], members[1:]):
            if wavel[q] > wavel[p]:
                a.append(p)
                b.append(q)
    return onp.asarray(a, int), onp.asarray(b, int)


def lag1_correlation(z, pairs):
    """The correlation of neighbouring residuals, ``Σ z_a z_b`` over
    ``½ Σ (z_a² + z_b²)``, and the number of pairs.

    It is scale free: inflating every error by one factor leaves it
    unchanged. NaN for no pairs or a zero residual.
    """
    a, b = pairs
    if len(a) == 0:
        return float("nan"), 0
    z = onp.asarray(z, float)
    den = 0.5 * onp.sum(z[a] ** 2 + z[b] ** 2)
    if not den > 0:
        return float("nan"), len(a)
    return float(onp.sum(z[a] * z[b]) / den), len(a)


def factor_from_correlation(rho, n_pairs):
    """``f = (1 + ρ)/(1 - ρ)`` for ρ significant at two standard errors
    (1/√m for ``n_pairs`` = m), else 1; limited to ``[F_MIN, F_MAX]``.

    NaN (no information) gives 1.
    """
    if not onp.isfinite(rho) or n_pairs < 1:
        return 1.0
    if abs(rho) < _SIGNIFICANCE / onp.sqrt(n_pairs):
        return 1.0
    rho = min(float(rho), (F_MAX - 1.0) / (F_MAX + 1.0))
    return float(onp.clip((1.0 + rho) / (1.0 - rho), F_MIN, F_MAX))


def estimate_factors(residuals, pairs, names=()):
    """The effective-channel factors from residuals at the best fit.

    Parameters
    ----------
    residuals : sequence of dict
        One dict per dataset, ``{block name: whitened residuals}``, each
        array in the block's own order (the order ``adjacent_channels``
        indexes).
    pairs : sequence of dict
        One dict per dataset from ``adjacent_channels``.
    names : sequence of str, optional
        Block names to report beyond those of ``residuals`` (blocks
        without residuals, such as extras, take the pooled ``f``).

    Returns
    -------
    ChannelFactors
        The estimator is described in the module docstring. Pooling sums
        the numerators and denominators of ρ over all blocks and datasets.
    """
    num, den, count = {}, {}, {}
    for resid, pair in zip(residuals, pairs):
        for name, z in resid.items():
            a, b = pair.get(name, _EMPTY)
            if len(a) == 0:
                continue
            z = onp.asarray(z, float)
            num[name] = num.get(name, 0.0) + float(onp.sum(z[a] * z[b]))
            den[name] = den.get(name, 0.0) + 0.5 * float(
                onp.sum(z[a] ** 2 + z[b] ** 2)
            )
            count[name] = count.get(name, 0) + len(a)
    total = sum(count.values())
    pooled = (
        sum(num.values()) / sum(den.values())
        if total and sum(den.values()) > 0
        else float("nan")
    )
    f_pooled = factor_from_correlation(pooled, total)
    everything = set(names) | {n for r in residuals for n in r}
    blocks, rho = {}, {"pooled": pooled}
    for name in sorted(everything):
        m = count.get(name, 0)
        own = num.get(name, 0.0) / den[name] if m and den[name] > 0 else None
        if m >= MIN_PAIRS and own is not None:
            rho[name] = own
            blocks[name] = factor_from_correlation(own, m)
        else:
            blocks[name] = f_pooled
    return ChannelFactors(
        f=f_pooled,
        blocks=blocks,
        rho=rho,
        n_pairs={**count, "pooled": total},
    )
