"""Helpers shared by the pipelines: χ², residual moments, quantiles,
geometry of the data, prior-bound fractions and figure saving."""

from __future__ import annotations

import math

import numpy as onp
from scipy import stats

from . import _io

# Fraction of a prior's range, at either end, counted as "at the bound".
_BOUND_FRACTION = 0.01


def _bound_samples(prior, samples):
    """Fraction of ``samples`` in the outer ``_BOUND_FRACTION`` of the
    prior's finite range at either end, or ``None`` if it has no edges.

    The range is the one ``fit`` reports (``fitting._bound_fraction``): a
    LogUniform's in log x, an interval support's, and none for tied terms,
    angle vectors, or priors with a flat coordinate (their ends are poles).
    """
    from ..fitting import _bound_fraction

    x = onp.asarray(samples, dtype=float).ravel()
    fraction = _bound_fraction(prior, x)
    if fraction is None:
        return None
    return float(
        onp.mean(
            (fraction < _BOUND_FRACTION) | (fraction > 1 - _BOUND_FRACTION)
        )
    )


def _close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def _save(fig, path):
    with _io.atomic_path(path) as tmp:
        fig.savefig(tmp, dpi=110, bbox_inches="tight")
    _close(fig)


def _geometry(data):
    """Baseline and resolution figures of ``data`` (metres, mas)."""
    from .._geometry import fringe_scales

    u = onp.asarray(data.u, dtype=float)
    v = onp.asarray(data.v, dtype=float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, dtype=float), u.shape)
    baseline = onp.hypot(u, v)
    positive = baseline > 0
    finest, coarsest = fringe_scales(data)
    return {
        "baseline_min_m": float(baseline[positive].min()),
        "baseline_max_m": float(baseline[positive].max()),
        "wavel_min_m": float(wavel.min()),
        "wavel_max_m": float(wavel.max()),
        # Half of the finest λ/B: inside it, separation and flux are
        # degenerate.
        "resolution_mas": float(0.5 * finest),
        # The coarsest λ/B: the field over which the coverage is
        # unambiguous.
        "fov_mas": float(coarsest),
        "lambda_over_b_mas": float(finest),
    }


def _chi2(model, data, **noise):
    from ..likelihood import whitened_residuals

    r = onp.asarray(whitened_residuals(model, data, **noise), dtype=float)
    return float(onp.sum(r**2)), r


def _moments(r):
    r = onp.asarray(r, dtype=float)
    if r.std() <= 1e-12 * max(abs(r.mean()), 1.0):
        return 0.0, -3.0  # scipy gives NaN for (near-)constant residuals
    return float(stats.skew(r)), float(stats.kurtosis(r))


def _wrap(angle):
    return (onp.asarray(angle) + 360.0) % 360.0


def _quantiles(x, angle=False):
    """Median, 16th and 84th percentiles; angles about their mean direction."""
    x = onp.asarray(x, dtype=float).ravel()
    if angle:
        mean = math.degrees(
            math.atan2(
                onp.mean(onp.sin(onp.radians(x))),
                onp.mean(onp.cos(onp.radians(x))),
            )
        )
        dev = (x - mean + 180.0) % 360.0 - 180.0
        q = onp.percentile(dev, [50, 16, 84])
        median = float(_wrap(q[0] + mean))
        q = onp.array([median, median + q[1] - q[0], median + q[2] - q[0]])
    else:
        q = onp.percentile(x, [50, 16, 84])
    return {"median": float(q[0]), "q16": float(q[1]), "q84": float(q[2])}
