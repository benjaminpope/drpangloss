"""Log-space quadrature shared by the orbit scores and the flux grids (private).

[`log_integral`][virgil._marginal.log_integral] integrates a possibly steep
log integrand adaptively (the bounded scale marginal of
[`virgil.epochs`][virgil.epochs]); [`gauss_legendre`][virgil._marginal.gauss_legendre]
and [`log_quadrature`][virgil._marginal.log_quadrature] are the node and
summation steps that the flux marginal of [`virgil.orbit_search`][virgil.orbit_search]
and the log-uniform evidence of [`virgil.grid_fit`][virgil.grid_fit] also use.
"""

import functools

import jax
import jax.numpy as np
import numpy as onp

# Scans of _N_SCAN nodes narrow the range to where the integrand is within
# exp(-depth) of its peak, then _N_QUADRATURE Gauss-Legendre nodes integrate it.
_N_SCAN = 64
_N_NARROW = 4
_N_QUADRATURE = 64


@functools.cache
# deliberate: cache boundary, one node set per order
def _leggauss(order):
    return onp.polynomial.legendre.leggauss(order)


def gauss_legendre(order, dtype=None):
    """Gauss-Legendre nodes and weights on [-1, 1], as arrays of ``dtype``
    (JAX's default float if None)."""
    return tuple(np.asarray(v, dtype) for v in _leggauss(order))


def log_quadrature(values, log_w):
    """``log Σ w exp(values)`` for log weights ``log_w``."""
    return jax.nn.logsumexp(values + log_w)


def log_integral(log_f, lo, hi, guess):
    """``log ∫ exp(log_f(t)) dt`` over ``[lo, hi]``, for ``log_f`` mapping
    an array of ``t`` to the log integrand, which may be steep.

    A fixed grid in t misses steep edges and narrow peaks. Each scan
    evaluates ``log_f`` on a uniform grid over the current window (plus
    ``guess``, the expected peak) and keeps the span of nodes within
    ``depth`` e-folds of the maximum, widened by one node on each side;
    Gauss-Legendre then integrates over the final window. The window does
    not carry gradients: its edges are where the integrand is
    ~exp(-depth) of its peak, or the fixed bounds.
    """
    guess = jax.lax.stop_gradient(guess)
    dtype = guess.dtype
    depth = -onp.log(float(np.finfo(dtype).eps)) + 10.0
    lo = np.asarray(lo, dtype)
    hi = np.asarray(hi, dtype)
    a, b = lo, hi
    uniform = np.linspace(0.0, 1.0, _N_SCAN, dtype=dtype)
    for k in range(_N_NARROW):
        nodes = a + (b - a) * uniform
        if k == 0:
            nodes = np.sort(
                np.concatenate([nodes, np.clip(guess, lo, hi)[None]])
            )
        values = jax.lax.stop_gradient(log_f(jax.lax.stop_gradient(nodes)))
        values = np.where(np.isnan(values), -np.inf, values)
        keep = values >= np.max(values) - depth
        index = np.arange(nodes.size)
        first = np.min(np.where(keep, index, nodes.size))
        last = np.max(np.where(keep, index, -1))
        a = nodes[np.maximum(first - 1, 0)]
        b = nodes[np.minimum(last + 1, nodes.size - 1)]
    x, w = gauss_legendre(_N_QUADRATURE, dtype)
    half = 0.5 * (b - a)
    t = a + half * (x + 1.0)
    return log_quadrature(log_f(t), np.log(w)) + np.log(half)
