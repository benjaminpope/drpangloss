"""The shared log-space quadrature (virgil._marginal)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.special import erf

from virgil._marginal import gauss_legendre, log_integral, log_quadrature


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def test_gauss_legendre_integrates_polynomials_exactly():
    x, w = gauss_legendre(8, np.float64)
    assert x.dtype == w.dtype == np.float64
    assert float(np.sum(w * x**6)) == pytest.approx(2.0 / 7.0)


def test_log_quadrature_is_a_log_weighted_sum():
    values, w = onp.array([-1.0, 0.5, -3.0]), onp.array([0.2, 1.0, 0.7])
    got = log_quadrature(np.asarray(values), np.log(w))
    assert float(got) == pytest.approx(onp.log(onp.sum(w * onp.exp(values))))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "mu, sigma", [(0.3, 0.5), (1.2, 0.02), (-1.5, 1e-3), (0.0, 5.0)]
)
def test_log_integral_of_a_gaussian_in_bounds(mu, sigma, dtype):
    """Narrow peaks and a peak at an edge are resolved (exact in closed form)."""
    lo, hi = -1.6, 1.6

    def log_f(t):
        return -0.5 * ((t - mu) / sigma) ** 2

    root = sigma * onp.sqrt(onp.pi / 2.0)
    exact = onp.log(
        root * (erf((hi - mu) / (onp.sqrt(2) * sigma))
                - erf((lo - mu) / (onp.sqrt(2) * sigma)))
    )  # fmt: skip
    got = log_integral(log_f, lo, hi, np.asarray(mu, dtype))
    assert float(got) == pytest.approx(
        exact, abs=2e-4 if dtype is np.float32 else 1e-9
    )


def test_log_integral_of_a_steep_cutoff_edge():
    """An integrand falling 1e4 per unit past 0.4 is a step, then e^-t."""

    def log_f(t):
        return np.where(t > 0.4, -1e4 * (t - 0.4), 0.0) - t

    exact = onp.log(onp.exp(1.6) - onp.exp(-0.4) + onp.exp(-0.4) * 1e-4)
    got = log_integral(log_f, -1.6, 1.6, np.asarray(0.0))
    assert float(got) == pytest.approx(exact, abs=1e-3)


def test_log_integral_gradient_matches_a_finite_difference():
    """The window carries no gradient, and the quadrature's own is exact:
    d/dm of log ∫ exp(-(t-m)²/2) (the peak guess moves with m) is 0."""

    def value(m):
        return log_integral(lambda t: -0.5 * (t - m) ** 2, -8.0, 8.0, m)

    m, h = np.asarray(0.5), 1e-4
    fd = (value(m + h) - value(m - h)) / (2 * h)
    assert abs(float(jax.grad(value)(m)) - float(fd)) < 1e-6
