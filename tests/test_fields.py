import warnings

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from drpangloss.coverage import vlti_oidata
from drpangloss.fields import GaussianField, field_spectrum
from drpangloss.fitting import fit
from drpangloss.imaging import diagnose, image_priors
from drpangloss.models import GaussianDisk, Image, PointSource, System


def _neumann_laplacian(n, h):
    """The 1D graph Laplacian with reflecting ends, divided by h²."""
    lap = 2.0 * onp.eye(n) - onp.eye(n, k=1) - onp.eye(n, k=-1)
    lap[0, 0] = lap[-1, -1] = 1.0
    return lap / h**2


def _covariance(field, h):
    """Covariance of η = f(latent) under a standard-normal latent."""
    jac = jax.jacfwd(lambda z: field.set("latent", z).evaluate(h).ravel())(
        field.latent
    )
    jac = onp.asarray(jac).reshape(field.latent.size, -1)
    return jac @ jac.T


@pytest.mark.parametrize("order", [1, 2])
def test_covariance_is_the_inverse_of_a_matern_operator(order):
    nrow, ncol, h, length, sigma = 8, 6, 0.5, 1.5, 2.0
    field = GaussianField(onp.zeros((nrow, ncol)), sigma, length, order)
    lap = onp.kron(_neumann_laplacian(nrow, h), onp.eye(ncol)) + onp.kron(
        onp.eye(nrow), _neumann_laplacian(ncol, h)
    )
    operator = onp.linalg.matrix_power(
        lap + onp.eye(lap.shape[0]) / length**2, order
    )
    # The constant mode is removed; then scale to a mean variance of σ².
    centre = onp.eye(lap.shape[0]) - 1.0 / lap.shape[0]
    expected = centre @ onp.linalg.inv(operator) @ centre
    expected *= sigma**2 / onp.mean(onp.diag(expected))
    assert onp.allclose(_covariance(field, h), expected, atol=1e-5 * sigma**2)


def test_order_one_is_total_squared_variation_plus_l2():
    # For a zero-mean field, ½|z|² = c (κ²|η|² + |∇η|²/h²) / 2: TSV + L2 on η.
    n, h, length, sigma = 10, 0.7, 2.0, 1.3
    latent = onp.random.default_rng(0).normal(size=(n, n))
    latent[0, 0] = 0.0  # the constant mode carries no prior
    field = GaussianField(latent, sigma, length, order=1)
    eta = onp.asarray(field.evaluate(h))
    tsv = onp.sum(onp.diff(eta, axis=0) ** 2) + onp.sum(
        onp.diff(eta, axis=1) ** 2
    )
    lam = [(2 / h * onp.sin(onp.pi * onp.arange(n) / (2 * n))) ** 2] * 2
    unscaled = 1.0 / (length**-2.0 + lam[0][:, None] + lam[1][None, :])
    unscaled[0, 0] = 0.0
    c = unscaled.sum() / (sigma**2 * n * n)
    expected = 0.5 * c * (length**-2.0 * onp.sum(eta**2) + tsv / h**2)
    assert 0.5 * onp.sum(latent**2) == pytest.approx(expected, rel=1e-4)


def test_sigma_and_length_are_calibrated():
    n, h = 16, 1.0
    for sigma in (0.5, 2.0):
        cov = _covariance(GaussianField(onp.zeros((n, n)), sigma, 3.0), h)
        assert onp.mean(onp.diag(cov)) == pytest.approx(sigma**2, rel=1e-4)
        spectrum = field_spectrum((n, n), h, sigma, 3.0)
        assert float(spectrum[0, 0]) == 0.0
        assert float(onp.mean(spectrum)) == pytest.approx(sigma**2, rel=1e-5)

    def neighbour_correlation(length):
        cov = _covariance(GaussianField(onp.zeros((n, n)), 1.0, length), h)
        i = (n // 2) * n + n // 2
        return cov[i, i + 1] / onp.sqrt(cov[i, i] * cov[i + 1, i + 1])

    assert neighbour_correlation(1.0) < neighbour_correlation(4.0) < 1.0


def test_zero_latents_reproduce_the_template_image():
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    field = GaussianField(onp.zeros((24, 24)), 2.0, 3.0, mean=template)
    image = Image(field, pixel_scale_mas=1.0)
    expected = template / template.max() + 1e-3
    assert onp.allclose(image.brightness, expected / expected.sum(), atol=1e-7)


def test_field_parameters_have_finite_gradients():
    template = onp.asarray(GaussianDisk(4.0).render(npix=16, fov_mas=16.0))
    image = Image(
        GaussianField(onp.zeros((16, 16)), 1.0, 2.0, mean=template), 1.0
    )
    u, v = onp.array([20.0, -35.0]), onp.array([15.0, 40.0])

    def power(sigma, length):
        field = image.log_brightness.set(
            ["sigma", "length_mas", "latent"],
            [sigma, length, np.ones((16, 16))],
        )
        return np.sum(
            np.abs(image.set("log_brightness", field).model(u, v, 1.6e-6)) ** 2
        )

    grads = jax.grad(power, argnums=(0, 1))(1.0, 2.0)
    assert all(onp.isfinite(float(g)) and float(g) != 0.0 for g in grads)


def test_a_gaussian_field_image_fits_by_least_squares_and_diagnoses():
    data = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])
    truth = System(
        star=PointSource(), env=GaussianDisk(3.0, dra=2.0, flux=0.4)
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(0))
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    field = GaussianField(onp.zeros((24, 24)), 2.0, 2.0, mean=template)
    start = System(star=PointSource(), env=Image(field, 1.0, flux=0.4))
    priors = image_priors(start)
    assert list(priors) == ["env.log_brightness.latent"]
    result = fit(start, priors, data)
    assert result.info["method"] == "lm" and result.info["converged"]
    first = fit(start, priors, data, max_steps=1)
    assert result.info["chi2_red"] < first.info["chi2_red"]
    assert diagnose(result.model, data).checks["anchored"]


def test_fitting_field_hyperparameters_by_map_warns():
    field = GaussianField(onp.zeros((8, 8)), 1.0, 2.0)
    start = System(star=PointSource(), env=Image(field, 1.0, flux=0.4))
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    priors = image_priors(start) | {
        "env.log_brightness.sigma": dist.Uniform(0.1, 5.0)
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(start, priors, data, max_steps=1)
    assert any(
        "hyperparameters of a Gaussian field" in str(w.message) for w in caught
    )
