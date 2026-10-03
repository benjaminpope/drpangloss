"""Sampling a Gaussian-field image with numpyro's NUTS (Stage 5c)."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, init_to_value

from drpangloss.coverage import vlti_oidata
from drpangloss.fields import GaussianField
from drpangloss.fitting import fit, gauss_newton_mass
from drpangloss.imaging import image_priors
from drpangloss.likelihood import numpyro_model
from drpangloss.models import GaussianDisk, Image, PointSource, System

N, H = 16, 1.0
TEMPLATE = onp.asarray(GaussianDisk(4.0).render(N, N * H))


def _scene(latent, sigma, length):
    field = GaussianField(latent, sigma, length, mean=TEMPLATE)
    return System(star=PointSource(), env=Image(field, H, flux=0.4))


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="NUTS on an image needs x64"
)
def test_nuts_recovers_the_amplitude_of_an_injected_field():
    truth = _scene(jax.random.normal(jax.random.PRNGKey(10), (N, N)), 1.5, 2.0)
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    ).with_model(truth, key=jax.random.PRNGKey(100))
    start = _scene(onp.zeros((N, N)), 1.0, 2.0)
    priors = image_priors(start) | {
        "env.log_brightness.sigma": dist.LogNormal(0.0, 1.0),
        "env.log_brightness.length_mas": dist.LogNormal(np.log(2.0), 0.5),
    }
    # Start at the MAP image, with the hyperparameters at the prior medians.
    latent = fit(start, image_priors(start), data).values
    init = latent | {
        "env.log_brightness.sigma": 1.0,
        "env.log_brightness.length_mas": 2.0,
    }
    mcmc = MCMC(
        NUTS(
            numpyro_model(start, priors, data),
            init_strategy=init_to_value(values=init),
        ),
        num_warmup=300,
        num_samples=300,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0), extra_fields=("diverging",))
    sigma = onp.asarray(mcmc.get_samples()["env.log_brightness.sigma"])
    low, high = onp.percentile(sigma, [5, 95])
    assert low < 1.5 < high
    assert int(mcmc.get_extra_fields()["diverging"].sum()) <= 5


def test_gauss_newton_mass_matches_the_curvature_and_its_layout():
    # Fixed σ and ℓ, flux sampled: one dense block over both sites, in the
    # order of the priors, whose inverse is JᵀJ (+ I from the latents'
    # standard-normal priors) in numpyro's unconstrained coordinates.
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    ).with_model(_scene(onp.zeros((N, N)), 1.5, 2.0))
    scene = _scene(onp.zeros((N, N)), 1.5, 2.0)
    priors = image_priors(scene) | {"env.flux": dist.Uniform(0.0, 1.0)}
    result = fit(scene, priors, data)
    mass = gauss_newton_mass(scene, priors, data, result.values)
    (paths,) = mass["dense_mass"]
    assert set(paths) == set(priors)
    assert mass["adapt_mass_matrix"] is False
    covariance = mass["inverse_mass_matrix"][paths]
    assert covariance.shape == (N * N + 1, N * N + 1)
    onp.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
    # With no data, each latent's variance would be 1; the data shrink it.
    # Sites are laid out in the block's order; the flux is one number.
    start = 1 if paths[0] == "env.flux" else 0
    variances = onp.diag(covariance)[start : start + N * N]
    assert onp.all(variances <= 1.0 + 1e-9) and variances.min() < 0.5


def test_gauss_newton_mass_rejects_an_unconstrained_parameter():
    # A zero-flux companion's position leaves the data unchanged.
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    scene = System(star=PointSource(), c=PointSource(flux=0.0, dra=5.0))
    priors = {"c.dra": dist.Uniform(-10.0, 10.0)}
    with pytest.raises(ValueError, match="singular"):
        gauss_newton_mass(scene, priors, data, {"c.dra": 5.0})


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="NUTS on an image needs x64"
)
def test_gauss_newton_mass_shortens_nuts_trajectories():
    # The data make the posterior much narrower in the measured directions
    # than in the rest, so plain NUTS needs long trajectories (255 steps
    # per draw here); the Gauss–Newton mass matrix makes them short (31).
    truth = _scene(jax.random.normal(jax.random.PRNGKey(10), (N, N)), 1.5, 2.0)
    data = vlti_oidata(
        hour_angles_h=(-3.0, -1.5, 0.0, 1.5, 3.0),
        wavelengths_m=[3.2e-6, 3.5e-6, 3.8e-6],
    ).with_model(truth, key=jax.random.PRNGKey(100))
    scene = _scene(onp.zeros((N, N)), 1.5, 2.0)
    priors = image_priors(scene) | {"env.flux": dist.Uniform(0.0, 1.0)}
    result = fit(scene, priors, data)

    def median_steps(**kw):
        mcmc = MCMC(
            NUTS(
                numpyro_model(result.model, priors, data),
                init_strategy=init_to_value(values=result.values),
                **kw,
            ),
            num_warmup=100,
            num_samples=100,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(1), extra_fields=("num_steps",))
        return float(onp.median(mcmc.get_extra_fields()["num_steps"]))

    plain = median_steps()
    whitened = median_steps(
        **gauss_newton_mass(scene, priors, data, result.values)
    )
    assert whitened < plain / 4
