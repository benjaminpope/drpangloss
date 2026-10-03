"""Sampling a Gaussian-field image with numpyro's NUTS (Stage 5c)."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, init_to_value

from drpangloss.coverage import vlti_oidata
from drpangloss.fields import GaussianField
from drpangloss.fitting import fit
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
