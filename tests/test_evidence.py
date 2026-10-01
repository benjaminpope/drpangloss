import jax
import jax.numpy as np
import numpy as onp
import pytest

from drpangloss.coverage import vlti_oidata
from drpangloss.fields import GaussianField
from drpangloss.fitting import fit
from drpangloss.imaging import MaxEntropy, image_priors, l_curve, log_evidence
from drpangloss.models import GaussianDisk, Image, PointSource, System

DATA = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])
N, H = 16, 1.0
TEMPLATE = onp.asarray(GaussianDisk(4.0).render(N, N * H))


def _gp_scene(latent, sigma, length=2.0):
    field = GaussianField(latent, sigma, length, mean=TEMPLATE)
    return System(star=PointSource(), env=Image(field, H, flux=0.4))


def test_the_evidence_prefers_the_field_amplitude_the_data_came_from():
    latent = jax.random.normal(jax.random.PRNGKey(0), (N, N))
    data = DATA.with_model(_gp_scene(latent, 1.5), key=jax.random.PRNGKey(1))
    evidence = {}
    for sigma in (0.3, 1.5, 6.0):
        start = _gp_scene(onp.zeros((N, N)), sigma)
        result = fit(start, image_priors(start), data)
        evidence[sigma] = log_evidence(result.model, data)
    assert max(evidence, key=evidence.get) == 1.5


def test_the_evidence_needs_a_gaussian_field():
    scene = System(
        star=PointSource(), env=Image.from_model(GaussianDisk(3.0), N, H)
    )
    with pytest.raises(TypeError, match="GaussianField"):
        log_evidence(scene, DATA)


def test_classic_maxent_lies_inside_the_sweep_and_is_bracketed():
    truth = System(
        star=PointSource(), env=GaussianDisk(3.0, dra=2.0, flux=0.4)
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(2))
    start = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(4.0), N, H, flux=0.4),
    )
    weights = np.logspace(-1, 3, 5)
    curve = l_curve(
        start, image_priors(start), data, MaxEntropy(1.0, path="env"), weights
    )
    weight = curve.classic_maxent(data)
    assert weight is not None and 0.1 < weight < 1000.0
    # A sweep confined to very strong weights does not bracket it.
    strong = l_curve(
        start,
        image_priors(start),
        data,
        MaxEntropy(1.0, path="env"),
        [3e4, 1e5],
    )
    assert strong.classic_maxent(data) is None
