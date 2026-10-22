"""Smoke tests of plot_field_draws: every cut mode, prior and posterior input."""

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import jax.numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as onp  # noqa: E402
import pytest  # noqa: E402

from virgil.imaging import Beam  # noqa: E402
from virgil.fields import GaussianField, MultipoleField  # noqa: E402
from virgil.plotting import plot_field_draws  # noqa: E402

KEY = jax.random.PRNGKey(0)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _gp():
    return GaussianField(
        np.zeros((16, 16)), sigma=1.0, length_mas=np.array([4.0, 1.0])
    )


def _mp():
    return MultipoleField(
        np.zeros((6, 12)),
        image_shape=(24, 24),
        r_max_mas=10.0,
        inc=40.0,
        pa=30.0,
    )


def test_prior_principal():
    fig, ax = plot_field_draws(_gp(), 0.5, KEY, n=6, n_cuts=3)
    assert set(ax) == {"a", "b", "c", "d"}
    assert len(ax["d"].get_lines()) >= 2


def test_prior_axes():
    fig, ax = plot_field_draws(
        _gp(), 0.5, KEY, n=5, cuts="axes", axes_pa=((30, "major"),)
    )
    assert ax["d"].get_legend() is not None


def test_prior_multipole():
    fig, ax = plot_field_draws(_mp(), 1.0, KEY, n=5, mmax=2, n_cuts=2)
    assert ax["d"].get_xlabel() == "disk-plane radius [mas]"


def test_posterior_latents_and_images():
    field = _gp()
    z = jax.random.normal(KEY, (5, 16, 16))
    plot_field_draws(field, 0.5, latents=z, snr=True)
    imgs = onp.exp(onp.random.default_rng(0).normal(size=(5, 16, 16)))
    imgs[:, :2] = imgs[0, :2]  # no spread: masked in the S/N map
    fig, ax = plot_field_draws(
        images=imgs,
        pixel_scale_mas=0.5,
        truth=imgs[0],
        snr=True,
        snr_scale="log",
        beam=Beam(major_mas=2.0, minor_mas=1.0, pa_deg=20.0),
        axes_pa=((30, "major"), (120, "minor")),
    )
    assert "c" in ax


def test_posterior_multipole_latents():
    z = jax.random.normal(KEY, (4, 6, 12))
    plot_field_draws(_mp(), 1.0, latents=z, n_cuts=2)


def test_errors():
    with pytest.raises(ValueError, match="key"):
        plot_field_draws(_gp(), 0.5)
    with pytest.raises(ValueError, match="MultipoleField"):
        plot_field_draws(_gp(), 0.5, KEY, n=4, cuts="multipole")
    with pytest.raises(ValueError, match="posterior"):
        plot_field_draws(_gp(), 0.5, KEY, n=4, snr=True)
    with pytest.raises(ValueError, match="field or images"):
        plot_field_draws(pixel_scale_mas=0.5)
