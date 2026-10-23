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


def _bold(ax, colour):
    return next(
        line
        for line in ax.get_lines()
        if line.get_linewidth() == 1.8 and line.get_color() == colour
    )


def test_cut_orientation():
    from virgil.plotting import _cut

    s = onp.arange(-8.0, 9.0)
    north, east = onp.zeros((21, 21)), onp.zeros((21, 21))
    north[7, 10] = 1.0  # 3 pixels North of the centre
    east[10, 6] = 1.0  # 4 pixels East of the centre
    assert s[onp.argmax(_cut(north, 0.0, 1.0, s))] == 3.0
    assert s[onp.argmax(_cut(east, 90.0, 1.0, s))] == 4.0


def test_multipole_harmonic_track(monkeypatch):
    field = _mp()
    theta = np.linspace(0.0, 2 * np.pi, field.shape[1], endpoint=False)
    amp = 0.7
    monkeypatch.setattr(
        MultipoleField,
        "polar_values",
        lambda self: amp
        * np.cos(2 * theta)[None, :]
        * np.ones((field.shape[0], 1)),
    )
    _, ax = plot_field_draws(field, 1.0, KEY, n=4, mmax=3, n_cuts=0)
    step = 2.5 * float(field.sigma)
    for m, expected in enumerate((0.0, 0.0, amp, 0.0)):
        y = _bold(ax["d"], f"C{m}").get_ydata()
        onp.testing.assert_allclose(y, expected - step * m, atol=1e-6)


def test_posterior_principal_cut_is_log_brightness():
    field = _gp()
    z = jax.random.normal(KEY, (4, 16, 16))
    _, ax = plot_field_draws(field, 0.5, latents=z, n_cuts=0)
    img = onp.exp(
        onp.asarray(
            field.__class__(z[0], field.sigma, field.length_mas).evaluate(0.5)
        )
    )
    expected = onp.log10(onp.maximum(img[:, 8] / img.max(), 1e-3))
    onp.testing.assert_allclose(
        _bold(ax["d"], "C0").get_ydata(), expected, atol=1e-5
    )


def test_truth_needs_posterior():
    with pytest.raises(ValueError, match="truth"):
        plot_field_draws(_gp(), 0.5, KEY, n=4, truth=onp.ones((16, 16)))
