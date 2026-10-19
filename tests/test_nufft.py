"""The nufftax transform (``ft="nufft"``) against the exact sum."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import _geometry
from virgil._geometry import (
    UVGrid,
    _nufft,
    _use_nufft,
    image_visibilities,
)
from virgil.likelihood import model_loglike
from virgil.models import FlaredDiskHG, Image
from tests._shared import vlti_night
from tests.test_image_model import MAS2RAD, _dft_reference

pytest.importorskip("nufftax")

WAVEL = 4.8e-6


def _problem(shape, n=200, seed=0, max_freq=2e6):
    rng = onp.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, shape)
    image /= image.sum()
    uu, vv = rng.uniform(-max_freq, max_freq, (2, n))
    return image, uu, vv


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("shape", [(17, 17), (16, 16), (12, 15), (128, 128)])
def test_nufft_matches_the_direct_sum(shape, x64):
    n = 50 if shape[0] > 100 else 200
    image, uu, vv = _problem(shape, n=n)
    expected = _dft_reference(image, uu, vv, 3.0)
    with jax.enable_x64(x64):
        got = _nufft(np.asarray(image), np.asarray(uu), np.asarray(vv), 3.0)
        # The image sums to one: absolute error is relative to V(0).
        error = onp.max(onp.abs(onp.asarray(got) - expected))
    # eps is 1e-5 in float32, as accurate as the float32 DFT.
    assert error < (1e-10 if x64 else 5e-5)


@pytest.mark.parametrize("shape", [(16, 16), (12, 15), (17, 17)])
def test_long_baselines_beyond_the_nufft_period_agree(shape):
    # |2π s u| up to ~27 rad: the even-axis half-pixel phase is 4π-periodic.
    with jax.enable_x64(True):
        image, uu, vv = _problem(shape, max_freq=3e7)
        scale = 30.0
        assert onp.max(onp.abs(2 * onp.pi * scale * MAS2RAD * uu)) > 3 * onp.pi
        expected = _dft_reference(image, uu, vv, scale)
        got = _nufft(np.asarray(image), np.asarray(uu), np.asarray(vv), scale)
        assert onp.max(onp.abs(onp.asarray(got) - expected)) < 1e-10


def test_gradient_matches_the_dft_gradient():
    image, uu, vv = _problem((16, 15), n=60)
    weights = np.asarray(onp.random.default_rng(3).normal(size=60))

    def objective(ft):
        def f(b):
            vis = image_visibilities(b, uu, vv, 3.0, ft)
            return np.sum(weights * np.abs(vis) ** 2)

        return jax.grad(f)(np.asarray(image))

    dft = objective("dft")
    assert np.max(np.abs(objective("nufft") - dft)) < 5e-5 * np.max(
        np.abs(dft)
    )


def test_forward_mode_matches_the_dft():
    # CLEAN, the image Jacobian, LM and Gauss-Newton differentiate forward.
    image, uu, vv = _problem((16, 15), n=60)
    image = np.asarray(image)
    tangent = np.asarray(onp.random.default_rng(4).normal(size=image.shape))

    def run(ft):
        def f(b):
            return image_visibilities(b, uu, vv, 3.0, ft)

        return jax.jvp(f, (image,), (tangent,))[1], jax.jacfwd(f)(image)

    for got, expected in zip(run("nufft"), run("dft")):
        assert np.max(np.abs(got - expected)) < 5e-4  # eps 1e-5 of |V| ~ 13


@pytest.mark.xfail(
    strict=True,
    reason="nufftax 0.6 cannot batch its vjp: reverse-over-reverse fails",
)
def test_hessian_through_the_nufft():
    image, uu, vv = _problem((8, 8), n=10)

    def f(b):
        return np.sum(np.abs(image_visibilities(b, uu, vv, 3.0, "nufft")) ** 2)

    jax.hessian(f)(np.asarray(image))


def test_use_nufft_decisions(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert _use_nufft((256, 256), 10_000)
    assert not _use_nufft((256, 256), 100)  # below the work threshold
    assert not _use_nufft((64, 64), 10**6)  # too small an image
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert _use_nufft((64, 64), 500)
    assert not _use_nufft((64, 64), 499)
    assert not _use_nufft((32, 32), 10**6)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert _use_nufft((128, 128), 1_000)  # any other backend counts as cpu


def test_auto_follows_the_decision(monkeypatch):
    image, uu, vv = _problem((16, 16), n=20)
    calls = []
    real = _geometry._nufft
    monkeypatch.setattr(
        _geometry, "_nufft", lambda *a: calls.append(1) or real(*a)
    )
    monkeypatch.setattr(_geometry, "_use_nufft", lambda shape, m: True)
    image_visibilities(np.asarray(image), uu, vv, 3.0)
    image_visibilities(np.asarray(image), uu, vv, 3.0, "dft")
    image_visibilities(np.asarray(image), uu, vv, 3.0, "nufft")
    assert len(calls) == 2
    with pytest.raises(ValueError, match="ft must be"):
        image_visibilities(np.asarray(image), uu, vv, 3.0, "fft")


def test_image_loglike_agrees_between_transforms():
    data = vlti_night(60000.0, frames=60, d_vis=0.05, d_phi=2.0)
    eta = onp.random.default_rng(1).normal(size=(16, 16))
    values = [
        float(model_loglike(Image(eta, 5.0, ft=ft), data))
        for ft in ("dft", "nufft")
    ]
    assert abs(values[0] - values[1]) < 1e-3 * abs(values[0])


def test_vmap_over_a_batch_of_images():
    _, uu, vv = _problem((16, 16), n=30)
    batch = onp.random.default_rng(5).uniform(size=(3, 16, 16))
    batch = np.asarray(batch / batch.sum(axis=(1, 2), keepdims=True))
    run = lambda ft: jax.vmap(  # noqa: E731
        lambda b: image_visibilities(b, uu, vv, 3.0, ft)
    )(batch)
    assert np.max(np.abs(run("nufft") - run("dft"))) < 3e-5


def test_a_matching_uv_grid_keeps_the_matrix_transform(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("NUFFT used on a lattice")

    monkeypatch.setattr(_geometry, "_nufft", boom)
    monkeypatch.setattr(_geometry, "_use_nufft", lambda shape, m: True)
    axis = np.linspace(-3.0, 3.0, 5)
    grid = UVGrid(axis, axis, np.arange(25), 0.0)
    u, v = np.meshgrid(axis, axis)
    image = Image(onp.random.default_rng(0).normal(size=(8, 8)), 4.0)
    vis = image.model_on_grid(u.ravel(), v.ravel(), np.asarray(WAVEL), grid)
    assert vis.shape == (25,)


def test_absent_nufftax(monkeypatch):
    def missing():
        raise ImportError('needs nufftax: pip install "virgil-astro[nufft]"')

    monkeypatch.setattr(_geometry, "_nufftax", missing)
    image, uu, vv = _problem((16, 16), n=20)
    image = np.asarray(image)
    with pytest.raises(ImportError, match=r"virgil-astro\[nufft\]"):
        image_visibilities(image, uu, vv, 3.0, "nufft")
    # auto falls back to the exact sum, silently when the problem is small ...
    assert not _use_nufft((16, 16), 20)
    # ... and with a warning when nufftax would have been faster.
    with pytest.warns(UserWarning, match="nufftax would be faster"):
        assert not _use_nufft((256, 256), 10**6)


def test_flared_disk_takes_ft(monkeypatch):
    kwargs = dict(
        g=0.3, radius=40.0, fwhm=20.0, inc=40.0, pa=10.0,
        npix=16, pixel_scale_mas=8.0,
    )  # fmt: skip
    _, uu, vv = _problem((2, 2), n=20)
    u, v = uu * WAVEL, vv * WAVEL
    a = FlaredDiskHG(ft="dft", **kwargs).model(u, v, WAVEL)
    b = FlaredDiskHG(ft="nufft", **kwargs).model(u, v, WAVEL)
    assert np.max(np.abs(a - b)) < 3e-5
