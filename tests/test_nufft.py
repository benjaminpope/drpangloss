import jax
import jax.numpy as np
import numpy as onp
import pytest
from jax.test_util import check_grads

from drpangloss._geometry import image_visibilities
from drpangloss.models import Image, PointSource, System

pytest.importorskip("jax_finufft")

WAVEL = 4.8e-6


def _problem(shape, n=400, seed=0):
    rng = onp.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, shape)
    image /= image.sum()
    u, v = rng.uniform(-6.5, 6.5, (2, n))
    return np.asarray(image), u / WAVEL, v / WAVEL


@pytest.mark.parametrize("shape", [(17, 17), (16, 16), (12, 15)])
@pytest.mark.parametrize("x64", [False, True])
def test_nufft_matches_dft(shape, x64):
    with jax.enable_x64(x64):
        image, uu, vv = _problem(shape)
        image = image.astype(float)
        dft = image_visibilities(image, uu, vv, 10.0, "dft")
        nufft = image_visibilities(image, uu, vv, 10.0, "nufft")
        eps = 1e-7 if x64 else 1e-5
        # FINUFFT bounds each point by eps * sum|I| = eps * V(0).
        assert np.max(np.abs(nufft - dft)) <= 3 * eps


def test_nufft_phases_on_faint_visibilities_in_float64():
    with jax.enable_x64(True):
        image, uu, vv = _problem((32, 32), n=2000)
        image = image.astype(float)
        dft = image_visibilities(image, uu, vv, 10.0, "dft")
        nufft = image_visibilities(image, uu, vv, 10.0, "nufft")
        faint = np.abs(dft) < np.quantile(np.abs(dft), 0.1)
        phase_error = np.abs(np.angle(nufft[faint] / dft[faint]))
        assert np.max(phase_error) < 3e-7 / np.min(np.abs(dft[faint]))


def test_nufft_orientation_matches_point_source():
    npix, scale = 16, 5.0
    log_b = np.full((npix, npix), -np.inf).at[2, 12].set(0.0)
    image = Image(log_b, scale, backend="nufft")
    offsets = (0.5 * (npix - 1) - onp.arange(npix)) * scale
    point = PointSource(dra=offsets[12], ddec=offsets[2])
    _, uu, vv = _problem((1, 1))
    u, v = uu * WAVEL, vv * WAVEL
    assert np.allclose(
        image.model(u, v, WAVEL), point.model(u, v, WAVEL), atol=3e-5
    )


def test_nufft_gradients():
    with jax.enable_x64(True):
        image, uu, vv = _problem((9, 9), n=20)
        image = image.astype(float)

        def loss(img):
            vis = image_visibilities(img, uu, vv, 10.0, "nufft")
            return np.sum(np.abs(vis) ** 2)

        check_grads(loss, (image,), order=1, modes=["rev"], eps=1e-4)


def test_nufft_under_vmap():
    image, uu, vv = _problem((16, 16), n=50)
    stack = np.stack([image, image[::-1]])
    batched = jax.vmap(
        lambda img: image_visibilities(img, uu, vv, 10.0, "nufft")
    )(stack)
    single = image_visibilities(stack[1], uu, vv, 10.0, "nufft")
    assert np.allclose(batched[1], single, atol=1e-6)


def test_nufft_image_in_a_system_matches_dft():
    rng = onp.random.default_rng(4)
    log_b = rng.normal(size=(24, 24))

    def scene(backend):
        env = Image(log_b, 8.0, flux=0.3, backend=backend)
        return System(star=PointSource(), env=env)

    _, uu, vv = _problem((1, 1), n=100)
    u, v = uu * WAVEL, vv * WAVEL
    assert np.allclose(
        scene("nufft").model(u, v, WAVEL),
        scene("dft").model(u, v, WAVEL),
        atol=3e-5,
    )


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        Image(np.zeros((4, 4)), 1.0, backend="fft")
