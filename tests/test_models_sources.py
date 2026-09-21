import numpy as onp
import pytest
import jax.numpy as np

from drpangloss.models import (
    BinaryModelCartesian,
    GaussianDiskModel,
    HarmonixModel,
    _image_coordinates,
    cvis_gaussian_disk,
)
from tests._test_data import oidata


def test_cvis_gaussian_disk_is_well_behaved():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_gaussian_disk(uu, vv, sigma=20.0, dra=5.0, ddec=-3.0)
    assert cvis.shape == uu.shape
    assert np.all(np.isfinite(cvis))
    assert np.all(np.abs(cvis) <= 1.0 + 1e-12)


def test_gaussian_disk_oidata_and_render():
    model = GaussianDiskModel(sigma=30.0, dra=10.0, ddec=-10.0)
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_gaussian_disk_render_remains_finite_for_narrow_shifted_disk():
    image = GaussianDiskModel(sigma=1e-6, dra=1e6, ddec=-1e6).render(
        npix=32, fov_mas=20.0
    )

    assert image.shape == (32, 32)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_binary_render_is_available():
    image = BinaryModelCartesian(10.0, -5.0, 1e-3).render(
        npix=32, fov_mas=80.0
    )
    assert image.shape == (32, 32)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("npix", "fov_mas", "expected"),
    [
        (4, 8.0, onp.array([3.0, 1.0, -1.0, -3.0])),
        (5, 10.0, onp.array([4.0, 2.0, 0.0, -2.0, -4.0])),
    ],
)
def test_image_coordinates_use_pixel_centers(npix, fov_mas, expected):
    xx, yy = _image_coordinates(npix, fov_mas)

    assert xx.shape == (npix, npix)
    assert yy.shape == (npix, npix)
    assert onp.allclose(onp.asarray(xx[0]), expected)
    assert onp.allclose(onp.asarray(yy[:, 0]), expected)


def test_gaussian_disk_render_uses_interferometric_image_orientation():
    image = GaussianDiskModel(sigma=1e-3, dra=2.0, ddec=2.0).render(
        npix=5, fov_mas=10.0
    )

    assert onp.unravel_index(onp.asarray(image).argmax(), image.shape) == (
        1,
        1,
    )


def test_harmonix_model_for_external_visibility_models():
    class MockHarmonix:
        def __init__(self):
            self.calls = []

        def visibility(self, uu, vv, time):
            self.calls.append((uu, vv, time))
            return np.exp(-1e-6 * (uu**2 + vv**2))

        def render(self, npix, fov_mas):
            _ = fov_mas
            return np.ones((npix, npix))

    mock = MockHarmonix()
    wrapped = HarmonixModel(
        mock,
        visibility_method="visibility",
        observation_time=0.25,
    )
    cvis = wrapped.model(oidata.u, oidata.v, oidata.wavel)
    image = wrapped.render(npix=20, fov_mas=100.0)
    uu, vv, time = mock.calls[0]

    assert cvis.shape == oidata.u.shape
    assert np.all(np.isfinite(cvis))
    assert np.allclose(uu, oidata.u / oidata.wavel)
    assert np.allclose(vv, oidata.v / oidata.wavel)
    assert time == 0.25
    assert image.shape == (20, 20)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harmonix_model_random_spherical_harmonics_are_finite(seed):
    harmonix_module = pytest.importorskip(
        "harmonix.harmonix",
        reason="harmonix integration tests require a compatible harmonix install",
    )
    starry_module = pytest.importorskip("jaxoplanet.starry")

    rng = onp.random.default_rng(seed)
    degree = 3
    coeffs = onp.concatenate(
        ([1.0], 0.1 * rng.normal(size=(degree + 1) ** 2 - 1))
    )
    surface = starry_module.Surface(
        y=starry_module.Ylm.from_dense(np.asarray(coeffs)),
        inc=np.pi / 2.0,
        obl=0.0,
        period=1.0,
    )
    wrapped = HarmonixModel(
        harmonix_module.Harmonix(surface, 1.0),
        observation_time=0.0,
    )

    u = np.linspace(90.0, 190.0, 10)
    v = np.zeros_like(u)
    wavel = np.full_like(u, 1e-6)

    cvis = wrapped.model(u, v, wavel)
    image = wrapped.render(npix=64, fov_mas=20.0)

    assert cvis.shape == (10,)
    assert np.all(np.isfinite(cvis))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)
