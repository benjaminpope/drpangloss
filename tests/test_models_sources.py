import jax.numpy as np

from drpangloss.models import (
    BinaryModelCartesian,
    GaussianDiskModel,
    HarmonixAdapter,
    cvis_gaussian_disk,
)
from tests._test_data import oidata


def test_cvis_gaussian_disk_is_well_behaved():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_gaussian_disk(uu, vv, fwhm=20.0, dra=5.0, ddec=-3.0)
    assert cvis.shape == uu.shape
    assert np.all(np.isfinite(cvis))
    assert np.all(np.abs(cvis) <= 1.0 + 1e-12)


def test_gaussian_disk_oidata_and_render():
    model = GaussianDiskModel(fwhm=30.0, dra=10.0, ddec=-10.0)
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_binary_render_is_available():
    image = BinaryModelCartesian(10.0, -5.0, 1e-3).render(npix=32, fov_mas=80.0)
    assert image.shape == (32, 32)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_harmonix_adapter_for_external_visibility_models():
    class MockHarmonix:
        def visibility(self, uu, vv):
            return np.exp(-1e-6 * (uu**2 + vv**2))

        def render(self, npix, fov_mas):
            _ = fov_mas
            return np.ones((npix, npix))

    wrapped = HarmonixAdapter(
        MockHarmonix(),
        visibility_method="visibility",
        expects_wavelength_units=False,
    )
    cvis = wrapped.model(oidata.u, oidata.v, oidata.wavel)
    image = wrapped.render(npix=20, fov_mas=100.0)

    assert cvis.shape == oidata.u.shape
    assert np.all(np.isfinite(cvis))
    assert image.shape == (20, 20)
    assert np.all(image == 1.0)
