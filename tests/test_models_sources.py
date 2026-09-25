import numpy as onp
import pytest
import jax.numpy as np
from scipy.special import j0 as scipy_j0
from scipy.special import j1 as scipy_j1
from scipy.special import jn_zeros

from drpangloss.models import (
    BinaryModelCartesian,
    GaussianDiskModel,
    HarmonixModel,
    ModulatedGaussianRimModel,
    UniformDiskModel,
    _image_coordinates,
    cvis_gaussian_disk,
    cvis_gaussian_rim,
    cvis_uniform_disk,
)
from tests._test_data import oidata

# Independent reference conversion (not imported from drpangloss) so the
# analytic checks below don't just re-test the module's own constant.
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0


def test_cvis_gaussian_disk_is_well_behaved():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_gaussian_disk(uu, vv, sigma=20.0, flux=0.1, dra=5.0, ddec=-3.0)
    assert cvis.shape == uu.shape
    assert np.all(np.isfinite(cvis))
    assert np.all(np.abs(cvis) <= 1.0 + 1e-12)


def test_gaussian_disk_oidata_and_render():
    model = GaussianDiskModel(sigma=30.0, flux=0.1, dra=10.0, ddec=-10.0)
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_gaussian_disk_render_remains_finite_for_narrow_shifted_disk():
    image = GaussianDiskModel(sigma=1e-6, flux=0.1, dra=1e6, ddec=-1e6).render(
        npix=32, fov_mas=20.0
    )

    assert image.shape == (32, 32)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_cvis_uniform_disk_is_well_behaved():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_uniform_disk(uu, vv, ud=5.0, dra=5.0, ddec=-3.0)
    assert cvis.shape == uu.shape
    assert np.all(np.isfinite(cvis))
    assert np.all(np.abs(cvis) <= 1.0 + 1e-12)


def test_cvis_uniform_disk_zero_baseline_is_unity():
    cvis = cvis_uniform_disk(np.array([0.0]), np.array([0.0]), ud=10.0)
    assert np.allclose(cvis, 1.0 + 0j)


def test_cvis_uniform_disk_matches_analytic_airy_formula():
    """Visibility amplitude should follow 2*J1(pi*theta*B/lambda) /
    (pi*theta*B/lambda), with theta the disk diameter in mas and B/lambda
    the baseline length in wavelength units (here passed directly as
    ``u``, ``v``). Checked against an independent scipy.special.j1 call,
    not against the module's own Bessel implementation.
    """
    ud = 8.0
    base_norm = onp.array([5.0, 20.0, 45.0, 80.0])

    cvis = cvis_uniform_disk(
        np.asarray(base_norm), np.zeros_like(base_norm), ud
    )

    kernel = onp.pi * ud * _MAS2RAD_REF * base_norm
    expected = 2.0 * scipy_j1(kernel) / kernel

    assert onp.allclose(onp.asarray(cvis).real, expected, atol=1e-8)
    assert onp.allclose(onp.asarray(cvis).imag, 0.0, atol=1e-8)


def test_cvis_uniform_disk_vanishes_at_first_airy_null():
    ud = 8.0
    first_null_kernel = jn_zeros(1, 1)[0]
    base_norm = first_null_kernel / (onp.pi * ud * _MAS2RAD_REF)

    cvis = cvis_uniform_disk(np.asarray(base_norm), np.asarray(0.0), ud)

    assert onp.abs(onp.asarray(cvis)) < 1e-6


def test_cvis_uniform_disk_converges_to_point_source_for_small_ud():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_uniform_disk(uu, vv, ud=1e-6)
    assert np.allclose(cvis, 1.0 + 0j, atol=1e-6)


def test_uniform_disk_oidata_and_render():
    model = UniformDiskModel(ud=30.0, dra=10.0, ddec=-10.0)
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_uniform_disk_render_uses_interferometric_image_orientation():
    image = UniformDiskModel(ud=1e-3, dra=2.0, ddec=2.0).render(
        npix=5, fov_mas=10.0
    )

    assert onp.unravel_index(onp.asarray(image).argmax(), image.shape) == (
        1,
        1,
    )


def test_cvis_gaussian_rim_is_well_behaved():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_gaussian_rim(
        uu,
        vv,
        dra=5.0,
        ddec=-3.0,
        diam=20.0,
        fwhm=2.0,
        inc=30.0,
        pa=45.0,
        az_amps=np.array([0.3]),
        az_phis=np.array([10.0]),
        flux=0.5,
    )
    assert cvis.shape == uu.shape
    assert np.all(np.isfinite(cvis))
    # |1 + 0.3*cos(theta)| never drops below 0.7, i.e. flux stays
    # non-negative everywhere, so the rim's own visibility magnitude stays
    # bounded by 1, and the point-source mixture preserves that bound.
    assert np.all(np.abs(cvis) <= 1.0 + 1e-8)


def test_cvis_gaussian_rim_zero_flux_is_pure_point_source():
    """With flux=0 the rim contributes no light, so the visibility should
    be exactly that of an unresolved point source (unity everywhere),
    regardless of the rim's own geometry.
    """
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_gaussian_rim(
        uu,
        vv,
        dra=5.0,
        ddec=-3.0,
        diam=20.0,
        fwhm=2.0,
        inc=30.0,
        pa=45.0,
        az_amps=np.array([0.3]),
        az_phis=np.array([10.0]),
        flux=0.0,
    )
    assert np.allclose(cvis, 1.0 + 0j)


def test_cvis_gaussian_rim_symmetric_case_matches_bessel_j0():
    """An unmodulated, uninclined, infinitely-narrow rim is a plain thin
    ring, whose visibility is the classic J0(2*pi*r0*B/lambda) form; mixed
    with an unresolved point source (visibility 1 everywhere) via the
    flux (rim/star) contrast convention, the total visibility should be
    l1 + l2*J0(...). Checked against an independent scipy.special.j0 call.
    """
    diam = 20.0
    flux = 3.0
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel

    cvis = cvis_gaussian_rim(
        uu,
        vv,
        dra=0.0,
        ddec=0.0,
        diam=diam,
        fwhm=1e-6,
        inc=0.0,
        pa=0.0,
        az_amps=np.array([]),
        az_phis=np.array([]),
        flux=flux,
    )

    base_norm = onp.hypot(onp.asarray(uu), onp.asarray(vv))
    rim_expected = scipy_j0(
        2.0 * onp.pi * base_norm * (diam / 2.0) * _MAS2RAD_REF
    )
    l2 = flux / (flux + 1.0)
    l1 = 1.0 - l2
    expected = l1 + l2 * rim_expected

    assert onp.allclose(onp.asarray(cvis).real, expected, atol=1e-6)
    assert onp.allclose(onp.asarray(cvis).imag, 0.0, atol=1e-6)


def test_modulated_gaussian_rim_oidata_and_render():
    model = ModulatedGaussianRimModel(
        diam=30.0,
        fwhm=3.0,
        inc=25.0,
        pa=60.0,
        az_amps=np.array([0.2]),
        az_pas=np.array([15.0]),
        flux=0.5,
        dra=5.0,
        ddec=-5.0,
    )
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_modulated_gaussian_rim_symmetric_case_is_finite_and_normalized():
    image = ModulatedGaussianRimModel(
        diam=40.0, fwhm=2.0, inc=0.0, pa=0.0, flux=0.5
    ).render(npix=64, fov_mas=100.0)

    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_modulated_gaussian_rim_render_at_zero_flux_is_point_source_only():
    """With flux=0 the rim contributes no light, so the rendered image
    should peak exactly on the central (unresolved point-source) pixel,
    with none appearing out at the rim's own radius.
    """
    npix = 65
    fov_mas = 100.0
    diam = 40.0
    image = onp.asarray(
        ModulatedGaussianRimModel(
            diam=diam, fwhm=2.0, inc=0.0, pa=0.0, flux=0.0
        ).render(npix=npix, fov_mas=fov_mas)
    )
    center = npix // 2

    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)
    assert onp.unravel_index(image.argmax(), image.shape) == (center, center)

    # No flux should appear near the rim's own radius (diam/2 from center),
    # since flux=0 removes the rim entirely.
    pixel_scale_mas = fov_mas / npix
    ring_px = int(round((diam / 2.0) / pixel_scale_mas))
    assert image[center, center + ring_px] < 1e-6


def test_cvis_gaussian_rim_stays_finite_at_edge_on_inclination():
    # inc=90deg drives stretch = cos(inc) to 0; the 1e-8 floor on stretch
    # should keep the Fourier-domain visibility finite (unlike the
    # pixel-mask render below, this doesn't depend on grid resolution).
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_gaussian_rim(
        uu,
        vv,
        dra=0.0,
        ddec=0.0,
        diam=40.0,
        fwhm=2.0,
        inc=90.0,
        pa=0.0,
        az_amps=np.array([]),
        az_phis=np.array([]),
        flux=0.5,
    )
    assert np.all(np.isfinite(cvis))


def test_modulated_gaussian_rim_render_finite_at_moderate_inclination():
    # A close-to-edge-on render (e.g. inc=90) can legitimately miss the
    # (near-)zero-measure ring on a coarse pixel grid; that's a
    # rasterization limitation of the mask-based render, not a numerical
    # blow-up, so this checks a realistic, non-degenerate inclination.
    image = ModulatedGaussianRimModel(
        diam=40.0, fwhm=2.0, inc=60.0, pa=0.0, flux=0.5
    ).render(npix=64, fov_mas=100.0)

    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("az_pas", "bright_half", "faint_half"),
    [
        # PA=90deg modulation -> bright towards East (+x, low column index).
        (90.0, "east", "west"),
        # PA=0deg modulation -> bright towards North (+y, low row index).
        (0.0, "north", "south"),
    ],
)
def test_modulated_gaussian_rim_render_follows_north_to_east_pa_convention(
    az_pas, bright_half, faint_half
):
    image = onp.asarray(
        ModulatedGaussianRimModel(
            diam=40.0,
            fwhm=2.0,
            inc=0.0,
            pa=0.0,
            az_amps=np.array([1.0]),
            az_pas=np.array([az_pas]),
            flux=0.5,
        ).render(npix=81, fov_mas=100.0)
    )

    center = image.shape[0] // 2
    halves = {
        "east": image[:, :center].sum(),
        "west": image[:, center + 1 :].sum(),
        "north": image[:center, :].sum(),
        "south": image[center + 1 :, :].sum(),
    }
    assert halves[bright_half] > halves[faint_half]


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
    image = GaussianDiskModel(sigma=1e-3, flux=10.0, dra=2.0, ddec=2.0).render(
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
