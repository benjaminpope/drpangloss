import jax.numpy as np
import numpy as onp
import pytest

from drpangloss._geometry import image_coordinates
from drpangloss.models import GaussianDisk, Image
from drpangloss.scenes import gaussian_blob, ring, spiral

NPIX, SCALE = 64, 4.0
WAVEL = 4.8e-6


def _scenes():
    return {
        "ring": ring(NPIX, SCALE, 60.0, 8.0, 40.0, 30.0, 0.5, 120.0),
        "spiral": spiral(NPIX, SCALE, 50.0, 8.0, 2.5, 20.0, fade_mas=120.0),
        "blob": gaussian_blob(NPIX, SCALE, 10.0, 20.0, -30.0),
    }


@pytest.mark.parametrize("name", ["ring", "spiral", "blob"])
def test_scene_is_finite_and_unit_sum(name):
    image = _scenes()[name]
    assert image.shape == (NPIX, NPIX)
    assert np.all(np.isfinite(image)) and np.all(image >= 0.0)
    assert np.isclose(image.sum(), 1.0)


def test_ring_asymmetry_at_pa_90_is_brightest_on_the_east_left():
    image = ring(NPIX, SCALE, 60.0, 8.0, asymmetry=0.6, asymmetry_pa_deg=90.0)
    left, right = image[:, : NPIX // 2].sum(), image[:, NPIX // 2 :].sum()
    assert left > 1.5 * right
    assert np.isclose(image.sum(axis=0)[: NPIX // 2].sum(), left)
    # Rotating the asymmetry to North (PA 0) moves the flux to the top.
    north = ring(NPIX, SCALE, 60.0, 8.0, asymmetry=0.6)
    assert north[: NPIX // 2].sum() > 1.5 * north[NPIX // 2 :].sum()


def test_inclined_ring_is_narrower_across_its_major_axis():
    x, y = image_coordinates(NPIX, NPIX * SCALE)
    inc = 60.0

    def extent(image, direction):
        # rms extent along the sky direction at position angle `direction`
        angle = onp.deg2rad(direction)
        s = x * onp.sin(angle) + y * onp.cos(angle)
        return float(np.sqrt(np.sum(image * s**2)))

    # Major axis along PA 30 deg; the minor axis is at PA 120 deg.
    image = ring(NPIX, SCALE, 60.0, 4.0, inc_deg=inc, pa_deg=30.0)
    assert np.isclose(
        extent(image, 120.0) / extent(image, 30.0),
        np.cos(np.pi / 3),
        atol=0.02,
    )


def test_offset_scenes_land_on_the_east_left():
    blob = gaussian_blob(NPIX, SCALE, 8.0, dra=40.0)
    assert np.unravel_index(np.argmax(blob), blob.shape)[1] < NPIX // 2
    blob = gaussian_blob(NPIX, SCALE, 8.0, ddec=40.0)
    assert np.unravel_index(np.argmax(blob), blob.shape)[0] < NPIX // 2
    # A quarter turn starting due East runs East to South: lower left.
    arm = spiral(NPIX, SCALE, 200.0, 6.0, turns=0.25, pa_deg=90.0)
    rows, cols = np.nonzero(arm > 1e-3 * arm.max())
    # The rounded ends reach up to ~4 widths past the quadrant edges.
    margin = int(onp.ceil(4 * 6.0 / SCALE))
    assert np.all(cols <= NPIX // 2 + margin)
    assert np.all(rows >= NPIX // 2 - 1 - margin)
    x, y = image_coordinates(NPIX, NPIX * SCALE)
    assert np.sum(arm * x) > 0.0 and np.sum(arm * y) < 0.0  # East, South


def test_spiral_steps_outwards_by_step_per_turn():
    step = 60.0
    image = spiral(NPIX, SCALE, step, 3.0, turns=2.0, pa_deg=0.0)
    x, y = image_coordinates(NPIX, NPIX * SCALE)
    # Along the northern axis (PA 0) the arm crosses at r = step * k.
    column = image[:, NPIX // 2]
    heights = onp.asarray(y[:, NPIX // 2])
    is_peak = (column[1:-1] > column[:-2]) & (column[1:-1] > column[2:])
    peaks = heights[1:-1][is_peak & (heights[1:-1] > 0.0)]
    expected = np.array([step, 2 * step])
    assert np.allclose(np.sort(peaks)[:2], expected, atol=SCALE)


def test_blob_visibility_matches_gaussian_disk_offset():
    scale, npix = 2.0, 96
    blob = Image.from_brightness(
        gaussian_blob(npix, scale, 6.0, 10.0, -8.0), scale
    )
    disk = GaussianDisk(sigma=6.0, dra=10.0, ddec=-8.0)
    rng = onp.random.default_rng(0)
    u, v = rng.uniform(-4.0, 4.0, (2, 30))
    assert np.allclose(
        blob.model(u, v, WAVEL), disk.model(u, v, WAVEL), atol=2e-4
    )


def test_ring_rejects_asymmetry_beyond_one():
    with pytest.raises(ValueError, match="asymmetry"):
        ring(16, 1.0, 5.0, 1.0, asymmetry=1.5)
