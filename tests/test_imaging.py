import jax
import jax.numpy as np
import numpy as onp
import pytest

from drpangloss.amigo import simulated_disco_record
from drpangloss.fitting import Problem, fit
from drpangloss.imaging import (
    TSV,
    TV,
    Centroid,
    LCurve,
    MaxEntropy,
    image_priors,
    l_curve,
    nyquist_pixel_scale,
)
from drpangloss.models import GaussianDisk, Image, PointSource, System
from drpangloss.oidata import OIData
from drpangloss.scenes import gaussian_blob

NPIX, SCALE = 16, 12.0
DATA = OIData(simulated_disco_record(max_baseline_m=4.0))


def _image(brightness, **kwargs):
    return Image.from_brightness(brightness, SCALE, **kwargs)


def test_tsv_and_tv_penalise_edges_and_structure():
    flat = _image(np.ones((NPIX, NPIX)))
    blob = _image(gaussian_blob(NPIX, SCALE, 15.0))
    # A flat image only has steps at its (zero-padded) edges.
    b = flat.brightness
    assert np.isclose(TSV(1.0).value(flat), 4 * NPIX * b[0, 0] ** 2, rtol=1e-4)
    for regulariser in (TSV(2.0), TV(2.0)):
        assert regulariser.value(blob) > 0.0
        assert np.isclose(
            regulariser.value(blob), 2.0 * type(regulariser)(1.0).value(blob)
        )
    tsv = TSV(3.0)
    assert np.isclose(0.5 * np.sum(tsv.residuals(blob) ** 2), tsv.value(blob))


def test_max_entropy_is_zero_at_the_default_image():
    blob = _image(gaussian_blob(NPIX, SCALE, 15.0))
    assert np.isclose(
        MaxEntropy(1.0, prior=blob.brightness).value(blob), 0.0, atol=1e-5
    )
    assert MaxEntropy(1.0).value(blob) > 0.0
    flat = _image(np.ones((NPIX, NPIX)))
    assert np.isclose(MaxEntropy(1.0).value(flat), 0.0, atol=1e-5)


def test_centroid_measures_sky_offsets_with_rotation():
    npix = 9
    one = np.full((npix, npix), -np.inf).at[0, 4].set(0.0)  # top centre
    image = Image(one, 5.0, rotation_deg=90.0, dra=1.0)
    # The top of a frame rotated to PA 90 is due East: +x.
    assert np.allclose(
        Centroid(2.0).centroid(image), np.array([21.0, 0.0]), atol=1e-5
    )
    assert np.allclose(
        Centroid(2.0).residuals(image), np.array([10.5, 0.0]), atol=1e-5
    )


def test_regularisers_find_the_image_by_path():
    scene = System(star=PointSource(), env=_image(np.ones((NPIX, NPIX))))
    assert np.isclose(
        TSV(1.0, path="env").value(scene), TSV(1.0).value(scene.env)
    )
    with pytest.raises(TypeError, match="not an Image"):
        TSV(1.0, path="star").value(scene)


def test_image_priors_cover_every_image():
    scene = System(
        star=PointSource(),
        disk=_image(np.ones((NPIX, NPIX))),
        outer=System(ring=_image(np.ones((8, 8)))),
    )
    priors = image_priors(scene)
    assert set(priors) == {"disk.log_brightness", "outer.ring.log_brightness"}
    assert priors["outer.ring.log_brightness"].event_shape == (8, 8)
    with pytest.raises(ValueError, match="no Image"):
        image_priors(PointSource())


def test_nyquist_pixel_scale():
    longest = float(np.max(np.hypot(DATA.u, DATA.v)))
    expected = 4.3e-6 / (2 * longest) / (onp.pi / 180 / 3600 / 1000)
    assert 100.0 < expected < 125.0  # λ / 2B for B just under 4 m
    assert np.isclose(nyquist_pixel_scale(DATA), expected, rtol=1e-3)
    assert np.isclose(nyquist_pixel_scale([DATA, DATA]), expected, rtol=1e-3)


def test_translation_leaves_disco_data_unchanged():
    # DISCO phases are insensitive to shifts, and amplitudes too: rolling
    # an image by a pixel (away from its edges) changes nothing.
    blob = gaussian_blob(NPIX, SCALE, 12.0)
    image, rolled = _image(blob), _image(np.roll(blob, 1, axis=1))
    shifted = DATA.model(rolled) - DATA.model(image)
    assert np.max(np.abs(shifted)) < 0.1 * float(DATA.d_vis[0])  # float32
    assert (
        np.max(np.abs(np.angle(image.model(DATA.u, DATA.v, DATA.wavel))))
        < 1e-4
    )
    shift = np.angle(rolled.model(DATA.u, DATA.v, DATA.wavel))
    assert np.max(np.abs(shift)) > 0.05  # the raw phases do move


def test_centroid_prior_centres_an_image_only_fit():
    truth = _image(gaussian_blob(NPIX, SCALE, 25.0))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(2))
    start = _image(gaussian_blob(NPIX, SCALE, 40.0, dra=30.0, ddec=-20.0))
    problem = Problem(
        start, data, image_priors(start), [TSV(1e3), Centroid(1.0)]
    )
    result = fit(problem, "lm")
    assert np.all(np.abs(Centroid(1.0).centroid(result.model)) < 3.0)


def test_a_star_anchors_the_image_position():
    # With an analytic star at the origin, an off-centre blob is recovered
    # in place without any centroid prior.
    truth = System(
        star=PointSource(),
        env=_image(
            gaussian_blob(NPIX, SCALE, 20.0, dra=36.0, ddec=24.0), flux=0.2
        ),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(3))
    start = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(40.0), NPIX, SCALE, flux=0.2),
    )
    result = fit(
        Problem(start, data, image_priors(start), [TSV(1e3, path="env")]), "lm"
    )
    centroid = Centroid(1.0, path="env").centroid(result.model)
    assert np.allclose(centroid, np.array([36.0, 24.0]), atol=6.0)


def test_l_curve_trades_chi2_against_the_penalty():
    truth = System(
        star=PointSource(),
        env=_image(gaussian_blob(NPIX, SCALE, 20.0, dra=24.0), flux=0.2),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(4))
    start = System(
        star=PointSource(), env=_image(np.ones((NPIX, NPIX)), flux=0.2)
    )
    curve = l_curve(
        lambda w: Problem(
            start, data, image_priors(start), [TSV(w, path="env")]
        ),
        [1e1, 1e3, 1e5],
        method="lm",
    )
    assert list(curve.weights) == [1e5, 1e3, 1e1]
    assert np.all(np.diff(curve.chi2) <= 1e-3 * curve.chi2[:-1])  # falls
    assert np.all(np.diff(curve.penalty) >= 0.0)  # rises
    with pytest.raises(ValueError, match="exactly one"):
        l_curve(lambda w: Problem(start, data, image_priors(start)), [1.0])


def test_corner_and_discrepancy_on_a_known_curve():
    # An "L": chi2 falls fast as w drops to 1, then flattens; the penalty
    # grows slowly, then fast.
    w = np.logspace(3, -3, 13)
    t = np.log10(w)
    chi2 = 100.0 * (1.0 + 10.0 ** np.clip(t, 0, None))
    penalty = 1.0 + 10.0 ** np.clip(-t, 0, None)
    curve = LCurve(w, chi2, chi2 / 100.0, penalty, [])
    assert 0.1 <= curve.corner() <= 10.0
    assert np.isclose(curve.discrepancy(target=11.0), 10.0, rtol=0.05)
    assert curve.discrepancy(target=0.5) is None
