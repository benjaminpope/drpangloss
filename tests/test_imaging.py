import jax
import jax.numpy as np
import numpy as onp
import pytest

from drpangloss.coverage import ami_grid_record
from drpangloss.fitting import fit
from drpangloss.imaging import (
    TSV,
    Beam,
    beam,
    dirty_image,
    field_of_view,
    TV,
    Centroid,
    LCurve,
    MaxEntropy,
    image_priors,
    l_curve,
    nyquist_pixel_scale,
    starting_image,
)
from drpangloss.models import GaussianDisk, Image, PointSource, System
from drpangloss.oidata import OIData
from drpangloss.plotting import plot_model
from drpangloss.scenes import gaussian_blob

NPIX, SCALE = 16, 12.0
DATA = OIData(ami_grid_record(pitch_m=0.5))


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


def test_tv_has_no_preferred_direction():
    b = np.array([[1.0, 2.0], [3.0, 7.0]])
    values = [
        TV(1.0).value(_image(x))
        for x in (b, b[::-1], b[:, ::-1], b[::-1, ::-1])
    ]
    assert np.allclose(np.array(values), values[0], rtol=1e-5)


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
    assert 60.0 < expected < 90.0  # λ / 2B for B of about 6 m
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
    result = fit(
        start,
        image_priors(start),
        data,
        [TSV(1e3), Centroid(1.0)],
        method="lm",
    )
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
        start, image_priors(start), data, [TSV(1e3, path="env")], method="lm"
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
        start,
        image_priors(start),
        data,
        TSV(1.0, path="env"),
        [1e1, 1e3, 1e5],
        method="lm",
    )
    assert list(curve.weights) == [1e5, 1e3, 1e1]
    assert np.all(np.diff(curve.chi2) <= 1e-3 * curve.chi2[:-1])  # falls
    assert np.all(np.diff(curve.penalty) >= 0.0)  # rises


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
    # With two datasets, the worse-fitted one decides.
    joint = LCurve(
        w, chi2, np.stack([chi2 / 100.0, chi2 / 200.0], axis=1), penalty, []
    )
    assert np.isclose(joint.discrepancy(target=11.0), 10.0, rtol=0.05)


def _ring_of_baselines(radius_m, n=36, stretch=1.0):
    angle = onp.linspace(0.0, onp.pi, n, endpoint=False)
    u, v = radius_m * onp.cos(angle) * stretch, radius_m * onp.sin(angle)
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": 4.8e-6,
            "vis": onp.ones(n),
            "d_vis": onp.full(n, 0.01),
            "phi": onp.zeros(n),
            "d_phi": onp.full(n, 0.1),
        }
    )


def test_beam_of_a_ring_of_baselines_is_round_and_about_lambda_over_b():
    result = beam(_ring_of_baselines(6.0))
    b = 6.0 / 4.8e-6 * onp.pi / 180 / 3600 / 1000  # cycles per mas
    expected = 2 * onp.sqrt(2 * onp.log(2)) / (onp.pi * onp.sqrt(2) * b)
    assert np.isclose(result.major_mas, expected, rtol=1e-3)
    assert np.isclose(result.minor_mas, expected, rtol=1e-3)


def test_beam_is_long_across_the_long_baselines():
    # Baselines stretched East-West resolve finely East-West, so the beam's
    # major axis points North-South (PA 0 or 180).
    result = beam(_ring_of_baselines(4.0, stretch=2.0))
    assert result.major_mas > 1.5 * result.minor_mas
    assert min(result.pa_deg, 180.0 - result.pa_deg) < 1.0


def test_plot_model_draws_the_beam_in_the_lower_left():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_model(
        _image(np.ones((NPIX, NPIX))),
        fov_mas=NPIX * SCALE,
        npix=NPIX,
        ax=ax,
        beam=Beam(60.0, 30.0, 45.0),
    )
    (patch,) = ax.patches
    x, y = patch.center
    assert x > 0 and y < 0  # East (displayed left) and South (bottom)
    assert np.isclose(patch.width, 60.0) and np.isclose(patch.angle, 45.0)
    plt.close(fig)


def test_field_of_view_is_at_most_500_mas_or_lambda_over_b_min():
    # A ring of 6 m baselines measures nothing larger than λ/B ~ 165 mas.
    assert np.isclose(
        field_of_view(_ring_of_baselines(6.0)),
        4.8e-6 / 6.0 / (onp.pi / 180 / 3600 / 1000),
        rtol=1e-3,
    )
    assert field_of_view(DATA) == 500.0  # AMI-like data reach much further


def test_starting_image_is_sized_from_the_data():
    truth = System(star=PointSource(), env=GaussianDisk(120.0, flux=0.1))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(5))
    start = starting_image(data)
    env = start.env
    assert isinstance(start.star, PointSource) and isinstance(env, Image)
    assert np.isclose(env.flux, 0.1, rtol=0.1)
    fov = env.log_brightness.shape[0] * env.pixel_scale_mas
    assert fov >= 6 * 2.3548 * 120.0 * 0.8
    assert env.pixel_scale_mas <= nyquist_pixel_scale(data) / 4 + 1e-9
    # Without a star, the result is the Image alone.
    alone = starting_image(DATA.with_model(GaussianDisk(120.0)), star=False)
    assert isinstance(alone, Image)


def test_dirty_image_finds_a_companion_once_the_star_is_removed():
    scene = System(
        star=PointSource(), comp=PointSource(flux=0.1, dra=200.0, ddec=120.0)
    )
    data = DATA.with_model(scene)
    dirty = dirty_image(data, 64, 20.0, flux_ratio=0.1)
    row, col = np.unravel_index(np.argmax(dirty), dirty.shape)
    assert (
        abs(row - (31.5 - 120 / 20)) <= 1 and abs(col - (31.5 - 200 / 20)) <= 1
    )


def test_dirty_image_uses_absolute_phases_and_refuses_closure_phases():
    rng = onp.random.default_rng(6)
    u, v = rng.uniform(-6.0, 6.0, (2, 200))
    point = PointSource(dra=-110.0, ddec=70.0)  # on a pixel centre
    cvis = onp.asarray(point.model(u, v, 4.8e-6))
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": 4.8e-6,
            "vis": onp.abs(cvis),
            "d_vis": onp.full(200, 0.01),
            "phi": onp.angle(cvis),
            "d_phi": onp.full(200, 0.01),
            "v2_flag": False,
            "cp_flag": False,
        }
    )
    dirty = dirty_image(data, 32, 20.0)
    row, col = np.unravel_index(np.argmax(dirty), dirty.shape)
    assert (row, col) == (12, 21)  # 15.5 - 70 / 20, 15.5 + 110 / 20
    assert np.isclose(dirty.max(), 1.0, atol=0.05)
    from drpangloss.coverage import nrm_oidata

    with pytest.raises(ValueError, match="complex visibilities"):
        dirty_image(nrm_oidata(), 32, 20.0)


def test_a_dirty_start_is_a_positive_image():
    truth = System(
        star=PointSource(),
        env=_image(gaussian_blob(NPIX, SCALE, 40.0, dra=30.0), flux=0.1),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(7))
    start = starting_image(data, start="dirty")
    assert np.all(start.env.brightness > 0.0)
    moments = starting_image(data)
    assert start.env.log_brightness.shape == moments.env.log_brightness.shape
