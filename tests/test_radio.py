"""virgil.radio: complex-visibility data, calibration nuisances, helpers."""

import os

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy import stats

import virgil
from virgil import (
    CalibrationNuisance,
    ComplexVisData,
    GaussianDisk,
    Image,
    PointSource,
    dartboard_folds,
    fit,
    primary_beam,
)
from virgil._geometry import _FT_DEFAULTS
from virgil._utils import mas2rad, rad2mas
from virgil.radio import C_LIGHT

jax.config.update("jax_enable_x64", True)

PIXEL = 2.0  # mas
NPIX = 64


def _data(n=1500, n_eb=2, ddids_per_eb=(2, 1), seed=0, noise=0.0, scene=None):
    """Random uv points over ddids of several EBs; vis = scene or noise."""
    rng = onp.random.default_rng(seed)
    ddid_eb = [e for e, k in enumerate(ddids_per_eb) for _ in range(k)]
    ddid = rng.integers(0, len(ddid_eb), n)
    eb = onp.asarray(ddid_eb)[ddid]
    freq = onp.asarray([230e9, 238e9, 245e9])[ddid]
    u = rng.uniform(-4e6, 4e6, n)
    v = rng.uniform(-4e6, 4e6, n)
    weight = rng.uniform(0.5, 2.0, n) * 1e4
    vis = rng.normal(size=n) + 1j * rng.normal(size=n)
    return ComplexVisData.from_arrays(
        u, v, vis, weight, freq, ddid=ddid + 10, eb=eb + 3
    )


def _image(seed=1, ft="dft", **kw):
    eta = onp.random.default_rng(seed).normal(size=(NPIX, NPIX)) * 0.5
    return Image(eta, PIXEL, ft=ft, **kw)


def _values(nuis, **over):
    values = {k: np.asarray(v) for k, v in nuis.defaults().items()}
    values.update({k: np.asarray(v) for k, v in over.items()})
    return values


def _expected(data, scene, flux, gain, offset, alpha, nu0):
    """The model written out directly, per point."""
    lam = C_LIGHT / onp.asarray(data.freq)
    u, v = onp.asarray(data.u), onp.asarray(data.v)
    vis = onp.asarray(scene.model(u * lam, v * lam, lam))
    eb = onp.asarray(data.eb)
    ramp = onp.exp(
        -2j * onp.pi * mas2rad * (offset[eb, 0] * u + offset[eb, 1] * v)
    )
    return (
        flux * gain[eb] * (onp.asarray(data.freq) / nu0) ** alpha * ramp * vis
    )


def test_model_matches_dft_and_nufft():
    data = _data()
    nuis = CalibrationNuisance(data, nu0_hz=239e9)
    values = _values(
        nuis,
        **{
            "cal.flux": 0.37,
            "cal.gain": [1.2],
            "cal.offset": [[7.0, -3.0]],
            "cal.alpha": 2.3,
        },
    )
    dft = _image(ft="dft")
    got = onp.asarray(data.model(dft, values, nuis))
    want = _expected(
        data,
        dft,
        0.37,
        onp.array([1.0, 1.2]),
        onp.array([[0.0, 0.0], [7.0, -3.0]]),
        2.3,
        239e9,
    )
    assert onp.max(onp.abs(got - want) / onp.abs(want)) < 1e-10

    nufft = _image(ft="nufft")
    got_n = onp.asarray(data.model(nufft, values, nuis))
    eps = _FT_DEFAULTS["eps_f64"]
    assert onp.linalg.norm(got_n - got) / onp.linalg.norm(got) <= 3 * eps


def test_offset_ramp_is_a_shift_of_the_scene():
    data = _data()
    nuis = CalibrationNuisance(
        data, gains=False, sigmas=False, alpha=False, nu0_hz=239e9
    )
    off = onp.array([[11.0, -6.0]])
    values = _values(nuis, **{"cal.flux": 1.0, "cal.offset": off})
    got = onp.asarray(data.model(_image(), values, nuis))
    shifted = _image(dra=11.0, ddec=-6.0)
    lam = C_LIGHT / onp.asarray(data.freq)
    u, v = onp.asarray(data.u), onp.asarray(data.v)
    want = onp.asarray(shifted.model(u * lam, v * lam, lam))
    sel = onp.asarray(data.eb) == 1
    onp.testing.assert_allclose(got[sel], want[sel], rtol=1e-9, atol=1e-12)
    # the reference EB is not shifted
    ref = onp.asarray(_image().model(u * lam, v * lam, lam))
    onp.testing.assert_allclose(got[~sel], ref[~sel], rtol=1e-9, atol=1e-12)


def _nll(data, scene, nuis, values, mask=None):
    """-Σ log N(Re) - Σ log N(Im) by scipy, with the scaled errors."""
    model = onp.asarray(data.model(scene, values, nuis))
    s = onp.asarray(nuis._unpack(values)[3])[onp.asarray(data.ddid)]
    sigma = s / onp.sqrt(onp.asarray(data.weight))
    keep = onp.ones(len(data), bool) if mask is None else mask
    r = onp.asarray(data.vis) - model
    return -(
        stats.norm.logpdf(r.real[keep], scale=sigma[keep]).sum()
        + stats.norm.logpdf(r.imag[keep], scale=sigma[keep]).sum()
    )


def test_normalization_matches_gaussian_density():
    data = _data()
    nuis = CalibrationNuisance(data)
    scene = _image()
    term = data.term(lambda v: scene, nuis, chunk_size=400)
    assert term.has_log_norm
    shifts = []
    for s in (1.0, 1.7):
        values = _values(nuis, **{"cal.flux": 0.2, "cal.sigma": [s, 0.8, s]})
        chi2 = float(np.sum(term(values) ** 2))
        loss = 0.5 * chi2 + float(term.log_norm(values))
        nll = _nll(data, scene, nuis, values)
        shifts.append(loss - nll)
        assert float(term.loglike(values)) == pytest.approx(-nll, rel=1e-10)
    # the difference is independent of the parameters (it is Σ ln w / 2 - N ln 2π ...)
    assert shifts[0] == pytest.approx(shifts[1], rel=1e-10)
    assert term(values).shape == (2 * len(data),)


def test_masked_points_do_not_count():
    data = _data()
    nuis = CalibrationNuisance(data)
    scene = _image()
    mask = onp.random.default_rng(3).random(len(data)) < 0.7
    values = _values(nuis, **{"cal.flux": 0.2, "cal.sigma": [1.2, 0.9, 1.1]})
    full = data.term(lambda v: scene, nuis, mask=mask, chunk_size=500)
    sub = data.subset(mask).term(lambda v: scene, nuis, chunk_size=500)
    chi2 = float(np.sum(full(values) ** 2))
    assert chi2 == pytest.approx(float(np.sum(sub(values) ** 2)), rel=1e-12)
    assert float(full.log_norm(values)) == pytest.approx(
        float(sub.log_norm(values)), rel=1e-12
    )
    # corrupting masked points changes nothing
    bad = data.vis.at[np.asarray(~mask)].set(1e6)
    data2 = eqx.tree_at(lambda d: d.vis, data, bad)
    full2 = data2.term(lambda v: scene, nuis, mask=mask, chunk_size=500)
    assert float(np.sum(full2(values) ** 2)) == pytest.approx(chi2, rel=1e-12)
    c, n = data.chi2_by_ddid(lambda v: scene, values, nuis, mask=mask)
    assert n.sum() == mask.sum()
    assert c.sum() == pytest.approx(chi2, rel=1e-10)
    assert list(n) == [
        int(((onp.asarray(data.ddid) == d) & mask).sum()) for d in range(3)
    ]


def test_gradients_match_finite_differences():
    data = _data(n=300)
    nuis = CalibrationNuisance(data)
    eta0 = onp.random.default_rng(5).normal(size=(NPIX, NPIX)) * 0.5

    def loss(eta, values):
        scene = Image(eta, PIXEL, ft="dft")
        term = data.term(lambda v: scene, nuis, chunk_size=128)
        return 0.5 * np.sum(term(values) ** 2) + term.log_norm(values)

    values = _values(
        nuis,
        **{
            "cal.flux": 0.3,
            "cal.gain": [1.1],
            "cal.offset": [[4.0, -2.0]],
            "cal.sigma": [1.1, 0.9, 1.3],
            "cal.alpha": 2.2,
        },
    )
    g_eta, g_val = jax.grad(loss, argnums=(0, 1))(np.asarray(eta0), values)
    rng = onp.random.default_rng(6)
    for _ in range(3):  # random directions in the image
        d = rng.normal(size=eta0.shape)
        h = 1e-5
        fd = (
            loss(np.asarray(eta0 + h * d), values)
            - loss(np.asarray(eta0 - h * d), values)
        ) / (2 * h)
        assert float(np.sum(g_eta * d)) == pytest.approx(float(fd), rel=1e-6)
    for key, val in values.items():
        flat = onp.asarray(val, dtype=float).ravel()
        for i in range(flat.size):

            def f(x, i=i, key=key, val=val):
                new = flat.copy()
                new[i] = x
                return loss(
                    np.asarray(eta0),
                    {**values, key: np.asarray(new.reshape(onp.shape(val)))},
                )

            h = 1e-6 * max(1.0, abs(flat[i]))
            fd = (f(flat[i] + h) - f(flat[i] - h)) / (2 * h)
            got = onp.asarray(g_val[key]).ravel()[i]
            assert got == pytest.approx(float(fd), rel=1e-5, abs=1e-6), (
                key,
                i,
            )


def test_fit_recovers_nuisances():
    rng = onp.random.default_rng(11)
    n = 2000
    truth = dict(
        flux=0.5, gain=1.15, off=(6.0, -4.0), s=(1.3, 0.8, 1.0), alpha=2.6
    )
    ddid = rng.integers(0, 3, n)
    ddid_eb = onp.array([0, 0, 1])
    eb = ddid_eb[ddid]
    freq = onp.array([230e9, 240e9, 245e9])[ddid]
    u = rng.uniform(-3e6, 3e6, n)
    v = rng.uniform(-3e6, 3e6, n)
    weight = onp.full(n, 4.0e4)
    sigma = onp.array(truth["s"])[ddid] / onp.sqrt(weight)
    template = Image.from_model(
        GaussianDisk(sigma=15.0), NPIX, PIXEL, ft="dft"
    )
    lam = C_LIGHT / freq
    clean = onp.asarray(template.model(u * lam, v * lam, lam))
    ramp = onp.where(
        eb == 1,
        onp.exp(
            -2j
            * onp.pi
            * mas2rad
            * (truth["off"][0] * u + truth["off"][1] * v)
        ),
        1.0,
    )
    vis = (
        truth["flux"]
        * onp.where(eb == 1, truth["gain"], 1.0)
        * (freq / 239e9) ** truth["alpha"]
        * ramp
        * clean
    )
    vis = vis + sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = ComplexVisData.from_arrays(
        u, v, vis, weight, freq, ddid=ddid, eb=eb
    )
    nuis = CalibrationNuisance(data, nu0_hz=239e9)
    priors = nuis.priors()
    # vector-valued sites reach fit() as arrays
    assert priors["cal.gain"].batch_shape == (1,)
    term = data.term(lambda v: template, nuis, chunk_size=700)

    def model(**values):
        return template

    init = nuis.defaults()
    result = fit(
        model, priors, data=(), likelihoods=[term], init=init, method="lbfgs"
    )
    got = result.values
    # parameter uncertainties from the Gaussian approximation (Hessian)
    z_flux = float(got["cal.flux"])
    assert z_flux == pytest.approx(truth["flux"], rel=0.05)
    assert float(got["cal.gain"][0]) == pytest.approx(truth["gain"], rel=0.05)
    onp.testing.assert_allclose(
        onp.asarray(got["cal.offset"])[0], truth["off"], atol=1.0
    )
    onp.testing.assert_allclose(
        onp.asarray(got["cal.sigma"]), truth["s"], rtol=0.06
    )
    assert float(got["cal.alpha"]) == pytest.approx(truth["alpha"], abs=0.3)


def test_dartboard_folds():
    data = _data(n=3000)
    # add Hermitian partners
    u = onp.concatenate([onp.asarray(data.u), -onp.asarray(data.u)])
    v = onp.concatenate([onp.asarray(data.v), -onp.asarray(data.v)])
    n = len(data)
    full = ComplexVisData.from_arrays(
        u, v, onp.ones(2 * n), onp.ones(2 * n), 230e9
    )
    fold, annulus = dartboard_folds(full, k=5, n_q=12, n_phi=16, seed=2)
    assert onp.array_equal(fold[:n], fold[n:])
    assert onp.array_equal(annulus[:n], annulus[n:])
    assert set(fold) == set(range(5))
    assert annulus.min() == 0 and annulus.max() <= 11
    again, _ = dartboard_folds(full, k=5, n_q=12, n_phi=16, seed=2)
    assert onp.array_equal(fold, again)


def test_primary_beam_half_power():
    freq, dish, factor = 230e9, 12.0, 1.13
    fwhm = factor * C_LIGHT / (freq * dish) * rad2mas
    npix, scale = (
        257,
        0.5 * fwhm / 128,
    )  # odd grid: a pixel at r = FWHM/2 on the axis
    pb = primary_beam(npix, scale, freq, dish, factor)
    centre = npix // 2
    assert pb[centre, centre] == 0.0
    # half power at r = FWHM / 2, along either axis
    assert onp.exp(pb[centre, centre + 128]) == pytest.approx(0.5, rel=1e-12)
    assert onp.exp(pb[centre + 128, centre]) == pytest.approx(0.5, rel=1e-12)


def test_taper_is_a_gaussian_convolution():
    data = _data(n=200)
    theta = 30.0
    nuis = CalibrationNuisance(
        data,
        gains=False,
        offsets=False,
        sigmas=False,
        alpha=False,
        taper_fwhm_mas=theta,
    )
    values = _values(nuis, **{"cal.flux": 1.0})
    sigma = theta / (2 * onp.sqrt(2 * onp.log(2)))
    plain = PointSource()
    got = onp.asarray(data.model(plain, values, nuis))
    q2 = onp.asarray(data.u) ** 2 + onp.asarray(data.v) ** 2
    want = onp.exp(-2 * onp.pi**2 * (sigma * mas2rad) ** 2 * q2)
    onp.testing.assert_allclose(got.real, want, rtol=1e-12)


def test_from_arrays_validation():
    with pytest.raises(ValueError, match="weight"):
        ComplexVisData.from_arrays([1.0], [1.0], [1 + 0j], [0.0], 230e9)
    with pytest.raises(ValueError, match="finite"):
        ComplexVisData.from_arrays([1.0], [1.0], [onp.nan + 0j], [1.0], 230e9)
    with pytest.raises(ValueError, match="several execution blocks"):
        ComplexVisData.from_arrays(
            [1.0, 2.0],
            [1.0, 2.0],
            [1 + 0j] * 2,
            [1.0] * 2,
            230e9,
            ddid=[0, 0],
            eb=[0, 1],
        )


def test_select_keeps_labels():
    data = _data()
    sub = data.select(ddids=[11])
    assert set(onp.asarray(sub.ddid)) == {1}
    assert sub.ddid_ids == data.ddid_ids and sub.eb_ids == (3, 4)
    assert len(data.select(ebs=[4])) == int((onp.asarray(data.eb) == 1).sum())


# ----------------------------------------------------------------------
# A synthetic measurement set from CASA's simulator (needs casatools)
# ----------------------------------------------------------------------

# Phase centre and the CASA-predicted point source's offset (mas).
PHASE_CENTRE = (
    onp.deg2rad(15.0 * (15 + 56 / 60 + 9.20 / 3600)),
    -onp.deg2rad(37 + 56 / 60 + 6.42 / 3600),
)
POINT_MAS = (300.0, 100.0)  # East, North
SPW_FREQ = (230e9, 245e9)
NCHAN, CHAN_WIDTH = 4, 125e6

# Local East, North, Up (m) of 8 antennas: compact, so the w term and the
# curvature of the sky stay below the test's tolerance.
ANTENNAS = onp.array(
    [
        [0.0, 0.0, 0.0],
        [38.0, 11.0, 0.5],
        [-52.0, 64.0, -0.3],
        [75.0, -83.0, 0.2],
        [-17.0, -121.0, 0.8],
        [112.0, 45.0, -0.6],
        [-96.0, -22.0, 0.1],
        [23.0, 140.0, 0.4],
    ]
)


def _tangent_to_radec(ra0, dec0, east_mas, north_mas):
    """Gnomonic inverse: sky position of a tangent-plane offset."""
    xi, eta = east_mas * mas2rad, north_mas * mas2rad
    ra = ra0 + onp.arctan2(xi, onp.cos(dec0) - eta * onp.sin(dec0))
    dec = onp.arcsin(
        (onp.sin(dec0) + eta * onp.cos(dec0)) / onp.sqrt(1 + xi**2 + eta**2)
    )
    return ra, dec


@pytest.fixture(scope="module")
def synthetic_ms(tmp_path_factory):
    """(path, casatools version): a point source predicted by CASA."""
    measures = os.environ.get("VIRGIL_CASA_MEASURES")
    if measures:  # CASA's measures data, if not in ~/.casa
        from casaconfig import config

        config.measurespath = measures
        config.measures_auto_update = False
        config.data_auto_update = False
    try:
        import casatools

        me = casatools.measures()
        reference = me.observatory("ALMA")
    except ImportError:
        pytest.skip("casatools is not installed")
    except Exception as err:  # no measures data
        pytest.skip(f"casatools is unusable (measures data?): {err}")
    root = tmp_path_factory.mktemp("ms")
    casatools.logsink().setlogfile(str(root / "casa.log"))
    ms, cl = str(root / "sim.ms"), str(root / "point.cl")
    ra, dec = _tangent_to_radec(*PHASE_CENTRE, *POINT_MAS)
    comp = casatools.componentlist()
    comp.addcomponent(
        flux=[1.0, 0, 0, 0],
        fluxunit="Jy",
        polarization="Stokes",
        dir=f"J2000 {ra:.15f}rad {dec:.15f}rad",
        shape="point",
        freq="230GHz",
    )
    comp.rename(cl)
    comp.close()
    sm = casatools.simulator()
    sm.open(ms)
    sm.setconfig(
        telescopename="ALMA",
        x=list(ANTENNAS[:, 0]),
        y=list(ANTENNAS[:, 1]),
        z=list(ANTENNAS[:, 2]),
        dishdiameter=[12.0] * len(ANTENNAS),
        mount=["alt-az"] * len(ANTENNAS),
        antname=[f"A{i:02d}" for i in range(len(ANTENNAS))],
        coordsystem="local",
        referencelocation=reference,
    )
    sm.setfeed(mode="perfect X Y")
    sm.setfield(
        sourcename="src",
        sourcedirection=f"J2000 {PHASE_CENTRE[0]:.15f}rad {PHASE_CENTRE[1]:.15f}rad",
    )
    sm.setlimits(shadowlimit=0.001, elevationlimit="10deg")
    sm.setauto(autocorrwt=0.0)
    sm.settimes(
        integrationtime="30s",
        usehourangle=True,
        referencetime=me.epoch("utc", "2025/03/01"),
    )
    for i, f in enumerate(SPW_FREQ):
        sm.setspwindow(
            spwname=f"SPW{i}",
            freq=f"{f}Hz",
            deltafreq=f"{CHAN_WIDTH}Hz",
            freqresolution=f"{CHAN_WIDTH}Hz",
            nchannels=NCHAN,
            stokes="XX YY",
        )
    for i in range(len(SPW_FREQ)):
        sm.observe("src", f"SPW{i}", starttime="-0.25h", stoptime="0.25h")
    sm.setdata(spwid=list(range(len(SPW_FREQ))), fieldid=[0])
    sm.setvp(dovp=False)
    sm.predict(complist=cl)
    sm.close()
    return ms, casatools.version()


def _point(data, conjugate_model=False):
    values = {"cal.flux": 1.0}
    nuis = CalibrationNuisance(
        data, gains=False, offsets=False, sigmas=False, alpha=False
    )
    out = onp.asarray(
        data.model(
            PointSource(dra=POINT_MAS[0], ddec=POINT_MAS[1]), values, nuis
        )
    )
    return onp.conj(out) if conjugate_model else out


def test_read_ms_convention(synthetic_ms):
    """Counts, u against UVW and ν, and the DATA-column convention."""
    path, version = synthetic_ms
    from casatools import table

    data = ComplexVisData.read_ms(path)
    tb = table()
    tb.open(path)
    n_rows = tb.nrows()
    uvw = tb.getcol("UVW")
    ddid_col = tb.getcol("DATA_DESC_ID")
    tb.close()
    tb.open(f"{path}/SPECTRAL_WINDOW")
    chan_freq = [tb.getcell("CHAN_FREQ", i) for i in range(2)]
    tb.close()
    assert (
        len(data) == n_rows * NCHAN
    )  # nothing flagged, cross-correlations only
    assert data.ddid_ids == (0, 1)
    assert len(data.eb_ids) == 1
    # u, v from UVW and the channel frequency, per ddid
    for d in (0, 1):
        sub = data.select(ddids=[d])
        rows = onp.flatnonzero(ddid_col == d)
        chan = onp.asarray(sub.chan).reshape(-1)
        freq = onp.asarray(sub.freq)
        nrow = rows.size
        # samples are (channel, row) pairs, channel-major
        onp.testing.assert_allclose(
            onp.asarray(sub.u),
            uvw[0, rows[onp.tile(onp.arange(nrow), NCHAN)]] * freq / C_LIGHT,
            rtol=1e-12,
        )
        onp.testing.assert_allclose(
            onp.asarray(sub.v),
            uvw[1, rows[onp.tile(onp.arange(nrow), NCHAN)]] * freq / C_LIGHT,
            rtol=1e-12,
        )
        assert set(chan) == set(range(NCHAN))
        assert onp.allclose(onp.unique(freq), chan_freq[d], rtol=0, atol=1e-3)
    # the convention: the read data must equal the model, and its opposite not
    got = onp.asarray(data.vis)
    want = _point(data)
    err = onp.max(onp.abs(got - want))
    err_conj = onp.max(onp.abs(got - onp.conj(want)))
    print(
        f"casatools {version}: |data - model| = {err:.3e}, |data - model*| = {err_conj:.3e}"
    )
    assert err < 1e-6, (err, err_conj)
    assert err_conj > 1e-2
    # the opposite conjugation must fail
    wrong = ComplexVisData.read_ms(
        path, conjugate=not virgil.radio.MS_CONJUGATE
    )
    assert onp.max(onp.abs(onp.asarray(wrong.vis) - want)) > 1e-2
