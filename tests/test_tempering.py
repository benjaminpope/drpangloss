"""Tempering the orbit evidence for correlated spectral channels (plan 1d).

The solved problem is the untempered evidence: with correlated channels the
best alias band got p = 1.000 at χ²/N 12, and the posteriors were far too
narrow. The effective-channel factor f = N / N_eff is estimated from the
residuals of neighbouring channels (``virgil.tempering``) and the likelihood
is tempered by 1/f in ``fit_orbit_aliases``.
"""

import functools
import json
import types
import warnings

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import aliases, tempering  # noqa: E402
from virgil._precision import cast_tree  # noqa: E402
from virgil.aliases import AliasBandResult, AliasResult  # noqa: E402
from virgil.epochs import EpochPositions, _Surface  # noqa: E402
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.oidata import OIData, cp_indices  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402

T_REF = 60000.0
TRUTH = dict(
    period=12.38,
    dt_peri=2.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=10.0,
)
FLUX = 0.3
EPOCHS = T_REF + onp.array([0.0, 2.4, 5.1, 238.0, 240.6, 247.4])
STATIONS = onp.array([[0.0, 0.0], [60.0, 5.0], [25.0, 70.0], [-40.0, 45.0]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
WAVEL = onp.linspace(2.0e-6, 2.4e-6, 8)  # eight channels
SIGMA_V2, SIGMA_CP = 0.04, 0.08  # quoted errors
SCALE = 3.46  # the noise is this times the quoted errors: χ²/N = 12


def _night_dict(mjd, rotation=0.0, frames=3, vis=None, phi=None):
    """One night of V² and closure phases in eight channels (the layout is
    baseline-major with channels fastest, as OIData stores several
    channels), one time for every frame."""
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    angles = onp.deg2rad(rotation + 15.0 * onp.arange(frames))
    c, s = onp.cos(angles)[:, None], onp.sin(angles)[:, None]
    u = (delta[:, 0] * c - delta[:, 1] * s).reshape(-1)
    v = (delta[:, 0] * s + delta[:, 1] * c).reshape(-1)

    def shift(i):
        return onp.concatenate([i + k * n for k in range(frames)])

    nb, nt, nw = u.size, len(i1) * frames, len(WAVEL)
    return dict(
        u=u,
        v=v,
        wavel=WAVEL,
        vis=onp.ones((nb, nw)) if vis is None else vis,
        d_vis=onp.full((nb, nw), SIGMA_V2),
        phi=onp.zeros((nt, nw)) if phi is None else phi,
        d_phi=onp.full((nt, nw), SIGMA_CP),
        i_cps1=shift(i1),
        i_cps2=shift(i2),
        i_cps3=shift(i3),
        mjd=onp.full(nb, mjd),
    )


def _ar1(rng, shape, rho):
    """Noise along the last axis with neighbour correlation ``rho``."""
    z = rng.standard_normal(shape)
    out = onp.empty(shape)
    out[..., 0] = z[..., 0]
    for k in range(1, shape[-1]):
        out[..., k] = rho * out[..., k - 1] + onp.sqrt(1 - rho**2) * z[..., k]
    return out


def _nights(rho, scale=SCALE, seed=0, n=len(EPOCHS), period=12.38):
    """Nights of a binary orbit observed in eight channels, with noise of
    neighbour-channel correlation ``rho`` and ``scale`` times the quoted
    errors, and the true per-night positions."""
    orbit = KeplerOrbit(**{**TRUTH, "period": period}, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    rng = onp.random.default_rng(seed)
    data = []
    for k, mjd in enumerate(EPOCHS[:n]):
        shape = _night_dict(mjd, 7.0 * k)
        clean = OIData(shape)
        pred = onp.asarray(clean.model(scene))
        nv, nphi = clean.vis.size, clean.phi.size
        vis = pred[:nv].reshape(shape["vis"].shape)
        phi = pred[nv : nv + nphi].reshape(shape["phi"].shape)
        vis = vis + scale * SIGMA_V2 * _ar1(rng, vis.shape, rho)
        phi = phi + scale * SIGMA_CP * _ar1(rng, phi.shape, rho)
        data.append(OIData(_night_dict(mjd, 7.0 * k, vis=vis, phi=phi)))
    dra, ddec, _ = orbit.relative(EPOCHS[:n])
    return data, _positions(dra, ddec)


def _positions(dra, ddec):
    n = len(dra)
    return EpochPositions(
        names=tuple(f"n{k}" for k in range(n)),
        mjd=EPOCHS[:n],
        dra=onp.asarray(dra) + 0.2,  # off the truth: the pilot refines them
        ddec=onp.asarray(ddec) - 0.2,
        cov=onp.tile(onp.eye(2), (n, 1, 1)),
        flux=onp.full(n, FLUX),
        gap=onp.ones(n),
        gap_marginal=onp.full(n, 20.0),
        chi2_raw=({},) * n,
        scale=({},) * n,
        flags=((),) * n,
    )


_cached = functools.cache(_nights)


def _pilot(rho, n=3):
    data, positions = _cached(rho, n=n)
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        return aliases._temper_factors(
            data, positions, None, None, ["vis", "phi"]
        )


# --- the estimator -----------------------------------------------------------


def _chain_pairs(n):
    return onp.arange(n - 1), onp.arange(1, n)


def test_white_noise_gives_f_within_five_percent_of_one():
    # Plain white residuals: ρ = 0, f = 1 (a ρ below two standard errors is
    # taken as zero, so this is exact about 95% of the time and never far).
    for seed in range(20):
        z = onp.random.default_rng(seed).standard_normal(20000)
        factors = tempering.estimate_factors(
            [{"vis": z}], [{"vis": _chain_pairs(len(z))}]
        )
        assert factors.f == pytest.approx(1.0, rel=0.05)


def test_the_correlation_estimator_recovers_f_for_a_correlated_chain():
    # AR(1) with ρ = 0.6: f = (1 + ρ)/(1 - ρ) = 4. The inflation of the
    # errors (χ²/N = 12) does not matter: ρ is scale free.
    z = 3.46 * _ar1(onp.random.default_rng(1), (200, 100), 0.6)
    pairs = (
        (onp.arange(200)[:, None] * 100 + onp.arange(99)).reshape(-1),
        (onp.arange(200)[:, None] * 100 + onp.arange(1, 100)).reshape(-1),
    )
    factors = tempering.estimate_factors(
        [{"vis": z.reshape(-1)}], [{"vis": pairs}]
    )
    assert factors.f == pytest.approx(4.0, rel=0.1)
    assert factors.blocks["vis"] == pytest.approx(factors.f)


def test_blocks_with_few_pairs_take_the_pooled_factor():
    rng = onp.random.default_rng(0)
    z = 3.0 * _ar1(rng, (400, 10), 0.7).reshape(-1)  # the vis block
    a = (onp.arange(400)[:, None] * 10 + onp.arange(9)).reshape(-1)
    pairs = (a, a + 1)
    few = (onp.arange(5), onp.arange(1, 6))  # 5 pairs of the phi block
    factors = tempering.estimate_factors(
        [{"vis": z, "phi": rng.standard_normal(6)}],
        [{"vis": pairs, "phi": few}],
        names=["extra[0]"],
    )
    assert factors.blocks["phi"] == pytest.approx(factors.f)
    assert factors.blocks["extra[0]"] == pytest.approx(factors.f)
    assert factors.f > 3.0
    assert "phi" not in factors.rho and "vis" in factors.rho


def test_neighbouring_channels_are_found_from_the_data_layout():
    data = OIData(_night_dict(T_REF))
    pairs = tempering.adjacent_channels(data)
    nw = len(WAVEL)
    assert len(pairs["vis"][0]) == 18 * (nw - 1)  # 6 baselines x 3 frames
    assert len(pairs["phi"][0]) == 12 * (nw - 1)  # 4 triangles x 3 frames
    a, b = pairs["vis"]
    assert onp.all(b - a == 1)  # channels are fastest within a baseline
    assert onp.all(onp.asarray(data.u)[a] == onp.asarray(data.u)[b])
    assert onp.all(onp.asarray(data.wavel)[b] > onp.asarray(data.wavel)[a])


def test_data_with_one_wavelength_have_no_pairs_and_f_is_one():
    one = _night_dict(T_REF)
    one.update(
        wavel=2.2e-6,
        vis=onp.ones(one["u"].size),
        d_vis=onp.full(one["u"].size, 0.01),
    )
    one["phi"] = onp.zeros(len(one["i_cps1"]))
    one["d_phi"] = onp.full(len(one["i_cps1"]), 0.02)
    pairs = tempering.adjacent_channels(OIData(one))
    assert all(len(p[0]) == 0 for p in pairs.values())
    factors = tempering.estimate_factors([], [], names=["vis", "phi"])
    assert factors.f == 1.0 and factors.blocks == {"vis": 1.0, "phi": 1.0}


def test_the_pilot_on_white_channels_gives_f_of_one():
    # White noise at χ²/N = 12: the error scale absorbs the inflation and
    # the channels are independent, so no tempering.
    factors = _pilot(0.0)
    assert factors.f == pytest.approx(1.0, rel=0.05)
    assert factors.n_pairs["pooled"] > 500


def test_the_pilot_on_correlated_channels_gives_f_above_one():
    # ρ = 0.85: f = 1.85/0.15 = 12.3; the estimate is a little low because
    # the pilot fits the night's position and flux.
    factors = _pilot(0.85)
    assert 8.0 < factors.f < 14.0
    assert all(8.0 < v < 14.0 for v in factors.blocks.values())


# --- the tempered likelihood -------------------------------------------------


def _cfg(dof=()):
    return (T_REF, None, False, None, dof)


def test_temper_false_reproduces_the_untempered_evidence():
    # No dof (cfg[4] empty) is the old likelihood: every block of every
    # epoch scored with dof = 1.
    data, _ = _cached(0.85, n=2)
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        surfaces = tuple(_Surface(d) for d in data)
        free = aliases._Bounds(onp.full(8, -onp.inf), onp.full(8, onp.inf))
        band = types.SimpleNamespace(p_lo=11.0, p_hi=14.0)
        x = aliases._start_vector(
            KeplerOrbit(**TRUTH, t_ref=T_REF), FLUX, T_REF, band, free
        )
        model = aliases._model(x, _cfg())
        old = sum(
            s.score(aliases._epoch_chi2(s, model, d, None)[0], 1.0, d)
            for s, d in zip(surfaces, data)
        )
        new = aliases._loglike(surfaces, x, tuple(data), _cfg())
        assert float(new) == pytest.approx(float(old), rel=1e-12)
        # a dof of 1/f in every block divides the log likelihood by f
        f = 12.0
        temp = aliases._loglike(
            surfaces,
            x,
            tuple(data),
            _cfg((("phi", 1.0 / f), ("vis", 1.0 / f))),
        )
        assert float(temp) == pytest.approx(float(old) / f, rel=1e-12)


def test_the_dof_argument_is_f_one_or_a_fixed_f(monkeypatch):
    data, positions = _cached(0.0, n=3)

    class Stop(Exception):
        pass

    seen = {}

    def preflight(problem, settings, band):
        seen["settings"] = settings
        raise Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **kw: [dict(orbits=[]) for _ in bands],
    )

    def run(**kw):
        with pytest.raises(Stop), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # f = 4 is outside [0.5, 2]
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), positions=positions, t_ref=T_REF,
                primary_diam=None, n_random=5, **kw,
            )  # fmt: skip
        return seen["settings"]

    assert run(temper=False).dof == ()
    assert run(temper=1.0).dof == ()  # f = 1 is the untempered fit
    assert run(temper=4.0).dof == (("phi", 0.25), ("vis", 0.25))
    # white channels estimate f = 1: nothing to temper
    assert run(temper=True).dof == ()
    with pytest.raises(ValueError, match="temper must be"):
        run(temper=-2.0)


def test_correlated_channels_are_tempered_by_default(monkeypatch):
    data, positions = _cached(0.85, n=3)
    seen = {}

    class Stop(Exception):
        pass

    def preflight(problem, settings, band):
        seen["settings"] = settings
        raise Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **kw: [dict(orbits=[]) for _ in bands],
    )
    with pytest.warns(UserWarning, match="effective-channel factor"):
        with pytest.raises(Stop):
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), positions=positions, t_ref=T_REF,
                primary_diam=None, n_random=5,
            )  # fmt: skip
    dof = dict(seen["settings"].dof)
    assert set(dof) == {"vis", "phi"} and all(v < 0.2 for v in dof.values())


def test_several_channels_without_positions_warn_and_stay_untempered(
    monkeypatch,
):
    data, _ = _cached(0.85, n=3)

    class Stop(Exception):
        pass

    def preflight(problem, settings, band):
        raise Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    with pytest.warns(UserWarning, match="needs EpochPositions"):
        with pytest.raises(Stop):
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), t_ref=T_REF, primary_diam=None,
                n_random=5,
            )  # fmt: skip


# --- the result --------------------------------------------------------------


def _band(n):
    return AliasBandResult(
        n=n, p_lo=1.0, p_hi=2.0, modes=[], log_z=-1.0, log_z_is=-1.0,
        ess=10.0, flagged=False, epochs=[],
    )  # fmt: skip


def test_the_factor_is_saved_and_loaded(tmp_path):
    result = AliasResult(
        T_REF, onp.zeros(2), 1.0, [_band(3)], {},
        primary_diam=None, vis_amp=None,
        temper_f=7.5, temper_blocks={"vis": 7.0, "phi": 8.0},
    )  # fmt: skip
    assert result.tempering  # 7.5 is outside [0.5, 2]
    result.save(tmp_path / "r.json")
    loaded = AliasResult.load(tmp_path / "r.json")
    assert loaded.temper_f == 7.5
    assert loaded.temper_blocks == {"vis": 7.0, "phi": 8.0}
    assert loaded.tempering


def test_an_old_file_loads_without_a_factor(tmp_path):
    result = AliasResult(
        T_REF, onp.zeros(2), 1.0, [_band(3)], {},
        primary_diam=None, vis_amp=None,
    )  # fmt: skip
    result.save(tmp_path / "r.json")
    payload = json.loads((tmp_path / "r.json").read_text())
    del payload["temper_f"], payload["temper_blocks"]  # before plan 1d
    (tmp_path / "old.json").write_text(json.dumps(payload))
    loaded = AliasResult.load(tmp_path / "old.json")
    assert loaded.temper_f is None and loaded.temper_blocks is None
    assert not loaded.tempering


@pytest.mark.parametrize(
    "f, flagged", [(1.0, False), (0.5, False), (2.0, False), (2.5, True),
                   (0.3, True), (None, False)]
)  # fmt: skip
def test_the_tempering_flag_is_outside_half_to_two(f, flagged):
    result = AliasResult(T_REF, onp.zeros(2), 1.0, [_band(3)], {}, temper_f=f)
    assert result.tempering is flagged


# --- the thresholds ----------------------------------------------------------


def _modes(gap):
    """A band whose best mode has a point-reflected partner ``gap`` nats
    below it: the label-swap rule's input."""
    x = onp.array(
        [onp.log(12.0), 0.1, 0.2, 0.6, 60.0, 120.0, onp.log(10.0), -1.2]
    )
    mirror = x.copy()
    mirror[4] += 180.0  # ω + 180°: the reflected track
    return AliasBandResult(
        n=20, p_lo=1.0, p_hi=2.0, log_z=0.0, log_z_is=0.0, ess=10.0,
        flagged=False, epochs=[],
        modes=[
            dict(x=x.tolist(), log_z=0.0, flags=[]),
            dict(x=mirror.tolist(), log_z=-gap, flags=[]),
        ],
    )  # fmt: skip


def test_the_label_threshold_is_on_tempered_nats_and_unchanged_for_f_one():
    # The rule compares the evidences the fit reports, which are tempered:
    # a gap of 3 nats there is 3 f raw nats. On white noise (f = 1) a gap
    # of 2 nats is free and one of 4 is not, as before tempering existed.
    settings = aliases._Settings(
        p_range=(1.0, 2.0), a_range=(1.0, 30.0), flux_range=(0.01, 1.0),
        ecc_max=0.9, s_max=None, t_ref=T_REF, n_candidates=1, n_refine=1,
        n_random=1, pool_size=64, k=3.0, eccs=None, min_distance_mas=0.5,
        max_steps=1, n_is=1, flag_nats=1.0, ess_min=1.0, seed=0,
    )  # fmt: skip
    times = onp.array([0.0, 1.0, 2.0])
    assert aliases.LABELS_FREE_NATS == 3.0
    for gap, free in ((2.0, True), (4.0, False)):
        got = aliases._labels_free(_modes(gap), settings, times)
        assert got is free or got == free
    # a raw gap of 12 nats at f = 4 is a tempered gap of 3: still free
    raw, f = 11.0, 4.0
    assert aliases._labels_free(_modes(raw / f), settings, times)
    assert not aliases._labels_free(_modes(raw), settings, times)


@pytest.mark.slow
def test_the_resolution_threshold_is_on_the_tempered_score():
    # estimate_primary_diameter's gain over a point primary is dof times the
    # untempered one: the same 5-nat threshold then asks for f times the raw
    # gain, and is unchanged for f = 1.
    from virgil.epochs import RESOLVE_NATS, estimate_primary_diameter
    from virgil.models import PointSource, System, UniformDisk
    from virgil.simulate import simulate

    truth = (5.0, -3.0, 0.25)
    scene = System(
        primary=UniformDisk(1.05),
        companion=PointSource(truth[2], truth[0], truth[1]),
    )
    data = [
        simulate(scene, OIData(_single(k)), jax.random.PRNGKey(k))
        for k in range(4)
    ]
    start = _positions(onp.full(4, 5.3 - 0.2), onp.full(4, -2.8 + 0.2))
    start = EpochPositions(**{**start.__dict__, "flux": onp.full(4, 0.2)})
    full = estimate_primary_diameter(data, start)
    assert full.resolved and full.delta_nats > RESOLVE_NATS
    quarter = estimate_primary_diameter(data, start, dof=0.25)
    assert quarter.delta_nats == pytest.approx(full.delta_nats / 4, rel=0.05)
    blocks = estimate_primary_diameter(
        data, start, dof={"vis": 0.25, "phi": 0.25}
    )
    assert blocks.delta_nats == pytest.approx(quarter.delta_nats, rel=1e-3)


def _single(k):
    d = _night_dict(T_REF + k, 7.0 * k, frames=2)
    nb = d["u"].size
    nt = len(d["i_cps1"])
    d.update(
        wavel=2.2e-6,
        vis=onp.ones(nb),
        d_vis=onp.full(nb, 0.01),
        phi=onp.zeros(nt),
        d_phi=onp.full(nt, 0.02),
    )
    return d


# --- the engine --------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:fit_orbit_aliases")  # f, poor-fit
def test_tempering_turns_an_overconfident_band_probability_into_a_calibrated_one():
    # Eight correlated channels (ρ = 0.85, f ≈ 12) with noise 3.5 times the
    # quoted errors (χ²/N = 11 at the best band). The untempered evidence
    # separates the bands by a gap of about f times the tempered one, tens of
    # nats: p = 1.000, which the data do not support. Tempered, the same
    # bands are 5 nats apart: p < 1.
    data, positions = _cached(0.85, period=12.1)
    result = aliases.fit_orbit_aliases(
        data,
        (11.8, 12.9),
        positions=positions,
        t_ref=T_REF,
        n_candidates=10,
        n_refine=2,
        n_is=200,
        n_samples=20,
        primary_diam=None,
    )
    assert result.temper_f > 8.0 and result.tempering
    best, rival = result.bands[0], result.bands[1]
    assert best.n == 20
    assert 9.0 < best.chi2_red < 14.0
    assert 0.5 < best.p < 1.0 - 1e-4
    # the untempered gap is f times larger: p = 1.000 to eight places
    gap = best.log_z - rival.log_z
    assert gap < 10.0 < gap * result.temper_f
