"""Period aliases of an undersampled orbit: bands, window and evidences."""

import copy
import functools
import json

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import aliases  # noqa: E402
from virgil.aliases import (  # noqa: E402
    AliasBandResult,
    AliasResult,
    alias_bands,
    fit_orbit_aliases,
    position_profile,
    spectral_window,
)
from virgil._precision import in_float64, cast_tree  # noqa: E402
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.oidata import OIData, cp_indices  # noqa: E402
from virgil.orbits import KeplerOrbit, PositionData  # noqa: E402
from virgil.simulate import simulate  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

T_REF = 60000.0
TRUTH = dict(
    period=12.1,
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


def _night(mjd, frames=3, rotation=0.0, d_phi=0.02):
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    angles = onp.deg2rad(rotation + 15.0 * onp.arange(frames))
    u = onp.concatenate(
        [delta[:, 0] * onp.cos(a) - delta[:, 1] * onp.sin(a) for a in angles]
    )
    v = onp.concatenate(
        [delta[:, 0] * onp.sin(a) + delta[:, 1] * onp.cos(a) for a in angles]
    )
    shift = lambda i: onp.concatenate([i + k * n for k in range(frames)])  # noqa: E731
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": 2.2e-6,
            "vis": onp.ones(u.size),
            "d_vis": onp.full(u.size, 0.01),
            "phi": onp.zeros(len(i1) * frames),
            "d_phi": onp.full(len(i1) * frames, d_phi),
            "i_cps1": shift(i1),
            "i_cps2": shift(i2),
            "i_cps3": shift(i3),
            "mjd": onp.repeat(mjd + onp.arange(frames) / 24.0, n),
        }
    )


def test_alias_bands_tile_the_period_range():
    times = [0.0, 100.0, 413.89]
    bands = alias_bands(times, (10.0, 14.0))
    assert [b.n for b in bands] == list(range(30, 42))
    n34 = next(b for b in bands if b.n == 34)
    assert n34.p_lo < 12.137 < n34.p_hi
    for lo, hi in zip(bands[1:], bands[:-1]):  # contiguous in period
        assert onp.isclose(lo.p_hi, hi.p_lo)


def test_spectral_window_is_one_at_zero_and_returns_at_aliases():
    t = onp.array([0.0, 1.0, 2.0, 3.0])  # a unit grid: aliases at integers
    w = spectral_window(t, [0.0, 0.5, 1.0])
    assert onp.allclose(w[[0, 2]], 1.0) and w[1] < 0.2


def test_position_profile_has_a_chi2_per_band():
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    dra, ddec, _ = orbit.relative(EPOCHS)
    cov = onp.tile(0.04 * onp.eye(2), (len(EPOCHS), 1, 1))
    pos = PositionData(EPOCHS, dra, ddec, cov, t_ref=T_REF)
    bands = alias_bands(EPOCHS, (11.0, 13.0))
    rows = position_profile(pos, bands, k=3.0, n_best=1)
    assert [r["n"] for r in rows] == [b.n for b in bands]
    assert all(onp.isfinite(r["chi2"]) for r in rows)


@functools.cache
def _synthetic(period):
    truth = {**TRUTH, "period": period}
    orbit = KeplerOrbit(**truth, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    key = jax.random.PRNGKey(3)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(EPOCHS)
    ]
    dra, ddec, _ = orbit.relative(EPOCHS)
    rng = onp.random.default_rng(1)
    pos = PositionData(
        EPOCHS,
        onp.asarray(dra) + 0.2 * rng.standard_normal(len(EPOCHS)),
        onp.asarray(ddec) + 0.2 * rng.standard_normal(len(EPOCHS)),
        onp.tile(0.04 * onp.eye(2), (len(EPOCHS), 1, 1)),
        t_ref=T_REF,
    )
    return data, pos


def test_alias_bands_include_the_long_period_band():
    bands = alias_bands([0.0, 100.0], (50.0, 500.0))
    assert (
        bands[0].n == 0 and bands[0].p_lo == 200.0 and bands[0].p_hi == 500.0
    )


# 12.38 d sits mid-band N = 20; 12.1 d is 0.05 cycles from the N = 20/21 edge.
# A few seeds suffice now that the optimizer restarts (aliases._optimize);
# without restarts L-BFGS-B stalled short of the ridge and 1 in 4 noise
# draws settled in a wrong alias at chi2_red ~ 100-800.
@pytest.mark.slow
@pytest.mark.filterwarnings("error:fit_orbit_aliases")  # a good fit is quiet
@pytest.mark.parametrize("period", [12.38, 12.1])
def test_injected_alias_wins_in_closure_phases(period, tmp_path):
    data, pos = _synthetic(period)
    result = fit_orbit_aliases(
        data,
        (11.3, 12.9),
        positions=pos,
        t_ref=T_REF,
        n_candidates=30,
        n_refine=3,
        n_is=400,
        n_samples=50,
    )
    n_true = round(float(onp.ptp(result.times)) / period)
    assert result.best.n == n_true
    assert result.best.p > 0.9
    assert abs(result.best.best["period"] - period) < 0.05
    assert result.best.chi2_red < 3.0  # raw χ²/ν on the quoted errors
    assert (
        n_true in result.samples
        and len(result.samples[n_true]["period"]) == 50
    )
    s = result.samples[n_true]
    assert onp.all((s["Omega"] >= 0) & (s["Omega"] < 180))  # mirror folded
    row = result.table()[0]
    assert row["n"] == n_true and "flags" in row
    result.to_json(tmp_path / "bands.json", n_samples=5)
    json.loads(
        (tmp_path / "bands.json").read_text()
    )  # strict JSON: no Infinity


@pytest.mark.slow
def test_a_band_that_cannot_fit_the_data_warns_loudly():
    """No orbit in 13.5-14.5 d fits a 12.38 d orbit's data, so the best
    band is flagged poor-fit and the call warns rather than failing
    silently."""
    data, pos = _synthetic(12.38)
    with pytest.warns(UserWarning, match="missed the narrow"):
        result = fit_orbit_aliases(
            data,
            (13.5, 14.5),
            positions=pos,
            t_ref=T_REF,
            n_candidates=10,
            n_refine=2,
            n_is=100,
            n_samples=10,
        )
    assert result.best.chi2_red > 10
    assert "poor-fit" in result.best.flags and result.best.flagged


# Few random starts and no positions: this checks the run completes, and
# the fit is often poor, which fit_orbit_aliases rightly warns about.
@pytest.mark.filterwarnings("ignore:fit_orbit_aliases:UserWarning")
def test_random_starts_and_s_max_without_positions():
    data, _ = _synthetic(12.38)
    result = fit_orbit_aliases(
        data,
        (12.1, 12.7),
        t_ref=T_REF,
        n_random=64,
        n_refine=2,
        n_is=100,
        n_samples=20,
        s_max=50.0,
    )
    # Random starts rarely find a narrow ridge in 8-D: only check that it runs.
    assert result.bands and all(b.flags is not None for b in result.bands)


def _settings(**kw):
    base = dict(
        p_range=(11.0, 13.0), a_range=(1.0, 30.0), flux_range=(0.01, 1.0),
        ecc_max=0.9, s_max=None, t_ref=T_REF, n_candidates=2, n_refine=3,
        n_random=30, pool_size=64, k=3.0, eccs=None, min_distance_mas=0.5,
        max_steps=5, n_is=20, flag_nats=1.0, ess_min=50.0, seed=0,
    )  # fmt: skip
    return aliases._Settings(**{**base, **kw})


def _bounds(lower=None, upper=None):
    inf = onp.inf
    lo = [-inf, -inf, 0.0, -1.0, -inf, -inf, -inf, -inf]
    hi = [inf, inf, 0.9, 1.0, inf, inf, inf, inf]
    return aliases._Bounds(
        onp.array(lo if lower is None else lower),
        onp.array(hi if upper is None else upper),
    )


# --- robustness: the failures of OzSTAR job 18296568 ----------------------------------


def _band(n, epochs, **kw):
    return AliasBandResult(
        n=n, p_lo=1.0, p_hi=2.0, modes=[], log_z=-1.0, log_z_is=-1.0, ess=10.0,
        flagged=False, epochs=epochs, **kw
    )  # fmt: skip


def test_table_scales_are_flat_floats_for_multi_block_epochs():
    # An epoch with V2 and closure-phase blocks has one scale per block (a list).
    epochs = [dict(scale=[1.0, 2.5]), dict(scale=[1.5])]
    res = AliasResult(0.0, onp.zeros(2), 1.0, [_band(3, epochs)], {})
    row = res.table()[0]
    assert row["scales"] == [1.0, 2.5, 1.5]
    assert row["epoch_scales"] == [[1.0, 2.5], [1.5]]
    assert f"{min(row['scales']):>6.2f}" and f"{max(row['scales']):>6.2f}"


def test_singular_hessian_is_flagged_not_raised():
    class Flat:
        def hessian(self, x):
            return -onp.diag([1.0, 1.0, 1e-20, 1.0, 1.0, 1.0, 1.0, 1.0])

    x = onp.zeros(8)
    rec, draw = aliases._mode_evidence(
        Flat(), _bounds(), x, -1.0, 10, onp.random.default_rng(0)
    )
    assert draw is None and rec["flags"] == ["hessian-singular"]
    assert rec["log_z_laplace"] == -onp.inf


def test_band_checkpoint_roundtrip_and_key(tmp_path):
    band = alias_bands([0.0, 100.0, 413.89], (10.0, 14.0))[0]
    key = dict(n_is=10, edges=[[1.0, 2.0]])
    done = (
        _band(band.n, [dict(scale=[1.0])]),
        (onp.ones((3, 2)), onp.zeros(3)),
    )
    assert aliases._load_band(tmp_path, band, key) is None
    aliases._save_band(tmp_path, band, key, done)
    json.loads((tmp_path / f"band_{band.n}.json").read_text())
    got = aliases._load_band(tmp_path, band, key)
    assert got[0].n == band.n and got[1][0].shape == (3, 2)
    assert aliases._load_band(tmp_path, band, {**key, "n_is": 11}) is None
    assert aliases._load_band(None, band, key) is None


@pytest.mark.slow
def test_one_failing_band_is_isolated_checkpointed_and_not_refit(
    monkeypatch, tmp_path
):
    data, _ = _synthetic(12.38)
    first = alias_bands(EPOCHS, (12.1, 12.7))[0].n
    calls = []
    real = aliases._fit_band

    def fit_band(problem, settings, band, orbits, times):
        calls.append(band.n)
        if band.n == first:
            raise RuntimeError("boom")
        return real(problem, settings, band, orbits, times)

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    kw = dict(
        t_ref=T_REF,
        n_random=20,
        n_refine=1,
        max_steps=3,
        n_is=20,
        n_samples=5,
        checkpoint_dir=tmp_path,
    )
    seen = []
    result = fit_orbit_aliases(data, (12.1, 12.7), on_band=seen.append, **kw)
    assert len(seen) == len(result.bands) > 1
    bad = [b for b in result.bands if "band-failed" in b.flags]
    assert [b.n for b in bad] == [first] and "boom" in bad[0].error
    others = [b for b in result.bands if b.n != first]
    assert {p.name for p in tmp_path.glob("band_*.pkl")} == {
        f"band_{b.n}.pkl" for b in others
    }
    assert not (
        tmp_path / f"band_{first}.pkl"
    ).exists()  # a failed band is retried
    result.to_json(tmp_path / "bands.json")
    json.loads((tmp_path / "bands.json").read_text())
    # A rerun with the same arguments loads the good bands and refits only the failed one.
    calls.clear()
    fit_orbit_aliases(data, (12.1, 12.7), **kw)
    assert calls == [first]
    # A changed setting (s_max) invalidates the checkpoints.
    calls.clear()
    fit_orbit_aliases(data, (12.1, 12.7), s_max=50.0, **kw)
    assert len(calls) == len(result.bands)


@pytest.mark.slow
def test_resumed_run_equals_uninterrupted_run(tmp_path):
    data, pos = _synthetic(12.1)
    kw = dict(
        positions=pos, t_ref=T_REF, k=3.0, n_candidates=40, n_refine=4,
        n_is=400, n_samples=10,
    )  # fmt: skip
    full = fit_orbit_aliases(
        data, (11.3, 12.9), checkpoint_dir=tmp_path / "full", **kw
    )

    class Interrupted(Exception):
        pass

    def stop(band):
        raise Interrupted

    with pytest.raises(Interrupted):
        fit_orbit_aliases(
            data, (11.3, 12.9), checkpoint_dir=tmp_path / "run", on_band=stop,
            **kw,
        )  # fmt: skip
    assert len(list((tmp_path / "run").glob("band_*.pkl"))) == 1
    loaded = []
    resumed = fit_orbit_aliases(
        data, (11.3, 12.9), checkpoint_dir=tmp_path / "run",
        on_band=loaded.append, **kw,
    )  # fmt: skip
    assert len(loaded) == len(full.bands) > 1
    assert [b.n for b in resumed.bands] == [b.n for b in full.bands]
    for a, b in zip(resumed.bands, full.bands):
        assert a.log_z == b.log_z and a.log_z_is == b.log_z_is
        assert a.p == b.p
    assert resumed.samples.keys() == full.samples.keys()
    assert any(onp.isfinite(b.log_z) for b in full.bands)
    for n, draws in full.samples.items():
        for name, values in draws.items():
            assert onp.array_equal(resumed.samples[n][name], values)


# --- failure semantics --------------------------------------------------------


def test_every_band_failing_raises_chained(monkeypatch):
    data, _ = _synthetic(12.38)
    boom = RuntimeError("boom")

    def fit_band(*args):
        raise boom

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    with pytest.raises(RuntimeError, match="Every band failed") as caught:
        fit_orbit_aliases(data, (12.1, 12.7), t_ref=T_REF)
    assert caught.value.__cause__ is boom


def test_type_error_in_a_band_propagates(monkeypatch):
    data, _ = _synthetic(12.38)
    calls = []

    def fit_band(problem, settings, band, orbits, times):
        calls.append(band.n)
        raise TypeError("a misspelt option")

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    with pytest.raises(TypeError, match="misspelt"):
        fit_orbit_aliases(data, (12.1, 12.7), t_ref=T_REF)
    assert len(calls) == 1  # not isolated as a failed band


def test_value_error_in_one_band_gives_a_band_failed_row(monkeypatch):
    data, _ = _synthetic(12.38)
    seen = []

    def fit_band(problem, settings, band, orbits, times):
        if not seen:
            seen.append(band.n)
            raise ValueError("a bad band")
        return _band(band.n, []), (None, None)

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    result = fit_orbit_aliases(
        data, (11.3, 12.9), t_ref=T_REF, on_band=lambda b: None
    )
    failed = [b for b in result.bands if "band-failed" in b.flags]
    assert [b.n for b in failed] == seen
    assert "a bad band" in failed[0].error
    assert len(failed) < len(result.bands)


def test_misconfiguration_surfaces_in_the_preflight(monkeypatch):
    data, _ = _synthetic(12.38)
    calls = []

    def value_and_grad(self, x):
        raise ValueError("bad data shape")

    monkeypatch.setattr(aliases._Problem, "value_and_grad", value_and_grad)
    monkeypatch.setattr(aliases, "_fit_band", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="bad data shape"):
        fit_orbit_aliases(data, (12.1, 12.7), t_ref=T_REF)
    assert calls == []  # no band was attempted


@pytest.mark.filterwarnings("error:Python iterable type")
def test_data_given_as_a_generator_is_cast_to_float64(monkeypatch):
    data, _ = _synthetic(12.38)
    seen = {}

    def preflight(problem, settings, band):
        seen["dtypes"] = {
            leaf.dtype
            for leaf in jax.tree.leaves(problem.data)
            if hasattr(leaf, "dtype") and leaf.dtype.kind == "f"
        }
        raise KeyboardInterrupt

    monkeypatch.setattr(aliases, "_preflight", preflight)
    with pytest.raises(KeyboardInterrupt):
        fit_orbit_aliases(iter(data), (12.1, 12.7), t_ref=T_REF)
    assert seen["dtypes"] == {onp.dtype("float64")}


class _Scores:
    """A problem whose candidates score by their position in the pool."""

    def batch(self, xs):
        return -onp.arange(len(xs), dtype=float)


def _refine_inputs():
    band = alias_bands(EPOCHS, (12.1, 12.7))[0]
    settings = _settings()
    bounds = aliases._Bounds.of(band, settings)
    rng = onp.random.default_rng(0)
    pool = aliases._candidates(settings, band, bounds, rng, None)
    return band, settings, bounds, pool


def test_failed_starts_are_counted_and_the_others_go_on(monkeypatch):
    band, settings, bounds, pool = _refine_inputs()
    calls = []

    def optimize(problem, bounds, x0, max_steps):
        calls.append(x0)
        if len(calls) == 1:
            raise RuntimeError("one bad start")
        return x0, -1.0

    monkeypatch.setattr(aliases, "_optimize", optimize)
    modes, n_failed = aliases._refine(
        _Scores(), settings, bounds, pool, EPOCHS
    )
    assert n_failed == 1 and len(calls) == settings.n_refine
    assert len(modes) == settings.n_refine - 1
    result = aliases._band_result(
        None, settings, band, [], -onp.inf, -onp.inf, n_failed
    )
    assert "starts-failed" in result.flags and result.n_starts_failed == 1


def test_a_non_finite_value_drops_the_start_without_counting_it(
    monkeypatch,
):
    band, settings, bounds, pool = _refine_inputs()
    monkeypatch.setattr(
        aliases, "_optimize", lambda p, b, x0, m: (x0, float("nan"))
    )
    modes_xy, n_failed = aliases._refine(
        _Scores(), settings, bounds, pool, EPOCHS
    )
    assert modes_xy == [] and n_failed == 0
    modes, z_lap, z_is, draws = aliases._band_evidence(
        None, settings, bounds, modes_xy, onp.random.default_rng(0)
    )
    assert modes == [] and z_lap == -onp.inf and z_is == -onp.inf
    assert draws == (None, None)
    result = aliases._band_result(
        None, settings, band, modes, z_lap, z_is, n_failed
    )
    assert result.flags == ["no-valid-mode"] and result.log_z == -onp.inf


@pytest.mark.parametrize("error", [ValueError, onp.linalg.LinAlgError])
def test_value_and_linalg_errors_count_as_failed_starts(monkeypatch, error):
    _, settings, bounds, pool = _refine_inputs()
    calls = []

    def optimize(problem, bounds, x0, max_steps):
        calls.append(x0)
        if len(calls) == 1:
            raise error("singular")
        return x0, -1.0

    monkeypatch.setattr(aliases, "_optimize", optimize)
    modes, n_failed = aliases._refine(
        _Scores(), settings, bounds, pool, EPOCHS
    )
    assert n_failed == 1 and len(calls) == settings.n_refine
    assert len(modes) == settings.n_refine - 1


def test_every_start_failing_raises_and_type_error_propagates(
    monkeypatch,
):
    _, settings, bounds, pool = _refine_inputs()
    error = RuntimeError("solver")

    def failing(problem, bounds, x0, max_steps):
        raise error

    monkeypatch.setattr(aliases, "_optimize", failing)
    with pytest.raises(RuntimeError, match="starts failed") as caught:
        aliases._refine(_Scores(), settings, bounds, pool, EPOCHS)
    assert caught.value.__cause__ is error

    def value_error(problem, bounds, x0, max_steps):
        raise ValueError("singular")

    monkeypatch.setattr(aliases, "_optimize", value_error)
    with pytest.raises(RuntimeError, match="starts failed") as caught:
        aliases._refine(_Scores(), settings, bounds, pool, EPOCHS)
    assert isinstance(caught.value.__cause__, ValueError)

    def misconfigured(problem, bounds, x0, max_steps):
        raise TypeError("bad option")

    monkeypatch.setattr(aliases, "_optimize", misconfigured)
    with pytest.raises(TypeError, match="bad option"):
        aliases._refine(_Scores(), settings, bounds, pool, EPOCHS)


# --- the checkpoint key ----------------------------------------------------------


def _with(obj, **fields):
    """A copy of an equinox module with (static) fields replaced."""
    new = copy.copy(obj)
    for name, value in fields.items():
        object.__setattr__(new, name, value)
    return new


def test_checkpoint_key_sees_static_fields(tmp_path):
    data, pos = _synthetic(12.38)
    bands = alias_bands(EPOCHS, (12.1, 12.7))
    settings = _settings()

    def key(data=data, pos=pos):
        return aliases._checkpoint_key(data, pos, bands, EPOCHS, settings)

    base = key()
    assert base["format"] == aliases._CHECKPOINT_FORMAT
    assert key() == base
    flipped = [_with(data[0], v2_flag=not data[0].v2_flag)] + data[1:]
    assert key(data=flipped)["data"] != base["data"]
    moved = [_with(data[0], t_ref=(data[0].t_ref or 0.0) + 1.0)] + data[1:]
    assert key(data=moved)["data"] != base["data"]
    moved_pos = _with(pos, t_ref=pos.t_ref + 1.0)
    assert key(pos=moved_pos)["positions"] != base["positions"]
    # A checkpoint saved under one key is not loaded under another.
    band = bands[0]
    done = (_band(band.n, [dict(scale=[1.0])]), (None, None))
    aliases._save_band(tmp_path, band, base, done)
    assert aliases._load_band(tmp_path, band, base) is not None
    assert aliases._load_band(tmp_path, band, key(data=flipped)) is None
    assert aliases._load_band(tmp_path, band, {**base, "format": 1}) is None


def test_checkpoint_key_refuses_unhashable_objects():
    data, _ = _synthetic(12.38)
    bands = alias_bands(EPOCHS, (12.1, 12.7))
    with pytest.raises(ValueError, match="hashed"):
        aliases._checkpoint_key(data, lambda: 0, bands, EPOCHS, _settings())


# --- the evidence ----------------------------------------------------------------

N_IS = 4000


class _GaussianProblem:
    """An 8-D Gaussian likelihood, whose evidence is known."""

    log_prior = -3.0
    peak = 2.5

    def __init__(self):
        sd = onp.array([0.02, 0.01, 0.02, 0.02, 1.0, 0.5, 0.05, 0.05])
        corr = 0.7 * onp.eye(8) + 0.3
        self.cov = sd[:, None] * corr * sd[None, :]
        self.prec = onp.linalg.inv(self.cov)
        self.mu = onp.array([2.5, 0.1, 0.4, 0.2, 100.0, 60.0, 2.3, -1.2])

    def value(self, x):
        d = onp.asarray(x) - self.mu
        quad = onp.einsum("...i,ij,...j->...", d, self.prec, d)
        return self.peak - 0.5 * quad

    def value_and_grad(self, x):
        return self.value(x), -self.prec @ (onp.asarray(x) - self.mu)

    def hessian(self, x):
        return -self.prec

    def batch(self, xs):
        assert len(xs) == N_IS  # every draw, at one shape
        return self.value(onp.asarray(xs))

    def log_evidence(self):
        logdet = onp.linalg.slogdet(self.cov)[1]
        return (
            self.peak + self.log_prior + 4 * onp.log(2 * onp.pi) + 0.5 * logdet
        )


def test_gaussian_evidence_laplace_exact_and_importance_sampling_close():
    problem = _GaussianProblem()
    bounds = _bounds(
        lower=[2.0, -onp.inf, 0.0, -1.0, -onp.inf, -onp.inf, 1.8, -1.7]
    )
    rec, draws = aliases._mode_evidence(
        problem, bounds, problem.mu, problem.peak, N_IS,
        onp.random.default_rng(1),
    )  # fmt: skip
    assert rec["flags"] == []
    assert abs(rec["log_z_laplace"] - problem.log_evidence()) < 1e-8
    assert rec["ess"] > 1000
    sigma = 1.0 / onp.sqrt(rec["ess"])  # relative error of the mean weight
    assert abs(rec["log_z_is"] - problem.log_evidence()) < 4 * sigma
    # The mixture of one mode is its own proposal.
    xs, log_w = aliases._mixture_weights([draws])
    z_is = logsumexp(log_w) - onp.log(len(xs))
    assert abs(z_is - rec["log_z_is"]) < 1e-9


def test_newton_polish_reaches_a_quadratic_optimum_inside_the_bounds():
    centre = onp.array([0.3, -0.2, 0.5, 0.1, 2.0, 1.0, 0.0, -0.5])
    scale = onp.array([1.0, 4.0, 0.5, 2.0, 1.0, 3.0, 1.0, 1.0])

    class Quad:
        def value_and_grad(self, x):
            x = onp.asarray(x)
            return -0.5 * onp.sum(scale * (x - centre) ** 2), -scale * (
                x - centre
            )

        def hessian(self, x):
            return -onp.diag(scale)

    lim = [[-5.0] * 8, [5.0] * 8]
    x0 = centre + 0.5
    x, value = aliases._newton_polish(
        Quad(), lim, x0.copy(), Quad().value_and_grad(x0)[0]
    )
    assert value > Quad().value_and_grad(x0)[0]
    onp.testing.assert_allclose(
        x[[0, 1, 2, 3, 6, 7]], centre[[0, 1, 2, 3, 6, 7]], atol=1e-6
    )
    # An optimum is left alone, and a start at a bound stays inside it.
    x1, v1 = aliases._newton_polish(Quad(), lim, centre.copy(), 0.0)
    onp.testing.assert_allclose(
        x1[[0, 1, 2, 3, 6, 7]], centre[[0, 1, 2, 3, 6, 7]], atol=1e-9
    )
    tight = [[-5.0] * 8, [0.2] + [5.0] * 7]
    x2, _ = aliases._newton_polish(Quad(), tight, centre.copy() - 0.1, -1.0)
    assert x2[0] <= 0.2 + 1e-12


# --- UD primary, V² amplitudes, the flux flip and the warm start -----------------------


def _problem(data, **kw):
    settings = _settings(**kw)
    return settings, aliases._Problem(
        [cast_tree(d, "float64") for d in data], settings
    )


def _x_truth(flux=FLUX, diam=None):
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    band = aliases.AliasBand(20, 12.0, 12.2)
    settings = _settings(primary_diam=None if diam is None else "fit")
    bounds = aliases._Bounds.of(band, settings)
    return aliases._start_vector(orbit, flux, T_REF, band, bounds, diam)


@in_float64
def test_a_vanishing_ud_primary_is_the_point_binary():
    data, _ = _synthetic(12.1)
    x = _x_truth()
    _, point = _problem(data)
    _, ud = _problem(data, primary_diam=1e-4)
    assert float(ud.value_and_grad(jax.numpy.asarray(x))[0]) == pytest.approx(
        float(point.value_and_grad(jax.numpy.asarray(x))[0]), abs=1e-3
    )


@in_float64
def test_a_fitted_diameter_is_recovered():
    truth = 3.0
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = aliases._UDBinary(orbit, FLUX, truth)
    key = jax.random.PRNGKey(5)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(EPOCHS)
    ]
    settings, problem = _problem(data, primary_diam="fit", max_steps=60)
    band = aliases.AliasBand(20, 12.0, 12.2)
    bounds = aliases._Bounds.of(band, settings)
    x0 = _x_truth(diam=1.0)
    x, value = aliases._optimize(problem, bounds, x0, 60)
    assert onp.exp(x[8]) == pytest.approx(truth, rel=0.1)
    assert value > float(problem.value_and_grad(jax.numpy.asarray(x0))[0])


@in_float64
def test_the_epoch_v2_amplitude_is_profiled():
    data, _ = _synthetic(12.1)
    # Epoch 0's V² lost half its contrast.
    lossy = [data[0].set("vis", data[0].vis * 0.5), *data[1:]]
    x = jax.numpy.asarray(_x_truth())
    _, plain = _problem(lossy)
    _, amp = _problem(lossy, vis_amp=(0.01, 1.5))
    assert float(amp.value_and_grad(x)[0]) > float(plain.value_and_grad(x)[0])
    rows = amp.per_epoch(x)
    assert rows[0]["vis_amp"] == pytest.approx(0.5, rel=0.1)
    assert rows[1]["vis_amp"] == pytest.approx(1.0, rel=0.1)
    assert rows[0]["chi2_red"][0] < plain.per_epoch(x)[0]["chi2_red"][0]


@in_float64
def test_relabelling_the_stars_leaves_the_likelihood_unchanged():
    data, _ = _synthetic(12.1)
    settings, problem = _problem(data, flux_range=(0.01, 2.0))
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 12.0, 12.2), settings)
    x = _x_truth(flux=0.97)
    y = aliases._flipped(x, settings, bounds)
    assert onp.exp(y[7]) == pytest.approx(1 / 0.97, rel=1e-3)
    value = lambda z: float(problem.value_and_grad(jax.numpy.asarray(z))[0])  # noqa: E731
    assert value(y) == pytest.approx(value(x), abs=0.05)


@pytest.mark.slow
def test_a_flux_ratio_above_one_is_refit_with_the_labels_swapped():
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    # The companion is the brighter star: ratio 1.4 is outside (0.01, 1].
    scene = OrbitalBinary(orbit, 1.4)
    key = jax.random.PRNGKey(9)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(EPOCHS)
    ]
    kw = dict(
        t_ref=T_REF, n_random=0, n_refine=1, n_is=200, n_samples=20,
        init=(orbit, 1.0),
    )  # fmt: skip
    plain = fit_orbit_aliases(data, (12.0, 12.2), flip_flux=False, **kw)
    flip = fit_orbit_aliases(data, (12.0, 12.2), **kw)
    assert flip.best.flipped
    assert flip.best.chi2_red < 3.0
    assert not plain.best.chi2_red < 3.0  # pinned at the bound, or no mode
    assert flip.best.best["flux"] == pytest.approx(1 / 1.4, rel=0.1)


@pytest.mark.slow
def test_a_warm_start_reproduces_the_optimum_without_other_seeds():
    data, _ = _synthetic(12.1)
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    result = fit_orbit_aliases(
        data, (12.0, 12.2), t_ref=T_REF, n_random=0, n_refine=1, n_is=200,
        n_samples=20, init=(orbit, FLUX),
    )  # fmt: skip
    best = result.best.best
    assert result.best.chi2_red < 3.0
    assert best["period"] == pytest.approx(12.1, abs=0.02)
    assert best["flux"] == pytest.approx(FLUX, rel=0.2)
    assert not result.best.flipped


def test_the_flip_acts_only_on_a_flux_pinned_at_the_top():
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 12.0, 12.2), _settings())
    assert not aliases._pinned(
        _x_truth(flux=0.95), bounds
    )  # near-equal, not pinned
    assert aliases._pinned(_x_truth(flux=0.995), bounds)


@in_float64
def test_the_flip_is_skipped_for_a_resolved_primary():
    data, _ = _synthetic(12.1)
    for kw, expect in (({}, True), ({"primary_diam": 0.5}, False)):
        _, problem = _problem(data, **kw)
        assert aliases._flip_is_relabelling(problem, _settings(**kw)) is expect


@in_float64
def test_a_warm_start_is_forced_into_its_band_whatever_its_rank(monkeypatch):
    data, _ = _synthetic(12.1)
    seen = []
    monkeypatch.setattr(
        aliases,
        "_refine",
        lambda *a: seen.append(a[5] if len(a) > 5 else ()) or ([], 0),
    )
    init = (12.1, 2.0, 0.3, 50.0, 60.0, 120.0, 10.0, 0.3)
    for p_lo, p_hi, expect in ((12.0, 12.2, 1), (11.0, 11.5, 0)):
        settings, problem = _problem(data, init=init)
        band = aliases.AliasBand(20, p_lo, p_hi)
        aliases._fit_band(problem, settings, band, None, EPOCHS)
        assert len(seen[-1]) == expect


def test_vis_amp_refuses_data_that_are_not_squared_visibilities():
    data, _ = _synthetic(12.1)
    d = copy.copy(data[0])
    object.__setattr__(d, "vis_mode", "amp")
    _, problem = _problem([d, *data[1:]], vis_amp=(0.01, 1.5))
    with pytest.raises(NotImplementedError, match="vis_amp"):
        problem.per_epoch(jax.numpy.asarray(_x_truth()))
