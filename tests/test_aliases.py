"""Period aliases of an undersampled orbit: bands, window and evidences."""

import copy
import dataclasses
import functools
import json
import warnings

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import _gauss_newton, aliases  # noqa: E402
from virgil.epochs import Epochs, epoch_positions  # noqa: E402
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


def _simulate_orbit(period):
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


_synthetic = functools.cache(_simulate_orbit)


@functools.cache
def _synthetic_as(period, x64):
    """The same data simulated in a fixed precision, which sets the noise
    draws: ``_synthetic`` takes whichever precision its first caller has."""
    with jax.enable_x64(x64):
        return _simulate_orbit(period)


def test_alias_bands_include_the_long_period_band():
    bands = alias_bands([0.0, 100.0], (50.0, 500.0))
    assert (
        bands[0].n == 0 and bands[0].p_lo == 200.0 and bands[0].p_hi == 500.0
    )


# 12.38 d sits mid-band N = 20; 12.1 d is 0.05 cycles from the N = 20/21 edge.
# A few seeds suffice now that the optimizer is damped Newton on the IRLS
# curvature (aliases._optimize); L-BFGS-B stalled short of the ridge and
# 1 in 4 noise draws settled in a wrong alias at chi2_red ~ 100-800.
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


# The log likelihood at the optimum of each near start below, from main's
# L-BFGS-B-with-restarts ``_optimize`` (commit 5f600a3); all six reach it.
_OLD_NEAR = [-193.286626] * 6


@in_float64
def test_the_kernel_optimizer_matches_the_old_one_from_near_starts():
    data, _ = _synthetic_as(12.1, True)
    settings, problem = _problem(data, max_steps=200)
    band = aliases.AliasBand(20, 12.0, 12.2)
    bounds = aliases._Bounds.of(band, settings)
    rng = onp.random.default_rng(0)
    sigma = onp.array([2e-3, 0.02, 0.05, 0.05, 10.0, 10.0, 0.1, 0.1])
    for old in _OLD_NEAR:
        x0 = bounds.clip(_x_truth() + sigma * rng.standard_normal(8))
        x, value = aliases._optimize(problem, bounds, x0, 200)
        assert value >= old - 0.01
        assert value == pytest.approx(
            float(problem.value_and_grad(jax.numpy.asarray(x))[0]), abs=1e-9
        )
        assert onp.all(bounds.inside(x))


@in_float64
@pytest.mark.parametrize("vis_amp", [None, (0.01, 1.5)])
def test_the_kernel_rows_carry_the_gradient_of_the_likelihood(vis_amp):
    # With the weights frozen, Jᵀρ is the gradient of -log likelihood
    # exactly (Σ_b w_b/2 ∇χ²_b), also through the live V² amplitude.
    data, _ = _synthetic(12.1)
    _, problem = _problem(data, vis_amp=vis_amp)
    x = jax.numpy.asarray(_x_truth() + 0.01)
    kernel = problem.kernel
    rows = kernel.rows(x)
    jac = jax.jacfwd(kernel.rows)(x)
    onp.testing.assert_allclose(
        jac.T @ rows, jax.grad(kernel.loss)(x), rtol=1e-8
    )
    assert float(kernel.loss(x)) == pytest.approx(
        -float(problem.value_and_grad(x)[0])
    )


@in_float64
def test_a_kernel_that_stalls_falls_back_to_lbfgs_and_then_the_kernel(
    monkeypatch,
):
    data, _ = _synthetic_as(12.1, True)
    settings, problem = _problem(data, max_steps=200)
    band = aliases.AliasBand(20, 12.0, 12.2)
    bounds = aliases._Bounds.of(band, settings)
    x0 = bounds.clip(
        _x_truth() + onp.array([1e-3, 0.01, 0.02, 0.02, 3, 3, 0.03, 0.03])
    )
    real, calls = _gauss_newton.solve, []

    def truncate_first(problem, start, **kw):
        calls.append(start)
        if len(calls) == 1:  # a real stall: stopped short of the optimum
            kw = {**kw, "max_steps": 2}
        return real(problem, start, **kw)

    monkeypatch.setattr(_gauss_newton, "solve", truncate_first)
    x, value = aliases._optimize(problem, bounds, x0, 200)
    assert len(calls) == 2  # the kernel (cut short), L-BFGS-B, the kernel
    assert value == pytest.approx(-193.286626, abs=0.01)


# Band log evidences (Laplace) of the fit below from main's optimizer (commit
# 5f600a3), by N. Band 22 (-460.15, three modes) is left out: its modes come
# from starts at the edge of the period range, and which local optimum such
# a start reaches differs between the optimizers (the kernel finds -414.3
# and -416.3 where main also found -406.4).
_OLD_LOG_Z = {19: -388.837124, 20: -262.393553, 21: -400.143249}


@pytest.mark.slow
def test_band_evidences_are_no_lower_than_with_the_old_optimizer():
    data, pos = _synthetic_as(12.1, False)
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
    by_n = {b.n: b for b in result.bands}
    for n, old in _OLD_LOG_Z.items():
        assert by_n[n].log_z >= old - 0.05


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


# --- the result API: save/load, scene(), residuals(), chi2_per_night() --------------

DIAM = 0.5  # mas: a fixed uniform-disk primary, so the UD path is covered
VIS_AMP = (0.01, 1.5)
FAST = dict(n_candidates=10, n_refine=1, n_is=100, n_samples=8)


@functools.cache
def _tiny_fit(primary_diam=DIAM):
    """The result of a fit, built without running the optimizer (which
    costs tens of seconds): the truth orbit as the one mode of one band,
    through the fit's own ``_band_result``. float32 data, as a user has
    them; V² amplitudes are fitted."""
    data, _ = _synthetic_as(12.1, False)
    fitted = primary_diam == "fit"
    with jax.enable_x64(True):
        x = _x_truth(diam=1.0 if fitted else None)
        settings, problem = _problem(
            data, primary_diam=primary_diam, vis_amp=VIS_AMP
        )
        value = float(problem.value_and_grad(jax.numpy.asarray(x))[0])
        mode = dict(
            x=onp.asarray(x).tolist(), loglike=value, log_z=value,
            log_z_is=value, ess=100.0, flags=[],
        )  # fmt: skip
        band = aliases._band_result(
            problem, settings, aliases.AliasBand(20, 12.0, 12.2), [mode],
            value, value, 0,
        )  # fmt: skip
        band.p = 1.0
    result = AliasResult(
        T_REF, onp.asarray(EPOCHS), float(onp.ptp(EPOCHS)), [band],
        {20: aliases._physical(onp.asarray(x), T_REF)},
        primary_diam=primary_diam, vis_amp=VIS_AMP,
    )  # fmt: skip
    return data, result


@functools.cache
def _real_fit(primary_diam):
    """A real, slow fit (the optimizer, evidences and draws included)."""
    data, pos = _synthetic_as(12.1, False)
    result = fit_orbit_aliases(
        data, (11.95, 12.25), positions=pos, t_ref=T_REF,
        primary_diam=primary_diam, vis_amp=VIS_AMP, **FAST,
    )  # fmt: skip
    return data, result


def _best_x(band):
    """The sampled vector of the band's best mode."""
    live = [m for m in band.modes if onp.isfinite(m["log_z"])]
    return onp.array(max(live, key=lambda m: m["log_z"])["x"])


def _objective(data, x, primary_diam=DIAM):
    """The fit's own scale-marginalized log likelihood at ``x``, in float64."""
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        _, problem = _problem(data, primary_diam=primary_diam, vis_amp=VIS_AMP)
        return float(problem.value_and_grad(jax.numpy.asarray(x))[0])


def _score(data, nights, primary_diam=DIAM):
    """The same log likelihood from ``chi2_per_night``'s ``chi2_fit``."""
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        _, problem = _problem(data, primary_diam=primary_diam, vis_amp=VIS_AMP)
        total = 0.0
        for s, d, row in zip(problem.surfaces, data, nights):
            chi2 = jax.numpy.asarray(
                [row["blocks"][n]["chi2_fit"] for n in s.names]
            )
            total += float(s.score(chi2, 1.0, d))
        return total


def _same_epochs(a, b):
    """Per-field comparison of two ``epochs`` lists (NaN equal to NaN)."""
    assert len(a) == len(b)  # empty for a band with no valid mode
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            if k == "blocks":  # block names
                assert x[k] == y[k]
            else:
                assert onp.allclose(x[k], y[k], rtol=1e-9, equal_nan=True), k


def test_a_saved_result_loads_back_equal(tmp_path):
    _, result = _tiny_fit()
    result.save(tmp_path / "fit.json")
    json.loads((tmp_path / "fit.json").read_text())  # strict JSON
    loaded = AliasResult.load(tmp_path / "fit.json")
    assert loaded.t_ref == result.t_ref and loaded.span == result.span
    assert onp.array_equal(loaded.times, result.times)
    assert loaded.primary_diam == DIAM and loaded.vis_amp == VIS_AMP
    assert loaded.positions is None
    assert [b.n for b in loaded.bands] == [b.n for b in result.bands]
    for a, b in zip(loaded.bands, result.bands):
        for name in ("log_z", "log_z_is", "ess", "p", "chi2_red"):
            got, want = getattr(a, name), getattr(b, name)
            assert got == pytest.approx(want, nan_ok=True), name
        assert a.flags == b.flags and a.flipped == b.flipped
        assert a.best.keys() == b.best.keys()
        for k in b.best:
            assert a.best[k] == pytest.approx(b.best[k])
        _same_epochs(a.epochs, b.epochs)
        assert a.modes or not b.modes
        assert len(a.modes) == len(b.modes)
        for m, n in zip(a.modes, b.modes):
            assert onp.allclose(m["x"], n["x"])
            assert m["flags"] == n["flags"]
            assert m["loglike"] == pytest.approx(n["loglike"], nan_ok=True)
    assert result.best.epochs and result.best.modes
    assert loaded.samples.keys() == result.samples.keys()
    for n, draws in result.samples.items():
        for name, values in draws.items():
            assert onp.allclose(loaded.samples[n][name], values)
    assert [r["flags"] for r in loaded.table()] == [
        r["flags"] for r in result.table()
    ]


def test_non_finite_numbers_survive_a_round_trip(tmp_path):
    # ``clean_json`` writes NaN and infinity as null: loading restores -inf
    # for evidences and NaN for everything else, and keeps them distinct.
    failed = aliases._failed_band(
        AliasBandResult(
            n=4, p_lo=1.0, p_hi=2.0, modes=[], log_z=0.0, log_z_is=0.0,
            ess=0.0, flagged=True,
        ),
        ValueError("no"),
    )  # fmt: skip
    nan = float("nan")
    ok = _band(3, [dict(scale=[nan], chi2_red=[1.0])], p=0.5, flipped=True)
    ok.modes = [
        dict(
            x=[0.0, nan], loglike=nan, log_z=-onp.inf, log_z_is=-onp.inf,
            flags=[],
        )
    ]  # fmt: skip
    res = AliasResult(
        60000.0, onp.zeros(2), 1.0, [ok, failed], {5: {"period": onp.array([1.0, nan])}}
    )  # fmt: skip
    res.save(tmp_path / "r.json")
    back = AliasResult.load(tmp_path / "r.json")
    assert back.bands[1].log_z == -onp.inf
    assert back.bands[1].error == failed.error
    assert onp.isnan(back.bands[1].p) and onp.isnan(back.bands[1].chi2_red)
    assert onp.isnan(back.bands[0].epochs[0]["scale"][0])
    mode = back.bands[0].modes[0]
    assert mode["log_z"] == -onp.inf and onp.isnan(mode["loglike"])
    assert mode["x"][0] == 0.0 and onp.isnan(mode["x"][1])
    assert back.bands[0].flipped is True  # the flip survives
    assert onp.isnan(back.samples[5]["period"][1])
    assert back.samples[5]["period"].dtype == float


def test_scene_reproduces_the_model_of_the_fit():
    # No float64 wrapper: the accessors run in float64 themselves.
    data, result = _tiny_fit()
    band = result.best
    fit_value = _objective(data, _best_x(band))
    assert _score(data, band.chi2_per_night(data)) == pytest.approx(
        fit_value, rel=1e-4, abs=1e-3
    )
    # It is the model the fit builds: a UD primary (not a point source).
    assert isinstance(band.scene(), aliases._UDBinary)
    assert float(band.scene().diam) == DIAM
    assert result.scene().orbit.period == pytest.approx(band.best["period"])


def test_params_reprofile_the_amplitude_and_match_the_fit_objective():
    # At a perturbed orbit the V² amplitude is profiled again there, so the
    # χ² equals the fit's likelihood at that orbit, not best's amplitude.
    data, result = _tiny_fit()
    x = _best_x(result.best).copy()
    x[6] += 0.1  # a 10% larger semimajor axis
    params = {k: float(v[0]) for k, v in aliases._physical(x, T_REF).items()}
    nights = result.best.chi2_per_night(data, params=params)
    assert _score(data, nights) == pytest.approx(
        _objective(data, x), rel=1e-4, abs=1e-3
    )
    best = result.best.chi2_per_night(data)
    assert any(
        a["vis_amp"] != pytest.approx(b["vis_amp"], rel=1e-3)
        for a, b in zip(nights, best)
    )  # the amplitude moved with the orbit
    assert sum(n["chi2_fit"] for n in nights) > sum(
        n["chi2_fit"] for n in best
    )


def test_scene_primary_follows_the_fit_settings():
    _, result = _tiny_fit()
    band = copy.copy(result.best)
    band.primary_diam = None
    assert type(band.scene()) is OrbitalBinary  # a point primary
    band.primary_diam = "fit"
    with pytest.raises(ValueError, match="diam"):
        band.scene()
    assert float(band.scene(dict(band.best, diam=2.0)).diam) == 2.0
    band.best = {}
    with pytest.raises(ValueError, match="no best orbit"):
        band.scene()


def test_per_night_chi2_equals_a_direct_computation():
    data, result = _tiny_fit()
    band = result.best
    nights = band.chi2_per_night(data)
    assert len(nights) == len(data)
    with jax.enable_x64(True):
        model = band.scene()
        for d, e, night in zip(data, band.epochs, nights):
            d = cast_tree(d, "float64")
            # Direct: the visibilities, scaled by this epoch's fitted
            # amplitude, and the closure phases, from virgil's whitened
            # residuals; the phase block includes the penalty rows.
            r = onp.asarray(aliases.whitened_residuals(model, d))
            n_vis = int(onp.asarray(d.vis).size)
            obs, err = onp.ravel(d.vis), onp.ravel(d.d_vis)
            mod = obs + r[:n_vis] * err
            vis = ((e["vis_amp"] * mod - obs) / err) ** 2
            phi = r[n_vis:] ** 2
            blocks = night["blocks"]
            assert blocks["vis"]["chi2_fit"] == pytest.approx(
                vis.sum(), rel=1e-5
            )
            assert blocks["phi"]["chi2_fit"] == pytest.approx(
                phi.sum(), rel=1e-5
            )
            # the data χ² leaves the penalty rows out and counts ν rows
            nu = blocks["phi"]["n"]
            assert blocks["phi"]["chi2"] == pytest.approx(
                phi[:nu].sum(), rel=1e-5
            )
            assert blocks["phi"]["chi2"] <= blocks["phi"]["chi2_fit"]
            assert night["n"] == d.n_independent
            assert night["chi2_per_n"] == pytest.approx(
                night["chi2"] / night["n"]
            )
            # and what the fit recorded for this epoch (raw χ²/ν per block)
            for name, red in zip(blocks, e["chi2_red"]):
                got = blocks[name]["chi2_per_n_fit"]
                assert got == pytest.approx(red, rel=1e-3)
    assert nights == result.chi2_per_night(data)


def test_residuals_are_whitened_blocks_in_vector_order():
    data, result = _tiny_fit()
    # Epoch 0 lost half its V² contrast, so its fitted amplitude is ~0.5.
    data = [data[0].set("vis", data[0].vis * 0.5), *data[1:]]
    blocks = result.residuals(data)
    assert len(blocks) == len(data) and list(blocks[0]) == ["vis", "phi"]
    amps = [n["vis_amp"] for n in result.chi2_per_night(data)]
    assert amps[0] == pytest.approx(0.5, rel=0.1) and amps[1] != amps[0]
    flat = onp.concatenate(list(blocks[0].values()))
    with jax.enable_x64(True):
        d = cast_tree(data[0], "float64")
        raw = onp.asarray(aliases.whitened_residuals(result.scene(), d))
        obs, err = onp.ravel(d.vis), onp.ravel(d.d_vis)
    assert flat.shape == raw.shape
    n_vis = blocks[0]["vis"].size
    assert onp.array_equal(flat[n_vis:], raw[n_vis:])  # closure phases as is
    # The visibility rows: (amp * model - obs) / err, not the raw rows.
    model = obs + raw[:n_vis] * err
    assert onp.allclose(
        blocks[0]["vis"], (amps[0] * model - obs) / err, rtol=1e-6, atol=1e-6
    )
    assert not onp.allclose(blocks[0]["vis"], raw[:n_vis], atol=1e-3)
    with pytest.raises(ValueError, match="epochs"):
        result.residuals(data[:2])


def test_a_diameter_in_params_needs_a_fitted_primary():
    _, result = _tiny_fit()  # a fixed-disk primary
    with pytest.raises(ValueError, match="not 'fit'"):
        result.scene(dict(result.best.best, diam=2.0))
    band = copy.copy(result.best)
    band.primary_diam = None
    with pytest.raises(ValueError, match="not 'fit'"):
        band.scene(dict(band.best, diam=2.0))
    _, fitted = _tiny_fit("fit")
    assert float(fitted.scene(dict(fitted.best.best, diam=2.0)).diam) == 2.0


def test_a_fitted_diameter_is_recognised_in_an_old_file(tmp_path):
    _, result = _tiny_fit("fit")
    result.save(tmp_path / "new.json")
    old = json.loads((tmp_path / "new.json").read_text())
    del old["primary_diam"], old["vis_amp"]
    for band in old["band_details"]:
        del band["primary_diam"], band["vis_amp"]
    (tmp_path / "old.json").write_text(json.dumps(old))
    loaded = AliasResult.load(tmp_path / "old.json")
    assert loaded.primary_diam == "fit"
    assert float(loaded.scene().diam) == pytest.approx(
        loaded.best.best["diam"]
    )
    assert loaded.bands[0].vis_amp == aliases.UNRECORDED  # not recoverable


def test_a_loaded_result_scores_like_the_original(tmp_path):
    data, result = _tiny_fit()
    result.save(tmp_path / "fit.json")
    loaded = AliasResult.load(tmp_path / "fit.json")
    a, b = result.chi2_per_night(data), loaded.chi2_per_night(data)
    assert [x["chi2"] for x in b] == pytest.approx(
        [x["chi2"] for x in a], rel=1e-5
    )


def test_a_file_that_lacks_the_settings_is_not_read_as_a_point_primary(
    tmp_path,
):
    # Files written before ``primary_diam`` and ``vis_amp`` were saved.
    data, result = _tiny_fit()
    result.save(tmp_path / "new.json")
    old = json.loads((tmp_path / "new.json").read_text())
    for key in ("primary_diam", "vis_amp"):
        del old[key]
    for band in old["band_details"]:
        for key in ("primary_diam", "vis_amp", "t_ref"):
            del band[key]
    (tmp_path / "old.json").write_text(json.dumps(old))
    loaded = AliasResult.load(tmp_path / "old.json")
    assert loaded.primary_diam == aliases.UNRECORDED
    assert loaded.bands[0].vis_amp == aliases.UNRECORDED
    for call in (
        lambda: loaded.scene(),
        lambda: loaded.residuals(data),
        lambda: loaded.chi2_per_night(data),
    ):
        with pytest.raises(ValueError, match="does not record"):
            call()
    # A diameter alone is not enough: the V² amplitude is unrecorded too.
    half = AliasResult.load(tmp_path / "old.json", primary_diam=DIAM)
    assert float(half.scene().diam) == DIAM
    with pytest.raises(ValueError, match="vis_amp"):
        half.chi2_per_night(data)
    full = AliasResult.load(
        tmp_path / "old.json", primary_diam=DIAM, vis_amp=list(VIS_AMP)
    )
    got = full.chi2_per_night(data)
    want = result.chi2_per_night(data)
    assert [g["chi2"] for g in got] == pytest.approx([w["chi2"] for w in want])
    # None is a recorded answer (point primary, no amplitude), not unrecorded.
    point = AliasResult.load(
        tmp_path / "old.json", primary_diam=None, vis_amp=None
    )
    assert type(point.scene()) is OrbitalBinary


def _check_fitted_diameter(data, result, tmp_path):
    band = result.best
    assert result.primary_diam == "fit" and "diam" in band.best
    assert float(band.scene().diam) == pytest.approx(band.best["diam"])
    x = _best_x(band)
    assert _score(data, band.chi2_per_night(data), "fit") == pytest.approx(
        _objective(data, x, "fit"), rel=1e-4, abs=1e-3
    )
    result.save(tmp_path / "fit.json")
    loaded = AliasResult.load(tmp_path / "fit.json")
    assert loaded.primary_diam == "fit"
    assert float(loaded.scene().diam) == pytest.approx(band.best["diam"])
    for a, b in zip(loaded.chi2_per_night(data), band.chi2_per_night(data)):
        assert a["chi2_fit"] == pytest.approx(b["chi2_fit"], rel=1e-5)


# --- night flags: a flagged night seeds nothing, but its data still count -----


@functools.cache
def _alias_night_data():
    """Six single-frame nights of the true orbit, drawn so that the last
    one's best static-binary peak is a single-night alias (24 mas from the
    truth: the Gl 229 2024-02-27 situation), found by scanning seeds."""
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    key = jax.random.PRNGKey(4)
    data = [
        simulate(
            scene,
            _night(m, frames=1, rotation=7.0 * k),
            key=jax.random.fold_in(key, k),
        )
        for k, m in enumerate(EPOCHS)
    ]
    dra, ddec, _ = orbit.relative(EPOCHS)
    return data, onp.asarray(dra), onp.asarray(ddec)


@functools.cache
def _alias_night_positions():
    data, *_ = _alias_night_data()
    axis = onp.arange(-14.0, 14.01, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return epoch_positions(
            Epochs({f"n{k}": d for k, d in enumerate(data)}),
            {"dra": axis, "ddec": axis, "flux": [0.1, 0.3, 0.6]},
        )


@pytest.mark.slow
def test_a_single_night_alias_is_flagged_and_the_alias_fit_finds_the_band(
    monkeypatch,
):
    data, dra, ddec = _alias_night_data()
    found = _alias_night_positions()
    miss = onp.hypot(found.dra - dra, found.ddec - ddec)
    # The last night's best peak is an alias, 3 mas or more from the truth,
    # and it is flagged; every other night is right and unflagged.
    assert miss[5] > 3.0 and onp.all(miss[:5] < 0.5)
    assert found.flagged.tolist() == [False] * 5 + [True]
    assert "alias-peaks" in found.flag_codes()["n5"]
    seeded = []
    real = aliases.position_profile

    def spy(positions, *args, **kw):
        seeded.append(positions)
        return real(positions, *args, **kw)

    monkeypatch.setattr(aliases, "position_profile", spy)
    result = fit_orbit_aliases(
        data,
        (11.3, 12.9),
        positions=found,
        t_ref=T_REF,
        n_candidates=30,
        n_refine=3,
        n_is=400,
        n_samples=50,
    )
    n_true = round(float(onp.ptp(result.times)) / TRUTH["period"])
    assert result.best.n == n_true
    assert abs(result.best.best["period"] - TRUTH["period"]) < 0.05
    # The flagged night seeded nothing ...
    assert len(seeded) == 1 and seeded[0].dra.shape == (5,)
    assert [u["name"] for u in result.unseeded] == ["n5"]
    assert "alias-peaks" in [f["code"] for f in result.unseeded[0]["flags"]]
    # ... but its data are in the likelihood: the best orbit predicts the
    # true position on that night (within 0.5 mas), not the alias peak
    # (more than 3 mas away).
    assert len(result.best.epochs) == 6
    best = result.best.best
    orbit = KeplerOrbit(**{k: best[k] for k in TRUTH}, t_ref=result.t_ref)
    pred_dra, pred_ddec, _ = orbit.relative(EPOCHS)
    assert onp.hypot(pred_dra[5] - dra[5], pred_ddec[5] - ddec[5]) < 0.5
    assert (
        onp.hypot(pred_dra[5] - found.dra[5], pred_ddec[5] - found.ddec[5])
        > 3.0
    )
    # Negative control: seeding with the flagged night's alias peak
    # misleads the position-level profile of the true band.
    bands = [
        b for b in alias_bands(result.times, (11.3, 12.9)) if b.n == n_true
    ]
    with_alias = position_profile(
        found.positions(t_ref=T_REF, skip_flagged=False), bands
    )
    without = position_profile(found.positions(t_ref=T_REF), bands)
    assert with_alias[0]["chi2"] > 5 * without[0]["chi2"]


@pytest.mark.slow
def test_skip_flagged_can_be_switched_off_and_unseeded_survives_save_load(
    monkeypatch, tmp_path
):
    data, *_ = _alias_night_data()
    found = _alias_night_positions()
    seeded = []
    real = aliases.position_profile
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda positions, *a, **k: (
            seeded.append(positions) or real(positions, *a, **k)
        ),
    )
    kw = dict(t_ref=T_REF, n_candidates=10, n_refine=2, n_is=100, n_samples=10)
    everyone = fit_orbit_aliases(
        data, (12.0, 12.4), positions=found, skip_flagged=False, **kw
    )
    assert seeded[0].dra.shape == (6,) and everyone.unseeded == ()
    skipped = fit_orbit_aliases(data, (12.0, 12.4), positions=found, **kw)
    assert seeded[1].dra.shape == (5,)
    skipped.save(tmp_path / "r.json")
    loaded = AliasResult.load(tmp_path / "r.json")
    assert loaded.unseeded == skipped.unseeded
    assert loaded.unseeded[0]["flags"][0]["reason"]
    # A file written before the record was kept has none.
    payload = json.loads((tmp_path / "r.json").read_text())
    del payload["unseeded"]
    (tmp_path / "old.json").write_text(json.dumps(payload))
    assert AliasResult.load(tmp_path / "old.json").unseeded == ()


class _Stop(Exception):
    """Raised by the spy that ends a fit before the optimizer starts."""


def _stub_positions(flagged=(), primary_diam=None):
    """``EpochPositions`` of the synthetic data without running
    ``epoch_positions``: only the names, times and flags matter to the
    seeding logic these tests exercise."""
    from virgil.epochs import EpochPositions, NightFlag

    data, _ = _synthetic(12.1)
    n = len(data)
    return EpochPositions(
        names=tuple(f"n{k}" for k in range(n)),
        mjd=onp.array([onp.mean(onp.asarray(d.mjd, float)) for d in data]),
        dra=onp.zeros(n),
        ddec=onp.zeros(n),
        cov=onp.tile(onp.eye(2), (n, 1, 1)),
        flux=onp.full(n, 0.3),
        gap=onp.ones(n),
        gap_marginal=onp.full(n, 20.0),
        chi2_raw=({},) * n,
        scale=({},) * n,
        flags=tuple(
            (NightFlag("manual", "flagged by the test"),)
            if f"n{k}" in flagged
            else ()
            for k in range(n)
        ),
        primary_diam=primary_diam,
    )


def _seed_settings(monkeypatch, positions, **options):
    """Run ``fit_orbit_aliases`` up to its preflight and return the
    ``n_random`` it settled on and whether any position-level seeding
    (``position_profile``) ran, without running the optimizer."""
    data, _ = _synthetic(12.1)
    seen = {}

    def preflight(problem, settings, band):
        seen["n_random"] = settings.n_random
        raise _Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **kw: seen.setdefault("seeded", len(pos.dt))
        and [dict(orbits=[]) for _ in bands],
    )
    with pytest.raises(_Stop):
        fit_orbit_aliases(
            data, (12.0, 12.4), positions=positions, t_ref=T_REF, **options
        )
    return seen


@pytest.mark.parametrize(
    "flagged",
    [("n0", "n1", "n2", "n3", "n4"), ("n0", "n1", "n2", "n3", "n4", "n5")],
    ids=["one-unflagged-night", "every-night-flagged"],
)
def test_fewer_than_two_seeding_nights_fall_back_to_random_starts(
    flagged, monkeypatch
):
    # Fewer than two positions give an underdetermined Thiele-Innes fit
    # whose chi2 = 0 seeds mean nothing, and used to leave n_random = 0
    # ("No candidate orbits" when every night was flagged).
    for n_random in (None, 0):
        with pytest.warns(UserWarning, match="falling back to random"):
            seen = _seed_settings(
                monkeypatch, _stub_positions(flagged), n_random=n_random
            )
        assert "seeded" not in seen and seen["n_random"] >= 200
    # The same for PositionData with a single row.
    one = PositionData([60000.0], [1.0], [1.0], [0.04 * onp.eye(2)])
    with pytest.warns(UserWarning, match="falling back to random"):
        seen = _seed_settings(monkeypatch, one, n_random=0)
    assert seen["n_random"] >= 200


def test_seeds_from_three_or_fewer_nights_keep_random_candidates(
    monkeypatch,
):
    seen = _seed_settings(monkeypatch, _stub_positions(("n3", "n4", "n5")))
    assert seen["seeded"] == 3 and seen["n_random"] == 200
    seen = _seed_settings(monkeypatch, _stub_positions(("n4", "n5")))
    assert seen["seeded"] == 4 and seen["n_random"] == 0
    # An explicit n_random is respected when there are seeds.
    seen = _seed_settings(
        monkeypatch, _stub_positions(("n3", "n4", "n5")), n_random=7
    )
    assert seen["n_random"] == 7


def test_epoch_positions_must_match_the_data_and_the_primary():
    found = _stub_positions(("n5",))
    times = found.mjd
    with pytest.raises(ValueError, match="not the epoch_positions"):
        aliases._seeding_positions(found, T_REF, True, times[:-1])
    shifted = dataclasses.replace(found, mjd=found.mjd + 5.0)
    with pytest.raises(ValueError, match="not the epoch_positions"):
        aliases._seeding_positions(shifted, T_REF, True, times)
    seeds, skipped, fell_back = aliases._seeding_positions(
        found, T_REF, True, times
    )
    assert len(seeds.dt) == 5 and not fell_back
    assert [u["name"] for u in skipped] == ["n5"]
    # Seeds fitted with another primary than the engine's warn.
    ud = _stub_positions(primary_diam=1.05)
    with pytest.warns(UserWarning, match="primary_diam=1.05"):
        aliases._check_seed_diameter(ud, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aliases._check_seed_diameter(ud, 1.05)
        aliases._check_seed_diameter(found, None)


def test_a_fitted_primary_diameter_works_end_to_end(tmp_path):
    _check_fitted_diameter(*_tiny_fit("fit"), tmp_path)


@pytest.mark.slow
def test_a_fitted_primary_diameter_after_a_real_fit(tmp_path):
    _check_fitted_diameter(*_real_fit("fit"), tmp_path)


@pytest.mark.slow
def test_a_real_fit_saves_and_loads_back(tmp_path):
    data, result = _real_fit(DIAM)
    result.save(tmp_path / "fit.json")
    loaded = AliasResult.load(tmp_path / "fit.json")
    assert loaded.best.n == result.best.n
    assert loaded.samples.keys() == result.samples.keys()
    for a, b in zip(loaded.chi2_per_night(data), result.chi2_per_night(data)):
        assert a["chi2_fit"] == pytest.approx(b["chi2_fit"], rel=1e-5)
