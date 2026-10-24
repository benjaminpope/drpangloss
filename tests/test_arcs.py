"""Short arcs: night velocities and the cycle counts they admit."""

import types

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import _gauss_newton, arcs  # noqa: E402
from virgil._precision import run_in  # noqa: E402
from virgil.aliases import alias_bands  # noqa: E402
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.orbits import KeplerOrbit, PositionData, _thiele_innes_fit  # noqa: E402
from virgil.simulate import simulate  # noqa: E402
from tests._shared import vlti_night  # noqa: E402

T_REF = 60000.0
# A season of nights days apart, a gap of most of a year, and a second
# season: nights spaced like this leave tens of alias bands.
NIGHTS = T_REF + onp.array([0.0, 4.0, 125.7, 359.1, 413.9])
P_RANGE = (2.6, 4.0 * float(onp.ptp(NIGHTS)))
ECCS = onp.arange(0.0, 0.91, 0.15)  # a coarser grid, for speed
# Gl 229 Ba-Bb-like: a 12-day orbit some 34 cycles across the nights.
TRUTH = dict(
    period=12.13, dt_peri=-5.0, ecc=0.3, inc=40.0, omega=1.0,
    Omega=214.0, a_mas=7.3,
)  # fmt: skip


def _positions_and_motion(
    truth, nights=NIGHTS, sigma=0.05, sigma_v=0.3, seed=0, underestimate=1.0
):
    """Positions with errors ``sigma`` and a bias of 0.3 mas shared by each
    season's nights (not in the stated errors), and the velocities of the
    first and last nights with a shared bias of ``sigma_v``. With
    ``underestimate`` the actual random errors are that many times the
    stated ``sigma`` and ``sigma_v``."""
    orbit = KeplerOrbit(**truth, t_ref=T_REF)
    rng = onp.random.default_rng(seed)
    r = onp.stack(orbit.relative(nights)[:2], -1)
    season = onp.concatenate([[0], onp.cumsum(onp.diff(nights) > 30.0)])
    angle = rng.uniform(0.0, 2 * onp.pi, season.max() + 1)[season]
    bias = 0.3 * onp.stack([onp.cos(angle), onp.sin(angle)], -1)
    r = r + bias + underestimate * sigma * rng.standard_normal(r.shape)
    h = 1e-4
    v = (
        onp.stack(orbit.relative(nights + h)[:2], -1)
        - onp.stack(orbit.relative(nights - h)[:2], -1)
    ) / (2 * h)
    v = v + underestimate * sigma_v * (
        onp.array([0.6, 0.8]) + rng.standard_normal(v.shape)
    )
    measured = onp.zeros(len(nights), bool)
    measured[[0, -1]] = True
    n = len(nights)
    pos = PositionData(
        nights,
        r[:, 0],
        r[:, 1],
        onp.tile(sigma**2 * onp.eye(2), (n, 1, 1)),
        t_ref=T_REF,
    )
    motion = arcs.NightMotion(
        nights,
        onp.where(measured[:, None], v, 0.0),
        onp.tile(sigma_v**2 * onp.eye(2), (n, 1, 1)),
        measured,
    )
    return pos, motion


def _kept(counts, nights=NIGHTS):
    return [b.n for b in alias_bands(nights, P_RANGE) if counts.admits(b)]


def test_the_velocity_free_fit_is_the_position_fit():
    pos, motion = _positions_and_motion(TRUTH)
    arc = arcs._arc(pos, None, 0.0)
    with run_in("float64"):
        _, chi2 = arcs._pv_fit(
            arc.t, arc.r, arc.wr, arc.v, arc.wv, 12.0, 3.0, 0.4
        )
        _, ref = _thiele_innes_fit(
            *(onp.asarray(x, float) for x in (arc.t, *arc.r.T, pos.whitener)),
            12.0,
            3.0,
            0.4,
        )
    assert onp.isclose(float(chi2), float(ref), rtol=1e-8)
    assert arc.n_data == 2 * len(NIGHTS)
    assert arcs._arc(pos, motion, 0.0).n_data == 2 * len(NIGHTS) + 4


def test_the_cut_keeps_an_off_grid_period():
    # A coarse (τ, e) grid only bounds a period's χ² from above: at high e
    # the truth falls between grid points, and the cut must not exclude it.
    truth = {**TRUTH, "ecc": 0.8}
    orbit = KeplerOrbit(**truth, t_ref=T_REF)
    dra, ddec, _ = orbit.relative(NIGHTS)
    n = len(NIGHTS)
    pos = PositionData(
        NIGHTS, dra, ddec, onp.tile(0.01 * onp.eye(2), (n, 1, 1)), t_ref=T_REF
    )
    counts = arcs.unwrap_cycles(pos, p_range=P_RANGE, eccs=ECCS)
    assert any(lo <= truth["period"] <= hi for lo, hi in counts.intervals)
    # the polish lowers the minimum below the best grid point's
    arc = arcs._arc(pos, None, 0.0)
    _, _, grid_chi2 = arcs._pv_grid(arc, onp.array([truth["period"]]), ECCS)
    assert counts.steps[-1]["chi2"] < 0.1 * grid_chi2.min()


# The true cycle count survives for a range of eccentricities and
# inclinations (face-on to near edge-on), with each season's positions
# biased by 0.3 mas and the velocities by their error, and the default floor
# (0.1 of λ/B_max ≈ 3.5 mas): the velocities narrow the bands well below the
# window's.
@pytest.mark.parametrize("ecc, inc", [(0.0, 10.0), (0.5, 60.0), (0.8, 85.0)])
def test_the_true_cycle_count_survives(ecc, inc):
    truth = {**TRUTH, "ecc": ecc, "inc": inc}
    pos, motion = _positions_and_motion(truth)
    counts = arcs.unwrap_cycles(
        pos, motion, p_range=P_RANGE, jitter_mas=0.35, eccs=ECCS
    )
    kept = _kept(counts)
    n_true = round(float(onp.ptp(NIGHTS)) / truth["period"])
    assert n_true in kept
    assert len(kept) < len(alias_bands(NIGHTS, P_RANGE)) / 4
    assert counts.steps  # the short arcs constrained it


# Errors read off a surface that treats correlated channels as independent
# are too small by √f: here by 6 (f = 36). Scaled by f, as the evidence is
# tempered, the cut keeps the true cycle count. No floor, so that the stated
# errors are all there is.
@pytest.mark.parametrize("seed", range(4))
def test_errors_underestimated_sixfold_keep_the_true_cycle_count(seed):
    pos, motion = _positions_and_motion(TRUTH, seed=seed, underestimate=6.0)
    n_true = round(float(onp.ptp(NIGHTS)) / TRUTH["period"])
    counts = arcs.unwrap_cycles(
        pos, motion, p_range=P_RANGE, eccs=ECCS, cov_scale=36.0
    )
    kept = _kept(counts)
    assert n_true in kept
    assert len(kept) < len(alias_bands(NIGHTS, P_RANGE))  # not the window
    # (that the scale widens the cut is test_the_cov_scale_widens_the_cut:
    # here the unscaled χ² rescale already keeps the true band)
    profile = arcs.arc_profile(
        pos,
        motion,
        [b for b in alias_bands(NIGHTS, P_RANGE) if b.n == n_true],
        eccs=ECCS,
        cov_scale=36.0,
    )
    assert abs(profile[0]["period"] - TRUTH["period"]) < 0.05


# Well-fitting data (errors as stated, small enough that the fit's χ²/dof
# is near 1): the cut's margin is then 25 in χ², and scaling every
# covariance by f divides the χ² by f, so it keeps strictly more bands.
def test_the_cov_scale_widens_the_cut():
    pos, motion = _positions_and_motion(TRUTH, sigma=0.3, sigma_v=0.5)
    n_true = round(float(onp.ptp(NIGHTS)) / TRUTH["period"])
    stated = arcs.unwrap_cycles(pos, motion, p_range=P_RANGE, eccs=ECCS)
    scaled = arcs.unwrap_cycles(
        pos, motion, p_range=P_RANGE, eccs=ECCS, cov_scale=36.0
    )
    assert n_true in _kept(stated) and n_true in _kept(scaled)
    assert len(_kept(scaled)) > len(_kept(stated))
    # the same χ² in both, with its scale (χ²/dof) floored at 1 before the
    # margin: scaling moves the steps' χ² down by f
    for a, b in zip(stated.steps, scaled.steps):
        assert b["chi2"] < a["chi2"]


def test_a_night_velocity_covariance_scales_before_the_floor():
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = OrbitalBinary(orbit, 0.3)
    nights = NIGHTS[[0, 3]]
    data = [
        simulate(
            scene,
            vlti_night(m, 6, 20.0 * k, d_phi=0.05),
            key=jax.random.PRNGKey(k),
        )
        for k, m in enumerate(nights)
    ]
    mid = nights + 2.5 / 24.0
    dra, ddec, _ = orbit.relative(mid)
    pos = types.SimpleNamespace(
        names=["a", "b"], mjd=mid, dra=onp.asarray(dra),
        ddec=onp.asarray(ddec), flux=[0.3, 0.3], primary_diam=None,
    )  # fmt: skip
    bare = arcs.night_motion(data, pos, jitter=0.0)
    wide = arcs.night_motion(data, pos, jitter=0.0, cov_scale=9.0)
    floored = arcs.night_motion(data, pos, jitter=0.1, cov_scale=9.0)
    assert bare.measured.all() and wide.measured.all()
    assert onp.allclose(wide.cov, 9.0 * bare.cov, rtol=1e-6)
    assert onp.allclose(wide.vel, bare.vel)
    # the floor is added after: it is not multiplied
    extra = floored.cov - wide.cov
    assert onp.all(onp.linalg.eigvalsh(extra) > 0)
    unscaled_floor = arcs.night_motion(data, pos, jitter=0.1)
    assert onp.allclose(
        extra, unscaled_floor.cov - bare.cov, rtol=1e-6, atol=1e-12
    )


def test_the_sign_free_seeds_use_the_scaled_positions_too(monkeypatch):
    pos, motion = _positions_and_motion(TRUTH)
    seen = []

    def spy(positions, *a, **k):
        seen.append(positions)
        return []

    monkeypatch.setattr(arcs, "starting_orbits", spy)
    band = alias_bands(NIGHTS, P_RANGE)[:1]
    for scale in (1.0, 9.0):
        arcs.arc_profile(
            pos, motion, band, cov_scale=scale, eccs=ECCS, sign_free=True
        )
    assert seen[0] is pos
    assert onp.allclose(
        arcs._position_cov(seen[1], 0.0), 9.0 * arcs._position_cov(pos, 0.0)
    )


def test_the_cov_scale_widens_the_positions_and_velocities():
    pos, motion = _positions_and_motion(TRUTH)
    one = arcs._arc(pos, motion, 0.1)
    scaled = arcs._arc(pos, motion, 0.1, cov_scale=9.0)
    # (whiteners are inverse standard deviations: a floor of 0.1 mas is
    # added to the scaled σ² = 9 × 0.05², not scaled itself)
    sd = lambda a: 1.0 / a.wr[0, 0, 0]  # noqa: E731
    assert onp.isclose(sd(one) ** 2, 0.05**2 + 0.1**2, rtol=1e-6)
    assert onp.isclose(sd(scaled) ** 2, 9 * 0.05**2 + 0.1**2, rtol=1e-6)
    for bad in (0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cov_scale"):
            arcs.unwrap_cycles(pos, motion, p_range=P_RANGE, cov_scale=bad)


def test_too_few_data_keep_the_whole_window():
    pos, _ = _positions_and_motion(TRUTH, nights=NIGHTS[:3])
    counts = arcs.unwrap_cycles(pos, p_range=P_RANGE, eccs=ECCS)
    assert counts.intervals == (P_RANGE,) and counts.steps == ()


def test_unwrapping_and_seeds_are_deterministic():
    pos, motion = _positions_and_motion(TRUTH)
    runs = [
        arcs.unwrap_cycles(
            pos, motion, p_range=P_RANGE, jitter_mas=0.35, eccs=ECCS
        )
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    bands = [b for b in alias_bands(NIGHTS, P_RANGE) if runs[0].admits(b)][:2]
    seeds = [
        arcs.arc_profile(pos, motion, bands, jitter_mas=0.35, eccs=ECCS)
        for _ in range(2)
    ]
    for a, b in zip(*seeds):
        assert a["n"] == b["n"] and a["chi2"] == b["chi2"]
        assert [o.period for o, _ in a["orbits"]] == [
            o.period for o, _ in b["orbits"]
        ]


def test_a_night_velocity_is_measured_from_its_halves():
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = OrbitalBinary(orbit, 0.3)
    nights = NIGHTS[[0, 3]]
    data = [
        simulate(
            scene,
            vlti_night(m, 6, 20.0 * k, d_phi=0.05),
            key=jax.random.PRNGKey(k),
        )
        for k, m in enumerate(nights)
    ]
    mid = nights + 2.5 / 24.0  # the frames' mean time
    dra, ddec, _ = orbit.relative(mid)
    pos = types.SimpleNamespace(  # the fields of EpochPositions it reads
        names=["a", "b"], mjd=mid, dra=onp.asarray(dra),
        ddec=onp.asarray(ddec), flux=[0.3, 0.3], primary_diam=None,
    )  # fmt: skip
    motion = arcs.night_motion(data, pos)
    assert motion.measured.all()
    h = 1e-4
    v = (
        onp.stack(orbit.relative(mid + h)[:2], -1)
        - onp.stack(orbit.relative(mid - h)[:2], -1)
    ) / (2 * h)
    for i in range(2):
        resid = motion.vel[i] - v[i]
        chi2 = resid @ onp.linalg.solve(motion.cov[i], resid)
        assert chi2 < 16.0  # within 4σ


def _early_half(data):
    """The earlier half of a night's frames, and its mean time."""
    frames = onp.unique(onp.asarray(data.frame))
    pick = onp.isin(onp.asarray(data.frame), frames[: frames.size // 2])
    return data._subset(pick), float(onp.mean(onp.asarray(data.mjd)[pick]))


def test_a_half_night_position_is_refined_to_a_converged_peak():
    # _refine_marginal returns (x, value, stop); a half's position is kept
    # only from a converged refinement (not a stalled or step-limited one).
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    data = simulate(
        OrbitalBinary(orbit, 0.3),
        vlti_night(NIGHTS[0], 6, 0.0, d_phi=0.05),
        key=jax.random.PRNGKey(0),
    )
    early, t_early = _early_half(data)
    truth = onp.array([float(c) for c in orbit.relative(t_early)[:2]])
    scale = float(arcs.fringe_scales(early)[0])
    start = onp.append(truth + 0.1 * scale, 0.3)  # off the peak, in the window
    found = arcs._half_position(early, start, scale, None)
    assert found is not None
    x, cov = found
    assert onp.hypot(*(x - truth)) < 0.5 and cov.shape == (2, 2)


def test_a_half_night_that_does_not_converge_has_no_position(monkeypatch):
    # A stalled or step-limited refinement keeps its best point, which may be
    # far from the half's peak: it must not measure a velocity.
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    data = simulate(
        OrbitalBinary(orbit, 0.3),
        vlti_night(NIGHTS[0], 6, 0.0, d_phi=0.05),
        key=jax.random.PRNGKey(0),
    )
    early, _ = _early_half(data)
    scale = float(arcs.fringe_scales(early)[0])
    for stop in (_gauss_newton.STALLED, _gauss_newton.LIMIT):
        monkeypatch.setattr(
            arcs,
            "_refine_marginal",
            lambda x0, *a, stop=stop, **k: (onp.asarray(x0), 0.0, stop),
        )
        assert (
            arcs._half_position(early, onp.array([1.0, 1.0, 0.3]), scale, None)
            is None
        )
