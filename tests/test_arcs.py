"""Short arcs: night velocities and the cycle counts they admit."""

import types

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import arcs  # noqa: E402
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
    truth, nights=NIGHTS, sigma=0.05, sigma_v=0.3, seed=0
):
    """Positions with a bias of a fraction of the resolution, and the
    velocities of the first and last nights."""
    orbit = KeplerOrbit(**truth, t_ref=T_REF)
    rng = onp.random.default_rng(seed)
    r = onp.stack(orbit.relative(nights)[:2], -1)
    r = r + (sigma + 0.3) * rng.standard_normal(r.shape)
    h = 1e-4
    v = (
        onp.stack(orbit.relative(nights + h)[:2], -1)
        - onp.stack(orbit.relative(nights - h)[:2], -1)
    ) / (2 * h)
    v = v + sigma_v * rng.standard_normal(v.shape)
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


# The true cycle count survives for a range of eccentricities and
# inclinations (face-on to near edge-on), with the positions biased by 0.3
# mas, and the velocities narrow the bands well below the window's.
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
