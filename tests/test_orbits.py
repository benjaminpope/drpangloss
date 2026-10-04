"""Orbit conventions (design/orbit_scene_joint_fitting.md §2, §5.1–5.2)."""

import itertools

import jax
import numpy as onp
import pytest
from scipy.optimize import brentq

pytest.importorskip("jaxoplanet")

from virgil.orbits import KeplerOrbit, ThieleInnesOrbit  # noqa: E402

T_REF = 60500.0
ORBIT = dict(
    period=400.0, dt_peri=30.0, ecc=0.4, inc=60.0, omega=40.0, Omega=110.0
)


def _orbit(**changes):
    return KeplerOrbit(**{**ORBIT, "a_mas": 20.0, **changes}, t_ref=T_REF)


def _reference(mjd, period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref):
    """An independent NumPy ephemeris: Kepler's equation by root finding,
    then the orbit rotated onto the sky (no Thiele–Innes constants)."""
    out = []
    for t in onp.atleast_1d(mjd):
        mean = 2 * onp.pi * (t - t_ref - dt_peri) / period
        mean = onp.mod(mean, 2 * onp.pi)
        ecc_anomaly = brentq(
            lambda e: e - ecc * onp.sin(e) - mean, 0.0, 2 * onp.pi
        )
        # In the orbital plane: x towards periastron, y along the motion.
        x = a_mas * (onp.cos(ecc_anomaly) - ecc)
        y = a_mas * onp.sqrt(1 - ecc**2) * onp.sin(ecc_anomaly)
        w, n, i = onp.deg2rad([omega, Omega, inc])
        # Rotate by ω from the node, tilt by i about the line of nodes, then
        # turn the node to position angle Ω (North through East).
        u = x * onp.cos(w) - y * onp.sin(w)  # along the node
        v = x * onp.sin(w) + y * onp.cos(w)  # in the plane, ⟂ to the node
        north = u * onp.cos(n) - v * onp.cos(i) * onp.sin(n)
        east = u * onp.sin(n) + v * onp.cos(i) * onp.cos(n)
        away = v * onp.sin(i)
        out.append((east, north, away))
    return onp.array(out).T


def _angles(**changes):
    return {**ORBIT, "a_mas": 20.0, **changes, "t_ref": T_REF}


def test_positions_match_an_independent_ephemeris():
    mjd = T_REF + onp.linspace(-300.0, 500.0, 17)
    grid = itertools.product(
        (0.0, 0.3, 0.9), (5.0, 60.0, 120.0), (0.0, 140.0), (20.0, 250.0)
    )
    with jax.enable_x64(True):
        for ecc, inc, omega, Omega in grid:
            kw = _angles(ecc=ecc, inc=inc, omega=omega, Omega=Omega)
            ours = onp.array(KeplerOrbit(**kw).relative(mjd))
            assert onp.allclose(
                ours, _reference(mjd, **kw), rtol=0, atol=1e-10 * kw["a_mas"]
            )
    # float32, with times relative to t_ref computed on the host.
    kw = _angles(ecc=0.3, inc=60.0)
    ours = onp.array(KeplerOrbit(**kw).relative(mjd))
    assert onp.allclose(ours, _reference(mjd, **kw), rtol=0, atol=1e-5 * 20)


def test_inclination_below_90_turns_the_position_angle_forward():
    mjd = T_REF + onp.linspace(0.0, 400.0, 401)
    with jax.enable_x64(True):
        for inc, sign in ((30.0, 1), (150.0, -1)):
            _, pa = _orbit(inc=inc).separation_pa(mjd)
            steps = onp.diff(onp.unwrap(onp.deg2rad(onp.asarray(pa))))
            assert onp.all(sign * steps > 0)


def test_the_node_ambiguity_and_the_primary_swap():
    mjd = T_REF + onp.linspace(0.0, 400.0, 23)
    with jax.enable_x64(True):
        base = onp.array(_orbit().relative(mjd))
        both = onp.array(_orbit(omega=220.0, Omega=290.0).relative(mjd))
        swap = onp.array(_orbit(omega=220.0).relative(mjd))
    # (Ω + 180°, ω + 180°): the same sky, dz reversed.
    assert onp.allclose(both[:2], base[:2], atol=1e-10)
    assert onp.allclose(both[2], -base[2], atol=1e-10)
    # ω + 180° alone is r -> -r: the primary and secondary swapped.
    assert onp.allclose(swap, -base, atol=1e-10)


def test_the_secondary_recedes_at_the_node_with_position_angle_omega():
    orbit = _orbit()
    with jax.enable_x64(True):

        def dz(t):
            return float(orbit.relative(T_REF + t)[2])

        # The ascending node: dz crosses zero going up.
        t = onp.linspace(0.0, 400.0, 4001)
        z = onp.array([dz(x) for x in t])
        up = onp.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))[0]
        node = brentq(dz, t[up], t[up + 1])
        _, pa = orbit.separation_pa(T_REF + node)
        assert float(pa) == pytest.approx(ORBIT["Omega"], abs=1e-6)
        # The velocity is the exact derivative of the position.
        times = T_REF + onp.array([10.0, 123.4, 300.0])
        velocity = onp.array(orbit.relative_velocity(times))
        h = 1e-4
        step = (
            onp.array(orbit.relative(times + h))
            - onp.array(orbit.relative(times - h))
        ) / (2 * h)
        assert onp.allclose(velocity, step, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "changes",
    [{}, {"ecc": 0.0}, {"inc": 0.0}, {"inc": 150.0, "Omega": 300.0}],
)
def test_thiele_innes_round_trip(changes):
    mjd = T_REF + onp.linspace(-200.0, 600.0, 31)
    with jax.enable_x64(True):
        orbit = _orbit(**changes)
        ti = orbit.to_thiele_innes()
        assert isinstance(ti, ThieleInnesOrbit)
        assert onp.allclose(
            onp.array(ti.sky(mjd)), onp.array(orbit.relative(mjd))[:2]
        )
        back = ti.to_kepler()
        assert 0.0 <= float(back.Omega) < 180.0
        assert onp.allclose(
            onp.array(back.relative(mjd))[:2],
            onp.array(orbit.relative(mjd))[:2],
            atol=1e-9,
        )
        assert float(back.inc) == pytest.approx(float(orbit.inc), abs=1e-6)
        assert float(back.a_mas) == pytest.approx(20.0, rel=1e-10)


def test_jaxoplanet_round_trip_and_conventions():
    mjd = T_REF + onp.linspace(0.0, 400.0, 9)
    with jax.enable_x64(True):
        orbit = _orbit()
        body, scale = orbit.to_jaxoplanet()
        x, y, z = body.relative_position(mjd - T_REF)
        # (dra, ddec, dz) = (Y, X, -Z): jaxoplanet's Z points to us.
        assert onp.allclose(
            onp.array([y, x, -z]) * scale, onp.array(orbit.relative(mjd))
        )
        # Our ω is the secondary's, jaxoplanet's the primary's.
        assert float(
            onp.rad2deg(onp.arctan2(body.sin_omega_peri, body.cos_omega_peri))
        ) == pytest.approx(ORBIT["omega"] - 180.0)
        # The primary's radial velocity (redshift positive) is positive
        # while the secondary approaches (dz falling).
        vz = onp.array(orbit.relative_velocity(mjd)[2])
        rv = onp.array(body.radial_velocity(mjd - T_REF))
        assert onp.all(onp.sign(rv) == -onp.sign(vz))
        back = KeplerOrbit.from_jaxoplanet(body, a_mas=20.0, t_ref=T_REF)
        for name in ("period", "dt_peri", "ecc", "inc", "omega", "Omega"):
            assert float(getattr(back, name)) == pytest.approx(
                ORBIT[name], abs=1e-9
            )


def test_orbits_are_differentiable_under_jit():
    dt = onp.linspace(0.0, 400.0, 5)

    @jax.jit
    def separation(orbit, dt):
        dra, ddec, _ = orbit._relative(dt)
        return (dra**2 + ddec**2).sum()

    grads = jax.grad(separation)(_orbit(), dt)
    assert onp.all(onp.isfinite(onp.array([grads.ecc, grads.omega])))


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"period": 0.0}, "period"),
        ({"ecc": 1.0}, "ecc"),
        ({"inc": 200.0}, "inc"),
        ({"a_mas": -1.0}, "a_mas"),
        ({"omega": float("nan")}, "omega"),
    ],
)
def test_out_of_domain_orbits_are_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        _orbit(**changes)
    # Traced values are not checked (they may be mid-optimisation).
    jax.jit(lambda p: _orbit(period=p).period)(0.0)


def test_missing_jaxoplanet_names_the_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "jaxoplanet", None)
    with pytest.raises(ImportError, match=r"virgil-astro\[orbits\]"):
        _orbit().to_jaxoplanet()
    with pytest.raises(ImportError, match=r"virgil-astro\[orbits\]"):
        _orbit().relative(T_REF)
