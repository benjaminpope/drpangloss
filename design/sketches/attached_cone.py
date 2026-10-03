"""Sketch: Apep's truncated cone attached to the binary frame of an orbit.

Design sketch for design/orbit_scene_joint_fitting.md, not library code. It
shows the proposed ``KeplerOrbit.relative`` (on jaxoplanet) and
``Attached(...).at(mjd)`` interfaces, evaluated with the existing
``TruncatedCone`` from the Apep analysis. It needs local paths (the
apep-gravity worktree and ~/data/apep_gravity/scripts) and jaxoplanet::

    uv run --python .venv/bin/python --with jaxoplanet \
        python design/sketches/attached_cone.py

Conventions (section 2 of the note): the relative vector r points from the
primary (the origin, here the WN star) to the secondary (the WC star), as
(dra, ddec, dz) in mas with dz positive away from the observer; omega is the
secondary's argument of periastron; Omega is the position angle (North
through East) of the node where the secondary recedes.
"""

import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(
    0,
    os.path.join(HOME, "code/drpangloss/.claude/worktrees/apep-gravity/src"),
)
sys.path.insert(0, os.path.join(HOME, "data/apep_gravity/scripts"))
jax.config.update("jax_enable_x64", True)

from jaxoplanet.orbits.keplerian import Body, Central  # noqa: E402
from jaxoplanet.orbits.keplerian import OrbitalBody  # noqa: E402
from truncated_cone import TruncatedCone  # noqa: E402

from drpangloss.models import PointSource, Resolved, System  # noqa: E402
from drpangloss.spectra import BlackBody, PowerLaw  # noqa: E402

DEG = np.pi / 180.0
YEAR = 365.25
W0 = 2.3e-6


def mjd(year):
    """MJD of a decimal year (J2000-based, good to a day)."""
    return 51544.5 + (np.asarray(year) - 2000.0) * YEAR


class KeplerOrbit(eqx.Module):
    """Relative orbit of a secondary about a primary, in angular units."""

    period: jax.Array  # days
    t_peri: jax.Array  # MJD - t_ref
    ecc: jax.Array
    inc: jax.Array  # deg; < 90 means the position angle increases
    omega: jax.Array  # deg, the secondary's argument of periastron
    Omega: jax.Array  # deg, PA of the node where the secondary recedes
    a_mas: jax.Array
    t_ref: float = eqx.field(static=True)  # float64 MJD, kept on the host

    def _body(self):
        # jaxoplanet's omega_peri is the primary's: the secondary's minus 180.
        # Its units (R_sun, M_sun, days) cancel once divided by the semimajor
        # axis; never pass its parallax argument (it scales by au, not 1/au).
        body = Body(
            period=self.period,
            time_peri=self.t_peri,
            eccentricity=self.ecc,
            omega_peri=(self.omega - 180.0) * DEG,
            inclination=self.inc * DEG,
            asc_node=self.Omega * DEG,
        )
        return OrbitalBody(Central(mass=1.0, radius=1.0), body)

    def relative(self, t):
        """(dra, ddec, dz) of the secondary at MJD t, in mas."""
        body = self._body()
        north, east, toward = body.relative_position(t - self.t_ref)
        scale = self.a_mas / body.semimajor
        return east * scale, north * scale, -toward * scale


class Attached(eqx.Module):
    """A component placed and oriented in the binary frame of an orbit.

    The component's origin is at ``anchor`` ("primary" or "secondary"); its
    axis (attributes ``pa`` and ``tilt``) points along the line of centres,
    primary to secondary, plus a fitted skew ``dpa``. The tilt is the line's
    elevation out of the sky plane (positive: the secondary is farther away).
    """

    component: eqx.Module
    orbit: KeplerOrbit
    dpa: jax.Array = 0.0
    anchor: str = eqx.field(static=True, default="secondary")

    def at(self, t):
        dra, ddec, dz = self.orbit.relative(t)
        pa = jnp.degrees(jnp.arctan2(dra, ddec))
        tilt = jnp.degrees(jnp.arctan2(dz, jnp.hypot(dra, ddec)))
        if self.anchor == "primary":
            dra, ddec = 0.0 * dra, 0.0 * ddec
        return eqx.tree_at(
            lambda c: (c.dra, c.ddec, c.pa, c.tilt),
            self.component,
            (dra, ddec, pa + self.dpa, tilt),
        )


# The cone posterior (results/cone_posterior.json) and an orbit with White et
# al. (2025)'s P, e, i and T_peri. Their omega = 10 deg is read as jaxoplanet's
# (the primary's), so the secondary's is 190 deg. Omega and a_mas are chosen
# to put the WC star at the measured 28.05 mas, PA 96.1 deg, in 2024.5: their
# 164 deg would put it at PA 164 deg (the open convention question).
p = dict(
    flux=0.341,
    companion_index=-4.75,
    tip=11.89,
    alpha=62.41,
    s0=7.81,
    length=13.68,
    width=1.53,
    dpa=0.39,
    cone_flux=3.16,
    T_cone=2987.0,
    halo_flux=0.517,
    halo_index=-1.49,
)
T_REF = float(mjd(2024.5))
orbit = KeplerOrbit(
    period=jnp.asarray(193.0 * YEAR),
    t_peri=jnp.asarray(mjd(1956.0) - T_REF),
    ecc=jnp.asarray(0.82),
    inc=jnp.asarray(24.0),
    omega=jnp.asarray(10.0 + 180.0),
    Omega=jnp.asarray(95.6),
    a_mas=jnp.asarray(16.36),
    t_ref=T_REF,
)
cone = Attached(
    TruncatedCone(
        p["tip"],
        p["alpha"],
        p["s0"],
        p["length"],
        p["width"],
        0.0,
        0.0,
        flux=BlackBody(p["cone_flux"], p["T_cone"], W0),
    ),
    orbit,
    dpa=jnp.asarray(p["dpa"]),
)


def scene_at(t, co_rotating=True):
    """The whole scene at MJD t; frozen at 2024.5 if not co-rotating."""
    t = t if co_rotating else T_REF
    dra, ddec, _ = orbit.relative(t)
    return System(
        star=PointSource(PowerLaw(1.0, -4.0, W0)),
        companion=PointSource(
            PowerLaw(p["flux"], p["companion_index"], W0), dra, ddec
        ),
        cone=cone.at(t),
        halo=Resolved(PowerLaw(p["halo_flux"], p["halo_index"], W0)),
    )


# Per-datum evaluation: each sample carries its own MJD (one per frame).
rng = np.random.default_rng(1)
n = 400
length = rng.uniform(30.0, 130.0, n)
angle = rng.uniform(0.0, np.pi, n)
u, v = length * np.sin(angle), length * np.cos(angle)
wavel = rng.uniform(2.21e-6, 2.40e-6, n)


def visibilities(t, co_rotating):
    def one(u, v, wavel):
        return scene_at(t, co_rotating).model(u, v, wavel)

    return jax.vmap(one)(u, v, wavel)


print("epoch    sep(mas)  PA(deg)  tilt(deg)  cone pa  max|dV| vs static")
for year in (2023.7, 2024.5, 2025.5):
    t = float(mjd(year))
    dra, ddec, dz = orbit.relative(t)
    sep = float(jnp.hypot(dra, ddec))
    pa = float(jnp.degrees(jnp.arctan2(dra, ddec))) % 360
    tilt = float(jnp.degrees(jnp.arctan2(dz, sep)))
    attached = cone.at(t)
    change = jnp.max(jnp.abs(visibilities(t, True) - visibilities(t, False)))
    print(
        f"{year:6.1f}  {sep:8.3f}  {pa:7.2f}  {tilt:9.3f}  "
        f"{float(attached.pa) % 360:7.2f}  {float(change):.4f}"
    )
