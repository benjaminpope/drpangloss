"""Keplerian orbits of a binary's secondary about its primary.

The conventions are those of ``design/orbit_scene_joint_fitting.md`` §2.1:

* ``dra`` is positive East and ``ddec`` positive North (mas), as everywhere
  in virgil, and ``dz`` is positive **away from** the observer, so that
  (dra, ddec, dz) is right-handed and ``dz`` grows while the secondary
  recedes. The vector runs from the primary (the scene's reference
  component) to the secondary.
* ``inc`` in [0°, 180°): below 90° the position angle increases with time
  (counterclockwise on the sky, North through East).
* ``Omega``: the position angle of the ascending node, the node where the
  secondary recedes. Positions alone fix it only modulo 180°.
* ``omega``: the **secondary's** argument of periastron (the visual-binary
  convention); the spectroscopic ω of the primary is ``omega - 180°``.
* ``dt_peri``: the time of periastron minus the static float64 ``t_ref``
  (days), so that float32 keeps it precise; ``period`` in days; ``a_mas``
  the angular semimajor axis of the relative orbit.

Kepler's equation is solved by jaxoplanet (an optional dependency,
``pip install "virgil-astro[orbits]"``), with its exact derivatives; the
positions follow from the Thiele–Innes constants (§2.4). jaxoplanet's own
conventions differ (its ω is the primary's, and its third axis points
toward the observer) and stay inside :meth:`KeplerOrbit.to_jaxoplanet` and
:meth:`KeplerOrbit.from_jaxoplanet`.
"""

import jax
import jax.numpy as np
import numpy as onp

import equinox as eqx
import zodiax as zx


__all__ = ["KeplerOrbit", "ThieleInnesOrbit"]


def _kepler(mean_anomaly, ecc):
    """``(sin f, cos f)`` of the true anomaly, from jaxoplanet's solver."""
    try:
        from jaxoplanet.core import kepler
    except ImportError as err:
        raise ImportError(
            'Orbits need jaxoplanet: pip install "virgil-astro[orbits]".'
        ) from err
    return kepler(mean_anomaly, ecc)


def _days_since(mjd, t_ref):
    """``mjd - t_ref`` in days, in float64 on the host when ``mjd`` is known.

    Under ``jit`` the subtraction happens in the traced precision, so pass
    times already relative to ``t_ref`` there (float32 resolves MJD ≈ 60000
    to only about 0.004 d).
    """
    if isinstance(mjd, jax.core.Tracer):
        return mjd - t_ref
    return np.asarray(onp.asarray(mjd, dtype=onp.float64) - t_ref)


def _unit_orbit(dt, period, dt_peri, ecc):
    """Thiele–Innes ``X = cos E - e`` and ``Y = √(1 - e²) sin E``.

    Computed from the true anomaly: ``X = (r/a) cos f``, ``Y = (r/a) sin f``.
    """
    mean_anomaly = 2.0 * np.pi * (dt - dt_peri) / period
    sin_f, cos_f = _kepler(mean_anomaly, ecc * np.ones_like(mean_anomaly))
    radius = (1.0 - ecc**2) / (1.0 + ecc * cos_f)
    return radius * cos_f, radius * sin_f


def _thiele_innes(a_mas, inc, omega, Omega):
    """``(A, B, F, G, C, H)`` for angles in degrees (§2.4)."""
    i, w, n = np.deg2rad(inc), np.deg2rad(omega), np.deg2rad(Omega)
    ci = np.cos(i)
    return (
        a_mas * (np.cos(w) * np.cos(n) - np.sin(w) * np.sin(n) * ci),
        a_mas * (np.cos(w) * np.sin(n) + np.sin(w) * np.cos(n) * ci),
        a_mas * (-np.sin(w) * np.cos(n) - np.cos(w) * np.sin(n) * ci),
        a_mas * (-np.sin(w) * np.sin(n) + np.cos(w) * np.cos(n) * ci),
        a_mas * np.sin(w) * np.sin(i),
        a_mas * np.cos(w) * np.sin(i),
    )


def _velocity(position, dt):
    """Time derivative (per day) of ``position(dt)``, exactly, by a JVP."""
    return jax.jvp(position, (dt,), (np.ones_like(dt),))[1]


class KeplerOrbit(zx.Base):
    """A Keplerian orbit of the secondary relative to the primary.

    Parameters
    ----------
    period : float
        Orbital period (days).
    dt_peri : float
        Time of periastron minus ``t_ref`` (days).
    ecc : float
        Eccentricity, ``0 <= ecc < 1``.
    inc : float
        Inclination (degrees, ``0 <= inc < 180``; below 90 the position
        angle increases with time).
    omega : float
        The secondary's argument of periastron (degrees), measured from the
        ascending node in the direction of motion.
    Omega : float
        Position angle of the ascending node, where the secondary recedes
        (degrees, North through East).
    a_mas : float
        Angular semimajor axis of the relative orbit (mas).
    t_ref : float, optional
        Reference time (MJD, float64, static). Times are measured from it,
        so that float32 keeps them precise.

    Notes
    -----
    (Omega + 180°, omega + 180°) gives the same sky positions with ``dz``
    reversed: positions alone cannot tell them apart.
    """

    period: jax.Array
    dt_peri: jax.Array
    ecc: jax.Array
    inc: jax.Array
    omega: jax.Array
    Omega: jax.Array
    a_mas: jax.Array
    t_ref: float = eqx.field(static=True)

    def __init__(
        self, period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref=0.0
    ):
        self.period = np.asarray(period, dtype=float)
        self.dt_peri = np.asarray(dt_peri, dtype=float)
        self.ecc = np.asarray(ecc, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.Omega = np.asarray(Omega, dtype=float)
        self.a_mas = np.asarray(a_mas, dtype=float)
        self.t_ref = float(t_ref)

    def thiele_innes(self):
        """The Thiele–Innes constants ``(A, B, F, G, C, H)`` (mas).

        ``ddec = A X + F Y``, ``dra = B X + G Y`` and ``dz = C X + H Y``,
        with ``X = cos E - e`` and ``Y = √(1 - e²) sin E``.
        """
        return _thiele_innes(self.a_mas, self.inc, self.omega, self.Omega)

    def _relative(self, dt):
        x, y = _unit_orbit(dt, self.period, self.dt_peri, self.ecc)
        a, b, f, g, c, h = self.thiele_innes()
        return np.stack([b * x + g * y, a * x + f * y, c * x + h * y])

    def relative(self, mjd):
        """Position of the secondary relative to the primary (mas).

        Parameters
        ----------
        mjd : array-like
            Times (MJD). Concrete times are offset from ``t_ref`` in float64
            first. Under ``jit`` the offset is taken in the traced
            precision, which in float32 is good to only about 0.004 d near
            MJD 60000.

        Returns
        -------
        tuple of arrays
            ``(dra, ddec, dz)``, each shaped like ``mjd``.
        """
        return tuple(self._relative(_days_since(mjd, self.t_ref)))

    def relative_velocity(self, mjd):
        """``d(dra, ddec, dz)/dt`` (mas per day), exactly."""
        dt = _days_since(mjd, self.t_ref)
        return tuple(_velocity(self._relative, dt))

    def separation_pa(self, mjd):
        """Separation (mas) and position angle (degrees, North through East,
        in [0, 360)) of the secondary from the primary."""
        dra, ddec, _ = self.relative(mjd)
        pa = np.mod(np.rad2deg(np.arctan2(dra, ddec)), 360.0)
        return np.hypot(dra, ddec), pa

    def to_thiele_innes(self):
        """The same sky orbit as a :class:`ThieleInnesOrbit`."""
        a, b, f, g, _, _ = self.thiele_innes()
        return ThieleInnesOrbit(
            self.period, self.dt_peri, self.ecc, a, b, f, g, self.t_ref
        )

    def to_jaxoplanet(self):
        """A jaxoplanet ``OrbitalBody`` for this orbit, and the factor that
        turns its relative positions into mas.

        jaxoplanet's times are days since ``t_ref``, its axes are
        (X, Y, Z) = (North, East, toward the observer), and its ω is the
        primary's: ``(dra, ddec, dz) = (Y, X, -Z) * scale``.
        """
        from jaxoplanet.orbits.keplerian import Body, Central, OrbitalBody

        body = OrbitalBody(
            Central(mass=1.0, radius=1.0),
            Body(
                period=self.period,
                time_peri=self.dt_peri,
                eccentricity=self.ecc,
                inclination=np.deg2rad(self.inc),
                omega_peri=np.deg2rad(self.omega) - np.pi,
                asc_node=np.deg2rad(self.Omega),
            ),
        )
        return body, self.a_mas / body.semimajor

    @classmethod
    def from_jaxoplanet(cls, body, a_mas, t_ref=0.0):
        """The orbit of a jaxoplanet ``OrbitalBody`` whose times are days
        since ``t_ref``, with angular semimajor axis ``a_mas``."""
        omega = np.rad2deg(
            np.arctan2(body.sin_omega_peri, body.cos_omega_peri)
        )
        Omega = np.rad2deg(np.arctan2(body.sin_asc_node, body.cos_asc_node))
        return cls(
            period=body.period,
            dt_peri=body.time_peri,
            ecc=body.eccentricity,
            inc=np.rad2deg(
                np.arctan2(body.sin_inclination, body.cos_inclination)
            ),
            omega=np.mod(omega + 180.0, 360.0),
            Omega=np.mod(Omega, 360.0),
            a_mas=a_mas,
            t_ref=t_ref,
        )


class ThieleInnesOrbit(zx.Base):
    """A sky orbit in Thiele–Innes form: linear in ``A``, ``B``, ``F``, ``G``.

    ``ddec = A X + F Y`` and ``dra = B X + G Y``, with ``X = cos E - e`` and
    ``Y = √(1 - e²) sin E``. For fixed ``(period, dt_peri, ecc)`` the
    positions are linear in the four constants, so a starting orbit is a
    linear least-squares solve on a grid of those three.

    Parameters
    ----------
    period, dt_peri, ecc : float
        As in :class:`KeplerOrbit`.
    A, B, F, G : float
        Thiele–Innes constants (mas).
    t_ref : float, optional
        Reference time (MJD, static float64).
    """

    period: jax.Array
    dt_peri: jax.Array
    ecc: jax.Array
    A: jax.Array
    B: jax.Array
    F: jax.Array
    G: jax.Array
    t_ref: float = eqx.field(static=True)

    def __init__(self, period, dt_peri, ecc, A, B, F, G, t_ref=0.0):
        self.period = np.asarray(period, dtype=float)
        self.dt_peri = np.asarray(dt_peri, dtype=float)
        self.ecc = np.asarray(ecc, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.F = np.asarray(F, dtype=float)
        self.G = np.asarray(G, dtype=float)
        self.t_ref = float(t_ref)

    def sky(self, mjd):
        """``(dra, ddec)`` of the secondary from the primary (mas)."""
        dt = _days_since(mjd, self.t_ref)
        x, y = _unit_orbit(dt, self.period, self.dt_peri, self.ecc)
        return self.B * x + self.G * y, self.A * x + self.F * y

    def to_kepler(self):
        """The :class:`KeplerOrbit` with these sky positions.

        Positions fix ``Omega`` only modulo 180°: the result has
        ``0 <= Omega < 180``, and (``Omega + 180``, ``omega + 180``) is the
        other solution, with ``dz`` reversed.
        """
        a, b, f, g = self.A, self.B, self.F, self.G
        total = np.arctan2(b - f, a + g)  # omega + Omega
        difference = np.arctan2(-(b + f), a - g)  # omega - Omega
        omega, Omega = (total + difference) / 2, (total - difference) / 2
        # Omega into [0, π): shifting both angles by π keeps the sky orbit.
        shift = np.floor(Omega / np.pi) * np.pi
        omega, Omega = omega - shift, Omega - shift
        half = (a**2 + b**2 + f**2 + g**2) / 2
        cos_term = a * g - b * f  # a² cos i
        a_sq = half + np.sqrt(np.maximum(half**2 - cos_term**2, 0.0))
        return KeplerOrbit(
            period=self.period,
            dt_peri=self.dt_peri,
            ecc=self.ecc,
            inc=np.rad2deg(np.arccos(np.clip(cos_term / a_sq, -1.0, 1.0))),
            omega=np.mod(np.rad2deg(omega), 360.0),
            Omega=np.rad2deg(Omega),
            a_mas=np.sqrt(a_sq),
            t_ref=self.t_ref,
        )
