"""FlaredDisk* against a PolarDisk built to match (comparison only).

The findings are written up in ``design/flared_disk_vs_polar_disk.md``.
FlaredDisk is unchanged; these tests pin what the two agree on, and the size
of the difference where they do not.
"""

import jax.numpy as np
import numpy as onp
import pytest

from virgil._geometry import undo_elliptical_transf_coord
from virgil._utils import mas2rad
from virgil.models import (
    FlaredDiskPowerLaw,
    GaussianRing,
    Harmonics,
    PolarDisk,
    PowerLawHeight,
)

DTOR = onp.pi / 180.0
R0, FWHM, PA, WAVEL = 20.0, 8.0, 30.0, 1e-6
SIGMA = FWHM / 2.3548200450309493


def _baselines():
    q = onp.linspace(0.002, 0.06, 24)  # cycles per mas
    ang = onp.linspace(0.0, onp.pi, 8, endpoint=False)
    qq, aa = onp.meshgrid(q, ang)
    qq, aa = qq.ravel(), aa.ravel()
    scale = WAVEL / mas2rad
    return qq * onp.cos(aa) * scale, qq * onp.sin(aa) * scale


def _flared(inc, aspect=0.0, flaring=1.0, n=0.0, symmetric=1.0):
    return FlaredDiskPowerLaw(
        n=n,
        radius=R0,
        fwhm=FWHM,
        inc=inc,
        pa=PA,
        npix=192,
        pixel_scale_mas=0.5,
        aspect=aspect,
        flaring=flaring,
        symmetric=symmetric,
    )


def _polar(inc, height=None, harmonics=None):
    return PolarDisk(
        GaussianRing(R0, SIGMA),
        harmonics=harmonics,
        height=height,
        inc=inc,
        pa=PA,
        n_r=96,
    )


def _max_dv(a, b):
    u, v = _baselines()
    return float(
        onp.max(
            onp.abs(
                onp.asarray(a.model(u, v, WAVEL))
                - onp.asarray(b.model(u, v, WAVEL))
            )
        )
    )


@pytest.mark.parametrize("inc", [0.0, 50.0, 70.0])
def test_a_flat_axisymmetric_disk_is_a_gaussian_ring(inc):
    assert _max_dv(_flared(inc), _polar(inc)) < 1e-4


@pytest.mark.parametrize("symmetric", [0.5, 3.0])
def test_the_cos_squared_phase_function_is_a_first_harmonic(symmetric):
    # f(theta) = cos^2(theta / 2) = (1 + cos theta) / 2, with theta from the
    # near-side minor axis: sin of PolarDisk's azimuth from the major axis.
    amplitude = 0.5 / (0.5 + symmetric)
    harmonics = Harmonics(c=0.0, s=amplitude, form="linear")
    flared = _flared(50.0, n=2.0, symmetric=symmetric)
    assert _max_dv(flared, _polar(50.0, harmonics=harmonics)) < 1e-4


def test_flaring_agrees_with_a_polar_disk_height_to_a_few_percent():
    inc, aspect = 50.0, 0.1
    z = aspect * R0
    flared = _flared(inc, aspect=aspect, flaring=1.0)
    flat = _max_dv(flared, _polar(inc))
    # The surface lifts the ring towards the far side by about z cos(inc)
    # (not z sin(inc) towards the near side): a height of -0.75 z at 50 deg.
    matched = _polar(inc, PowerLawHeight(-0.75 * z, R0, 1.0))
    naive = _polar(inc, PowerLawHeight(z, R0, 1.0))
    assert _max_dv(flared, matched) < 0.05
    assert _max_dv(flared, matched) < 0.5 * flat
    assert _max_dv(flared, naive) > 0.15


def _thin(
    inc,
    aspect,
    flaring,
    use_tan=False,
    consistent=True,
    drop_z2=True,
    jacobian=True,
):
    """FlaredDisk with the exact thin-surface geometry (test reference).

    Differs from FlaredDisk in four ways, each switched by a flag: the
    shift of the ring in deprojected coordinates is z tan(i) rather than
    z sin(i); z is evaluated at the radius of the surface point itself
    (fixed-point iteration) rather than the pixel's mid-plane radius; z^2
    is not added to the radius; and the brightness is divided by the
    Jacobian dy/dy_mid so that each ring keeps its flux (PolarDisk's
    rings are flux-conserving).
    """

    class Thin(FlaredDiskPowerLaw):
        def _centred_image(self, xx, yy, pixel_scale_mas):
            cos_inc = np.cos(self.inc * DTOR)
            y, x = undo_elliptical_transf_coord(xx, yy, self.pa, cos_inc)
            tilt = self.inc * DTOR
            s = np.tan(tilt) if use_tan else np.sin(tilt)

            def z_of(rad):
                return self.aspect * R0 * (rad / R0) ** self.flaring

            rad = np.sqrt(x**2 + y**2)
            if consistent:
                for _ in range(40):
                    rad = np.sqrt(x**2 + (y + z_of(rad) * s) ** 2)
                z = z_of(rad)
                r2 = rad**2
            else:
                z = z_of(rad)
                r2 = x**2 + (y + z * s) ** 2
            if not drop_z2:
                r2 = r2 + z**2
            offset = (np.sqrt(r2) - R0) / SIGMA
            out = np.exp(-0.5 * offset**2)
            if jacobian:
                zp = (
                    self.flaring
                    * self.aspect
                    * (rad / R0) ** (self.flaring - 1)
                )
                out = out / (1.0 - zp * (y + z * s) / rad * s)
            return out

    return Thin(
        n=0.0,
        radius=R0,
        fwhm=FWHM,
        inc=inc,
        pa=PA,
        npix=192,
        pixel_scale_mas=0.5,
        aspect=aspect,
        flaring=flaring,
        symmetric=1.0,
    )


@pytest.mark.parametrize("flaring", [1.0, 1.25])
@pytest.mark.parametrize("inc", [20.0, 50.0, 70.0])
def test_the_exact_thin_surface_is_a_polar_disk_with_negative_height(
    inc, flaring
):
    # Verified in design/flared_disk_vs_polar_disk.md: with tan(i), a
    # self-consistent z, no z^2 and a flux-conserving Jacobian, FlaredDisk's
    # construction is PowerLawHeight(h0 = -aspect r0) to the transform error
    # (1.5e-9 at every inclination and aspect tried, float64).
    aspect = 0.1
    polar = _polar(inc, PowerLawHeight(-aspect * R0, R0, flaring))
    thin = _thin(inc, aspect, flaring, use_tan=True)
    assert _max_dv(thin, polar) < 1e-5


@pytest.mark.parametrize(
    "flag,value",
    [
        ("use_tan", False),
        ("consistent", False),
        ("drop_z2", False),
        ("jacobian", False),
    ],
)
def test_each_ingredient_of_the_thin_surface_matters(flag, value):
    # Undoing any one of the four differences re-opens a gap well above the
    # transform error (2e-3 at least, at inclination 50, aspect 0.2).
    inc, aspect = 50.0, 0.2
    polar = _polar(inc, PowerLawHeight(-aspect * R0, R0, 1.0))
    flags = {"use_tan": True, flag: value}
    assert _max_dv(_thin(inc, aspect, 1.0, **flags), polar) > 2e-3


def test_flared_disk_as_written_has_the_far_side_sign():
    # FlaredDisk follows Blakely et al. (2024) Eq. 3 literally (sin i, mid-plane
    # z): its ring is displaced towards the far side, so the matching
    # PowerLawHeight is negative, and the opposite sign is a worse match
    # than no height.
    inc, aspect = 50.0, 0.1
    flared = _flared(inc, aspect=aspect, flaring=1.0)
    z = aspect * R0
    for_far = _max_dv(flared, _polar(inc, PowerLawHeight(-0.75 * z, R0, 1.0)))
    for_near = _max_dv(flared, _polar(inc, PowerLawHeight(0.75 * z, R0, 1.0)))
    assert for_far < _max_dv(flared, _polar(inc)) < for_near
