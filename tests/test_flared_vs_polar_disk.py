"""FlaredDisk* against a PolarDisk built to match (comparison only).

The findings are written up in ``design/flared_disk_vs_polar_disk.md``.
FlaredDisk is unchanged; these tests pin what the two agree on, and the size
of the difference where they do not.
"""

import numpy as onp
import pytest

from virgil._utils import mas2rad
from virgil.models import (
    FlaredDiskPowerLaw,
    GaussianRing,
    Harmonics,
    PolarDisk,
    PowerLawHeight,
)

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
