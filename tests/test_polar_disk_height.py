"""PolarDisk with a height and harmonics, and TruncatedCone on the engine."""

import time

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from jaxbessel import bessel_jn, j0

from virgil import models
from virgil._geometry import image_coordinates, offset_phase
from virgil._utils import dtor, mas2rad
from virgil.models import (
    Cone,
    ExpTail,
    GaussianRing,
    Harmonics,
    ModulatedGaussianRim,
    PointSource,
    PolarDisk,
    PowerLawHeight,
    RadialLinear,
    RadialPower,
    ThinRing,
    TruncatedCone,
)

WAVEL = 2.2e-6
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0


def _uv(n=30, seed=0, qmax=0.3, qmin=0.01):
    """Baselines (metres) with spatial frequencies in cycles/mas."""
    rng = onp.random.default_rng(seed)
    q = rng.uniform(qmin, qmax, n) / _MAS2RAD_REF  # cycles/rad
    theta = rng.uniform(0.0, 2.0 * onp.pi, n)
    return q * onp.cos(theta) * WAVEL, q * onp.sin(theta) * WAVEL


def _cvis(model, u, v):
    return onp.asarray(model.model(np.asarray(u), np.asarray(v), WAVEL))


# --- the old midpoint implementation of TruncatedCone, as the reference --------


def _old_cone_cvis(u, v, n_rings, t_max=5.0, **p):
    """TruncatedCone's visibility before it ran on the engine: the midpoint
    rule on ``n_rings`` rings from ``s0`` to ``s0 + t_max length``, a copy of
    the pre-subsumption code. ``p`` holds the cone's parameters."""
    tip, alpha, s0, length, width = (
        p[k] for k in ("tip", "alpha", "s0", "length", "width")
    )
    tilt, pa, ratio = (
        p.get(k, d) for k, d in (("tilt", 0.0), ("pa", 0.0), ("ratio", 1.0))
    )
    amps, pas = p.get("az_amps"), p.get("az_pas")
    uu, vv = np.ravel(u / WAVEL)[:, None], np.ravel(v / WAVEL)[:, None]
    t = (np.arange(n_rings) + 0.5) / n_rings * t_max
    s = s0 + t * length
    rho = s * np.sin(alpha * dtor)
    along = (s * np.cos(alpha * dtor) - tip) * np.cos(tilt * dtor)
    weight = rho * np.exp(-(s - s0) / length)
    weight = weight / np.sum(weight)
    sin_pa, cos_pa = np.sin(pa * dtor), np.cos(pa * dtor)
    q_par, q_perp = uu * sin_pa + vv * cos_pa, uu * cos_pa - vv * sin_pa
    squash = ratio * np.sin(tilt * dtor)
    q = np.sqrt(q_perp**2 + (squash * q_par) ** 2 + 1e-30)
    shift = offset_phase(uu, vv, along * sin_pa, along * cos_pa)
    x = 2.0 * np.pi * mas2rad * rho * q
    if amps is None:
        rings = j0(x)
    else:
        amps, pas = (
            np.atleast_1d(np.asarray(amps)),
            np.atleast_1d(np.asarray(pas)),
        )
        turn = (q_perp + 1j * squash * q_par) / q
        bessel = bessel_jn(amps.size, x)
        rings = bessel[0]
        for m in range(1, amps.size + 1):
            phase = (pa + 90.0 - pas[m - 1]) * dtor
            rings = rings + (
                amps[m - 1]
                * (-1j) ** m
                * np.real(turn**m * np.exp(-1j * m * phase))
                * bessel[m]
            )
    rings = np.matmul(
        rings * shift, weight, precision=jax.lax.Precision.HIGHEST
    )
    fwhm = width * mas2rad
    envelope = np.exp(
        -(np.pi**2)
        * fwhm**2
        * (uu[:, 0] ** 2 + vv[:, 0] ** 2)
        / (4.0 * np.log(2.0))
    )
    return (rings * envelope).reshape(np.shape(u))


_CONES = {
    "plain": dict(
        tip=5.0, alpha=62.5, s0=4.0, length=13.8, width=2.0, tilt=40.0, pa=96.0
    ),
    "negative tilt": dict(
        tip=5.0,
        alpha=30.0,
        s0=4.0,
        length=8.0,
        width=1.0,
        tilt=-25.0,
        pa=40.0,
        ratio=0.8,
    ),
    "edge on": dict(
        tip=0.0, alpha=45.0, s0=2.0, length=10.0, width=0.5, tilt=0.0, pa=10.0
    ),
    "down the axis": dict(
        tip=3.0, alpha=40.0, s0=4.0, length=6.0, width=0.3, tilt=90.0, pa=70.0
    ),
    "modulated": dict(
        tip=5.0,
        alpha=40.0,
        s0=4.0,
        length=6.0,
        width=0.3,
        tilt=35.0,
        pa=70.0,
        az_amps=(0.4, 0.2, 0.1),
        az_pas=(100.0, 30.0, 250.0),
    ),
    "modulated, negative tilt, squashed": dict(
        tip=2.0,
        alpha=55.0,
        s0=3.0,
        length=9.0,
        width=1.0,
        tilt=-50.0,
        pa=200.0,
        ratio=1.3,
        az_amps=(0.3, 0.2),
        az_pas=(300.0, 10.0),
    ),
}


@pytest.mark.parametrize("name", list(_CONES))
def test_cone_agrees_with_the_old_midpoint_stack_at_large_n_rings(name):
    p = _CONES[name]
    u, v = _uv(40, seed=1, qmax=0.3)
    with jax.enable_x64(True):
        reference = onp.asarray(_old_cone_cvis(u, v, 8192, **p))
        new = onp.asarray(
            TruncatedCone(**p, n_rings=64).model(
                np.asarray(u), np.asarray(v), WAVEL
            )
        )
    assert onp.max(onp.abs(new - reference)) < 1e-6


def test_cone_agrees_with_the_old_stack_in_float32():
    p = _CONES["modulated"]
    u, v = _uv(40, seed=2, qmax=0.3)
    reference = onp.asarray(_old_cone_cvis(u, v, 4096, **p))
    new = _cvis(TruncatedCone(**p, n_rings=64), u, v)
    assert onp.max(onp.abs(new - reference)) < 5e-5


def test_the_new_rule_reaches_1e5_with_few_nodes_where_midpoint_needs_hundreds():
    p = _CONES["plain"]
    u, v = _uv(40, seed=3, qmax=0.3)
    with jax.enable_x64(True):
        reference = onp.asarray(_old_cone_cvis(u, v, 8192, **p))
        error = lambda x: onp.max(onp.abs(x - reference))  # noqa: E731
        new = {
            n: error(
                onp.asarray(
                    TruncatedCone(**p, n_rings=n).model(
                        np.asarray(u), np.asarray(v), WAVEL
                    )
                )
            )
            for n in (24, 48)
        }
        midpoint = {
            n: error(onp.asarray(_old_cone_cvis(u, v, n, **p)))
            for n in (48, 256)
        }
    # The spacing of 0.3 cycles/mas fringes needs about 48 nodes; 24 reach
    # 1e-5 for baselines to 0.1 cycles/mas (next assertion).
    assert new[48] < 1e-5
    assert midpoint[48] > 100.0 * new[48]
    assert midpoint[256] > 1e-6  # the midpoint error is still ~1e-5 here
    u, v = _uv(40, seed=3, qmax=0.1)
    with jax.enable_x64(True):
        reference = onp.asarray(_old_cone_cvis(u, v, 8192, **p))
        new24 = TruncatedCone(**p, n_rings=24).model(
            np.asarray(u), np.asarray(v), WAVEL
        )
        assert onp.max(onp.abs(onp.asarray(new24) - reference)) < 1e-5


def test_new_cone_is_faster_than_256_midpoint_rings():
    p = _CONES["plain"]
    rng = onp.random.default_rng(0)
    u = np.asarray(rng.uniform(-100.0, 100.0, 4000))
    v = np.asarray(rng.uniform(-100.0, 100.0, 4000))
    cone = TruncatedCone(**p, n_rings=48)
    new = jax.jit(lambda cone, u, v: cone.model(u, v, WAVEL))
    old = jax.jit(lambda u, v: _old_cone_cvis(u, v, 256, **p))

    def best(f, *args):
        jax.block_until_ready(f(*args))
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(f(*args))
            times.append(time.perf_counter() - t0)
        return min(times)

    assert best(new, cone, u, v) < best(old, u, v)


@pytest.mark.parametrize("name", ["plain", "modulated", "edge on"])
def test_cone_gradients_are_finite(name):
    p = {k: v for k, v in _CONES[name].items() if not k.startswith("az")}
    modulated = "az_amps" in _CONES[name]
    u, v = _uv(12, seed=4)
    u, v = np.asarray(onp.r_[0.0, u]), np.asarray(onp.r_[0.0, v])

    def loss(cone):
        vis = cone.model(u, v, WAVEL)
        return np.sum(np.abs(vis) ** 2)

    cone = TruncatedCone(
        **p,
        **(
            dict(
                az_amps=_CONES[name]["az_amps"], az_pas=_CONES[name]["az_pas"]
            )
            if modulated
            else {}
        ),
    )
    grads = eqx.filter_grad(loss)(cone)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert onp.all(onp.isfinite(onp.asarray(leaf)))
    # The edge-on cone has tilt 0: its tilt gradient must also be finite.


def test_cone_visibility_is_one_at_the_origin():
    for p in _CONES.values():
        vis = TruncatedCone(**p).model(np.zeros(2), np.zeros(2), WAVEL)
        assert onp.allclose(onp.asarray(vis), 1.0, atol=1e-6)


def test_cone_with_more_than_the_node_limit_is_still_valid():
    # The Gauss-Legendre rule has no node limit (the Laguerre rule of an
    # ExpTail does), so the old tests' n_rings=256 keep working.
    u, v = _uv(5)
    a = _cvis(TruncatedCone(**_CONES["plain"], n_rings=256), u, v)
    b = _cvis(TruncatedCone(**_CONES["plain"], n_rings=64), u, v)
    assert onp.allclose(a, b, atol=1e-5)


def test_cone_is_a_polar_disk_with_a_conical_height():
    p = dict(tip=3.0, alpha=40.0, s0=4.0, length=6.0, tilt=35.0, pa=70.0)
    cone = TruncatedCone(**p, width=1e-4, n_rings=48)
    sin_a = onp.sin(onp.radians(p["alpha"]))
    disk = PolarDisk(
        ExpTail(p["s0"] * sin_a, p["length"] * sin_a, cutoff=5.0),
        height=Cone(p["alpha"], p["tip"]),
        inc=90.0 - p["tilt"],
        pa=p["pa"] - 90.0,
        n_r=48,
    )
    u, v = _uv(30, seed=5)
    assert onp.max(onp.abs(_cvis(cone, u, v) - _cvis(disk, u, v))) < 2e-6


# --- harmonics ---------------------------------------------------------------------


def test_harmonics_of_a_thin_ring_match_the_modulated_rim():
    amps, phis = (0.3, 0.2, 0.1), (40.0, 100.0, 250.0)
    # The rim's az_pas are position angles on the sky; its modulation is
    # measured in the disk plane from the projected major axis, at PA pa.
    pa = 30.0
    angle = lambda m, p: m * onp.radians(p - pa)  # noqa: E731
    c = [
        a * onp.cos(angle(m, p)) for m, (a, p) in enumerate(zip(amps, phis), 1)
    ]
    s = [
        a * onp.sin(angle(m, p)) for m, (a, p) in enumerate(zip(amps, phis), 1)
    ]
    rim = ModulatedGaussianRim(
        diam=12.0, fwhm=1.5, inc=50.0, pa=30.0, az_amps=amps, az_pas=phis
    )
    disk = PolarDisk(
        ThinRing(6.0),
        harmonics=Harmonics(c=c, s=s),
        inc=50.0,
        pa=30.0,
        blur_fwhm=1.5,
    )
    u, v = _uv(40, seed=6, qmax=0.2)
    assert onp.max(onp.abs(_cvis(rim, u, v) - _cvis(disk, u, v))) < 2e-6


@pytest.mark.parametrize("axis", ["c", "s"])
def test_harmonics_sign_first_order(axis):
    # c1 brightens the side of the projected major axis at PA pa; s1 the side
    # at PA pa + 90. Face-on, the bright side is where the centroid goes,
    # and V = 1 - 2 pi i (u x + v y) for a source with centroid (x, y).
    pa = 30.0
    harmonics = Harmonics(**{axis: 0.6})
    disk = PolarDisk(
        GaussianRing(5.0, 0.5), harmonics=harmonics, pa=pa, n_r=32
    )
    bright = pa if axis == "c" else pa + 90.0
    direction = onp.array(
        [onp.sin(onp.radians(bright)), onp.cos(onp.radians(bright))]
    )
    u, v = 1e5 * direction[0] * WAVEL, 1e5 * direction[1] * WAVEL  # tiny q
    vis = onp.asarray(disk.model(np.array([u]), np.array([v]), WAVEL))[0]
    # Along the bright direction the centroid is positive: Im V < 0.
    assert vis.imag < -1e-4
    # And perpendicular to it nothing shifts.
    vis_perp = onp.asarray(disk.model(np.array([-v]), np.array([u]), WAVEL))[0]
    assert abs(vis_perp.imag) < 0.01 * abs(vis.imag)
    image = onp.asarray(disk.render(npix=129, fov_mas=24.0))
    xx, yy = (onp.asarray(a) for a in image_coordinates(129, 24.0))
    centroid = (
        onp.array([onp.sum(image * xx), onp.sum(image * yy)]) / image.sum()
    )
    assert onp.dot(centroid, direction) > 0.5
    assert abs(centroid @ onp.array([direction[1], -direction[0]])) < 0.05


def _render_visibilities(model, u, v, npix=384, fov_mas=60.0):
    image = onp.asarray(model.render(npix=npix, fov_mas=fov_mas)).ravel()
    xx, yy = (onp.asarray(a).ravel() for a in image_coordinates(npix, fov_mas))
    phase = onp.exp(
        -2j
        * onp.pi
        * _MAS2RAD_REF
        * (onp.outer(u, xx) + onp.outer(v, yy))
        / WAVEL
    )
    return phase @ image / image.sum()


_RENDERED = {
    "harmonics": PolarDisk(
        GaussianRing(9.0, 1.5),
        harmonics=Harmonics(c=(0.3, -0.2, 0.1), s=(-0.4, 0.15)),
        inc=45.0,
        pa=70.0,
        dra=1.0,
        ddec=-2.0,
    ),
    "radial harmonics": PolarDisk(
        GaussianRing(9.0, 2.5),
        harmonics=Harmonics(
            c=RadialLinear(0.0, 0.05, 9.0), s=(RadialPower(0.2, 1.0, 9.0), 0.1)
        ),
        inc=30.0,
        pa=20.0,
    ),
    "flared": PolarDisk(
        GaussianRing(9.0, 1.5),
        height=PowerLawHeight(2.5, 9.0, 1.0),
        inc=60.0,
        pa=110.0,
    ),
    "cone": PolarDisk(
        ExpTail(3.0, 3.0, cutoff=6.0),
        height=Cone(35.0, 2.0),
        inc=50.0,
        pa=15.0,
        n_r=64,
    ),
    "cone with harmonics": PolarDisk(
        ExpTail(3.0, 3.0, cutoff=6.0),
        harmonics=Harmonics(c=0.3, s=(-0.5, 0.2)),
        height=Cone(35.0, 2.0),
        inc=50.0,
        pa=15.0,
        blur_fwhm=1.5,
        n_r=64,
    ),
}


@pytest.mark.parametrize("name", list(_RENDERED))
def test_render_fourier_transform_matches_model_visibilities(name):
    disk = _RENDERED[name]
    u, v = _uv(30, seed=7, qmax=0.1)
    cvis_render = _render_visibilities(disk, u, v)
    assert onp.max(onp.abs(cvis_render - _cvis(disk, u, v))) < 1.2e-2


def test_height_shifts_towards_pa_plus_90_by_h_sin_i():
    h, inc, pa = 3.0, 55.0, 30.0
    ring = PolarDisk(
        GaussianRing(0.0, 0.01),
        height=PowerLawHeight(h, 1.0, 0.0),
        inc=inc,
        pa=pa,
        n_r=16,
    )
    shift = h * onp.sin(onp.radians(inc))
    point = PointSource(
        dra=shift * onp.cos(onp.radians(pa)),
        ddec=-shift * onp.sin(onp.radians(pa)),
    )
    u, v = _uv(20, seed=8, qmax=0.05)
    assert onp.allclose(_cvis(ring, u, v), _cvis(point, u, v), atol=2e-5)
    # Face-on, a height changes nothing; the render moves with the shift.
    flat = PolarDisk(
        GaussianRing(0.0, 0.01), height=PowerLawHeight(h, 1.0, 0.0)
    )
    assert onp.allclose(
        _cvis(flat, u, v), 1.0 * _cvis(PointSource(), u, v), atol=2e-5
    )
    image = onp.asarray(ring.render(npix=129, fov_mas=24.0))
    xx, yy = (onp.asarray(a) for a in image_coordinates(129, 24.0))
    centroid = (
        onp.array([onp.sum(image * xx), onp.sum(image * yy)]) / image.sum()
    )
    assert onp.allclose(centroid, [point.dra, point.ddec], atol=0.2)


def test_cone_height_lifts_rings_by_r_cot_alpha_minus_tip():
    alpha, tip, r0, inc, pa = 40.0, 1.5, 6.0, 65.0, 120.0
    ring = PolarDisk(
        ThinRing(r0), height=Cone(alpha, tip), inc=inc, pa=pa, blur_fwhm=0.5
    )
    lift = (r0 / onp.tan(onp.radians(alpha)) - tip) * onp.sin(onp.radians(inc))
    flat = PolarDisk(
        ThinRing(r0),
        inc=inc,
        pa=pa,
        blur_fwhm=0.5,
        dra=lift * onp.cos(onp.radians(pa)),
        ddec=-lift * onp.sin(onp.radians(pa)),
    )
    u, v = _uv(20, seed=9)
    assert onp.allclose(_cvis(ring, u, v), _cvis(flat, u, v), atol=2e-6)


def test_visibility_is_one_at_zero_baseline_with_height_and_harmonics():
    for disk in _RENDERED.values():
        vis = disk.model(np.zeros(3), np.zeros(3), WAVEL)
        assert onp.allclose(onp.asarray(vis), 1.0, atol=1e-6)


def test_gradients_are_finite_with_height_and_harmonics():
    u, v = _uv(10, seed=10)
    u, v = np.asarray(onp.r_[0.0, u]), np.asarray(onp.r_[0.0, v])

    def loss(disk):
        return np.sum(np.abs(disk.model(u, v, WAVEL)) ** 2)

    for disk in _RENDERED.values():
        grads = eqx.filter_grad(loss)(disk)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert onp.all(onp.isfinite(onp.asarray(leaf)))


def test_changing_height_and_harmonics_values_does_not_recompile():
    traces = []

    @eqx.filter_jit
    def evaluate(model, u, v):
        traces.append(1)
        return model.model(u, v, WAVEL)

    u, v = (np.asarray(a) for a in _uv(10))
    for c, h0 in [(0.1, 1.0), (0.3, 2.0), (-0.2, 3.0)]:
        disk = PolarDisk(
            GaussianRing(5.0, 1.0),
            harmonics=Harmonics(c=c, s=(0.1, c)),
            height=PowerLawHeight(h0, 5.0, 1.0),
            inc=30.0,
            n_r=24,
        )
        evaluate(disk, u, v)
    assert len(traces) == 1
    # A different degree is a different program.
    evaluate(
        PolarDisk(
            GaussianRing(5.0, 1.0),
            harmonics=Harmonics(c=0.1, degree=3),
            n_r=24,
        ),
        u,
        v,
    )
    assert len(traces) == 2


def test_chunked_evaluation_matches_with_height_and_harmonics(monkeypatch):
    disk = _RENDERED["cone with harmonics"]
    u, v = _uv(37)
    whole = _cvis(disk, u, v)
    monkeypatch.setattr(models, "_POLAR_CHUNK_ELEMENTS", 5 * 64 * 3)
    assert onp.allclose(whole, _cvis(disk, u, v), atol=1e-6)


def test_parameters_are_reachable_through_paths():
    disk = PolarDisk(
        GaussianRing(5.0, 1.0),
        harmonics=Harmonics(c=(0.3, 0.1), s=0.2),
        height=Cone(40.0, 1.0),
    )
    assert float(disk.get("harmonics.c.0")) == pytest.approx(0.3)
    assert float(disk.get("height.alpha")) == 40.0
    assert float(disk.set("harmonics.s.1", 0.4).harmonics.s[1]) == 0.4


def test_min_intensity_reports_positivity_violations():
    ring = GaussianRing(5.0, 1.0)
    assert float(PolarDisk(ring).min_intensity()) == 1.0
    ok = PolarDisk(ring, harmonics=Harmonics(c=0.5, s=0.5))
    assert float(ok.min_intensity()) == pytest.approx(
        1.0 - onp.sqrt(0.5), abs=1e-4
    )
    assert bool(ok.is_physical())
    bad = PolarDisk(ring, harmonics=Harmonics(c=1.2))
    assert float(bad.min_intensity()) == pytest.approx(-0.2, abs=1e-4)
    assert not bool(bad.is_physical())
    # A radial function: only the outer nodes violate it.
    grows = PolarDisk(ring, harmonics=Harmonics(c=RadialLinear(0.0, 0.3, 5.0)))
    assert float(grows.min_intensity()) < 0.0


def test_harmonics_and_height_inputs_are_validated():
    with pytest.raises(ValueError, match="linear"):
        Harmonics(c=0.1, form="positive")
    with pytest.raises(ValueError, match="degree"):
        Harmonics(c=(0.1, 0.2), degree=1)
    with pytest.raises(ValueError, match="scalar"):
        Harmonics(c=[[0.1, 0.2]])
    with pytest.raises(TypeError, match="Harmonics"):
        PolarDisk(GaussianRing(5.0, 1.0), harmonics=0.3)
    with pytest.raises(TypeError, match="height"):
        PolarDisk(GaussianRing(5.0, 1.0), height=2.0)
    assert not bool(
        PolarDisk(GaussianRing(5.0, 1.0), height=Cone(95.0)).is_physical()
    )
    assert Harmonics(degree=2).degree == 2


@pytest.mark.parametrize("modulated", [False, True])
def test_apep_like_cone_old_against_new(modulated):
    # Edge-on, 130 m baselines at 2.2 um (0.29 cycles/mas): the converged
    # midpoint stack is the reference; 64 midpoint rings are off by ~5e-4,
    # while 48 Gauss-Legendre nodes are exact to ~1e-8 (40 reach 1e-6).
    p = dict(
        tip=12.0, alpha=63.0, s0=8.0, length=14.0, width=2.0, tilt=0.0, pa=96.0
    )
    if modulated:
        p.update(az_amps=(0.4,), az_pas=(150.0,))
    rng = onp.random.default_rng(0)
    radius = 130.0 * onp.sqrt(rng.uniform(0.0, 1.0, 300))
    angle = rng.uniform(0.0, 2.0 * onp.pi, 300)
    u, v = radius * onp.cos(angle), radius * onp.sin(angle)
    with jax.enable_x64(True):
        reference = onp.asarray(_old_cone_cvis(u, v, 8192, **p))
        error = lambda x: onp.max(onp.abs(onp.asarray(x) - reference))  # noqa: E731
        midpoint64 = error(_old_cone_cvis(u, v, 64, **p))
        new = {
            n: error(
                TruncatedCone(**p, n_rings=n).model(
                    np.asarray(u), np.asarray(v), WAVEL
                )
            )
            for n in (24, 40, 48)
        }
    assert midpoint64 > 1e-4
    assert new[24] > 1e-3  # too few nodes for these baselines
    assert new[40] < 1e-5 and new[48] < 1e-6
