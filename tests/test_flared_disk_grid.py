"""The matrix Fourier transform of pixel-rendered components on a uv lattice."""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil import models
from virgil.coverage import ami_grid_record
from virgil.fitting import fit
from virgil.models import (
    FlaredDiskHG,
    FlaredDiskPowerLaw,
    PointSource,
    Rotated,
    System,
)
from virgil.oidata import OIData

ROTATION = -6.9


def _without_grid(data):
    return eqx.tree_at(
        lambda d: d.uv_grid, data, None, is_leaf=lambda x: x is None
    )


def _disk(**options):
    geometry = dict(
        n=3.0,
        radius=80.0,
        fwhm=40.0,
        inc=50.0,
        pa=160.0,
        npix=32,
        pixel_scale_mas=8.0,
        skew=2.0,
        aspect=0.1,
        flux=0.2,
        dra=3.0,
        ddec=-2.0,
    )
    return FlaredDiskPowerLaw(**(geometry | options))


def _disco_record(u, v, n_modes=40, seed=0):
    """A drpangloss-style DISCO dictionary (orthonormal modes) at ``(u, v)``."""
    rng = onp.random.default_rng(seed)
    modes = onp.linalg.qr(rng.normal(size=(u.size, n_modes)))[0].T
    return {
        "u": u,
        "v": v,
        "wavel": 4.3e-6,
        "vis": onp.zeros(n_modes),
        "d_vis": onp.ones(n_modes),
        "phi": onp.zeros(n_modes),
        "d_phi": onp.ones(n_modes),
        "disco_vis_mat": modes,
        "disco_phi_mat": modes,
        "vis_mode": "logamp",
        "cp_flag": False,
    }


def _lattice(rotation_deg, pitch=0.26, n=(25, 13), dtype=float):
    col, row = onp.meshgrid(onp.arange(n[0]), onp.arange(n[1]))
    a, b = pitch * (col.ravel() - 12), pitch * (row.ravel() + 1)
    c, s = (
        onp.cos(onp.radians(rotation_deg)),
        onp.sin(onp.radians(rotation_deg)),
    )
    u, v = c * a + s * b, -s * a + c * b
    return u.astype(dtype), v.astype(dtype)


@pytest.mark.validates(
    "virgil.models.FlaredDisk.model_on_grid", roots=["self-consistency"]
)
@pytest.mark.parametrize("rotation", [ROTATION, "grid"])
def test_matrix_transform_matches_the_direct_sum_in_x64(rotation):
    with jax.enable_x64(True):
        data = OIData(ami_grid_record(rotation_deg=ROTATION))
        assert data.uv_grid is not None
        exact = data.uv_grid.rotation_deg if rotation != "grid" else rotation
        disk = _disk(rotation_deg=exact, ft="dft")
        fast = disk.model_on_grid(data.u, data.v, data.wavel, data.uv_grid)
        # The direct sum on the same (rotated) pixels.
        frame = data.uv_grid.rotation_deg
        slow = _disk(rotation_deg=frame, ft="dft").model(
            data.u, data.v, data.wavel
        )
        assert np.max(np.abs(fast - slow)) < 1e-10 * np.max(np.abs(slow))


@pytest.mark.validates(
    "virgil.models.FlaredDisk.model_on_grid", roots=["self-consistency"]
)
def test_a_north_up_disk_on_a_rotated_lattice_keeps_the_direct_sum():
    data = OIData(ami_grid_record(rotation_deg=ROTATION))
    disk = _disk(rotation_deg=0.0, ft="dft")
    on_grid = disk.model_on_grid(data.u, data.v, data.wavel, data.uv_grid)
    assert np.array_equal(on_grid, disk.model(data.u, data.v, data.wavel))


def test_rotation_grid_is_north_up_without_a_lattice():
    rng = onp.random.default_rng(1)
    u, v = rng.uniform(-6.0, 6.0, (2, 40))
    follow, north = _disk(), _disk(rotation_deg=0.0)
    assert np.array_equal(
        follow.model(u, v, 4.3e-6), north.model(u, v, 4.3e-6)
    )
    with pytest.raises(ValueError, match="rotation_deg"):
        _disk(rotation_deg="sky")


def test_rotating_the_pixels_changes_only_the_discretisation():
    # A finely sampled disk looks the same in any pixel frame.
    rng = onp.random.default_rng(2)
    u, v = rng.uniform(-3.0, 3.0, (2, 40))
    fine = dict(npix=96, pixel_scale_mas=3.0, ft="dft")
    north = _disk(rotation_deg=0.0, **fine).model(u, v, 4.3e-6)
    turned = _disk(rotation_deg=31.7, **fine).model(u, v, 4.3e-6)
    assert np.max(np.abs(turned - north)) < 2e-3


def test_system_scenes_and_fits_use_the_matrix_transform(monkeypatch):
    calls = []
    original = models.grid_visibilities

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def boom(*args, **kwargs):
        raise AssertionError("per-point transform used on a matched lattice")

    monkeypatch.setattr(models, "grid_visibilities", counting)
    template = OIData(ami_grid_record(rotation_deg=ROTATION))

    def scene_at(rotation_deg):
        disk = FlaredDiskHG(
            g=0.3,
            radius=80.0,
            fwhm=40.0,
            inc=50.0,
            pa=160.0,
            npix=32,
            pixel_scale_mas=8.0,
            flux=0.2,
            rotation_deg=rotation_deg,
        )
        return System(star=PointSource(), disk=disk)

    scene = scene_at("grid")
    data = template.with_model(scene)
    expected = _without_grid(data).model(scene_at(data.uv_grid.rotation_deg))
    assert np.allclose(data.model(scene), expected, atol=1e-5)
    assert calls
    monkeypatch.setattr(models, "image_visibilities", boom)
    calls.clear()
    result = fit(
        scene,
        {"disk.flux": dist.Uniform(0.0, 1.0)},
        data,
        init={"disk.flux": 0.15},
        max_steps=5,
    )
    assert calls
    assert abs(float(result.values["disk.flux"]) - 0.2) < 0.05


def test_a_disco_dictionary_on_a_lattice_gets_a_uv_grid():
    # float32 coordinates, as in AMIGO's drpangloss-format products.
    u, v = _lattice(31.7, dtype=onp.float32)
    data = OIData(_disco_record(u, v))
    grid = data.uv_grid
    assert grid is not None
    assert np.isclose(grid.rotation_deg, 31.7, atol=1e-4)
    assert grid.u_axis.size * grid.v_axis.size == u.size
    with jax.enable_x64(True):
        data = OIData(_disco_record(u, v))
        disk = _disk(rotation_deg=data.uv_grid.rotation_deg, ft="dft")
        fast = data.model(disk)
        slow = _without_grid(data).model(disk)
        # Up to the float32 rounding of the stored coordinates.
        assert np.max(np.abs(fast - slow)) < 1e-5


@pytest.mark.parametrize(
    "case", ["irregular", "two wavelengths", "no DISCO", "sparse"]
)
def test_dictionaries_off_a_dense_disco_lattice_get_none(case):
    u, v = _lattice(10.0)
    record = _disco_record(u, v)
    if case == "irregular":
        rng = onp.random.default_rng(3)
        record["u"] = u + 1e-3 * rng.normal(size=u.size)
    elif case == "two wavelengths":
        record["wavel"] = onp.full(u.size, 4.3e-6)
        record["wavel"][::2] = 3.8e-6
    elif case == "no DISCO":
        record["vis_mat"] = record.pop("disco_vis_mat")
        record["phi_mat"] = record.pop("disco_phi_mat")
    else:
        # Two patches of one lattice, 400 cells apart: mostly empty.
        du, dv = (
            104.0 * onp.cos(onp.radians(10.0)),
            -104.0 * onp.sin(onp.radians(10.0)),
        )
        record = _disco_record(onp.r_[u, u + du], onp.r_[v, v + dv])
    assert OIData(record).uv_grid is None


def test_a_rotated_disk_takes_the_direct_path_eagerly_and_under_jit():
    # Rotated's angle is traced inside every fit, so it never forwards the
    # lattice: the disk renders North up in its own frame either way.
    data = OIData(ami_grid_record(rotation_deg=ROTATION))
    scene = Rotated(_disk(ft="dft"), 20.0)
    eager = data.model(scene)
    jitted = eqx.filter_jit(lambda m: data.model(m))(scene)
    direct = _without_grid(data).model(
        Rotated(_disk(rotation_deg=0.0, ft="dft"), 20.0)
    )
    assert np.array_equal(eager, direct)
    assert np.allclose(jitted, direct, rtol=1e-6, atol=1e-6)


def test_ensemble_fields_are_capped_on_a_disco_dictionary_lattice():
    from virgil.ensemble import EnsembleSpec, draw_groups

    u, v = _lattice(31.7, dtype=onp.float32)
    data = OIData(_disco_record(u, v))
    assert data.uv_grid is not None
    spec = EnsembleSpec(field_factors=(4.0,), field_mas=100.0, max_npix=1024)
    for d in draw_groups(data, 4, jax.random.PRNGKey(0), spec):
        assert d.npix * d.pixel_scale_mas <= 100.0 + d.pixel_scale_mas
