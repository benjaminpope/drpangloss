"""FlaredDisk phase functions from cos(theta) = y/rho match the arctan2 formulas."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record
from virgil.models import (
    FlaredDiskGaussian,
    FlaredDiskHG,
    FlaredDiskPowerLaw,
)
from virgil.oidata import OIData

GEOMETRY = dict(
    radius=80.0,
    fwhm=40.0,
    inc=50.0,
    pa=160.0,
    npix=32,
    pixel_scale_mas=8.0,
    skew=2.0,
    aspect=0.1,
    flux=0.2,
    dra=0.0,
    ddec=0.0,
    ft="dft",
)


def _old(theta, cls, p):
    """The phase functions as written before cos(theta) was formed from y / rho."""
    if cls is FlaredDiskHG:
        g = p
        return (1.0 - g**2) / (
            4.0 * np.pi * (1.0 + g**2 - 2.0 * g * np.cos(theta)) ** 1.5
        )
    if cls is FlaredDiskPowerLaw:
        return np.maximum(np.cos(0.5 * theta), 0.0) ** p
    return np.exp(-0.5 * (theta / (p * np.pi / 180.0)) ** 2)


def _reference(cls):
    class Old(cls):
        def _phase_function(self, cos_theta, x, y):
            p = {
                FlaredDiskHG: lambda: self.g,
                FlaredDiskPowerLaw: lambda: self.n,
                FlaredDiskGaussian: lambda: self.sigma_theta,
            }[cls]()
            return _old(np.arctan2(x, y), cls, p)

    return Old


CASES = [
    (FlaredDiskHG, "g", 0.4),
    (FlaredDiskHG, "g", -0.3),
    (FlaredDiskGaussian, "sigma_theta", 70.0),
    (FlaredDiskPowerLaw, "n", 3.0),
    (FlaredDiskPowerLaw, "n", 2.5),
    (FlaredDiskPowerLaw, "n", 0.0),
]


def _close(a, b):
    a, b = onp.asarray(a), onp.asarray(b)
    return onp.max(onp.abs(a - b)) <= 1e-12 * onp.max(onp.abs(b))


@pytest.mark.parametrize("cls,name,value", CASES)
@pytest.mark.parametrize("dra", [0.0, 3.0])
def test_images_and_visibilities_match_the_arctan2_formula(
    cls, name, value, dra
):
    # npix=32 with dra = ddec = 0 puts the star on a pixel centre only for odd
    # grids, so also render on a grid containing the exact origin.
    with jax.enable_x64(True):
        geometry = GEOMETRY | {name: value, "dra": dra}
        new, old = cls(**geometry), _reference(cls)(**geometry)
        for npix, fov in [(32, 256.0), (33, 264.0)]:
            assert _close(
                new.render(npix=npix, fov_mas=fov),
                old.render(npix=npix, fov_mas=fov),
            )
        data = OIData(ami_grid_record(rotation_deg=-6.9))
        assert _close(
            new.model_on_grid(data.u, data.v, data.wavel, data.uv_grid),
            old.model_on_grid(data.u, data.v, data.wavel, data.uv_grid),
        )
        assert _close(
            new.model(data.u, data.v, data.wavel),
            old.model(data.u, data.v, data.wavel),
        )


@pytest.mark.parametrize("cls,name,value", CASES)
def test_gradients_match_the_arctan2_formula(cls, name, value):
    with jax.enable_x64(True):
        geometry = GEOMETRY | {name: value}
        new, old = cls(**geometry), _reference(cls)(**geometry)
        # The old formula's gradient is NaN at a pixel exactly on the star
        # (arctan2(0, 0)); the new one is finite there, so compare on a grid
        # without a star pixel and check finiteness on one with it.
        names = [name, "inc", "pa", "radius"]

        def loss(model, values, npix=32, fov=256.0):
            for k, v in zip(names, values):
                model = model.set(k, v)
            return np.sum(model.render(npix=npix, fov_mas=fov) ** 2)

        values = np.array([getattr(new, k) for k in names], dtype=float)
        g_new = jax.grad(lambda v: loss(new, v))(values)
        g_old = jax.grad(lambda v: loss(old, v))(values)
        assert onp.all(onp.isfinite(g_new))
        if cls is not FlaredDiskGaussian:  # which still needs arctan2
            at_star = jax.grad(lambda v: loss(new, v, 33, 264.0))(values)
            assert onp.all(onp.isfinite(at_star))
        assert onp.allclose(
            g_new, g_old, rtol=1e-9, atol=1e-12 * onp.abs(g_old).max()
        )
