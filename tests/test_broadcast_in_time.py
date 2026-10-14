"""Time-dependent scenes are evaluated for all samples at once (broadcast),
with the serial per-sample loop kept only where it is needed; the two agree
in value, gradient and Hessian."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil.aliases import _UDBinary  # noqa: E402
from virgil.likelihood import model_loglike  # noqa: E402
from virgil.models import (  # noqa: E402
    Attached,
    ModulatedGaussianRim,
    OrbitalBinary,
    PointSource,
    System,
    UniformDisk,
)
from virgil.orbits import KeplerOrbit  # noqa: E402
from tests.test_attached import ORBIT, T_REF, _epochs  # noqa: E402


def _orbit(x):
    return KeplerOrbit(x[0], -100.0, 0.3, x[1], 60.0, 120.0, x[2], t_ref=T_REF)


def _binary(x):
    return OrbitalBinary(_orbit(x), x[3])


def _ud_binary(x):
    return _UDBinary(_orbit(x), x[3], x[4])


def _ud_attached(x):
    return System(
        primary=UniformDisk(x[4]),
        companion=Attached(UniformDisk(0.3, flux=x[3]), _orbit(x)),
    )


X = onp.array([730.0, 50.0, 20.0, 0.1, 0.8])


def _derivatives(build, data):
    def loglike(x):
        return model_loglike(build(x), data)

    x = np.asarray(X)
    return (
        loglike(x),
        jax.grad(loglike)(x),
        jax.hessian(loglike)(x),
    )


@pytest.mark.parametrize("build", [_binary, _ud_binary, _ud_attached])
def test_broadcast_matches_the_serial_loop_in_float64(build, monkeypatch):
    with jax.enable_x64(True):
        mjds = T_REF + onp.array([0.0, 150.0, 400.0, 900.0])
        data = _epochs(mjds)
        scene = build(np.asarray(X))
        assert scene.broadcast_in_time
        fast = _derivatives(build, data)
        # Force the serial path through the one hook that selects it.
        monkeypatch.setattr(
            type(scene), "broadcast_in_time", property(lambda self: False)
        )
        assert not scene.broadcast_in_time
        serial = _derivatives(build, data)
    for a, b in zip(fast, serial):
        assert onp.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_bound_angles_keep_the_serial_loop():
    disc = Attached(
        ModulatedGaussianRim(
            4.0, 1.5, 30.0, 10.0, az_amps=0.5, az_pas=0.0, flux=0.02
        ),
        ORBIT,
        bind={"inc": "apparent_inc"},
    )
    assert not disc.broadcast_in_time
    assert not System(primary=PointSource(), disc=disc).broadcast_in_time
    assert System(
        primary=PointSource(), comp=Attached(PointSource(0.1), ORBIT)
    ).broadcast_in_time
