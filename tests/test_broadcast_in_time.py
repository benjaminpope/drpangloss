"""Time-dependent scenes are evaluated for all samples at once (broadcast),
with the serial per-sample loop kept only where it is needed; the two agree
in value, gradient and Hessian."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil.aliases import _UDBinary  # noqa: E402
from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.likelihood import model_loglike  # noqa: E402
from virgil.models import (  # noqa: E402
    Attached,
    ModulatedGaussianRim,
    OrbitalBinary,
    PointSource,
    System,
    UniformDisk,
)
from virgil.spectra import PowerLaw  # noqa: E402
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


def _chromatic(x):
    # a chromatic companion: its weight depends on each sample's wavelength
    return System(
        primary=UniformDisk(x[4]),
        companion=Attached(
            PointSource(PowerLaw(x[3], 2.0, 2.2e-6)), _orbit(x)
        ),
    )


X = onp.array([730.0, 50.0, 20.0, 0.1, 0.8])


def _channels():
    """Several channels per baseline: ``mjd`` repeated, ``wavel`` varying."""
    return vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, 4),
        nights_mjd=T_REF + onp.array([0.0, 150.0, 400.0]),
    )


def _derivatives(build, data):
    def loglike(x):
        return model_loglike(build(x), data)

    x = np.asarray(X)
    return (
        loglike(x),
        jax.grad(loglike)(x),
        jax.hessian(loglike)(x),
    )


@pytest.mark.parametrize(
    "build, channels",
    [
        (_binary, False),
        (_ud_binary, False),
        (_ud_attached, False),
        (_binary, True),
        (_chromatic, True),
    ],
)
def test_broadcast_matches_the_serial_loop_in_float64(
    build, channels, monkeypatch
):
    with jax.enable_x64(True):
        if channels:
            data = _channels()
            assert np.size(data.wavel) > 1
        else:
            data = _epochs(T_REF + onp.array([0.0, 150.0, 400.0, 900.0]))
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


def test_orbit_batches_are_vectorized_by_default_and_change_nothing():
    # lax.map without a batch_size is a serial loop; the default now batches.
    from virgil.epochs import Epochs, _orbit_batch_size, rank_orbits
    from virgil.orbit_search import SharedFlux, score_orbits

    data = _epochs(T_REF + onp.array([0.0, 150.0]))
    assert _orbit_batch_size(None, [data], 5000) == 5000
    assert _orbit_batch_size(None, [data], 3) == 3
    assert _orbit_batch_size(7, [data], 3) == 3
    # The batch holds about a million visibilities in all: it shrinks with
    # the data, with the number of datasets and with work per candidate,
    # and is never below one.
    from virgil._grid import batch_visibilities

    many = [data] * 40
    big = batch_visibilities() // (40 * data.u.size)
    assert _orbit_batch_size(None, many, 10**9) == big
    assert big < _orbit_batch_size(None, [data], 10**9)
    assert _orbit_batch_size(None, many, 10**9, per_candidate=8) == big // 8
    assert _orbit_batch_size(None, many, 10**9, per_candidate=10**9) == 1
    epochs = Epochs({"a": data})
    periods = onp.linspace(600.0, 800.0, 7)
    orbits = [
        KeplerOrbit(p, -100.0, 0.3, 50.0, 60.0, 120.0, 20.0, t_ref=T_REF)
        for p in periods
    ]
    for scales in ("quoted", "marginal"):
        default = rank_orbits(
            lambda o: OrbitalBinary(o, 0.1), epochs, orbits, scales=scales
        )
        single = rank_orbits(
            lambda o: OrbitalBinary(o, 0.1),
            epochs,
            orbits,
            scales=scales,
            batch_size=1,
        )
        assert onp.array_equal(default.order, single.order)
        assert onp.allclose(default.loglike, single.loglike, rtol=1e-10)
    kwargs = dict(shared=SharedFlux([0.1]), s_max=5.0)
    default = score_orbits(OrbitalBinary, epochs, orbits, **kwargs)
    single = score_orbits(
        OrbitalBinary, epochs, orbits, batch_size=2, **kwargs
    )
    assert onp.allclose(default.score, single.score, rtol=1e-10)
