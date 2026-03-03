import jax.numpy as np

from drpangloss.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    closure_phases,
    cvis_binary,
    fisher,
    laplace_cov,
    loglike,
)

from tests._test_data import i_cps1, i_cps2, i_cps3, oidata, u, v


ddec, dra, planet = 0.1, 0.2, 10


def test_cvis_binary():
    vis = cvis_binary(u, v, ddec, dra, planet)
    vis2 = np.abs(vis) ** 2
    assert vis.shape == (u.shape[0],)
    assert np.all(vis2 >= 0.0)
    assert np.all(vis2 <= 1.0)
    assert np.all(np.isfinite(vis))


def test_closure_phases():
    vis = cvis_binary(u, v, ddec, dra, planet)
    cps = closure_phases(vis, i_cps1, i_cps2, i_cps3)
    assert cps.shape == (35,)
    assert np.all(np.isfinite(cps))


def test_likelihood():
    binary = BinaryModelAngular(50, 45, 10)
    model_data = oidata.model(binary)
    data, errors = oidata.flatten_data()

    like = -0.5 * np.sum((data - model_data) ** 2 / errors**2)
    assert np.all(np.isfinite(like))


def test_BinaryModelAngular():
    binary = BinaryModelAngular(50, 45, 0.1)
    model_data = oidata.model(binary)
    assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_data))


def test_BinaryModelCartesian():
    binary = BinaryModelCartesian(150, 150, 1e-3)
    model_data = oidata.model(binary)
    assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_data))


def test_laplace_and_fisher_wrappers_are_finite():
    params = ["dra", "ddec", "flux"]
    values = np.array([120.0, -80.0, 2e-3])

    cov = laplace_cov(values, params, oidata, BinaryModelCartesian)
    fmat = fisher(values, params, oidata, BinaryModelCartesian, ridge=1e-10)
    like = loglike(values, params, oidata, BinaryModelCartesian)

    assert cov.shape == (3, 3)
    assert fmat.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    assert np.all(np.isfinite(fmat))
    assert np.allclose(fmat, fmat.T)
    assert np.isfinite(like)
