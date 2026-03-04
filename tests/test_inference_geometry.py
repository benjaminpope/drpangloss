import jax.numpy as np

from drpangloss.inference import (
    fisher_matrix,
    fisher_projection,
    hessian_matrix,
    laplace_covariance,
)


def test_hessian_and_fisher_shapes_and_symmetry():
    objective = lambda x: (x[0] - 1.0) ** 2 + 3.0 * (x[1] + 2.0) ** 2
    x0 = np.array([0.2, -1.1])

    hess = hessian_matrix(objective, x0)
    fmat = fisher_matrix(objective, x0)

    assert hess.shape == (2, 2)
    assert fmat.shape == (2, 2)
    assert np.allclose(hess, hess.T)
    assert np.allclose(fmat, fmat.T)
    assert np.allclose(hess, fmat)


def test_laplace_covariance_is_finite_and_positive_diagonal():
    objective = lambda x: (x[0] / 2.0) ** 2 + (x[1] / 3.0) ** 2
    x0 = np.array([0.1, -0.3])

    cov = laplace_covariance(objective, x0, ridge=1e-8)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0.0)


def test_fisher_projection_whitens_local_metric():
    fmat = np.array([[5.0, 1.0], [1.0, 2.0]])
    proj = fisher_projection(fmat)
    ident = proj.T @ fmat @ proj

    assert proj.shape == (2, 2)
    assert np.all(np.isfinite(proj))
    assert np.allclose(ident, np.eye(2), atol=1e-5)
