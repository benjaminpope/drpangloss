import jax.numpy as np

from drpangloss.inference import (
    fisher_matrix,
    fisher_projection,
    gaussian_fisher,
    hessian_matrix,
    laplace_covariance,
    observed_information,
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


def test_gaussian_fisher_supports_parameter_pytrees():
    params = {"offset": np.array(0.2), "slopes": np.array([1.0, -0.5])}
    design = np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    errors = np.array([0.5, 2.0])

    def prediction(values):
        vector = np.concatenate(
            [np.atleast_1d(values["offset"]), values["slopes"]]
        )
        return design @ vector

    fmat, unravel = gaussian_fisher(prediction, params, errors)
    expected = design.T @ np.diag(errors**-2) @ design
    restored = unravel(np.array([0.3, 1.1, -0.4]))

    assert np.allclose(fmat, expected)
    assert np.allclose(restored["offset"], 0.3)
    assert np.allclose(restored["slopes"], np.array([1.1, -0.4]))
    assert np.all(np.linalg.eigvalsh(fmat) >= -1e-6)


def test_expected_fisher_matches_noiseless_observed_information():
    params = np.array([0.7, -0.2])
    errors = np.array([0.3, 0.5])
    prediction = lambda values: np.array(
        [values[0] ** 2 + values[1], np.sin(values[0]) - values[1]]
    )
    data = prediction(params)
    objective = lambda values: 0.5 * np.sum(
        ((data - prediction(values)) / errors) ** 2
    )

    expected, _ = gaussian_fisher(prediction, params, errors)
    observed = observed_information(objective, params)

    assert np.allclose(expected, observed, rtol=1e-5, atol=1e-6)


def test_nonlinear_residual_curvature_changes_observed_information():
    params = np.array([0.7, -0.2])
    errors = np.array([0.3, 0.5])
    prediction = lambda values: np.array(
        [values[0] ** 2 + values[1], np.sin(values[0]) - values[1]]
    )
    data = prediction(params) + np.array([0.4, -0.2])
    objective = lambda values: 0.5 * np.sum(
        ((data - prediction(values)) / errors) ** 2
    )

    expected, _ = gaussian_fisher(prediction, params, errors)
    observed = observed_information(objective, params)

    assert not np.allclose(expected, observed, rtol=1e-4, atol=1e-5)
