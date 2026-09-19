import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree


def hessian_matrix(objective, x):
    """Return the Hessian matrix of ``objective`` evaluated at ``x``."""
    return np.asarray(jax.hessian(objective)(x), dtype=float)


def regularized_inverse(matrix, ridge=1e-10):
    """Return a numerically stabilized inverse with diagonal ridge regularization."""
    matrix = np.asarray(matrix, dtype=float)
    ridge = float(max(ridge, 0.0))
    ident = np.eye(matrix.shape[-1], dtype=matrix.dtype)
    return np.linalg.inv(matrix + ridge * ident)


def laplace_covariance(objective, x, ridge=1e-10):
    """Return Laplace covariance from the Hessian of a negative log-likelihood objective."""
    hess = hessian_matrix(objective, x)
    return regularized_inverse(hess, ridge=ridge)


def observed_information(objective, x, ridge=0.0):
    """Return observed information from the Hessian of a negative log likelihood.

    Unlike expected Fisher information, this quantity depends on the observed
    residuals and includes curvature of a nonlinear forward model.
    """
    information = hessian_matrix(objective, x)
    if ridge > 0.0:
        ident = np.eye(information.shape[-1], dtype=information.dtype)
        information = information + ridge * ident
    return information


def fisher_matrix(objective, x, ridge=0.0):
    """Return observed information for backward compatibility.

    This historical name computes the Hessian of ``objective``. Use
    :func:`observed_information` when the distinction from expected Fisher
    information matters.
    """
    return observed_information(objective, x, ridge=ridge)


def gaussian_fisher(prediction_fn, params, errors, ridge=0.0):
    """Return expected Fisher information for fixed independent Gaussian errors.

    Parameters
    ----------
    prediction_fn : callable
        Function mapping the parameter pytree to a one-dimensional prediction.
    params : pytree
        Parameter values at which to evaluate the local model sensitivity.
    errors : array-like
        Standard deviations corresponding to the prediction vector.
    ridge : float, optional
        Diagonal regularization term.

    Returns
    -------
    tuple[array-like, callable]
        Expected Fisher matrix and a function restoring a flat parameter vector
        to the structure of ``params``.
    """
    flat_params, unravel = ravel_pytree(params)
    errors = np.asarray(errors, dtype=float).reshape(-1)

    def flat_prediction(values):
        return np.asarray(prediction_fn(unravel(values)), dtype=float).reshape(
            -1
        )

    jacobian = jax.jacrev(flat_prediction)(flat_params)
    if jacobian.shape[0] != errors.size:
        raise ValueError(
            "Prediction and error vectors must have the same length; "
            f"got {jacobian.shape[0]} and {errors.size}."
        )
    if bool(np.any(errors <= 0.0)):
        raise ValueError("Gaussian errors must be strictly positive.")

    weighted_jacobian = jacobian / errors[:, None]
    information = weighted_jacobian.T @ weighted_jacobian
    if ridge > 0.0:
        ident = np.eye(information.shape[-1], dtype=information.dtype)
        information = information + ridge * ident
    return information, unravel


def fisher_projection(fmat, eps=1e-12):
    """Return projection matrix mapping unit-normal latent vectors to parameter steps.

    If ``u ~ N(0, I)``, then ``x = x0 + P @ u`` has local covariance approximately
    ``F^{-1}`` for Fisher matrix ``F``.
    """
    evals, evecs = np.linalg.eigh(np.asarray(fmat, dtype=float))
    safe = np.clip(evals, eps, np.inf)
    inv_sqrt = np.diag(1.0 / np.sqrt(safe))
    return evecs @ inv_sqrt
