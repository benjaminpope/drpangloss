"""Local curvature tools: Hessians, Laplace covariances and Fisher matrices.

Objectives are negative log likelihoods of a flat 1D parameter vector,
except for [`gaussian_fisher`][drpangloss.inference.gaussian_fisher], which accepts any parameter pytree.
Checks that need concrete values (positive errors, positive-definite
matrices) are skipped inside ``jax.jit``.
"""

import warnings

import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree


def _concrete(value):
    """``value`` as a NumPy array, or ``None`` inside a traced computation."""
    import numpy as onp

    try:
        return onp.asarray(value)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


def _warn_if_not_positive_definite(matrix, what):
    """Warn if a concrete symmetric ``matrix`` has non-positive eigenvalues."""
    import numpy as onp

    value = _concrete(matrix)
    if value is None or not onp.all(onp.isfinite(value)):
        return
    if onp.min(onp.linalg.eigvalsh(value)) <= 0.0:
        warnings.warn(
            f"The {what} is not positive definite (the point is not a "
            "likelihood maximum, or a parameter is unconstrained); the "
            "result has negative or infinite variances.",
            RuntimeWarning,
            stacklevel=3,
        )


def hessian_matrix(objective, x):
    """Return the Hessian matrix of ``objective`` evaluated at ``x``.

    ``x`` must be a flat 1D parameter vector; use
    ``jax.flatten_util.ravel_pytree`` to flatten a pytree first.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"hessian_matrix expects a 1D parameter vector; got shape "
            f"{x.shape}."
        )
    return np.asarray(jax.hessian(objective)(x), dtype=float)


def regularized_inverse(matrix, ridge=1e-10):
    """Return ``inv(matrix + ridge * I)``.

    A ``RuntimeWarning`` is raised (outside ``jax.jit``) if the regularized
    matrix is not positive definite, since its inverse is then not a
    covariance.
    """
    matrix = np.asarray(matrix, dtype=float)
    ident = np.eye(matrix.shape[-1], dtype=matrix.dtype)
    regularized = matrix + np.maximum(ridge, 0.0) * ident
    _warn_if_not_positive_definite(regularized, "matrix to invert")
    return np.linalg.inv(regularized)


def laplace_covariance(objective, x, ridge=1e-10):
    """Return the Laplace covariance, the inverse Hessian of ``objective``.

    ``objective`` is a negative log likelihood of the flat vector ``x``,
    which should be at (or near) its minimum. A small ``ridge`` (default
    1e-10) is added to the diagonal before inverting.
    """
    hess = hessian_matrix(objective, x)
    return regularized_inverse(hess, ridge=ridge)


def observed_information(objective, x, ridge=0.0):
    """Return observed information from the Hessian of a negative log likelihood.

    Unlike expected Fisher information, this quantity depends on the observed
    residuals and includes curvature of a nonlinear forward model. No ridge
    is added by default (compare [`laplace_covariance`][drpangloss.inference.laplace_covariance]).
    """
    information = hessian_matrix(objective, x)
    ident = np.eye(information.shape[-1], dtype=information.dtype)
    return information + np.maximum(ridge, 0.0) * ident


def fisher_matrix(objective, x, ridge=0.0):
    """Return observed information for backward compatibility.

    This historical name computes the Hessian of ``objective``. Use
    [`observed_information`][drpangloss.inference.observed_information] when the distinction from expected Fisher
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
        Expected Fisher matrix ``Jᵀ Σ⁻¹ J`` and a function restoring a flat
        parameter vector to the structure of ``params``.
    """
    flat_params, unravel = ravel_pytree(params)
    errors = np.asarray(errors, dtype=float).reshape(-1)
    concrete_errors = _concrete(errors)
    if concrete_errors is not None and (concrete_errors <= 0.0).any():
        raise ValueError("Gaussian errors must be strictly positive.")

    def flat_prediction(values):
        return np.asarray(prediction_fn(unravel(values)), dtype=float).reshape(
            -1
        )

    # Forward mode: one pass per parameter, and there are usually far fewer
    # parameters than data points.
    jacobian = jax.jacfwd(flat_prediction)(flat_params)
    if jacobian.shape[0] != errors.size:
        raise ValueError(
            "Prediction and error vectors must have the same length; "
            f"got {jacobian.shape[0]} and {errors.size}."
        )

    weighted_jacobian = jacobian / errors[:, None]
    information = weighted_jacobian.T @ weighted_jacobian
    ident = np.eye(information.shape[-1], dtype=information.dtype)
    return information + np.maximum(ridge, 0.0) * ident, unravel


def fisher_projection(fmat, eps=1e-12):
    """Return projection matrix mapping unit-normal latent vectors to parameter steps.

    If ``u ~ N(0, I)``, then ``x = x0 + P @ u`` has local covariance approximately
    ``F^{-1}`` for Fisher matrix ``F``.

    Parameters
    ----------
    fmat : array-like
        Symmetric Fisher (or observed information) matrix.
    eps : float, optional
        Eigenvalues below ``eps`` times the largest eigenvalue are raised to
        that floor, with a ``RuntimeWarning`` (outside ``jax.jit``), so that
        flat directions get large but finite steps.

    Returns
    -------
    array-like
        ``P = V diag(λ^-1/2)``, so that ``P Pᵀ = F⁻¹``.
    """
    evals, evecs = np.linalg.eigh(np.asarray(fmat, dtype=float))
    floor = eps * np.max(np.abs(evals))
    concrete = _concrete(evals)
    if concrete is not None and (concrete < _concrete(floor)).any():
        warnings.warn(
            "fisher_projection(): the Fisher matrix has eigenvalues below "
            f"eps={eps} times its largest; they were raised to that floor.",
            RuntimeWarning,
            stacklevel=2,
        )
    safe = np.maximum(evals, floor)
    return evecs * (1.0 / np.sqrt(safe))[None, :]
