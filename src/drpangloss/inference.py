import jax
import jax.numpy as np


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


def fisher_matrix(objective, x, ridge=0.0):
    """Approximate local Fisher matrix using the Hessian of ``objective`` at ``x``."""
    fmat = hessian_matrix(objective, x)
    if ridge > 0.0:
        ident = np.eye(fmat.shape[-1], dtype=fmat.dtype)
        fmat = fmat + ridge * ident
    return fmat


def fisher_projection(fmat, eps=1e-12):
    """Return projection matrix mapping unit-normal latent vectors to parameter steps.

    If ``u ~ N(0, I)``, then ``x = x0 + P @ u`` has local covariance approximately
    ``F^{-1}`` for Fisher matrix ``F``.
    """
    evals, evecs = np.linalg.eigh(np.asarray(fmat, dtype=float))
    safe = np.clip(evals, eps, np.inf)
    inv_sqrt = np.diag(1.0 / np.sqrt(safe))
    return evecs @ inv_sqrt
