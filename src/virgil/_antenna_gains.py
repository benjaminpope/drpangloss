"""Laplace-marginalized antenna gains for complex visibilities (private).

Source: Broderick et al. (2020), [references](../references.md#broderick2020);
Blackburn et al. (2020), [references](../references.md#blackburn2020).
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp


def gain_intervals(data, solint):
    """Solution intervals and antenna indices of ``data``.

    Returns ``(interval, a1, a2, n_interval, n_ant)``: int32 arrays per point
    and the static counts. Intervals never cross a scan (or an observation).
    """
    missing = [n for n in ("time", "ant1", "ant2") if getattr(data, n) is None]
    if solint == "scan" and data.scan is None:
        missing.append("scan")
    if missing:
        raise ValueError(
            f"antenna_gains needs {', '.join(missing)}; build the data with "
            "those fields (read_ms fills them)."
        )
    time = onp.asarray(data.time, dtype=onp.float64)
    scan = (
        onp.zeros(time.size, dtype=onp.int64)
        if data.scan is None
        else onp.asarray(data.scan, dtype=onp.int64)
    )
    group = onp.stack([onp.asarray(data.obs, dtype=onp.int64), scan], axis=1)
    if solint == "scan":
        cols = [group]
    elif solint == "int":
        cols = [group, time[:, None]]
    elif isinstance(solint, str) or not solint > 0:
        raise ValueError(
            "solint must be 'int', 'scan' or a positive number of seconds."
        )
    else:
        _, g = onp.unique(group, axis=0, return_inverse=True)
        t0 = onp.full(g.max() + 1, onp.inf)
        onp.minimum.at(t0, g.ravel(), time)
        cols = [group, onp.floor((time - t0[g.ravel()]) / solint)[:, None]]
    key = onp.concatenate(cols, axis=1)
    _, interval = onp.unique(key, axis=0, return_inverse=True)
    a1, a2 = onp.asarray(data.ant1), onp.asarray(data.ant2)
    if onp.any(a1 == a2):
        raise ValueError("antenna_gains needs cross-correlations only.")
    ants, inv = onp.unique(onp.concatenate([a1, a2]), return_inverse=True)
    a1, a2 = inv[: a1.size], inv[a1.size :]
    return (
        interval.ravel().astype(onp.int32),
        a1.astype(onp.int32),
        a2.astype(onp.int32),
        int(interval.max()) + 1,
        int(ants.size),
    )


def _sums(x, index, size):
    return jax.ops.segment_sum(x, index, size)


def _hessians(c, k1, k2, n_s, n_a, tau_a, tau_p):
    """Amplitude and phase blocks ``(S, N, N)`` of the Gauss–Newton Hessian.

    ``c = w |V|²`` per point. The blocks are ``D ± A`` plus the prior
    precision, ``A`` the symmetric baseline-weight matrix: the λ–φ cross terms
    vanish.
    """
    p = n_s * n_a
    s, i, j = k1 // n_a, k1 % n_a, k2 % n_a
    flat = np.concatenate(
        [s * n_a * n_a + i * n_a + j, s * n_a * n_a + j * n_a + i]
    )
    adj = _sums(np.concatenate([c, c]), flat, p * n_a).reshape(n_s, n_a, n_a)
    deg = adj.sum(axis=2)
    eye = np.eye(n_a)
    diag = deg[:, :, None] * eye
    return diag + adj + eye / tau_a**2, diag - adj + eye / tau_p**2


def _solve(h, rhs):
    """Batched ``H⁻¹ rhs`` per interval (Cholesky)."""
    chol = np.linalg.cholesky(h)
    return jax.vmap(lambda c, r: jax.scipy.linalg.cho_solve((c, True), r))(
        chol, rhs
    )


def _logdet(h):
    chol = np.linalg.cholesky(h)
    return 2.0 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)))


def marginal(model, vis, w, interval, a1, a2, n_s, n_a, tau_a, tau_p, steps):
    """Gain solution and Laplace terms of the flat arrays.

    ``K = steps`` Gauss–Newton steps from g = 1, then the Hessian blocks at
    the solution. Returns ``(lam, phi, norm, step)``, ``lam`` and ``phi`` of shape
    ``(S, N)`` and ``norm`` the ``½ log det H`` terms plus the prior
    normalization (without the ½ Σ prior χ², which the residuals carry);
    ``step`` is the size of the last step (small once converged).
    """
    k1 = interval * n_a + a1
    k2 = interval * n_a + a2
    p = n_s * n_a

    def state(lam, phi):
        v = np.exp(lam[k1] + lam[k2] + 1j * (phi[k1] - phi[k2])) * model
        return v, vis - v

    def step(carry, _):
        lam, phi = carry
        v, r = state(lam, phi)
        z = w * np.conj(r) * v
        a, b = np.real(z), np.imag(z)
        g_lam = -_sums(a, k1, p) - _sums(a, k2, p) + lam / tau_a**2
        g_phi = _sums(b, k1, p) - _sums(b, k2, p) + phi / tau_p**2
        h_lam, h_phi = _hessians(
            w * np.abs(v) ** 2, k1, k2, n_s, n_a, tau_a, tau_p
        )
        d_lam = _solve(h_lam, g_lam.reshape(n_s, n_a))
        d_phi = _solve(h_phi, g_phi.reshape(n_s, n_a))
        # a step beyond one e-fold or one radian is not trusted
        d_lam = np.clip(d_lam.reshape(-1), -1.0, 1.0)
        d_phi = np.clip(d_phi.reshape(-1), -1.0, 1.0)
        size = np.maximum(np.max(np.abs(d_lam)), np.max(np.abs(d_phi)))
        return (lam - d_lam, phi - d_phi), size

    zero = np.zeros(p, dtype=np.real(model).dtype)
    (lam, phi), sizes = jax.lax.scan(step, (zero, zero), None, length=steps)
    v, _ = state(lam, phi)
    h_lam, h_phi = _hessians(
        w * np.abs(v) ** 2, k1, k2, n_s, n_a, tau_a, tau_p
    )
    norm = 0.5 * (_logdet(h_lam) + _logdet(h_phi)) + n_s * n_a * (
        np.log(tau_a) + np.log(tau_p)
    )
    return lam.reshape(n_s, n_a), phi.reshape(n_s, n_a), norm, sizes[-1]
