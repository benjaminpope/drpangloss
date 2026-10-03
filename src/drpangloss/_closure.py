"""Independent, whitened closure phases.

The closure phase of triangle (a, b, c) is φ_ab + φ_bc − φ_ac. If the noise
on the baseline phases φ is independent with variances s_b, the closure
phases of triangles that share baselines are correlated, with covariance

    C = T diag(s) Tᵀ,

where T is the triangle-by-baseline matrix of +1, +1, −1 entries. With four
telescopes, the four triangles of one frame and channel span only three
independent combinations, and C has rank 3; with equal noise, neighbouring
triangles correlate at ±1/3 (Kammerer et al. 2020, A&A 644, A110, §2.2).
Treating all four as independent counts the closure phases 4/3 times.

``ClosureNoise`` groups the triangles that share baselines (one frame and
channel each), keeps the independent combinations Q (an orthonormal basis of
the column space of T), and whitens them with the Cholesky factor of
Q C Qᵀ. The baseline variances s are the minimum-norm solution of
Σ_{b∈t} s_b = σ_t² for the reported closure-phase errors σ_t, a linear map
P with s = P σ², so that rescaled or inflated errors carry through.
"""

import equinox as eqx
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp


def _groups(i1, i2, i3):
    """Triangles connected through shared baseline samples, as index lists."""
    n = len(i1)
    parent = onp.arange(n)

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    owner = {}
    for t in range(n):
        for b in (int(i1[t]), int(i2[t]), int(i3[t])):
            if b in owner:
                parent[find(t)] = find(owner[b])
            else:
                owner[b] = t
    roots = onp.array([find(t) for t in range(n)])
    return [onp.flatnonzero(roots == r) for r in onp.unique(roots)]


class ClosureNoise(eqx.Module):
    """Correlated noise of closure phases from independent baseline noise."""

    groups: onp.ndarray  # (n_group, m) closure-phase indices, padded with 0
    mask: onp.ndarray  # (n_group, m) True for real triangles
    incidence: onp.ndarray  # (n_group, m, n_base) T
    variance_map: onp.ndarray  # (n_group, n_base, m) P: s = P σ²
    basis: onp.ndarray  # (n_group, k, m) Q, orthonormal rows
    valid: onp.ndarray  # (n_group, k) True for real basis rows
    keep: onp.ndarray  # flat indices of the real rows, in group order

    @classmethod
    def from_indices(cls, i1, i2, i3):
        """Build from closure-phase indices, or ``None`` if none correlate."""
        i1, i2, i3 = (onp.asarray(i, dtype=int) for i in (i1, i2, i3))
        groups = _groups(i1, i2, i3)
        if all(g.size == 1 for g in groups):
            return None  # three telescopes: nothing to decorrelate
        blocks = []
        for g in groups:
            baselines = onp.unique(onp.concatenate([i1[g], i2[g], i3[g]]))
            col = {b: j for j, b in enumerate(baselines)}
            t = onp.zeros((g.size, baselines.size))
            for row, tri in enumerate(g):
                t[row, col[i1[tri]]] += 1.0
                t[row, col[i2[tri]]] += 1.0
                t[row, col[i3[tri]]] -= 1.0
            left, singular, _ = onp.linalg.svd(t, full_matrices=False)
            rank = int(onp.sum(singular > 1e-9 * singular.max()))
            blocks.append((g, t, onp.linalg.pinv(t**2), left[:, :rank].T))
        m = max(b[0].size for b in blocks)
        n_base = max(b[1].shape[1] for b in blocks)
        k = max(b[3].shape[0] for b in blocks)
        n = len(blocks)
        groups_out = onp.zeros((n, m), dtype=int)
        mask = onp.zeros((n, m), dtype=bool)
        incidence = onp.zeros((n, m, n_base))
        variance_map = onp.zeros((n, n_base, m))
        basis = onp.zeros((n, k, m))
        valid = onp.zeros((n, k), dtype=bool)
        for j, (g, t, p, q) in enumerate(blocks):
            groups_out[j, : g.size] = g
            mask[j, : g.size] = True
            incidence[j, : g.size, : t.shape[1]] = t
            variance_map[j, : t.shape[1], : g.size] = p
            basis[j, : q.shape[0], : g.size] = q
            valid[j, : q.shape[0]] = True
        return cls(
            groups_out,
            mask,
            incidence,
            variance_map,
            basis,
            valid,
            onp.flatnonzero(valid.reshape(-1)),
        )

    @property
    def size(self):
        """Number of independent closure phases."""
        return int(self.keep.size)

    def _cholesky(self, sigma):
        var = np.where(self.mask, np.asarray(sigma)[self.groups] ** 2, 0.0)
        s = np.einsum("gbm,gm->gb", self.variance_map, var)
        # The minimum-norm baseline variances can dip below zero when the
        # closure-phase errors are very unequal; floor them.
        floor = 1e-6 * np.max(var, axis=1, keepdims=True)
        s = np.maximum(s, floor)
        cov = np.einsum("gmb,gb,gnb->gmn", self.incidence, s, self.incidence)
        cov_q = np.einsum("gkm,gmn,gln->gkl", self.basis, cov, self.basis)
        pad = 1.0 - self.valid.astype(cov_q.dtype)
        cov_q = cov_q + pad[:, :, None] * np.eye(pad.shape[1])
        return np.linalg.cholesky(cov_q)

    def whiten(self, residuals, sigma):
        """Whitened independent combinations, and their effective errors.

        ``residuals`` and ``sigma`` have one entry per closure phase. The
        returned errors (the Cholesky diagonal) give the Gaussian
        normalisation: their log-sum is ½ log det of the covariance.
        """
        chol = self._cholesky(sigma)
        r = np.where(self.mask, np.asarray(residuals)[self.groups], 0.0)
        y = np.einsum("gkm,gm->gk", self.basis, r)
        w = jsl.solve_triangular(chol, y[..., None], lower=True)[..., 0]
        d = np.diagonal(chol, axis1=1, axis2=2)
        return w.reshape(-1)[self.keep], d.reshape(-1)[self.keep]

    def sample(self, key, sigma, n_phase):
        """Closure-phase noise drawn from independent baseline-phase noise."""
        import jax

        var = np.where(self.mask, np.asarray(sigma)[self.groups] ** 2, 0.0)
        s = np.einsum("gbm,gm->gb", self.variance_map, var)
        s = np.maximum(s, 0.0)
        e = np.sqrt(s) * jax.random.normal(key, s.shape)
        noise = np.einsum("gmb,gb->gm", self.incidence, e)
        # Each closure phase sits in exactly one group; padded slots add 0.
        out = np.zeros(n_phase, dtype=noise.dtype)
        return out.at[self.groups].add(np.where(self.mask, noise, 0.0))
