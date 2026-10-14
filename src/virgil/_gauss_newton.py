"""A dense damped-Newton (Levenberg–Marquardt) kernel for negative log posteriors.

The objective Φ(z) is any scalar function of a flat vector z. Its gradient
is the exact ∇Φ from autodiff. Only the curvature is approximated:

    F(z) = J(z)ᵀ J(z) + diag(P(z)),    J = ∂ρ/∂z,

where ρ(z) is a vector of *curvature rows* (for a least-squares objective,
the whitened residuals, with Φ = ½|ρ|² + …) and P ≥ 0 is the diagonal from
elementwise priors and coordinate transforms. F is positive semi-definite
by construction. Because steps are accepted only if Φ itself falls, an
error in F costs speed, never correctness.

A *kernel problem* is an ``eqx.Module`` with three methods of a flat 1-D
float vector: ``loss(z)`` (scalar Φ), ``rows(z)`` (curvature rows, any
length, possibly 0) and ``diag(z)`` (P, length n). Data are module fields,
so they are traced; layouts that must be static are
``eqx.field(static=True)``. (A ``functools.partial`` would be static to
``eqx.filter_jit``: it would fail to hash or recompile per dataset.)

**Step** (`step`). At z with gradient g and
curvature F, find the coordinates held at a bound (below), set
``M = diag(max(F_ii, 1e-12 max_j F_jj, tiny))`` over the free ones and
solve ``(F + λM) δ = -g`` by Cholesky on the free coordinates (δ = 0 on
held ones). The trial is ``clip(z + δ, lower, upper)``. It is accepted if
δ, Φ, ∇Φ and F at the trial are all finite and Φ has fallen. A failed
Cholesky gives NaNs in JAX rather than an exception, so it counts as a
rejection. ``steps`` counts accepted and rejected iterations.

**Damping.** λ starts at 1e-3. An accepted step divides it by 3 (floor
1e-12), a rejected step multiplies it by 4. Small λ is the Gauss–Newton
step; large λ is a short step down the gradient with each coordinate
scaled by its own curvature. M is Marquardt's scaling, floored so that a
coordinate the rows ignore (F_ii = 0) still gives a positive-definite
system and no NaN.

**Active set** (ε-active, Bertsekas 1982). With
``ε = 1e-10 (1 + |bound|)``, coordinate i is held if
``z_i ≤ lower_i + ε`` and ``g_i > 0``, or ``z_i ≥ upper_i - ε`` and
``g_i < 0``: the descent direction ``-g_i`` points out of the box. Holding
only coordinates exactly on a bound makes a coordinate that sits just
inside one clip and be rejected over and over. A coordinate on a bound
whose gradient points inward is free, so it leaves. The projected gradient
is g with held entries zeroed.

**Stop codes**, tested after each step (the first that applies):

- ``CONVERGED``: the projected gradient's infinity norm is ≤ ``gtol``, or
  the Newton decrement ``½ gᵀF⁻¹g`` on the free coordinates is ≤ ``dtol``;
- once λ > 1e10: ``CONVERGED`` if the decrement is ≤ ``10 eps max(|Φ|, 1)``,
  so that the rounding floor (e.g. in float32) does not read as a failure;
  ``STALLED`` otherwise;
- ``LIMIT``: ``steps`` has reached the limit.

``NON_FINITE`` is set by `init` when Φ, g or F
is not finite at the start. A stopped state is left unchanged by ``step``,
so a ``lax.scan`` of k steps equals ``solve(max_steps=k)`` and ``vmap``
over starts is well defined. The default tolerances (0) rely on the
rounding-floor rule; pass ``gtol`` or ``dtol`` to stop earlier.

`solve` is a ``lax.while_loop`` over ``step``
under ``eqx.filter_jit``. Given a ``State`` it resumes: ``stop`` is reset
to ``RUNNING`` (``NON_FINITE`` is kept) and ``max_steps`` counts from
``state.steps``, so a caller can run in chunks with λ carried over.

Private; imports nothing from virgil. See ``design/gauss_newton_plan.md``.
"""

import equinox as eqx
import jax
import jax.numpy as np

RUNNING, CONVERGED, STALLED, LIMIT, NON_FINITE = 0, 1, 2, 3, 4

_LAM_START = 1e-3
_LAM_FLOOR = 1e-12
_LAM_STALL = 1e10
_BOUND_EPS = 1e-10
_MARQUARDT_FLOOR = 1e-12


class State(eqx.Module):
    """The state of a damped-Newton run at one point; see the module docstring."""

    z: jax.Array
    value: jax.Array
    grad: jax.Array
    curvature: jax.Array
    lam: jax.Array
    steps: jax.Array
    rejects: jax.Array
    stop: jax.Array  # int32 stop code


def curvature(problem, z):
    """``JᵀJ + diag(P)`` at z, with J by forward mode, symmetrized."""
    J = jax.jacfwd(problem.rows)(z)
    F = J.T @ J + np.diag(problem.diag(z))
    return 0.5 * (F + F.T)


def _held(z, g, lower, upper):
    """The ε-active set: coordinates whose descent direction leaves the box."""
    eps_lo = _BOUND_EPS * (
        1.0 + np.abs(np.where(np.isfinite(lower), lower, 0.0))
    )
    eps_hi = _BOUND_EPS * (
        1.0 + np.abs(np.where(np.isfinite(upper), upper, 0.0))
    )
    at_lower = np.isfinite(lower) & (z <= lower + eps_lo) & (g > 0)
    at_upper = np.isfinite(upper) & (z >= upper - eps_hi) & (g < 0)
    return at_lower | at_upper


def _marquardt(F, free):
    """Marquardt's scaling M (the diagonal), floored, on the free coordinates."""
    d = np.diag(F)
    biggest = np.max(np.where(free, d, 0.0))
    tiny = np.finfo(F.dtype).tiny
    return np.maximum(np.maximum(d, _MARQUARDT_FLOOR * biggest), tiny)


def _solve_free(F, g, free, lam):
    """δ of ``(F + λM) δ = -g`` on the free coordinates, 0 on held ones.

    NaN when the system is not positive definite (Cholesky does not raise).
    """
    A = F + lam * np.diag(_marquardt(F, free))
    both = free[:, None] & free[None, :]
    A = np.where(both, A, 0.0) + np.diag(np.where(free, 0.0, 1.0))
    rhs = np.where(free, -g, 0.0)
    L = np.linalg.cholesky(A)
    delta = jax.scipy.linalg.cho_solve((L, True), rhs)
    return np.where(free, delta, 0.0)


def _decrement(F, g, free):
    """Newton decrement ½ gᵀF⁻¹g on the free coordinates (inf if F is not PD).

    F carries the smallest damping, so a direction the rows ignore (and with
    a zero gradient) contributes nothing instead of NaN.
    """
    delta = _solve_free(F, g, free, _LAM_FLOOR)
    dec = -0.5 * np.sum(np.where(free, g * delta, 0.0))
    return np.where(np.isfinite(dec), dec, np.inf)


def _evaluate(problem, z):
    value, grad = jax.value_and_grad(problem.loss)(z)
    return value, grad, curvature(problem, z)


def init(problem, z0, lower, upper):
    """The starting `State` at ``z0`` (clipped into the box).

    ``stop`` is ``NON_FINITE`` if Φ, its gradient or F is not finite there.
    """
    z = np.clip(np.asarray(z0), lower, upper)
    value, grad, F = _evaluate(problem, z)
    finite = (
        np.isfinite(value) & np.all(np.isfinite(grad)) & np.all(np.isfinite(F))
    )
    return State(
        z=z,
        value=value,
        grad=grad,
        curvature=F,
        lam=np.asarray(_LAM_START, z.dtype),
        steps=np.zeros((), np.int32),
        rejects=np.zeros((), np.int32),
        stop=np.where(finite, RUNNING, NON_FINITE).astype(np.int32),
    )


def _advance(problem, state, lower, upper, gtol, dtol, limit):
    z, g, F = state.z, state.grad, state.curvature
    free = ~_held(z, g, lower, upper)
    delta = _solve_free(F, g, free, state.lam)
    trial = np.clip(z + delta, lower, upper)
    value, grad, F_trial = _evaluate(problem, trial)
    ok = (
        np.all(np.isfinite(delta))
        & np.isfinite(value)
        & np.all(np.isfinite(grad))
        & np.all(np.isfinite(F_trial))
    )
    accept = ok & (value < state.value)
    z = np.where(accept, trial, z)
    value = np.where(accept, value, state.value)
    g = np.where(accept, grad, g)
    F = np.where(accept, F_trial, F)
    lam = np.where(
        accept, np.maximum(state.lam / 3, _LAM_FLOOR), state.lam * 4
    )
    steps = state.steps + 1
    rejects = state.rejects + (~accept).astype(np.int32)

    free = ~_held(z, g, lower, upper)
    gnorm = np.max(np.where(free, np.abs(g), 0.0))
    dec = _decrement(F, g, free)
    eps = np.finfo(z.dtype).eps
    floor = lam > _LAM_STALL
    converged = (gnorm <= gtol) | (dec <= dtol)
    floor_ok = floor & (dec <= 10 * eps * np.maximum(np.abs(value), 1.0))
    code = np.select(
        [converged, floor_ok, floor, steps >= limit],
        [CONVERGED, CONVERGED, STALLED, LIMIT],
        default=RUNNING,
    ).astype(np.int32)
    return State(z, value, g, F, lam, steps, rejects, code)


def step(problem, state, lower, upper, gtol, dtol, max_steps):
    """One damped-Newton iteration; the identity once ``state.stop != RUNNING``.

    ``max_steps`` is the total ``steps`` at which the run stops with
    ``LIMIT``.
    """
    return jax.lax.cond(
        state.stop == RUNNING,
        lambda s: _advance(problem, s, lower, upper, gtol, dtol, max_steps),
        lambda s: s,
        state,
    )


@eqx.filter_jit
def _solve(problem, start, lower, upper, gtol, dtol, max_steps):
    if isinstance(start, State):
        keep = start.stop == NON_FINITE
        state = eqx.tree_at(
            lambda s: s.stop,
            start,
            np.where(keep, NON_FINITE, RUNNING).astype(np.int32),
        )
    else:
        state = init(problem, start, lower, upper)
    limit = state.steps + max_steps
    return jax.lax.while_loop(
        lambda s: s.stop == RUNNING,
        lambda s: step(problem, s, lower, upper, gtol, dtol, limit),
        state,
    )


def solve(
    problem,
    start,
    *,
    lower=None,
    upper=None,
    gtol=0.0,
    dtol=0.0,
    max_steps=200,
):
    """Run damped Newton on ``problem`` from ``start`` and return the `State`.

    Parameters
    ----------
    problem
        A kernel problem (module docstring).
    start
        A flat starting vector, or a `State` to resume (``max_steps`` more
        iterations from its ``steps``).
    lower, upper
        Box bounds, scalars or vectors; default ±inf.
    gtol, dtol
        Stop when the projected gradient's infinity norm is ≤ ``gtol`` or the
        Newton decrement is ≤ ``dtol``.
    max_steps
        Maximum iterations (accepted and rejected) in this call, at least 1.
    """
    if not isinstance(start, State):
        start = np.asarray(start)
        if not np.issubdtype(start.dtype, np.floating):
            start = start.astype(float)
    z = start.z if isinstance(start, State) else start
    dtype = z.dtype

    def bound(b, default):
        return np.broadcast_to(
            np.asarray(default if b is None else b, dtype), z.shape
        )

    return _solve(
        problem,
        start,
        bound(lower, -np.inf),
        bound(upper, np.inf),
        np.asarray(gtol, dtype),
        np.asarray(dtol, dtype),
        np.asarray(max_steps, np.int32),
    )
