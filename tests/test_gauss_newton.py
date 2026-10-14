"""The damped-Newton kernel of `virgil._gauss_newton`.

Everything here is tiny (2 to 6 coordinates, seconds on a laptop): linear and
Rosenbrock least squares, an iteratively reweighted (IRLS) marginal
objective that is not a sum of squares, boxes, and the `step`/`solve`
contracts (vmap, scan, resume, monotone Φ, stop codes). See
design/gauss_newton_plan.md, GN1.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.optimize import minimize

from virgil._gauss_newton import (
    CONVERGED,
    LIMIT,
    NON_FINITE,
    RUNNING,
    STALLED,
    State,
    curvature,
    init,
    solve,
    step,
)


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


class LeastSquares(eqx.Module):
    """Φ = ½|ρ|² with ρ = A z − b, a linear problem."""

    A: jax.Array
    b: jax.Array

    def rows(self, z):
        return self.A @ z - self.b

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class Rosenbrock(eqx.Module):
    def rows(self, z):
        return np.stack([10.0 * (z[1] - z[0] ** 2), 1.0 - z[0]])

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class Quadratic(eqx.Module):
    """Φ = ½ (z − c)ᵀ H (z − c) with H = LᵀL, curvature rows L (z − c)."""

    L: jax.Array
    c: jax.Array

    def rows(self, z):
        return self.L @ (z - self.c)

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class Marginal(eqx.Module):
    """Φ = Σ_b (ν_b/2) ln χ²_b for two blocks of a linear model.

    Not a sum of squares. The rows carry the IRLS weight ν_b/χ²_b, frozen,
    so JᵀJ is the Gauss–Newton curvature of Φ (the majorise–minimise form).
    """

    A: tuple
    y: tuple

    def _chi2(self, z):
        return [np.sum((A @ z - y) ** 2) for A, y in zip(self.A, self.y)]

    def loss(self, z):
        return sum(
            0.5 * y.size * np.log(c) for y, c in zip(self.y, self._chi2(z))
        )

    def rows(self, z):
        out = []
        for A, y, c in zip(self.A, self.y, self._chi2(z)):
            w = jax.lax.stop_gradient(y.size / c)
            out.append(np.sqrt(w) * (A @ z - y))
        return np.concatenate(out)

    def diag(self, z):
        return np.zeros_like(z)


class Ignores(eqx.Module):
    """Rows that ignore coordinate 1, with P = 0 there."""

    def rows(self, z):
        return z[:1] - 1.0

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class LogSum(eqx.Module):
    """Φ = Σ log z: not finite for z ≤ 0."""

    def rows(self, z):
        return z

    def loss(self, z):
        return np.sum(np.log(z))

    def diag(self, z):
        return np.zeros_like(z)


class NoRows(eqx.Module):
    """A pure-prior problem: no rows at all, P = 1."""

    def rows(self, z):
        return np.zeros((0,), z.dtype)

    def loss(self, z):
        return 0.5 * np.sum(z**2)

    def diag(self, z):
        return np.ones_like(z)


def _no_bounds(n):
    return np.full(n, -np.inf), np.full(n, np.inf)


def test_linear_least_squares_matches_lstsq_in_five_steps():
    rng = onp.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    b = rng.normal(size=30)
    state = solve(
        LeastSquares(np.asarray(A), np.asarray(b)),
        np.zeros(4),
        gtol=1e-10,
        max_steps=5,
    )
    assert int(state.stop) == CONVERGED
    assert int(state.steps) <= 5
    onp.testing.assert_allclose(
        state.z, onp.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10
    )


def test_rosenbrock_residuals_reach_the_minimum_in_fifty_steps():
    state = solve(
        Rosenbrock(), np.array([-1.2, 1.0]), gtol=1e-10, max_steps=50
    )
    assert int(state.stop) == CONVERGED
    assert int(state.steps) <= 50
    onp.testing.assert_allclose(state.z, [1.0, 1.0], atol=1e-8)


def _marginal_problem():
    rng = onp.random.default_rng(1)
    A = (rng.normal(size=(20, 3)), rng.normal(size=(15, 3)))
    z_true = onp.array([1.0, -2.0, 0.5])
    # the second block is five times noisier, so the weights are far from 1
    y = (
        A[0] @ z_true + 0.1 * rng.normal(size=20),
        A[1] @ z_true + 0.5 * rng.normal(size=15),
    )
    return Marginal(tuple(map(np.asarray, A)), tuple(map(np.asarray, y)))


def test_marginal_objective_with_irls_rows_matches_scipy_bfgs():
    problem = _marginal_problem()
    z0 = np.zeros(3)
    state = solve(problem, z0, gtol=1e-12, max_steps=100)
    assert int(state.stop) == CONVERGED
    ref = minimize(
        lambda z: float(problem.loss(np.asarray(z))),
        onp.zeros(3),
        jac=lambda z: onp.asarray(jax.grad(problem.loss)(np.asarray(z))),
        method="BFGS",
        options={"gtol": 1e-13},
    )
    onp.testing.assert_allclose(state.z, ref.x, atol=1e-8)
    assert float(state.value) <= ref.fun + 1e-12


def test_curvature_is_jtj_plus_diag_and_handles_no_rows():
    problem = _marginal_problem()
    z = np.array([0.3, -1.0, 0.2])
    J = jax.jacfwd(problem.rows)(z)
    onp.testing.assert_allclose(curvature(problem, z), J.T @ J, atol=1e-12)
    F = curvature(NoRows(), np.ones(3))
    onp.testing.assert_allclose(F, onp.eye(3))
    state = solve(NoRows(), np.ones(3), gtol=1e-12, max_steps=20)
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, 0.0, atol=1e-10)


def test_bound_holds_the_coordinate_whose_minimum_is_outside_the_box():
    problem = Quadratic(np.diag(np.array([1.0, 2.0])), np.array([2.0, 0.5]))
    state = solve(
        problem,
        np.array([0.2, 0.2]),
        lower=0.0,
        upper=1.0,
        gtol=1e-10,
        max_steps=30,
    )
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, [1.0, 0.5], atol=1e-9)
    # held at the upper bound: the gradient points out of the box
    assert float(state.grad[0]) < 0


def test_coordinate_on_a_bound_with_inward_gradient_leaves_it():
    problem = Quadratic(np.eye(2), np.array([1.0, 1.0]))
    state = solve(
        problem,
        np.array([0.0, 0.0]),
        lower=np.array([0.0, 0.0]),
        upper=np.array([5.0, 5.0]),
        gtol=1e-10,
        max_steps=20,
    )
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, [1.0, 1.0], atol=1e-9)


def test_strongly_coupled_coordinates_with_one_at_a_bound_do_not_cycle():
    rho = 0.99
    H = onp.array([[1.0, rho], [rho, 1.0]])
    L = onp.linalg.cholesky(H).T
    c = onp.array([-1.0, 2.0])
    problem = Quadratic(np.asarray(L), np.asarray(c))
    state = solve(
        problem,
        np.array([0.5, 5.0]),
        lower=np.array([0.0, -np.inf]),
        upper=np.array([np.inf, np.inf]),
        gtol=1e-10,
        max_steps=20,
    )
    assert int(state.stop) == CONVERGED
    assert int(state.steps) <= 20
    # coordinate 0 held at its bound; coordinate 1 minimizes over the rest
    onp.testing.assert_allclose(state.z, [0.0, c[1] + rho * c[0]], atol=1e-9)
    assert float(state.grad[0]) > 0
    assert abs(float(state.grad[1])) < 1e-9


def _starts():
    rng = onp.random.default_rng(2)
    return np.asarray(rng.uniform(-2, 2, size=(8, 2)))


def test_vmap_over_starts_equals_a_loop():
    problem = Rosenbrock()
    z0s = _starts()

    def run(z0):
        return solve(problem, z0, gtol=1e-10, max_steps=100)

    batched = jax.vmap(run)(z0s)
    for k in range(8):
        one = run(z0s[k])
        onp.testing.assert_allclose(batched.z[k], one.z, atol=1e-12, rtol=0)
        onp.testing.assert_allclose(
            batched.value[k], one.value, atol=1e-12, rtol=0
        )
        assert int(batched.steps[k]) == int(one.steps)
        assert int(batched.stop[k]) == int(one.stop)


def test_scan_of_steps_equals_solve_with_the_same_limit():
    problem, z0 = Rosenbrock(), np.array([-1.2, 1.0])
    lower, upper = _no_bounds(2)
    tol = np.asarray(0.0)

    def body(state, _):
        return step(
            problem, state, lower, upper, tol, tol, np.int32(20)
        ), state.value

    scanned, _ = jax.lax.scan(
        body, init(problem, z0, lower, upper), None, length=20
    )
    solved = solve(problem, z0, max_steps=20)
    assert int(solved.stop) == LIMIT
    for a, b in zip(jax.tree.leaves(scanned), jax.tree.leaves(solved)):
        onp.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_resuming_after_seven_steps_equals_an_uninterrupted_solve():
    problem, z0 = Rosenbrock(), np.array([-1.2, 1.0])
    first = solve(problem, z0, gtol=1e-10, max_steps=7)
    assert int(first.stop) == LIMIT
    assert int(first.steps) == 7
    resumed = solve(problem, first, gtol=1e-10, max_steps=93)
    whole = solve(problem, z0, gtol=1e-10, max_steps=100)
    assert int(resumed.stop) == int(whole.stop) == CONVERGED
    assert isinstance(resumed, State)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole)):
        onp.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


def test_loss_never_increases_over_a_run():
    problem = Rosenbrock()
    lower, upper = _no_bounds(2)
    tol = np.asarray(0.0)

    def body(state, _):
        new = step(problem, state, lower, upper, tol, tol, np.int32(60))
        return new, new.value

    start = init(problem, np.array([-1.2, 1.0]), lower, upper)
    _, values = jax.lax.scan(body, start, None, length=60)
    values = onp.concatenate([[float(start.value)], onp.asarray(values)])
    assert onp.all(onp.diff(values) <= 0)
    assert values[-1] < values[0]


def test_step_is_the_identity_once_stopped():
    problem = Rosenbrock()
    lower, upper = _no_bounds(2)
    tol = np.asarray(0.0)
    done = solve(problem, np.array([-1.2, 1.0]), gtol=1e-10, max_steps=100)
    again = step(problem, done, lower, upper, tol, tol, np.int32(1000))
    for a, b in zip(jax.tree.leaves(again), jax.tree.leaves(done)):
        onp.testing.assert_array_equal(a, b)


def test_non_finite_start_stops_with_non_finite():
    state = solve(LogSum(), np.array([-1.0, 1.0]), max_steps=10)
    assert int(state.stop) == NON_FINITE
    assert int(state.steps) == 0


def test_a_direction_the_rows_ignore_gives_no_nan_and_converges_in_the_others():
    state = solve(Ignores(), np.array([3.0, 7.0]), gtol=1e-10, max_steps=20)
    assert int(state.stop) == CONVERGED
    assert onp.all(onp.isfinite(onp.asarray(state.z)))
    assert onp.all(onp.isfinite(onp.asarray(state.curvature)))
    onp.testing.assert_allclose(state.z, [1.0, 7.0], atol=1e-9)


def test_float32_rosenbrock_ends_converged_not_stalled():
    with jax.enable_x64(False):
        state = solve(Rosenbrock(), np.array([-1.2, 1.0], dtype=np.float32))
        assert state.z.dtype == np.float32
        assert int(state.stop) == CONVERGED
        assert int(state.stop) not in (STALLED, RUNNING)
        onp.testing.assert_allclose(state.z, [1.0, 1.0], atol=1e-3)
