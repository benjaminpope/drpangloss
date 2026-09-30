"""Fitting problems and optimisers.

A [`Problem`][drpangloss.fitting.Problem] is the single specification shared
by fitting and sampling: a template model, the data, a prior for every free
parameter, and optional regularisers (see [`drpangloss.imaging`][drpangloss.imaging]).
Free parameters are exactly the keys of ``priors``, which are numpyro
distributions; each is optimised or sampled in unconstrained coordinates,
through the bijection to the prior's support.

The problem exposes three views of the same objective:

* ``residuals(z)``: a vector whose half sum of squares is the loss, for
  least-squares solvers such as Levenberg–Marquardt;
* ``loss(z)``: the negative log posterior (up to a constant), with priors
  and regularisers evaluated in the constrained parameters;
* ``logdensity(z)``: the log posterior density in the unconstrained
  coordinates, including the Jacobian of the bijections, for samplers.

[`fit`][drpangloss.fitting.fit] minimises it with Levenberg–Marquardt,
L-BFGS or Adam, in float64 by default.
"""

import dataclasses
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import lineax as lx
import optax
import optimistix as optx

from ._precision import cast_tree, run_in
from ._utils import is_flux_param
from .likelihood import _check_positive_flux_prior, whitened_residuals


def _bijection(distribution):
    from numpyro.distributions.transforms import biject_to

    return biject_to(distribution.support)


def _prior_residuals(path, distribution, value):
    """Residuals ``r`` with ``-log p(value) = 0.5 sum r² + const``, or None.

    ``None`` means the prior is flat on its support (Uniform,
    ImproperUniform), so it adds nothing to a least-squares objective.
    """
    import numpyro.distributions as dist

    if isinstance(distribution, (dist.Uniform, dist.ImproperUniform)):
        return None
    if isinstance(distribution, dist.Normal):
        return np.ravel((value - distribution.loc) / distribution.scale)
    raise TypeError(
        f"The {type(distribution).__name__} prior on {path!r} has no "
        "least-squares form; use Normal, Uniform or ImproperUniform "
        "priors, or fit with method='lbfgs' or 'adam'."
    )


class Problem(eqx.Module):
    """A model, data, priors and regularisers: what to fit or sample.

    Parameters
    ----------
    model : SourceModel
        Template model. Its leaves at the paths in ``priors`` are the free
        parameters (they also give the starting point); every other leaf is
        fixed.
    data : OIData or sequence of OIData
        The data, fitted jointly with the one model.
    priors : dict[str, numpyro.distributions.Distribution]
        A prior for each free parameter, keyed by its path in ``model``
        (e.g. ``"env.log_brightness"``, ``"comp.flux"``). Priors on fluxes
        must have non-negative support.
    regularisers : sequence, optional
        Penalties on the model, e.g. from [`drpangloss.imaging`][drpangloss.imaging]. Each
        has ``value(model)``, added to the loss; ``residuals(model)`` if it
        can be written as a least-squares term (``value = 0.5 sum r²``);
        and ``probabilistic``, whether it is a genuine log prior density
        that may enter ``logdensity``.

    Examples
    --------
    >>> problem = Problem(
    ...     BinaryModelCartesian(100.0, 50.0, 0.01),
    ...     data,
    ...     {"dra": dist.Uniform(-300, 300), "ddec": dist.Uniform(-300, 300),
    ...      "flux": dist.Uniform(0.0, 1.0)},
    ... )
    >>> result = fit(problem)
    """

    model: object
    data: tuple
    priors: dict
    regularisers: tuple

    def __init__(self, model, data, priors, regularisers=()):
        self.model = model
        self.data = tuple(data) if isinstance(data, (list, tuple)) else (data,)
        self.priors = dict(priors)
        self.regularisers = tuple(regularisers)
        if not self.priors:
            raise ValueError("priors must name at least one free parameter.")
        for path, prior in self.priors.items():
            model.get(path)  # raises for an unknown path
            if is_flux_param(path):
                _check_positive_flux_prior(path, prior)

    @property
    def paths(self):
        """The free parameters' paths, in the order of ``priors``."""
        return tuple(self.priors)

    @property
    def has_residuals(self):
        """Whether the whole objective has a least-squares form."""
        try:
            self.residuals(self.init())
        except TypeError:
            return False
        return True

    def init(self):
        """Unconstrained coordinates of the template model's values."""
        z = {}
        for path, prior in self.priors.items():
            value = np.asarray(self.model.get(path), dtype=float)
            z[path] = _bijection(prior).inv(value)
        return z

    def constrain(self, z):
        """Map unconstrained coordinates to parameter values."""
        return {
            path: _bijection(prior)(z[path])
            for path, prior in self.priors.items()
        }

    def build(self, z):
        """The model with the parameters at unconstrained coordinates ``z``."""
        values = self.constrain(z)
        return self.model.set(
            list(self.paths), [values[p] for p in self.paths]
        )

    def data_residuals(self, model):
        """Whitened residuals of ``model`` for each dataset, as a list."""
        return [whitened_residuals(model, d) for d in self.data]

    def residuals(self, z):
        """Residual vector whose half sum of squares is ``loss(z)`` + const.

        Raises ``TypeError`` if a regulariser or prior has no least-squares
        form (then use L-BFGS or Adam).
        """
        model = self.build(z)
        values = self.constrain(z)
        parts = self.data_residuals(model)
        for regulariser in self.regularisers:
            if not hasattr(regulariser, "residuals"):
                raise TypeError(
                    f"{type(regulariser).__name__} has no least-squares "
                    "form; fit with method='lbfgs' or 'adam'."
                )
            parts.append(np.ravel(regulariser.residuals(model)))
        for path, prior in self.priors.items():
            r = _prior_residuals(path, prior, values[path])
            if r is not None:
                parts.append(r)
        return np.concatenate(parts)

    def loss(self, z):
        """Negative log posterior (up to a constant) at ``z``.

        Priors are evaluated at the constrained values, without the
        Jacobian of the bijection, so the minimum is the maximum a
        posteriori in the model's own parameters.
        """
        model = self.build(z)
        values = self.constrain(z)
        chi2 = sum(np.sum(r**2) for r in self.data_residuals(model))
        penalty = sum(r.value(model) for r in self.regularisers)
        log_prior = sum(
            np.sum(prior.log_prob(values[path]))
            for path, prior in self.priors.items()
        )
        return 0.5 * chi2 + penalty - log_prior

    def logdensity(self, z):
        """Log posterior density in unconstrained coordinates, for samplers.

        Includes the log Jacobian of each bijection. Raises ``ValueError``
        if a regulariser is not a probability density (e.g. maximum
        entropy, total variation or total squared variation).
        """
        improper = [
            type(r).__name__ for r in self.regularisers if not r.probabilistic
        ]
        if improper:
            raise ValueError(
                f"{', '.join(improper)} are penalties, not log prior "
                "densities, so the problem has no posterior to sample."
            )
        model = self.build(z)
        values = self.constrain(z)
        chi2 = sum(np.sum(r**2) for r in self.data_residuals(model))
        density = -0.5 * chi2 - sum(r.value(model) for r in self.regularisers)
        for path, prior in self.priors.items():
            transform = _bijection(prior)
            density += np.sum(prior.log_prob(values[path]))
            density += np.sum(
                transform.log_abs_det_jacobian(z[path], values[path])
            )
        return density


@dataclasses.dataclass(frozen=True)
class FitResult:
    """The result of [`fit`][drpangloss.fitting.fit].

    Attributes
    ----------
    model : SourceModel
        The fitted model.
    values : dict
        The fitted parameter values, keyed by path.
    info : dict
        ``method``; ``converged`` (``None`` for Adam, which has no
        convergence test); ``steps``; ``loss`` (the unscaled negative log
        posterior); ``chi2`` and ``ndata``, per dataset; and ``chi2_red``,
        the total χ² per data point.
    """

    model: object
    values: dict
    info: dict


def fit(
    problem,
    method=None,
    *,
    max_steps=None,
    rtol=1e-5,
    atol=1e-5,
    gtol=1e-4,
    learning_rate=1e-2,
    cg_steps=50,
    dtype="float64",
):
    """Find the maximum a posteriori parameters of a problem.

    Parameters
    ----------
    problem : Problem
        What to fit.
    method : {"lm", "lbfgs", "adam"}, optional
        ``"lm"``: Levenberg–Marquardt on ``problem.residuals``, with a
        matrix-free inner solve (``cg_steps`` conjugate-gradient steps on
        the normal equations), so the Jacobian is never formed. The default
        when the whole objective has a least-squares form.
        ``"lbfgs"``: L-BFGS (optax) on ``problem.loss``, for penalties
        such as maximum entropy and total variation; the default otherwise.
        ``"adam"``: Adam with ``learning_rate``, run for ``max_steps``.
    max_steps : int, optional
        Step limit (defaults: 1000 for LM, 20000 for L-BFGS, 2000 for Adam).
    rtol, atol : float, optional
        LM's convergence tolerances, on the change of the residuals between
        steps.
    gtol : float, optional
        L-BFGS stops when no component of the gradient of the loss per data
        point exceeds ``gtol``.
    learning_rate : float, optional
        Adam's learning rate, in unconstrained coordinates.
    cg_steps : int, optional
        Conjugate-gradient steps per LM step. The inner solve runs for
        exactly this many steps (its tolerances are zero), because an inner
        solve that stops at a step limit would abort the outer one.
    dtype : {"float64", "float32"}, optional
        Precision of the fit. The default runs in float64 inside a local
        ``jax.enable_x64`` context. The returned model and values are cast
        back to JAX's precision outside the fit (float32, unless x64 is
        enabled), so that they work with the rest of your code.

    Returns
    -------
    FitResult
        The fitted model, parameter values and diagnostics. A warning is
        raised if LM or L-BFGS did not converge.
    """
    with run_in(dtype):
        problem = cast_tree(problem, dtype)
        method = method or ("lm" if problem.has_residuals else "lbfgs")
        z0 = problem.init()
        # Optimisers see the loss per data point, so step sizes and
        # tolerances do not depend on the size of the dataset.
        ndata = [int(np.size(d.flatten_data()[0])) for d in problem.data]
        scale = float(max(sum(ndata), 1))
        if method == "lm":
            solver = optx.LevenbergMarquardt(
                rtol=rtol,
                atol=atol,
                norm=_loss_norm,
                linear_solver=lx.Normal(
                    lx.CG(rtol=0.0, atol=0.0, max_steps=cg_steps)
                ),
            )
            solution = optx.least_squares(
                lambda z, p: p.residuals(z) / np.sqrt(scale),
                solver,
                z0,
                args=problem,
                max_steps=max_steps or 1000,
                throw=False,
            )
            z, converged = solution.value, _succeeded(solution)
            steps = int(solution.stats["num_steps"])
        elif method == "lbfgs":
            z, steps, converged = _lbfgs(
                problem, z0, scale, max_steps or 20_000, gtol
            )
        elif method == "adam":
            steps = max_steps or 2000
            z, converged = _adam(problem, z0, scale, learning_rate, steps)
        else:
            raise ValueError(
                f"method must be 'lm', 'lbfgs' or 'adam', not {method!r}."
            )
        if converged is False:
            warnings.warn(
                f"fit(method={method!r}) did not converge in {steps} steps.",
                RuntimeWarning,
                stacklevel=2,
            )
        model = problem.build(z)
        values = problem.constrain(z)
        chi2 = [float(np.sum(r**2)) for r in problem.data_residuals(model)]
        info = {
            "method": method,
            "converged": converged,
            "steps": steps,
            "loss": float(problem.loss(z)),
            "chi2": chi2,
            "ndata": ndata,
            "chi2_red": sum(chi2) / scale,
        }
    ambient = "float64" if jax.config.jax_enable_x64 else "float32"
    return FitResult(
        cast_tree(model, ambient), cast_tree(values, ambient), info
    )


def _loss_norm(tree):
    """Norm for the convergence test that looks at the loss, not the parameters.

    optimistix stops when both the parameter change and the loss change are
    small. The log-brightnesses of image pixels that should be dark keep
    drifting towards -inf without changing the image, so the parameter test
    would never pass; we therefore let only the loss (a residual vector or
    scalar) decide, through the root-mean-square of its relative change.
    The parameters are a dict, the loss is an array.
    """
    if isinstance(tree, dict):
        return np.asarray(0.0)
    return optx.rms_norm(tree)


def _succeeded(solution):
    return bool(solution.result == optx.RESULTS.successful)


def _lbfgs(problem, z0, scale, max_steps, gtol):
    # optax's L-BFGS (with a zoom line search), stopped on the gradient.
    # A gradient test suits log-brightness pixels: the gradient for a pixel
    # that should be dark vanishes with its flux, while its value keeps
    # drifting, and a test on the loss change can stop at the first short
    # line-search step.
    optimiser = optax.lbfgs()

    @eqx.filter_jit
    def run(problem, z0):
        def loss(z):
            return problem.loss(z) / scale

        value_and_grad = optax.value_and_grad_from_state(loss)

        def largest(tree):
            return np.max(
                np.stack([np.max(np.abs(x)) for x in jax.tree.leaves(tree)])
            )

        def keep_going(carry):
            step, _, _, gradient = carry
            return (step < max_steps) & (gradient > gtol)

        def step(carry):
            count, z, state, _ = carry
            value, grad = value_and_grad(z, state=state)
            updates, state = optimiser.update(
                grad, state, z, value=value, grad=grad, value_fn=loss
            )
            return (
                count + 1,
                optax.apply_updates(z, updates),
                state,
                largest(grad),
            )

        start = (0, z0, optimiser.init(z0), np.asarray(np.inf, dtype=float))
        count, z, _, gradient = jax.lax.while_loop(keep_going, step, start)
        return z, count, gradient <= gtol

    z, count, converged = run(problem, z0)
    return z, int(count), bool(converged)


def _adam(problem, z0, scale, learning_rate, steps):
    optimiser = optax.adam(learning_rate)

    @eqx.filter_jit
    def run(problem, z0):
        grad = jax.grad(lambda z: problem.loss(z) / scale)

        def step(carry, _):
            z, state = carry
            updates, state = optimiser.update(grad(z), state, z)
            return (optax.apply_updates(z, updates), state), None

        (z, _), _ = jax.lax.scan(
            step, (z0, optimiser.init(z0)), None, length=steps
        )
        return z

    return run(problem, z0), None
