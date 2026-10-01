"""Maximum a posteriori fits of models, including images.

[`fit`][drpangloss.fitting.fit] takes the same arguments as
[`numpyro_model`][drpangloss.likelihood.numpyro_model]: a model (a template,
or a function of the parameters), a dict of numpyro priors whose keys are the
free parameters, and the data, plus optional regularisers (see
[`drpangloss.imaging`][drpangloss.imaging]). It finds the maximum a
posteriori parameters with Levenberg–Marquardt, L-BFGS or Adam, optimising
each parameter in unconstrained coordinates through the bijection to its
prior's support, in float64 by default. To sample the same posterior, pass
the same arguments to ``numpyro_model``.
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
from .fields import GaussianField
from .likelihood import (
    _check_positive_flux_prior,
    build_model,
    whitened_residuals,
)
from .models import SourceModel


def _bijection(distribution):
    from numpyro.distributions.transforms import biject_to

    return biject_to(distribution.support)


def _prior_residuals(path, distribution, value):
    """Residuals ``r`` with ``-log p(value) = 0.5 sum r² + const``, or None.

    ``None`` means the prior is flat on its support (Uniform,
    ImproperUniform), so it adds nothing to a least-squares objective.
    """
    import numpyro.distributions as dist

    if isinstance(distribution, dist.Independent):
        distribution = distribution.base_dist
    if isinstance(distribution, (dist.Uniform, dist.ImproperUniform)):
        return None
    if isinstance(distribution, dist.Normal):
        return np.ravel((value - distribution.loc) / distribution.scale)
    raise TypeError(
        f"The {type(distribution).__name__} prior on {path!r} has no "
        "least-squares form; use Normal, Uniform or ImproperUniform "
        "priors, or fit with method='lbfgs' or 'adam'."
    )


def _warn_if_field_hyperparameter(model, path):
    """Warn when a GaussianField's sigma or length is fitted by MAP."""
    parent, _, name = path.rpartition(".")
    if name in ("sigma", "length_mas") and parent:
        if isinstance(model.get(parent), GaussianField):
            warnings.warn(
                f"Fitting {path!r} by MAP: the hyperparameters of a Gaussian "
                "field are biased at the MAP (towards a flat field). Fix "
                "them, choose them from the evidence, or sample them.",
                UserWarning,
                stacklevel=4,
            )


class _Objective(eqx.Module):
    """The negative log posterior of a fit, as a loss and as residuals.

    ``residuals(z)`` is a vector whose half sum of squares is ``loss(z)``
    (up to a constant), for least-squares solvers; it raises ``TypeError``
    if a regulariser or prior has no least-squares form. ``z`` are the
    unconstrained coordinates of the parameters.
    """

    model: object
    data: tuple
    priors: dict
    regularisers: tuple

    def __init__(self, model, priors, data, regularisers=()):
        self.model = model
        self.data = tuple(data) if isinstance(data, (list, tuple)) else (data,)
        self.priors = dict(priors)
        self.regularisers = tuple(regularisers)
        if not self.priors:
            raise ValueError("priors must name at least one free parameter.")
        for path, prior in self.priors.items():
            if isinstance(model, SourceModel):
                model.get(path)  # raises for an unknown path
                _warn_if_field_hyperparameter(model, path)
            if is_flux_param(path):
                _check_positive_flux_prior(path, prior)

    @property
    def paths(self):
        """The free parameters' paths, in the order of ``priors``."""
        return tuple(self.priors)

    def init(self, values=None):
        """Unconstrained coordinates of ``values`` (default: the template's)."""
        values = {} if values is None else dict(values)
        z = {}
        for path, prior in self.priors.items():
            if path not in values:
                if not isinstance(self.model, SourceModel):
                    raise ValueError(
                        f"No starting value for {path!r}: pass init= when "
                        "the model is a function."
                    )
                values[path] = self.model.get(path)  # raises if unknown
            z[path] = _bijection(prior).inv(np.asarray(values[path], float))
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
        return build_model(
            self.model, self.paths, [values[p] for p in self.paths]
        )

    def data_residuals(self, model):
        """Whitened residuals of ``model`` for each dataset, as a list.

        ``model`` is one model for all the datasets, or a list of them, one
        per dataset.
        """
        if not isinstance(model, (list, tuple)):
            model = [model] * len(self.data)
        if len(model) != len(self.data):
            raise ValueError(
                f"The model function returned {len(model)} models for "
                f"{len(self.data)} datasets."
            )
        return [whitened_residuals(m, d) for m, d in zip(model, self.data)]

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
            parts.append(np.ravel(regulariser.residuals(_reference(model))))
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
        penalty = sum(r.value(_reference(model)) for r in self.regularisers)
        log_prior = sum(
            np.sum(prior.log_prob(values[path]))
            for path, prior in self.priors.items()
        )
        return 0.5 * chi2 + penalty - log_prior


def _reference(model):
    """The model regularisers act on: the first if there is one per dataset."""
    return model[0] if isinstance(model, (list, tuple)) else model


@dataclasses.dataclass(frozen=True)
class FitResult:
    """The result of [`fit`][drpangloss.fitting.fit].

    Attributes
    ----------
    model : SourceModel or list
        The fitted model, or models (one per dataset) if the model function
        returned a list.
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
    model,
    priors,
    data,
    regularisers=(),
    *,
    init=None,
    method=None,
    max_steps=None,
    gtol=1e-4,
    learning_rate=1e-2,
    cg_steps=50,
    dtype="float64",
):
    """Find the maximum a posteriori parameters of a model given data.

    Parameters
    ----------
    model : SourceModel or callable
        A template model whose leaves at the paths in ``priors`` are fitted
        (their values are the starting point), or a function called with the
        parameters as keyword arguments, as for
        [`numpyro_model`][drpangloss.likelihood.numpyro_model]. The function
        may return a list of models, one per dataset, sharing parameters:
        for example a scene and a [`Rotated`][drpangloss.models.Rotated]
        copy of it, for two epochs between which it turns. Regularisers
        then act on the first.
    priors : dict[str, numpyro.distributions.Distribution]
        A prior for each free parameter, keyed by its path (e.g.
        ``"comp.flux"`` or ``"env.log_brightness"``; see
        [`image_priors`][drpangloss.imaging.image_priors]). Priors on
        fluxes must have non-negative support.
    data : OIData or sequence of OIData
        The data, fitted jointly.
    regularisers : sequence, optional
        Penalties added to the loss, e.g. from
        [`drpangloss.imaging`][drpangloss.imaging].
    init : dict, optional
        Starting values by path, overriding the template's (required for
        a function model).
    method : {"lm", "lbfgs", "adam"}, optional
        ``"lm"``: Levenberg–Marquardt (optimistix) on the residuals, with
        a matrix-free inner solve (``cg_steps`` conjugate-gradient steps on
        the normal equations), so the Jacobian is never formed. The default
        when the whole objective has a least-squares form.
        ``"lbfgs"``: L-BFGS (optax) on the loss, for penalties
        such as maximum entropy and total variation; the default otherwise.
        ``"adam"``: Adam with ``learning_rate``, run for ``max_steps``.
    max_steps : int, optional
        Step limit (defaults: 1000 for LM, 20000 for L-BFGS, 2000 for Adam).
    gtol : float, optional
        LM and L-BFGS stop when no component of the gradient of the loss
        per data point exceeds ``gtol``, nor 1/1000 of its largest starting
        value (so that a fit started near a solution, e.g. along an
        L-curve, still converges).
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
        problem = cast_tree(
            _Objective(model, priors, data, regularisers), dtype
        )
        z0 = problem.init(cast_tree(init, dtype))
        method = method or ("lm" if _has_residuals(problem, z0) else "lbfgs")
        # Optimisers see the loss per data point, so step sizes and
        # tolerances do not depend on the size of the dataset.
        ndata = [int(np.size(d.flatten_data()[0])) for d in problem.data]
        scale = float(max(sum(ndata), 1))
        if method == "lm":
            z, steps, converged = _lm(
                problem, z0, scale, max_steps or 1000, gtol, cg_steps
            )
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


def _has_residuals(problem, z):
    """Whether the whole objective has a least-squares form."""
    try:
        problem.residuals(z)
    except TypeError:
        return False
    return True


def _largest(tree):
    """The largest absolute value in a pytree of arrays."""
    return np.max(np.stack([np.max(np.abs(x)) for x in jax.tree.leaves(tree)]))


class _GradientStoppedLM(optx.LevenbergMarquardt):
    """Levenberg-Marquardt that stops when the gradient of the loss is small.

    optimistix's own test, on the change of the residuals, passes at once
    from a warm start, and a test on the parameters never passes for
    log-brightness pixels that should be dark (they drift towards -inf
    without changing the image). This stops like the L-BFGS path instead.
    """

    tolerance: jax.Array
    scale: float

    def __init__(self, tolerance, scale, cg_steps):
        super().__init__(
            rtol=0.0,
            atol=0.0,
            linear_solver=lx.Normal(
                lx.CG(rtol=0.0, atol=0.0, max_steps=cg_steps)
            ),
        )
        self.tolerance = tolerance
        self.scale = scale

    def terminate(self, fn, y, args, options, state, tags):
        gradient = jax.grad(lambda z: args.loss(z) / self.scale)(y)
        return _largest(gradient) <= self.tolerance, optx.RESULTS.successful


def _lm(problem, z0, scale, max_steps, gtol, cg_steps):
    g0 = _largest(jax.grad(lambda z: problem.loss(z) / scale)(z0))
    solver = _GradientStoppedLM(np.minimum(gtol, 1e-3 * g0), scale, cg_steps)
    solution = optx.least_squares(
        lambda z, p: p.residuals(z) / np.sqrt(scale),
        solver,
        z0,
        args=problem,
        max_steps=max_steps,
        throw=False,
    )
    converged = bool(solution.result == optx.RESULTS.successful)
    return solution.value, int(solution.stats["num_steps"]), converged


def _lbfgs(problem, z0, scale, max_steps, gtol):
    # optax's L-BFGS (with a zoom line search), stopped on the gradient.
    # A gradient test suits log-brightness pixels: the gradient for a pixel
    # that should be dark vanishes with its flux, while its value keeps
    # drifting, and a test on the loss change can stop at the first short
    # line-search step. The gradient must also fall by a factor of 1000 from
    # where it started, so that a warm start (e.g. along an L-curve, whose
    # gradient is small from the outset) still converges rather than
    # stopping at once.
    optimiser = optax.lbfgs()

    @eqx.filter_jit
    def run(problem, z0):
        def loss(z):
            return problem.loss(z) / scale

        value_and_grad = optax.value_and_grad_from_state(loss)

        tolerance = np.minimum(gtol, 1e-3 * _largest(jax.grad(loss)(z0)))

        def keep_going(carry):
            step, _, _, gradient = carry
            return (step < max_steps) & (gradient > tolerance)

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
                _largest(grad),
            )

        start = (0, z0, optimiser.init(z0), np.asarray(np.inf, dtype=float))
        count, z, _, gradient = jax.lax.while_loop(keep_going, step, start)
        return z, count, gradient <= tolerance

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
