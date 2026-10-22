"""Maximum a posteriori fits of models, including images.

[`fit`][virgil.fitting.fit] takes the same arguments as
[`numpyro_model`][virgil.likelihood.numpyro_model]: a model (a template,
or a function of the parameters), a dict of numpyro priors whose keys are the
free parameters, and the data, plus optional regularizers (see
[`virgil.imaging`][virgil.imaging]). It finds the maximum a
posteriori parameters with Levenberg–Marquardt, L-BFGS or Adam, optimizing
each parameter in unconstrained coordinates through the bijection to its
prior's support, in float64 by default. A parameter whose prior is uniform in
some coordinate (a log-uniform scale, an isotropic inclination) is fitted in
that flat coordinate, where its prior adds nothing to the loss, so that
Levenberg–Marquardt works with the Jeffreys priors. To sample the same
posterior, pass the same arguments to ``numpyro_model``; it samples in numpyro's
coordinates unless told ``flat_coordinates=True``.
"""

import dataclasses
import math
import time
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import lineax as lx
import numpy as onp
import optax
import optimistix as optx
from jax.flatten_util import ravel_pytree

from ._deprecate import renamed
from ._flat import _base, _flat_coordinate, _FlatBijection
from ._precision import cast_tree, run_in
from ._utils import _per_dataset, _reference, is_flux_param
from .angles import is_angle_vector, vector_angle, vector_site
from .fields import GaussianField, MultipoleField
from .likelihood import (
    _check_positive_flux_prior,
    _curvature_rows,
    _whitened_and_errors,
    build_model,
    is_tied,
    noise_for,
    noise_sites,
    tied_log_prior,
)
from .models import (
    Image,
    MultipoleKernel,
    RadialGP,
    RadialGPFunction,
    SourceModel,
)
from ._deprecate import external_stacklevel


def _bijection(distribution):
    from numpyro.distributions.transforms import biject_to

    return biject_to(distribution.support)


class NoLeastSquaresForm(TypeError):
    """The objective has no residual vector (use L-BFGS or Adam).

    A ``TypeError`` for backward compatibility; ``fit`` picks L-BFGS on it
    and ``variational`` falls back to a diagonal start, while other errors
    propagate.
    """


class SingularCurvature(ValueError):
    """The Gauss–Newton curvature is singular at the given values: some
    direction is constrained by neither the data nor a proper prior (an
    ``ImproperUniform`` or flat-in-coordinate parameter the data do not
    fix), or the matrix is not positive definite in the working precision.
    """


class NoStartingValue(ValueError):
    """A free parameter has no starting value (a function model needs one)."""


class GaussNewtonMass(dict):
    """NUTS keyword arguments from ``gauss_newton_mass``.

    A ``dict`` that unpacks into ``NUTS(...)`` as usual, and records the
    ``flat_coordinates`` of its matrix as an attribute for
    [`chain_init_params`][virgil.likelihood.chain_init_params]. ``condition``
    is the condition number of the matrix's correlation form (``None`` for
    a diagonal mass).
    """

    flat_coordinates = False
    condition = None


def _prior_residuals(path, distribution, value):
    """Residuals ``r`` with ``-log p(value) = 0.5 sum r² + const``, or None.

    ``None`` means the prior is flat on its support or in the coordinate ``fit``
    optimizes it in, so it adds nothing to a least-squares objective.
    """
    import numpyro.distributions as dist

    if _flat_coordinate(distribution) is not None:
        return None
    distribution = _base(distribution)
    if is_angle_vector(distribution):
        return distribution.residuals(value)  # ``value`` is the vector
    if isinstance(distribution, (dist.Uniform, dist.ImproperUniform)):
        return None
    if isinstance(distribution, dist.Normal):
        return np.ravel((value - distribution.loc) / distribution.scale)
    raise NoLeastSquaresForm(
        f"The {type(distribution).__name__} prior on {path!r} has no "
        "least-squares form; use Normal, Uniform, ImproperUniform or "
        "LogUniform or AngleVector priors (or one with a flat coordinate, "
        "such as an isotropic inclination; an AngleVector takes von Mises "
        "priors on angles), or fit with method='lbfgs' or 'adam'."
    )


def _traced(prior):
    """``prior`` with its Python-number parameters as JAX arrays, so that a new prior
    bound does not recompile the jitted solvers (equinox treats numbers as static).
    """

    def to_array(leaf):
        if isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            return np.asarray(float(leaf))
        return leaf

    return jax.tree.map(to_array, prior)


def _warn_if_field_hyperparameter(model, path):
    """Warn when a Gaussian process's sigma or length is fitted by MAP."""
    parent, _, name = path.rpartition(".")
    hyper = (
        "sigma",
        "length_mas",
        "ell",
        "ell_r",
        "ell_theta",
        "length_r",
        "length_theta",
    )
    if name in hyper and parent:
        if isinstance(
            model.get(parent),
            (
                GaussianField,
                RadialGP,
                RadialGPFunction,
                MultipoleKernel,
                MultipoleField,
            ),
        ):
            warnings.warn(
                f"Fitting {path!r} by MAP: the hyperparameters of a Gaussian "
                "field are biased at the MAP (towards a flat field). Fix "
                "them, choose them from the evidence, or sample them.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )


def _log_jacobian(bijection, z):
    """``log |dx/dz|`` of a bijection (virgil's flat one or numpyro's)."""
    if isinstance(bijection, _FlatBijection):
        return bijection.log_abs_det_jacobian(z)
    return bijection.log_abs_det_jacobian(z, bijection(z))


class _Objective(eqx.Module):
    """The negative log posterior of a fit, as a loss and as residuals.

    ``residuals(z)`` has half sum of squares ``loss(z)`` (up to a constant), for
    least-squares solvers; it raises ``TypeError`` if a regularizer or prior has
    no least-squares form, or if error terms are fitted. ``z`` are the
    unconstrained coordinates of the parameters and error terms. With ``flat=True``
    (for ``fit``) a prior with a flat coordinate is unconstrained through it; with
    ``flat=False``, as ``numpyro_model`` samples by default, every parameter uses
    numpyro's bijection of its prior's support.
    """

    model: object
    data: tuple
    priors: dict
    regularizers: tuple
    noise: dict
    likelihoods: tuple
    flat: bool = eqx.field(static=True)

    def __init__(
        self,
        model,
        priors,
        data,
        regularizers=(),
        noise=None,
        likelihoods=(),
        flat=True,
    ):
        self.model = model
        self.flat = flat
        self.data = tuple(data) if isinstance(data, (list, tuple)) else (data,)
        self.likelihoods = tuple(likelihoods)
        self.priors = {path: _traced(p) for path, p in priors.items()}
        self.regularizers = tuple(regularizers)
        self.noise = {
            site: (prior if is_tied(prior) else _traced(prior), datasets, term)
            for site, (prior, datasets, term) in noise_sites(
                noise, len(self.data)
            ).items()
        }
        if not self.priors:
            raise ValueError("priors must name at least one free parameter.")
        for path, prior in self.priors.items():
            if isinstance(model, SourceModel):
                model.get(path)  # raises for an unknown path
                _warn_if_field_hyperparameter(model, path)
            if is_flux_param(path):
                _check_positive_flux_prior(path, prior)

    @property
    def _has_gains(self):
        """Whether any dataset's covariance depends on the model: gains, or
        marginalized flux scales (``OIData.has_model_covariance``)."""
        return any(
            getattr(d, "has_model_covariance", False) for d in self.data
        )

    @property
    def paths(self):
        """The free parameters' paths, in the order of ``priors``."""
        return tuple(self.priors)

    def _in_flat_coordinate(self, prior):
        """Whether ``prior``'s parameter is fitted in its flat coordinate."""
        return self.flat and _flat_coordinate(prior) is not None

    def bijection(self, prior):
        """The map from unconstrained coordinates to ``prior``'s support."""
        if self._in_flat_coordinate(prior):
            return _FlatBijection(*_flat_coordinate(prior))
        return _bijection(prior)

    @property
    def sites(self):
        """The numpyro sites of the free parameters, in the order of
        ``priors``: ``"<path>_vec"`` for an angle vector, else its path."""
        return tuple(
            vector_site(p) if is_angle_vector(q) else p
            for p, q in self.priors.items()
        )

    def _prior_value(self, values, path):
        """The value a prior's density is evaluated at: the vector for an
        angle vector, else the parameter."""
        if is_angle_vector(self.priors[path]):
            return values[vector_site(path)]
        return values[path]

    def init(self, values=None):
        """Unconstrained coordinates of ``values`` (default: the template's).

        Error terms start at 1 (scales and supplied gain widths), 0 (``wavel_offset``,
        ``north_angle``) or 0.01 (added errors, other gain widths), or at their
        prior's mean if that is outside the support or on its boundary.
        """
        values = {} if values is None else dict(values)
        z = {}
        for site, (prior, _, term) in self.noise.items():
            if is_tied(prior):
                continue  # a function of the parameters, not a coordinate
            unit = term.endswith("scale") or term.endswith("_modes")
            zero = term in ("wavel_offset", "north_angle")
            default = 0.0 if zero else 1.0 if unit else 0.01
            start = values.get(site, default)
            if not bool(prior.support(np.asarray(start, float))):
                start = prior.mean
            z[site] = self.bijection(prior).inv(np.asarray(start, float))
            if not bool(np.all(np.isfinite(z[site]))):
                # On the boundary of a bounded prior (e.g. 0 for
                # Uniform(0, ...)), the unconstrained coordinate is
                # infinite: start inside it, at the prior's mean.
                mean = np.asarray(prior.mean, float)
                z[site] = self.bijection(prior).inv(mean)
        for path, prior in self.priors.items():
            if is_angle_vector(prior) and vector_site(path) in values:
                z[path] = np.asarray(values[vector_site(path)], float)
                continue
            if path not in values:
                if not isinstance(self.model, SourceModel):
                    raise NoStartingValue(
                        f"No starting value for {path!r}: pass init= when "
                        "the model is a function."
                    )
                values[path] = self.model.get(path)  # raises if unknown
            if is_angle_vector(prior):
                # The unit vector of the starting angle (degrees).
                angle = np.deg2rad(np.asarray(values[path], float))
                z[path] = np.stack([np.cos(angle), np.sin(angle)])
                continue
            z[path] = self.bijection(prior).inv(
                np.asarray(values[path], float)
            )
        return z

    def constrain(self, z):
        """Map unconstrained coordinates to parameter and error-term values.

        An ``AngleVector`` angle appears twice: in degrees at its path, and as the
        vector at ``"<path>_vec"``.
        """
        values = {}
        for path, prior in self.priors.items():
            if is_angle_vector(prior):
                values[vector_site(path)] = z[path]
                values[path] = vector_angle(z[path])
            else:
                values[path] = self.bijection(prior)(z[path])
        for site, (prior, _, _) in self.noise.items():
            if not is_tied(prior):
                values[site] = self.bijection(prior)(z[site])
        params = {path: values[path] for path in self.priors}
        for site, (prior, _, _) in self.noise.items():
            if is_tied(prior):
                values[site] = prior(params)
        return values

    def build(self, z):
        """The model with the parameters at unconstrained coordinates ``z``."""
        values = self.constrain(z)
        return build_model(
            self.model, self.paths, [values[p] for p in self.paths]
        )

    @property
    def _has_term_norms(self):
        return any(getattr(t, "has_log_norm", False) for t in self.likelihoods)

    def data_residuals(self, model, values=None):
        """Whitened residuals of ``model`` for each dataset, as a list.

        ``model`` is one model for all the datasets, or a list of them, one
        per dataset; ``values`` holds the error terms, if any are fitted.
        """
        return [w for w, _ in self._whitened(model, values)]

    def _whitened(self, model, values=None):
        """(whitened residuals, inflated errors) for each dataset."""
        models = _per_dataset(model, len(self.data))
        values = {} if values is None else values
        return [
            _whitened_and_errors(m, d, noise_for(self.noise, values, i))
            for i, (m, d) in enumerate(zip(models, self.data))
        ]

    def _check_least_squares(self, term_norms=True):
        """Raise :class:`NoLeastSquaresForm` if the objective has no residual
        vector: fitted error terms, data with gains, regularizers without
        ``residuals`` and, if ``term_norms``, likelihood terms with a
        ``log_norm``."""
        if self.noise:
            raise NoLeastSquaresForm(
                "Fitted error terms have no least-squares form (the "
                "likelihood's normalization depends on them); fit with "
                "method='lbfgs' or 'adam'."
            )
        if self._has_gains:
            raise NoLeastSquaresForm(
                "Data with gains or marginalized flux scales have no "
                "least-squares form (their covariance, and its determinant, "
                "depend on the model); fit "
                "with method='lbfgs' or 'adam'."
            )
        if term_norms and self._has_term_norms:
            raise NoLeastSquaresForm(
                "Likelihood terms with a log_norm (a fitted RV jitter, or "
                "the Jacobian of a position-angle prior) have no "
                "least-squares form; fit with method='lbfgs' or 'adam'."
            )
        for regularizer in self.regularizers:
            if not hasattr(regularizer, "residuals"):
                raise NoLeastSquaresForm(
                    f"{type(regularizer).__name__} has no least-squares "
                    "form; fit with method='lbfgs' or 'adam'."
                )

    def residuals(self, z):
        """Residual vector whose half sum of squares is ``loss(z)`` + const.

        Raises :class:`NoLeastSquaresForm` if a regularizer or prior has no
        least-squares form (use L-BFGS or Adam).
        """
        self._check_least_squares()
        model = self.build(z)
        values = self.constrain(z)
        parts = self.data_residuals(model)
        parts += [term(values) for term in self.likelihoods]
        for regularizer in self.regularizers:
            parts.append(np.ravel(regularizer.residuals(_reference(model))))
        for path, prior in self.priors.items():
            r = _prior_residuals(path, prior, self._prior_value(values, path))
            if r is not None:
                parts.append(r)
        return np.concatenate(parts)

    # The Gauss–Newton curvature F = JᵀJ + diag(P) (design/gauss_newton_plan.md).

    @property
    def coordinate_keys(self):
        """The keys of ``z`` in the order of the flat coordinate vector: the
        paths of ``priors``, then the non-tied noise sites."""
        return tuple(self.priors) + tuple(
            site
            for site, (prior, _, _) in self.noise.items()
            if not is_tied(prior)
        )

    @property
    def coordinate_sites(self):
        """The numpyro sites of the flat coordinates, in the same order:
        ``sites``, then the non-tied noise sites."""
        return self.sites + self.coordinate_keys[len(self.priors) :]

    def ravel(self, z):
        """``(flat, unravel)``: ``z`` as one vector in ``coordinate_keys``
        order, and the map back to a dict. (``ravel_pytree`` of a dict would
        sort the keys, so the arrays go in as a list.)"""
        keys = self.coordinate_keys
        flat, unravel_list = ravel_pytree([z[k] for k in keys])

        def unravel(x):
            return dict(zip(keys, unravel_list(x)))

        return flat, unravel

    def _coordinate_priors(self):
        priors = dict(self.priors)
        priors.update({site: p for site, (p, _, _) in self.noise.items()})
        return [(key, priors[key]) for key in self.coordinate_keys]

    def curvature_rows(self, z):
        """The curvature rows ρ(z), whose Jacobian J gives JᵀJ.

        In order: each dataset's mean and normalizer rows
        (``likelihood._curvature_rows``), each likelihood term's residuals, and the
        residuals of ``AngleVector`` priors. Regularizers are not included.
        """
        model = self.build(z)
        values = self.constrain(z)
        models = _per_dataset(model, len(self.data))
        rows = [
            _curvature_rows(m, d, noise_for(self.noise, values, i))
            for i, (m, d) in enumerate(zip(models, self.data))
        ]
        rows += [np.ravel(term(values)) for term in self.likelihoods]
        rows += [
            prior.residuals(z[path])
            for path, prior in self.priors.items()
            if is_angle_vector(prior)
        ]
        return np.concatenate(rows) if rows else np.zeros(0)

    def _no_prior_curvature(self, prior, jacobian):
        """Whether ``prior``'s coordinates have P = 0 by type (known when tracing).

        Angle vectors, ``ImproperUniform``, and, without the Jacobian, priors flat in
        their fitted coordinate and ``Uniform``. A prior that is neither elementwise
        nor an angle vector raises :class:`NoLeastSquaresForm`.
        """
        import numpyro.distributions as dist

        base = _base(prior)
        # ``_base`` strips ``.expand`` and ``.to_event``: an image's
        # ``Normal(0, 1).to_event(2)`` is still a separable, elementwise prior.
        if is_angle_vector(prior):
            return True
        if isinstance(base, dist.ImproperUniform):
            return True
        if base.event_shape != ():
            raise NoLeastSquaresForm(
                f"{type(base).__name__} has event_shape {base.event_shape}: "
                "its curvature is not diagonal, and the Gauss–Newton matrix "
                "takes only elementwise priors and angle vectors."
            )
        if jacobian:
            return False
        return self._in_flat_coordinate(prior) or isinstance(
            base, dist.Uniform
        )

    def curvature_zero(self, z, jacobian):
        """A boolean NumPy vector over the flat coordinates: those whose P
        is zero by type (``_no_prior_curvature``)."""
        return onp.concatenate(
            [
                onp.full(
                    np.size(z[key]), self._no_prior_curvature(p, jacobian)
                )
                for key, p in self._coordinate_priors()
            ]
        )

    def curvature_diag(self, z, jacobian):
        """P, the diagonal of curvature from elementwise priors, over the flat coordinates.

        For each elementwise prior, ``max(0, ∂²/∂z² [-log p(x(z)) - log|dx/dz|])``,
        the Jacobian term only if ``jacobian`` (NUTS's potential has it, ``fit``'s
        loss does not). An ``AngleVector`` gives 0 (it has rows); a prior that is
        neither elementwise nor an angle vector raises :class:`NoLeastSquaresForm`.
        """
        parts = []
        for key, prior in self._coordinate_priors():
            zk = np.asarray(z[key])
            if self._no_prior_curvature(prior, jacobian):
                parts.append(np.zeros(zk.size, zk.dtype))
                continue
            bijection = self.bijection(prior)

            def potential(v, prior=prior, bijection=bijection):
                term = -prior.log_prob(bijection(v))
                if jacobian:
                    term = term - _log_jacobian(bijection, v)
                return np.sum(term)

            ones = np.ones_like(zk)
            second = jax.jvp(jax.grad(potential), (zk,), (ones,))[1]
            parts.append(np.ravel(np.maximum(0.0, second)))
        return np.concatenate(parts) if parts else np.zeros(0)

    def penalty(self, z):
        """The summed regularizers at ``z``."""
        model = _reference(self.build(z))
        return sum((r.value(model) for r in self.regularizers), 0.0)

    def loss(self, z):
        """Negative log posterior (up to a constant) at ``z``.

        A prior with a flat coordinate adds nothing; other priors are evaluated at the
        constrained values, without the Jacobian of the bijection.
        """
        model = self.build(z)
        values = self.constrain(z)
        whitened = self._whitened(model, values)
        chi2 = sum(np.sum(w**2) for w, _ in whitened)
        norms = 0.0
        for t in self.likelihoods:
            if hasattr(t, "residuals_and_log_norm"):
                # One evaluation gives both (a profiled scale needs the
                # residuals to find its own normalization).
                r, n = t.residuals_and_log_norm(values)
                chi2, norms = chi2 + np.sum(r**2), norms + n
            else:
                chi2 = chi2 + np.sum(t(values) ** 2)
                if getattr(t, "has_log_norm", False):
                    norms = norms + t.log_norm(values)
        # With fitted error terms, the normalization of the Gaussian
        # likelihood, sum(log σ), is no longer a constant.
        log_norm = sum(np.sum(np.log(e)) for _, e in whitened)
        log_norm = log_norm if self.noise or self._has_gains else 0.0
        # A likelihood term with a fitted jitter reports its own Σ log σ.
        log_norm = log_norm + norms
        penalty = sum(r.value(_reference(model)) for r in self.regularizers)
        log_prior = sum(
            np.sum(prior.log_prob(self._prior_value(values, path)))
            for path, prior in self.priors.items()
            if not self._in_flat_coordinate(prior)
        )
        for site, (prior, _, _) in self.noise.items():
            if not is_tied(prior) and not self._in_flat_coordinate(prior):
                log_prior = log_prior + np.sum(prior.log_prob(values[site]))
        log_prior = log_prior + tied_log_prior(self.noise, values)
        return 0.5 * chi2 + log_norm + penalty - log_prior


@dataclasses.dataclass(frozen=True)
class FitResult:
    """The result of [`fit`][virgil.fitting.fit].

    Attributes
    ----------
    model : SourceModel or list
        The fitted model, or models (one per dataset) if the model function
        returned a list.
    values : dict
        The fitted parameter values, keyed by path, in the model's own
        parameters (not the flat coordinates some are fitted in). An angle with an
        [`AngleVector`][virgil.angles.AngleVector] prior is in degrees at
        its path and its vector at ``"<path>_vec"``, so that ``values`` can
        start numpyro (``init_to_value``).
    info : dict
        ``method``; ``converged`` (``None`` for Adam, which has no
        convergence test); ``steps``; ``loss`` (the unscaled negative log
        posterior); ``chi2`` and ``ndata``, per dataset and then per
        likelihood term; ``chi2_red``,
        the total χ² per data point; and ``grad_norm``, the infinity norm
        (largest absolute component) of the gradient of the loss per data
        point at the result, in the unconstrained coordinates of the
        convergence test (see ``gtol`` in [`fit`][virgil.fitting.fit]). With fitted error terms, χ² uses the
        inflated errors, and ``values`` holds the terms too. ``stop`` says
        why an unconverged fit ended (``None`` when it converged, and for
        an Adam fit with a finite loss, which has no convergence test):
        ``"limit"`` (``max_steps``), ``"time"``
        (``time_limit``), ``"stalled"`` (an L-BFGS step no longer moved
        the parameters), ``"non-finite"`` (a NaN or infinite loss or
        gradient; any method ending on a non-finite loss is unconverged)
        or ``"failed"`` (another LM failure). ``at_bound`` lists
        the parameters and error terms that ended within 1/1000 of their
        prior's range of one of its finite ends, in the coordinate they
        are fitted in: a fit that runs into a prior bound (say an
        eccentricity at 0.9) is often not at a mode. ``derived`` holds
        quantities that likelihood terms compute from the fit and that are not
        parameters, such as the profiled error scales ``ŝ`` of a radio
        term (``"cal.sigma"``).
    """

    model: object
    values: dict
    info: dict


@renamed()
def fit(
    model,
    priors,
    data,
    regularizers=(),
    *,
    noise=None,
    init=None,
    method=None,
    max_steps=None,
    gtol=1e-4,
    max_step_size=2.0,
    lbfgs_memory=50,
    learning_rate=1e-2,
    cg_steps=50,
    dtype="float64",
    likelihoods=(),
    time_limit=None,
    progress=None,
    precondition=None,
):
    """Find the maximum a posteriori parameters of a model given data.

    `model`, `priors`, `data`, `regularizers`, `noise`, `likelihoods` and `dtype`
    are shared with [`numpyro_model`][virgil.likelihood.numpyro_model]; see
    [common arguments](../common_arguments.md). A prior that is uniform in some
    coordinate (`LogUniform`, an isotropic inclination, ...) is optimized in it;
    the returned `values` are always in the model's own parameters.

    Parameters
    ----------
    model : SourceModel or callable
        A template whose leaves at the paths in ``priors`` are fitted (their
        values are the starting point), or a function of the parameters that
        returns a model or a list of models, one per dataset.
    priors : dict[str, numpyro.distributions.Distribution]
        A prior for each free parameter, keyed by its path.
    data : OIData or sequence of OIData
        The data, fitted jointly; may be ``()`` when ``likelihoods`` holds all
        the data.
    regularizers : sequence, optional
        Penalties added to the loss, e.g. from [`virgil.imaging`][virgil.imaging].
    noise : dict or list of dict, optional
        Priors on error and calibration terms fitted with the parameters; see
        [common arguments](../common_arguments.md#noise).
    init : dict, optional
        Starting values by path (or ``noise`` site); required for a function model.
    method : {"lm", "lbfgs", "adam"}, optional
        ``"lm"``: Levenberg–Marquardt, the default when the objective has a
        least-squares form (Normal priors and priors flat in their fitted
        coordinate); matrix-free (``cg_steps``) above 200 coordinates.
        ``"lbfgs"``: for penalties such as maximum entropy and total variation,
        and the default otherwise. ``"adam"``: with ``learning_rate``.
    max_steps : int, optional
        Step limit (defaults: 1000 for LM, 20000 for L-BFGS, 2000 for Adam).
    gtol : float, optional
        LM and L-BFGS stop when no component of the gradient of the loss per data
        point exceeds ``gtol``, nor 1/1000 of its largest starting value (floored
        at √eps of the dtype times that value, and at ``1e-6 * gtol``).
    max_step_size : float, optional
        L-BFGS moves no unconstrained coordinate by more than this per step (a
        factor ``exp(max_step_size)`` for a log-brightness pixel).
    lbfgs_memory : int, optional
        Past steps L-BFGS keeps to model the curvature.
    learning_rate : float, optional
        Adam's learning rate, in unconstrained coordinates.
    cg_steps : int, optional
        Conjugate-gradient steps per LM step, above 200 coordinates.
    dtype : {"float64", "float32"}, optional
        Precision of the fit; see [common arguments](../common_arguments.md#dtype).
    likelihoods : sequence, optional
        Further Gaussian likelihood terms, such as
        [`PositionData.term`][virgil.orbits.PositionData.term].
    time_limit : float, optional
        Wall-clock budget (seconds) for LM and L-BFGS, which run in chunks and
        stop after the first chunk that ends past it, with
        ``info["stop"] == "time"`` and a warning. Compilation counts.
    progress : callable or bool, optional
        Between chunks, call a function with a dict of ``steps``, ``loss``,
        ``grad_norm`` and ``elapsed`` (seconds), or print them if ``True``.
    precondition : {None, "diag"}, optional
        ``"diag"`` rescales each unconstrained coordinate by the square root of
        the diagonal of the Gauss–Newton curvature (see
        [`gauss_newton_diag`][virgil.fitting.gauss_newton_diag]) for L-BFGS and
        Adam, which helps when the coordinates' curvatures differ by many
        decades, as for the faint pixels of a flat-prior image. The diagonal is
        refreshed every L-BFGS round (Adam: once). The optimum, ``gtol``,
        ``max_step_size`` and ``info["grad_norm"]`` keep their meaning in the
        unconstrained coordinates. ``info["precondition_decades"]`` lists, per
        round, the spread of the diagonal in decades (the largest within a
        coordinate block). LM ignores it: it already uses JᵀJ.

    Returns
    -------
    FitResult
        The fitted model, values and diagnostics. A warning is raised if LM or
        L-BFGS did not converge; ``info["stop"]`` holds the reason, and
        ``info["grad_norm"]`` the largest absolute component of the gradient of
        the loss per data point, in the unconstrained coordinates.
    """
    if not math.isfinite(max_step_size) or max_step_size <= 0:
        raise ValueError(
            f"max_step_size must be finite and positive, not {max_step_size}."
        )
    if time_limit is not None and not time_limit >= 0:
        raise ValueError(
            f"time_limit must be a number of seconds, not {time_limit}."
        )
    if precondition not in (None, False, "diag"):
        raise ValueError(
            f"precondition must be None or 'diag', not {precondition!r}."
        )
    precondition = bool(precondition)
    budget = _Budget(time_limit, progress)
    with run_in(dtype):
        problem = cast_tree(
            _Objective(model, priors, data, regularizers, noise, likelihoods),
            dtype,
        )
        z0 = problem.init(cast_tree(init, dtype))
        method = method or ("lm" if _has_residuals(problem, z0) else "lbfgs")
        # Optimizers see the loss per data point, so step sizes and
        # tolerances do not depend on the size of the dataset.
        ndata = [d.n_independent for d in problem.data]
        # A term's size is the length of its residuals at the start.
        start = problem.constrain(z0)
        ndata += [int(np.size(term(start))) for term in problem.likelihoods]
        scale = float(max(sum(ndata), 1))
        # Numbers go to the jitted solvers as arrays, which are traced, so
        # that new values do not recompile. The step limits of LM and Adam
        # set a loop's length, so they stay static.
        traced_scale, gtol = np.asarray(scale), np.asarray(gtol)
        stop = grad_norm = None
        decades = []
        if method == "lm":
            limit = max_steps or 1000
            z, steps, converged, stop = _lm(
                problem, z0, traced_scale, limit, gtol, cg_steps, budget
            )
        elif method == "lbfgs":
            limit = max_steps or 20_000
            z, steps, converged, stop, grad_norm, decades = _lbfgs(
                problem,
                z0,
                traced_scale,
                np.asarray(limit),
                gtol,
                np.asarray(max_step_size),
                int(lbfgs_memory),
                budget,
                precondition,
            )
        elif method == "adam":
            if budget.chunked:
                raise ValueError(
                    "Adam runs exactly max_steps steps: set those, not "
                    "time_limit or progress (which apply to 'lm' and "
                    "'lbfgs')."
                )
            steps = limit = max_steps or 2000
            metric = (
                _metric(problem, z0, traced_scale) if precondition else None
            )
            z = _adam(
                problem,
                z0,
                traced_scale,
                np.asarray(learning_rate),
                steps,
                None if metric is None else metric.normalized(),
            )
            converged = None
        else:
            raise ValueError(
                f"method must be 'lm', 'lbfgs' or 'adam', not {method!r}."
            )
        model = problem.build(z)
        values = problem.constrain(z)
        loss, chi2, grad_norm = _summary(problem, z, traced_scale, grad_norm)
        loss = float(loss)
        if not math.isfinite(loss) and converged is not False:
            # E.g. LM or Adam ending on NaN parameters.
            converged, stop = False, "non-finite"
        if converged is False:
            warnings.warn(
                _not_converged(method, stop, steps, limit, dtype, time_limit),
                RuntimeWarning,
                stacklevel=external_stacklevel(),
            )
        chi2 = [float(c) for c in chi2]
        info = {
            "method": method,
            "converged": converged,
            "stop": stop,
            "steps": steps,
            "loss": loss,
            "chi2": chi2,
            "ndata": ndata,
            "chi2_red": sum(chi2) / scale,
            "grad_norm": float(grad_norm),
            "precondition": "diag"
            if precondition and method != "lm"
            else None,
            "precondition_decades": decades,
            "at_bound": _at_bound(problem, values),
            "derived": _derived(problem, values),
        }
    ambient = "float64" if jax.config.jax_enable_x64 else "float32"
    return FitResult(
        cast_tree(model, ambient), cast_tree(values, ambient), info
    )


def _print_progress(report):
    print(
        f"fit: {report['steps']} steps, loss {report['loss']:.6g}, "
        f"grad_norm {report['grad_norm']:.3g}, {report['elapsed']:.1f} s",
        flush=True,
    )


class _Budget:
    """The time limit and progress report of a fit (see ``fit``).

    The clock starts when the budget is made, at the start of ``fit``, so
    the time to set up and compile the fit counts.
    """

    def __init__(self, time_limit=None, progress=None):
        self.time_limit = time_limit
        self.progress = _print_progress if progress is True else progress
        self.progress = self.progress or None
        self.start = time.monotonic()
        self._reported = None

    @property
    def chunked(self):
        """Whether the optimizer must stop between chunks of steps."""
        return self.time_limit is not None or self.progress is not None

    def check(self, steps, loss_and_gradient):
        """Report progress after ``steps`` steps; whether time is up.

        ``loss_and_gradient()`` returns the loss and the gradient norm,
        and is called only to report them.
        """
        elapsed = self.finish(steps, loss_and_gradient)
        return self.time_limit is not None and elapsed >= self.time_limit

    def finish(self, steps, loss_and_gradient):
        """Report progress at ``steps`` unless just reported; return the elapsed time."""
        elapsed = time.monotonic() - self.start
        if self.progress is not None and self._reported != steps:
            self._reported = steps
            loss, grad_norm = loss_and_gradient()
            self.progress(
                {
                    "steps": steps,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "elapsed": elapsed,
                }
            )
        return elapsed


def _derived(problem, values):
    """Quantities a likelihood term computes from the fit that are not sites
    (a radio term's profiled error scales ``ŝ``), by name."""
    out = {}
    for term in problem.likelihoods:
        if hasattr(term, "derived"):
            out.update(
                {k: onp.asarray(v) for k, v in term.derived(values).items()}
            )
    return out


# A parameter is at a bound of its prior when it is within this fraction of
# the prior's range of one end, in the coordinate it is fitted in.
_AT_BOUND = 1e-3


def _bound_fraction(prior, value, flat=True):
    """Where ``value`` lies in its prior's finite interval, as a fraction.

    Only genuine edges of the support count (a LogUniform's, in log x if ``flat``,
    or an interval support's); ``None`` for unbounded priors, angle vectors, tied
    terms and other priors with a flat coordinate, whose ends are coordinate poles.
    """
    if is_tied(prior) or is_angle_vector(prior):
        return None
    import numpyro.distributions as dist

    if isinstance(_base(prior), dist.LogUniform):
        if flat:
            to_flat, _, low, high = _flat_coordinate(prior)
            value = to_flat(value)
        else:
            low, high = _base(prior).low, _base(prior).high
    elif _flat_coordinate(prior) is not None:
        return None
    else:
        support = _base(prior).support
        low = getattr(support, "lower_bound", None)
        high = getattr(support, "upper_bound", None)
        if low is None or high is None:
            return None
    low, high = onp.asarray(low, float), onp.asarray(high, float)
    if not (onp.all(onp.isfinite(low)) and onp.all(onp.isfinite(high))):
        return None
    return (onp.asarray(value, float) - low) / (high - low)


def _at_bound(problem, values):
    """The paths and noise sites within ``_AT_BOUND`` of a prior bound."""
    priors = dict(problem.priors)
    priors.update({site: p for site, (p, _, _) in problem.noise.items()})
    found = []
    for site, prior in priors.items():
        fraction = _bound_fraction(prior, values[site], problem.flat)
        if fraction is not None and onp.any(
            (fraction < _AT_BOUND) | (fraction > 1 - _AT_BOUND)
        ):
            found.append(site)
    return tuple(found)


# The one size limit for dense n × n matrices. Above this many coordinates
# ``gauss_newton_mass`` returns a diagonal mass (and no condition number), and
# ``svi.variational`` refuses the dense guides: n² entries are 134 MB in
# float64 here, and a 128² image (n = 16384) would be 2 GB, 256² 34 GB.
_DENSE_MASS_MAX = 4096


def _check_dense_size(n, what, hint=""):
    """Raise ``ValueError`` if ``what`` would need a dense n × n matrix."""
    if n > _DENSE_MASS_MAX:
        raise ValueError(
            f"{what} needs a dense {n} × {n} matrix ({n * n:.1e} entries), "
            f"more than the {_DENSE_MASS_MAX} coordinates virgil allows."
            + (f" {hint}" if hint else "")
        )


# P is floored at this fraction of its largest entry in the Woodbury branch,
# so that W = P^-½ is finite.
_P_FLOOR = 1e-12


def gauss_newton_mass(
    model,
    priors,
    data,
    values,
    *,
    likelihoods=(),
    noise=None,
    flat_coordinates=False,
):
    """A NUTS mass matrix from the Gauss–Newton curvature at a fit.

    Near the maximum a posteriori, a posterior with one peak is close to a
    Gaussian whose precision is F = JᵀJ + diag(P) in the unconstrained
    coordinates numpyro samples (J: Jacobian of the whitened residuals; P: curvature
    of the priors and transforms). Its inverse is the mass matrix, so that
    directions the data fix tightly take the same step size as those left to the
    prior. See [the Gauss–Newton mass matrix](../gauss_newton_mass.md) for the
    construction, fitted error terms and gains, what it leaves out, and
    [when not to use it](../gauss_newton_mass.md#when-not-to-use-it). Use it at
    fixed field hyperparameters, with every sampled parameter in ``priors``.

    With more than 4096 coordinates (images larger than 64²) the result is a
    diagonal mass, with a ``UserWarning``.

    Parameters
    ----------
    model, priors, data
        As for [`fit`][virgil.fitting.fit]. Any elementwise prior is accepted
        (``event_shape == ()``; a ``MultivariateNormal`` or ``Dirichlet`` raises
        ``NoLeastSquaresForm``). ``ImproperUniform`` priors add no curvature, nor
        do [`AngleVector`][virgil.angles.AngleVector] priors, whose residuals are
        rows instead; an angle vector's block is keyed by its site,
        ``"<path>_vec"``. Data may carry gains.
    values : dict
        The parameter values at which to take the curvature, normally
        ``fit(model, priors, data).values``, including the fitted error terms if
        ``noise`` is given.
    likelihoods : sequence, optional
        Further likelihood terms, as for ``fit``; ``data`` may then be ``()``.
    noise : dict or list of dict, optional
        Priors on error terms, as for ``fit``; they are coordinates of the matrix.
        A term tied to the parameters (a function instead of a prior) is not one.
    flat_coordinates : bool, optional
        Whether the sampler moves in the priors' flat coordinates, as for
        ``numpyro_model`` (default ``False``); pass the same value to both. See
        [common arguments](../common_arguments.md#flat_coordinates).

    Returns
    -------
    GaussNewtonMass
        A dict of keyword arguments for ``numpyro.infer.NUTS``:
        ``inverse_mass_matrix``, ``dense_mass`` and ``adapt_mass_matrix=False``
        (a covariance estimated from a few hundred draws in thousands of
        dimensions is far worse than this matrix). The attribute ``condition`` is
        the condition number of the correlation matrix, or ``None`` for a
        diagonal mass; above 1e10 the matrix is numerically unreliable. For the
        diagonal mass ``dense_mass`` is empty and ``inverse_mass_matrix`` holds a
        vector.

    Raises
    ------
    SingularCurvature
        If a direction is constrained by neither the data nor a proper prior (an
        ``ImproperUniform`` or flat-in-coordinate parameter the data do not
        fix), or the matrix is not positive definite. A parameter under a
        bounded prior is not one: it gets the curvature of the transform.

    Examples
    --------
    >>> result = fit(scene, priors, data)  # doctest: +SKIP
    >>> kernel = NUTS(numpyro_model(result.model, priors, data),
    ...               init_strategy=init_to_value(values=result.values),
    ...               **gauss_newton_mass(scene, priors, data, result.values))  # doctest: +SKIP
    """
    with run_in("float64"):
        # The curvature is taken in the unconstrained coordinates NUTS
        # samples: numpyro_model's, flat or not as it is told.
        problem = cast_tree(
            _Objective(
                model,
                priors,
                data,
                noise=noise,
                likelihoods=likelihoods,
                flat=flat_coordinates,
            ),
            "float64",
        )
        z = problem.init(cast_tree(values, "float64"))
        n = sum(np.size(z[k]) for k in problem.coordinate_keys)
        dense = n <= _DENSE_MASS_MAX
        covariance, ok, condition = _gauss_newton_covariance(problem, z, dense)
    if not ok:
        raise SingularCurvature(
            "The Gauss–Newton curvature is singular: a direction is "
            "constrained by neither the data nor a proper prior (an "
            "ImproperUniform or flat-in-coordinate parameter the data do "
            "not fix), or the matrix is not positive definite."
        )
    sites = problem.coordinate_sites
    if not dense:
        warnings.warn(
            f"The Gauss–Newton mass has {n} coordinates, more than "
            f"{_DENSE_MASS_MAX}: returning its diagonal, not the dense "
            "matrix, which would need n² entries.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    out = GaussNewtonMass(
        inverse_mass_matrix={sites: covariance},
        dense_mass=[sites] if dense else [],
        adapt_mass_matrix=False,
    )
    out.flat_coordinates = bool(flat_coordinates)
    out.condition = condition
    return out


def _correlation_condition(covariance):
    """The condition number of the covariance scaled to unit diagonal."""
    if not onp.all(onp.isfinite(covariance)):
        return onp.inf
    diagonal = onp.diag(covariance)
    if not onp.all(diagonal > 0):
        return onp.inf
    scale = onp.sqrt(diagonal)
    correlation = covariance / onp.outer(scale, scale)
    eigenvalues = onp.linalg.eigvalsh(0.5 * (correlation + correlation.T))
    if eigenvalues[0] <= 0.0:
        return onp.inf
    return float(eigenvalues[-1] / eigenvalues[0])


def _gauss_newton_covariance(problem, z, dense=True):
    """F⁻¹ for F = JᵀJ + diag(P) over the flat coordinates of ``problem``.

    Returns ``(covariance, ok, condition)``: the covariance (its diagonal if
    ``dense`` is false), whether it is positive definite and finite, and the
    condition number of its correlation matrix (``None`` unless ``dense``).
    """
    covariance, ok = _covariance(problem, z, dense)
    covariance, ok = onp.asarray(covariance), bool(ok)
    condition = _correlation_condition(covariance) if dense else None
    return covariance, ok, condition


@eqx.filter_jit
def _covariance(problem, z, dense):
    """``(covariance, ok)`` of ``_gauss_newton_covariance``, from J and P.

    Cholesky of F = JᵀJ + diag(P) with at least as many rows as coordinates,
    otherwise a Woodbury inverse of the size of the rows; positive definiteness is
    checked on the covariance itself. See ``design/fitting_internals.md``.
    """
    flat, unravel = problem.ravel(z)
    n = flat.size

    def rows(x):
        return problem.curvature_rows(unravel(x))

    n_rows = jax.eval_shape(rows, flat).size
    p = problem.curvature_diag(z, True)
    zero = problem.curvature_zero(z, True)
    # Forward mode costs a pass per coordinate, reverse a pass per row.
    jac = (jax.jacfwd if n_rows >= n else jax.jacrev)(rows)(flat)
    if n_rows >= n:
        if dense:
            factor = jax.scipy.linalg.cho_factor(jac.T @ jac + np.diag(p))
            covariance = jax.scipy.linalg.cho_solve(factor, np.eye(n))
            ok_factor = np.all(np.isfinite(factor[0]))
        else:
            # Never forming n × n: every coordinate must be constrained,
            # which a diagonal mass can check (not a rank deficiency).
            diagonal = np.sum(jac**2, axis=0) + p
            covariance = 1.0 / diagonal
            ok_factor = np.all(diagonal > 0)
    else:
        zero_np = onp.asarray(zero)
        largest = np.max(np.where(zero_np, 0.0, p), initial=0.0)
        floor = _P_FLOOR * np.where(largest > 0.0, largest, 1.0)
        w = 1.0 / np.sqrt(np.where(zero_np, 1.0, np.maximum(p, floor)))
        k = jac * w
        inner = jax.scipy.linalg.cho_factor(np.eye(n_rows) + k @ k.T)
        ok_factor = np.all(np.isfinite(inner[0]))
        flats = onp.flatnonzero(zero_np)
        if dense:
            shrink = k.T @ jax.scipy.linalg.cho_solve(inner, k)
            a_inv = w[:, None] * (np.eye(n) - shrink) * w[None, :]
            covariance = a_inv
            if flats.size:
                columns = a_inv[:, flats]
                schur = jax.scipy.linalg.cho_factor(
                    np.eye(flats.size) - columns[flats]
                )
                covariance = a_inv + columns @ jax.scipy.linalg.cho_solve(
                    schur, columns.T
                )
                ok_factor = ok_factor & np.all(np.isfinite(schur[0]))
        else:
            # The diagonal of the same expression, never forming n × n:
            # with Y = L⁻¹K and I + KKᵀ = LLᵀ, a_inv = W (I - YᵀY) W.
            # (cho_factor gives the upper factor U, I + KKᵀ = UᵀU, so L = Uᵀ.)
            y = jax.scipy.linalg.solve_triangular(
                inner[0], k, trans="T", lower=inner[1]
            )
            covariance = w**2 * (1.0 - np.sum(y**2, axis=0))
            if flats.size:
                e = (
                    np.zeros((n, flats.size))
                    .at[flats, np.arange(flats.size)]
                    .set(1.0)
                )
                columns = w[:, None] * (e - y.T @ y[:, flats])
                columns = columns * w[flats][None, :]
                schur = jax.scipy.linalg.cho_factor(
                    np.eye(flats.size) - columns[flats]
                )
                solved = jax.scipy.linalg.cho_solve(schur, columns.T)
                covariance = covariance + np.sum(columns * solved.T, axis=1)
                ok_factor = ok_factor & np.all(np.isfinite(schur[0]))
    finite = np.all(np.isfinite(covariance))
    if dense:
        covariance = 0.5 * (covariance + covariance.T)
        chol = np.linalg.cholesky(covariance)
        positive = np.all(np.isfinite(chol)) & np.all(np.diag(chol) > 0)
    else:
        positive = np.all(covariance > 0)
    return covariance, ok_factor & finite & positive


def _has_residuals(problem, z):
    """Whether the whole objective has a least-squares form."""
    try:
        jax.eval_shape(problem.residuals, z)
    except NoLeastSquaresForm:
        return False
    return True


# The solvers below are jitted at module level, with the problem as an
# argument, so that repeated fits of problems with the same structure (an
# L-curve, a grid of hyperparameters, a fit per dataset) reuse one
# compilation. A function defined inside the call would be a new function
# each time and recompile.


@eqx.filter_jit
def _forward_summary(problem, z):
    """The loss and each dataset's chi-squared (with inflated errors) at ``z``.

    Forward only: it does not differentiate the model.
    """
    values = problem.constrain(z)
    residuals = problem.data_residuals(problem.build(z), values)
    residuals += [term(values) for term in problem.likelihoods]
    return problem.loss(z), [np.sum(r**2) for r in residuals]


@eqx.filter_jit
def _gradient_norm(problem, z, scale):
    """The largest component of the loss-per-data-point gradient at ``z``."""
    return _largest(jax.grad(_scaled_loss(problem, scale))(z))


def _summary(problem, z, scale, grad_norm=None):
    """The loss, each dataset's chi-squared and the gradient norm at ``z``, as
    reported in ``FitResult.info``.

    ``grad_norm`` is the gradient norm if the optimizer already holds it (L-BFGS
    does, see ``_final_gradient``). Left as ``None`` (LM, Adam), it is computed by
    differentiating the whole model, which is costly to compile.
    """
    loss, chi2 = _forward_summary(problem, z)
    if grad_norm is None:
        grad_norm = _gradient_norm(problem, z, scale)
    return loss, chi2, grad_norm


def _loss_and_gradient(problem, z, scale, grad_norm=None):
    """The loss and gradient norm at ``z``, as ``_summary`` reports them."""
    loss, _, grad_norm = _summary(problem, z, scale, grad_norm)
    return float(loss), float(grad_norm)


def _final_gradient(carry, metric=None):
    """The gradient norm at the point an L-BFGS carry holds, without a new gradient.

    The carry's own gradient is that of the step's start, one step behind its
    point. The line search leaves the gradient at the new point in the optimizer
    state, from which the next step would reuse it. Before any step the carry's
    gradient is the one at the start.
    """
    count, _, state, _, gradient, _ = carry
    if int(count) == 0:
        return gradient
    return _state_gradient_norm(state, metric)


@jax.jit
def _state_gradient_norm(state, metric=None):
    gradient = optax.tree_utils.tree_get(state, "grad")
    return _largest(gradient if metric is None else metric.gradient(gradient))


def _largest(tree):
    """The largest absolute value in a pytree of arrays."""
    return np.max(np.stack([np.max(np.abs(x)) for x in jax.tree.leaves(tree)]))


def _scaled_loss(problem, scale):
    return lambda z: problem.loss(z) / scale


def _tolerance(gradient, gtol):
    """The gradient tolerance of LM and L-BFGS, from the starting gradient.

    ``gtol``, or 1/1000 of the starting gradient's largest component if smaller,
    but no less than √eps times it, nor below ``1e-6 * gtol``. See
    ``design/fitting_internals.md``.
    """
    largest = _largest(gradient)
    floor = np.sqrt(np.finfo(largest.dtype).eps) * largest
    # Both floors scale with the starting gradient, which at an exact,
    # zero-residual optimum is itself rounding noise; 1e-6 * gtol keeps
    # such a start reachable (converged at once).
    return np.maximum(
        np.minimum(gtol, 1e-3 * largest), np.maximum(floor, 1e-6 * gtol)
    )


class _DescentTrustRegion(optx.ClassicalTrustRegion):
    """optimistix's trust region, also rejecting any step that raises the loss
    (optimistix accepts one when its predicted reduction is positive)."""

    def step(self, first_step, y, y_eval, f_info, f_eval_info, state):
        step_size, accept, result, new_state = super().step(
            first_step, y, y_eval, f_info, f_eval_info, state
        )
        worse = f_eval_info.as_min() > f_info.as_min()
        reject = worse & ~first_step
        shrunk = state.step_size * self.low_constant
        new_size = np.where(reject, shrunk, step_size)
        new_state = eqx.tree_at(lambda s: s.step_size, new_state, new_size)
        return new_size, accept & ~reject, result, new_state


class _GradientStoppedLM(optx.LevenbergMarquardt):
    """Levenberg-Marquardt that stops when the gradient of the loss is small, like the
    L-BFGS path (optimistix's own tests pass at once from a warm start, or never for
    pixels drifting to -inf).

    ``limit`` is the step bound, traced rather than optimistix's static
    ``max_steps``, so that a shorter last chunk reuses the compiled kernel.
    """

    tolerance: jax.Array
    limit: jax.Array

    def __init__(self, tolerance, limit, linear_solver):
        super().__init__(rtol=0.0, atol=0.0, linear_solver=linear_solver)
        self.search = _DescentTrustRegion()
        self.tolerance = tolerance
        self.limit = limit

    def terminate(self, fn, y, args, options, state, tags):
        problem, scale = args
        gradient = jax.grad(_scaled_loss(problem, scale))(y)
        converged = _largest(gradient) <= self.tolerance
        spent = state.num_steps >= self.limit
        result = optx.RESULTS.where(
            spent & ~converged,
            optx.RESULTS.nonlinear_max_steps_reached,
            optx.RESULTS.successful,
        )
        return converged | spent, result


def _lm_residuals(z, args):
    problem, scale = args
    return problem.residuals(z) / np.sqrt(scale)


@eqx.filter_jit
def _lm_tolerance(problem, z0, scale, gtol):
    gradient = jax.grad(_scaled_loss(problem, scale))(z0)
    return _tolerance(gradient, gtol)


# Up to this many unconstrained coordinates, LM's inner solve is a dense QR
# of the Jacobian, which costs one Jacobian-vector product per coordinate.
_DENSE_LM = 200


def _lm_linear_solver(z0, cg_steps):
    """QR for a few coordinates, else CG on the normal equations, run for exactly
    ``cg_steps`` steps (no more than the number of coordinates).
    """
    n = sum(int(np.size(x)) for x in jax.tree.leaves(z0))
    if n <= _DENSE_LM:
        return lx.QR()
    return lx.Normal(lx.CG(rtol=0.0, atol=0.0, max_steps=min(cg_steps, n)))


# Steps per chunk when fit has a time limit or reports progress: LM steps
# (a Jacobian and a solve each) cost far more than L-BFGS steps.
_LM_CHUNK = 50
_LBFGS_CHUNK = 200
# L-BFGS restarts after this many steps without converging, while a round
# still gains more than _RESTART_GAIN in the loss per data point.
_LBFGS_ROUND = 2000
_RESTART_GAIN = 1e-7


def _lm(problem, z0, scale, max_steps, gtol, cg_steps, budget):
    """Levenberg–Marquardt, returning ``(z, steps, converged, stop)``.

    With a time limit or progress report (``budget.chunked``), optimistix runs
    ``_LM_CHUNK`` steps at a time, each restarting from the last point with its
    damping reset, so the path can differ slightly from an unchunked run.
    """
    tolerance = _lm_tolerance(problem, z0, scale, gtol)
    linear_solver = _lm_linear_solver(z0, cg_steps)
    chunk = min(_LM_CHUNK, max_steps) if budget.chunked else max_steps
    z, steps = z0, 0
    while True:
        limit = np.asarray(min(chunk, max_steps - steps))
        solution = optx.least_squares(
            _lm_residuals,
            _GradientStoppedLM(tolerance, limit, linear_solver),
            z,
            args=(problem, scale),
            max_steps=None,
            throw=False,
        )
        z = solution.value
        steps += int(solution.stats["num_steps"])
        limited = solution.result == optx.RESULTS.nonlinear_max_steps_reached
        done = (
            solution.result == optx.RESULTS.successful
            or not limited
            or steps >= max_steps
        )
        if done:
            budget.finish(steps, lambda: _loss_and_gradient(problem, z, scale))
        if solution.result == optx.RESULTS.successful:
            return z, steps, True, None
        if not limited:
            return z, steps, False, "failed"
        if steps >= max_steps:
            return z, steps, False, "limit"
        if budget.check(steps, lambda: _loss_and_gradient(problem, z, scale)):
            return z, steps, False, "time"


# Up to this many coordinates the Gauss–Newton diagonal is exact (one
# Jacobian-vector product each); above, it is estimated from _DIAG_PROBES probes.
_EXACT_DIAG = 1024
_DIAG_PROBES = 16
# A coordinate's curvature is floored at this fraction of its block's largest.
_DIAG_FLOOR = 1e-3


def _gn_diagonal(rows, x, probes):
    """The diagonal of JᵀJ for the rows ``rows(x)``: exact (``probes=None``),
    or the mean of ``(Jᵀu)²`` over Gaussian ``u`` in row space, whose mean is
    the diagonal and which is never negative."""
    n = x.size
    m = jax.eval_shape(rows, x).size
    if probes is None and n <= m:

        def column(i):
            e = np.zeros(n, x.dtype).at[i].set(1.0)
            return np.sum(jax.jvp(rows, (x,), (e,))[1] ** 2)

        return jax.lax.map(column, np.arange(n))
    _, pull = jax.vjp(rows, x)
    if probes is None:

        def add(total, i):
            return total + pull(np.zeros(m, x.dtype).at[i].set(1.0))[
                0
            ] ** 2, None

        return jax.lax.scan(add, np.zeros(n, x.dtype), np.arange(m))[0]

    def squared(key):
        return pull(jax.random.normal(key, (m,), x.dtype))[0] ** 2

    keys = jax.random.split(jax.random.PRNGKey(0), probes)
    return np.mean(jax.lax.map(squared, keys), axis=0)


def _hessian_diagonal(f, x, probes):
    """The diagonal of the Hessian of ``f`` at ``x``: exact (``probes=None``) or
    the mean of ``v ⊙ Hv`` over Rademacher ``v``."""
    n = x.size
    grad = jax.grad(f)

    def product(v):
        return jax.jvp(grad, (x,), (v,))[1]

    if probes is None:

        def entry(i):
            return product(np.zeros(n, x.dtype).at[i].set(1.0))[i]

        return jax.lax.map(entry, np.arange(n))

    def estimate(key):
        v = jax.random.rademacher(key, (n,), x.dtype)
        return v * product(v)

    keys = jax.random.split(jax.random.PRNGKey(1), probes)
    return np.mean(jax.lax.map(estimate, keys), axis=0)


@eqx.filter_jit
def _fisher_diagonal(problem, z, probes):
    """The diagonal of the Gauss–Newton curvature of ``problem.loss`` at ``z``,
    over the flat coordinates: JᵀJ of the data and likelihood rows, the
    curvature of the priors, and the Hessian diagonal of the regularizers
    (floored at zero, as for ``gauss_newton_mass``)."""
    flat, unravel = problem.ravel(z)
    diagonal = _gn_diagonal(
        lambda x: problem.curvature_rows(unravel(x)), flat, probes
    ) + problem.curvature_diag(z, False)
    if problem.regularizers:
        penalty = _hessian_diagonal(
            lambda x: problem.penalty(unravel(x)), flat, probes
        )
        diagonal = diagonal + np.maximum(penalty, 0.0)
    return diagonal


def _diagonal_probes(n, n_probes):
    """``None`` for an exact diagonal of ``n`` coordinates, else the probes."""
    if n_probes is None:
        return None if n <= _EXACT_DIAG else _DIAG_PROBES
    return int(n_probes) or None


def gauss_newton_diag(
    model,
    priors,
    data,
    values,
    *,
    regularizers=(),
    likelihoods=(),
    noise=None,
    n_probes=None,
    dtype="float64",
):
    """The diagonal of the Gauss–Newton curvature of a fit's loss.

    The curvature is F = JᵀJ + diag(P) + diag(R) in the unconstrained
    coordinates [`fit`][virgil.fitting.fit] optimizes: J is the Jacobian of the
    whitened residuals, P the curvature of the priors and R the Hessian diagonal
    of the regularizers, floored at zero. It is the scale
    ``fit(..., precondition="diag")`` divides each coordinate by; see
    [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] for the dense
    matrix.

    Parameters
    ----------
    model, priors, data, regularizers, noise, likelihoods, dtype
        As for [`fit`][virgil.fitting.fit].
    values : dict
        Parameter values at which to take the curvature (e.g. ``fit(...).values``).
    n_probes : int, optional
        Up to 1024 coordinates the diagonal is exact, one Jacobian-vector
        product per coordinate. Above, it is estimated: JᵀJ from the mean of
        ``(Jᵀu)²`` over 16 Gaussian vectors ``u``, R from 16 Rademacher
        Hessian-vector products, each good to a relative ``√(2 / probes)``.
        Give a number of probes to estimate at any size, or ``0`` for the exact
        diagonal at any size.

    Returns
    -------
    dict
        The diagonal by coordinate name, with each coordinate's shape.
    """
    with run_in(dtype):
        problem = cast_tree(
            _Objective(model, priors, data, regularizers, noise, likelihoods),
            dtype,
        )
        z = problem.init(cast_tree(values, dtype))
        n = sum(int(np.size(z[k])) for k in problem.coordinate_keys)
        flat = _fisher_diagonal(problem, z, _diagonal_probes(n, n_probes))
        _, unravel = problem.ravel(z)
        return {k: onp.asarray(v) for k, v in unravel(flat).items()}


class _Metric(eqx.Module):
    """A frozen change of variables ``z = z_ref + P(width ⊙ z̃)``.

    ``width`` is the reciprocal square root of the loss's curvature per data
    point (see ``gauss_newton_diag``), so L-BFGS in ``z̃`` starts from a
    diagonal Newton step. ``P`` removes the mean over the support from an
    image's log-brightness block (``gauge``: the support masks), the direction
    the loss does not depend on (a softmax image), which the scaling would
    otherwise let drift.
    """

    zref: dict
    width: dict
    gauge: dict
    decades: float = 0.0

    def to_z(self, zt):
        return {
            k: self.zref[k] + self._project(k, self.width[k] * zt[k])
            for k in self.zref
        }

    def step(self, updates):
        """Steps of ``z̃`` in the coordinates of ``z``."""
        return {
            k: self._project(k, self.width[k] * u) for k, u in updates.items()
        }

    def gradient(self, gradient):
        """A gradient with respect to ``z̃`` as one with respect to ``z``."""
        return jax.tree.map(lambda g, w: g / w, gradient, self.width)

    def normalized(self):
        """The same, with the narrowest width scaled to one."""
        smallest = np.min(
            np.stack([np.min(w) for w in jax.tree.leaves(self.width)])
        )
        width = jax.tree.map(lambda w: w / smallest, self.width)
        return _Metric(self.zref, width, self.gauge, self.decades)

    def _project(self, key, x):
        if key not in self.gauge:
            return x
        mask = self.gauge[key]
        mean = np.sum(np.where(mask, x, 0.0)) / np.sum(mask)
        return x - np.where(mask, mean, 0.0)


class _InMetric(eqx.Module):
    """A problem's loss as a function of ``z̃``, for ``_scaled_loss``."""

    problem: object
    metric: _Metric

    def loss(self, zt):
        return self.problem.loss(self.metric.to_z(zt))


def _gauge_masks(problem):
    """Support masks of the plain log-brightness arrays being fitted: an Image's
    brightness is a softmax of them, so adding a constant leaves it unchanged."""
    model = problem.model
    if not isinstance(model, SourceModel):
        return {}
    masks = {}
    for path in problem.priors:
        if not path.endswith("log_brightness"):
            continue
        parent = path[: -len("log_brightness")].rstrip(".")
        image = model.resolved()
        image = image.get(parent) if parent else image
        if not isinstance(image, Image) or hasattr(
            image.log_brightness, "evaluate"
        ):
            continue
        shape = np.shape(image.log_brightness)
        support = image.support
        masks[path] = (
            np.ones(shape, bool)
            if support is None
            else np.asarray(support != 0)
        )
    return masks


def _metric(problem, z, scale):
    """The ``_Metric`` at ``z`` for a loss per data point ``scale``."""
    n = sum(int(np.size(z[k])) for k in problem.coordinate_keys)
    flat = _fisher_diagonal(problem, z, _diagonal_probes(n, None))
    _, unravel = problem.ravel(z)
    width, decades = {}, 0.0
    for key, d in unravel(flat).items():
        floor = _DIAG_FLOOR * np.max(d)
        usable = np.isfinite(floor) & (floor > 0)
        d = np.where(usable, np.maximum(np.nan_to_num(d), floor), 1.0 * scale)
        width[key] = np.sqrt(scale / d)
        decades = max(decades, float(np.log10(np.max(d) / np.min(d))))
    return _Metric(z, width, _gauge_masks(problem), decades)


def _capped_lbfgs(max_step_size, memory, metric=None):
    """optax's L-BFGS, with no coordinate moving more than ``max_step_size``.

    The whole direction is scaled down to the cap before the zoom line search (so
    the quasi-Newton direction is kept and the line search evaluates the points
    actually taken). With a ``metric`` the optimizer's coordinates are scaled and
    the cap acts on the step in the unscaled ones.
    """

    def cap(updates, state, params=None):
        step = updates if metric is None else metric.step(updates)
        factor = np.minimum(1.0, max_step_size / _largest(step))
        return jax.tree.map(lambda u: u * factor, updates), state

    return optax.chain(
        optax.scale_by_lbfgs(memory_size=memory),
        optax.scale(-1.0),
        optax.GradientTransformation(lambda params: optax.EmptyState(), cap),
        optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20,
            max_learning_rate=1.0,
            initial_guess_strategy="one",
        ),
    )


@eqx.filter_jit
def _lbfgs_loop(
    problem, carry, scale, stop_at, tolerance, step_size, memory, metric=None
):
    """L-BFGS steps from ``carry`` until step ``stop_at`` or a stop.

    ``carry`` is ``(count, z, optimizer state, loss, gradient, moved)``. Stops at
    ``stop_at`` steps, within ``tolerance``, when a step no longer moves the
    parameters, or when the loss or gradient is not finite. ``stop_at`` is traced,
    so chunked fits compile the loop once. With a ``metric``, ``z`` is in its
    scaled coordinates, and ``gradient`` is still the unscaled one.
    """
    optimizer = _capped_lbfgs(step_size, memory, metric)
    loss = _scaled_loss(problem, scale)
    if metric is not None:
        loss = _scaled_loss(_InMetric(problem, metric), scale)
    value_and_grad = optax.value_and_grad_from_state(loss)

    def keep_going(carry):
        count, _, _, value, gradient, moved = carry
        running = (count < stop_at) & (gradient > tolerance) & moved
        return running & np.isfinite(value)

    def step(carry):
        count, z, state, _, _, _ = carry
        value, grad = value_and_grad(z, state=state)
        updates, state = optimizer.update(
            grad, state, z, value=value, grad=grad, value_fn=loss
        )
        new = optax.apply_updates(z, updates)
        moved = jax.tree.reduce(
            np.logical_or,
            jax.tree.map(lambda a, b: np.any(a != b), new, z),
        )
        size = _largest(grad if metric is None else metric.gradient(grad))
        return count + 1, new, state, value, size, moved

    return jax.lax.while_loop(keep_going, step, carry)


@eqx.filter_jit
def _lbfgs_run(
    problem,
    z0,
    scale,
    max_steps,
    gtol,
    max_step_size,
    memory,
    tolerance=None,
    metric=None,
):
    # optax's L-BFGS (with a zoom line search, and capped steps; see
    # _capped_lbfgs), stopped on the gradient.
    # A gradient test suits log-brightness pixels: the gradient for a pixel
    # that should be dark vanishes with its flux, while its value keeps
    # drifting, and a test on the loss change can stop at the first short
    # line-search step. The gradient must also fall by a factor of 1000 from
    # where it started, so that a warm start (e.g. along an L-curve, whose
    # gradient is small from the outset) still converges rather than
    # stopping at once. The fit also stops, unconverged, when a step no
    # longer changes the parameters (the line search has run out of
    # precision, as can happen in float32). Returns the loop's carry (see
    # _lbfgs_loop) after at most max_steps steps, and the tolerance. With a
    # metric (see _Metric) the loop runs in its coordinates, from zero.
    loss = _scaled_loss(problem, scale)
    value, gradient = jax.value_and_grad(loss)(z0)
    # A restart passes the first round's tolerance, so that every round
    # meets the same test.
    if tolerance is None:
        tolerance = _tolerance(gradient, gtol)
    # The start itself may already be a stationary point.
    if metric is not None:
        z0 = jax.tree.map(np.zeros_like, z0)
    state = _capped_lbfgs(max_step_size, memory, metric).init(z0)
    start = (0, z0, state, value, _largest(gradient), np.asarray(True))
    carry = _lbfgs_loop(
        problem,
        start,
        scale,
        max_steps,
        tolerance,
        max_step_size,
        memory,
        metric,
    )
    return carry, tolerance


def _not_converged(method, stop, steps, limit, dtype, time_limit):
    """The warning for an unconverged fit, saying why it stopped."""
    if stop == "time":
        return (
            f"fit(method={method!r}) did not converge: it reached its "
            f"time_limit of {time_limit} s after {steps} steps."
        )
    if stop == "non-finite":
        return (
            f"fit(method={method!r}) did not converge: its loss or gradient "
            f"was not finite (NaN or inf) after {steps} steps; check the "
            "model and priors for invalid values."
        )
    if method == "lbfgs":
        return _lbfgs_not_converged(method, stop, steps, limit, dtype)
    if stop == "failed":
        # The solver gave up before its step limit: no damping found a step
        # that lowers the loss, as at a minimum with a discontinuous or
        # flat loss, or when started on one.
        return (
            f"fit(method={method!r}) did not converge: the solver stopped "
            f"after {steps} steps, before its limit of {limit}, because no "
            "step lowered the loss (the start may already be at a minimum "
            "of a loss with jumps, such as a closure-phase flip)."
        )
    if stop == "limit":
        return (
            f"fit(method={method!r}) did not converge in {steps} steps, the "
            "step limit; raise max_steps."
        )
    return f"fit(method={method!r}) did not converge in {steps} steps."


def _lbfgs_not_converged(method, stop, steps, limit, dtype):
    """The warning for an unconverged L-BFGS fit, saying why it stopped."""
    head = f"fit(method={method!r}) did not converge"
    if stop == "limit":
        return f"{head} in {steps} steps, the step limit; raise max_steps."
    hint = "; dtype='float64' can go further" if dtype == "float32" else ""
    return (
        f"{head}: it stopped after {steps} of {limit} steps, when a step no "
        f"longer changed the parameters (the line search ran out of {dtype} "
        f"precision{hint})."
    )


def _lbfgs(
    problem,
    z0,
    scale,
    max_steps,
    gtol,
    max_step_size,
    memory,
    budget,
    precondition=False,
):
    """L-BFGS in rounds, returning ``(z, steps, converged, stop, grad_norm)``.

    A round that ends at its step limit without converging restarts from its last
    point with fresh curvature memory, until a round gains less than
    ``_RESTART_GAIN`` in the loss per data point. With ``precondition``, each
    round runs in coordinates scaled by the diagonal curvature at its start
    (see ``_Metric``).
    """
    limit, total, z, tolerance = int(max_steps), 0, z0, None
    decades = []
    size = _LBFGS_ROUND
    while True:
        before = float(_forward_summary(problem, z)[0])
        metric = _metric(problem, z, scale) if precondition else None
        if precondition:
            decades.append(metric.decades)
        z, steps, converged, stop, tolerance, grad_norm = _lbfgs_round(
            problem,
            z,
            scale,
            min(size, limit - total),
            gtol,
            max_step_size,
            memory,
            budget,
            total,
            tolerance,
            metric,
        )
        total += steps
        # Only a round that used all its steps restarts: a stalled one
        # (the line search out of precision) or a timed-out or non-finite
        # one ends the fit.
        if converged or stop != "limit" or total >= limit:
            return z, total, converged, stop, grad_norm, decades
        gain = before - float(_forward_summary(problem, z)[0])
        if not gain > _RESTART_GAIN:  # also true of a NaN
            size = limit  # little progress: the rest as one last round


def _lbfgs_round(
    problem,
    z0,
    scale,
    max_steps,
    gtol,
    max_step_size,
    memory,
    budget,
    offset,
    tolerance=None,
    metric=None,
):
    """One round of L-BFGS: ``(z, steps, converged, stop, tolerance, grad_norm)``.

    ``grad_norm`` is the gradient norm at ``z``, taken from the optimizer's state.

    ``tolerance`` is the first round's convergence test (``None`` there), kept by
    later rounds. ``stop`` is ``"limit"``, ``"time"``, ``"stalled"`` or
    ``"non-finite"`` for an unconverged fit, and ``None`` if it converged. With a
    time limit or progress report the loop runs ``_LBFGS_CHUNK`` steps per call,
    carrying the whole optimizer state, so the path is the unchunked one.
    """
    limit = int(max_steps)
    first = min(limit, _LBFGS_CHUNK) if budget.chunked else limit
    carry, traced_tolerance = _lbfgs_run(
        problem,
        z0,
        scale,
        np.asarray(first),
        gtol,
        max_step_size,
        memory,
        None if tolerance is None else np.asarray(tolerance),
        metric,
    )
    tolerance, stop = float(traced_tolerance), None
    to_z = (lambda x: x) if metric is None else metric.to_z
    while budget.chunked:
        count, _, _, value, gradient, moved = carry
        count, value, gradient = int(count), float(value), float(gradient)
        running = count < limit and gradient > tolerance and bool(moved)
        if not (running and math.isfinite(value)):
            break
        z = to_z(carry[1])
        if budget.check(
            offset + count,
            lambda: _loss_and_gradient(
                problem, z, scale, _final_gradient(carry, metric)
            ),
        ):
            stop = "time"
            break
        carry = _lbfgs_loop(
            problem,
            carry,
            scale,
            np.asarray(min(count + _LBFGS_CHUNK, limit)),
            traced_tolerance,
            max_step_size,
            memory,
            metric,
        )
    count, z, _, value, gradient, moved = carry
    z = to_z(z)
    count, value, gradient = int(count), float(value), float(gradient)
    grad_norm = _final_gradient(carry, metric)
    if budget.chunked:
        budget.finish(
            offset + count,
            lambda: _loss_and_gradient(problem, z, scale, grad_norm),
        )
    # A gradient seen only at the start (count 0) that is not finite gives
    # a non-finite tolerance too.
    finite = math.isfinite(tolerance) and math.isfinite(value)
    if not finite or (count > 0 and not math.isfinite(gradient)):
        return z, count, False, "non-finite", tolerance, grad_norm
    if gradient <= tolerance:
        return z, count, True, None, tolerance, grad_norm
    if stop == "time":
        return z, count, False, "time", tolerance, grad_norm
    stop = "limit" if bool(moved) else "stalled"
    return z, count, False, stop, tolerance, grad_norm


@eqx.filter_jit
def _adam(problem, z0, scale, learning_rate, steps, metric=None):
    # With a metric, Adam runs in its scaled coordinates: its rms divisor
    # follows the size of the gradient, the metric the width of the
    # coordinate, so each coordinate's step is the learning rate times its
    # width. See design/fit_preconditioning.md.
    if metric is not None:
        problem, start = (
            _InMetric(problem, metric),
            jax.tree.map(np.zeros_like, z0),
        )
    else:
        start = z0
    optimizer = optax.adam(learning_rate)
    grad = jax.grad(_scaled_loss(problem, scale))

    def step(carry, _):
        z, state = carry
        updates, state = optimizer.update(grad(z), state, z)
        return (optax.apply_updates(z, updates), state), None

    (z, _), _ = jax.lax.scan(
        step, (start, optimizer.init(start)), None, length=steps
    )
    return z if metric is None else metric.to_z(z)
