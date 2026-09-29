"""Gaussian likelihoods of source models given interferometric data.

A model is given either as a template [`SourceModel`][drpangloss.models.SourceModel],
whose parameters at dot-separated zodiax paths (e.g. ``"comp.flux"``) are
replaced, or as a class/callable called with the parameters as keyword
arguments (see [`build_model`][drpangloss.likelihood.build_model]). Phase
residuals are wrapped into ``[-π, π)`` by
[`OIData.residuals`][drpangloss.oidata.OIData.residuals].
"""

import jax
import jax.numpy as np
import numpy as onp

from ._utils import concrete, is_flux_param
from .models import SourceModel


def _gaussian_loglike(residuals, errors):
    """Sum of independent Gaussian log densities of ``residuals``."""
    return jax.scipy.stats.norm.logpdf(residuals, loc=0.0, scale=errors).sum()


def inflated_errors(data_obj, prediction, vis_error_rel=None, phi_error=None):
    """The data uncertainties with extra error terms added in quadrature.

    Parameters
    ----------
    data_obj : OIData
        Data whose uncertainties are inflated.
    prediction : array-like
        Model vector, e.g. from [`OIData.model`][drpangloss.oidata.OIData.model].
    vis_error_rel : float, optional
        Extra visibility error, as a fraction of the *model* visibility
        observable (e.g. of the model V² for squared visibilities).
    phi_error : float, optional
        Extra phase error in radians.

    Returns
    -------
    array-like
        Uncertainties matching [`flatten_data`][drpangloss.oidata.OIData.flatten_data].
    """
    _, errors = data_obj.flatten_data()
    if vis_error_rel is None and phi_error is None:
        return errors
    if data_obj.vis_mat is not None or data_obj.phi_mat is not None:
        raise ValueError(
            "Extra error terms are defined for the observed visibilities and "
            "phases, not for projected (vis_mat/phi_mat) observables."
        )
    n_vis = np.asarray(data_obj.vis).size
    d_vis, d_phi = errors[:n_vis], errors[n_vis:]
    if vis_error_rel is not None:
        d_vis = np.hypot(d_vis, vis_error_rel * np.asarray(prediction)[:n_vis])
    if phi_error is not None:
        d_phi = np.hypot(d_phi, phi_error)
    return np.concatenate([d_vis, d_phi])


def model_loglike(
    model_object,
    data_obj,
    *,
    vis_error_rel=None,
    phi_error=None,
    reject_unphysical=False,
):
    """Evaluate a Gaussian log likelihood for an instantiated model object.

    Phase residuals are wrapped into ``[-π, π)`` (see
    [`residuals`][drpangloss.oidata.OIData.residuals]).

    Parameters
    ----------
    model_object : SourceModel
        Model to evaluate.
    data_obj : OIData
        Data to compare with.
    vis_error_rel, phi_error : float, optional
        Extra error terms added in quadrature to the data uncertainties,
        e.g. fitted as nuisance parameters: a visibility error relative to
        the model visibility, and a phase error in radians (see
        [`inflated_errors`][drpangloss.likelihood.inflated_errors]). The
        Gaussian normalization uses the inflated errors.
    reject_unphysical : bool, optional
        If True, return ``-inf`` when
        [`is_physical`][drpangloss.models.SourceModel.is_physical] is false,
        e.g. for a negative flux or a rim whose brightness goes negative.
        This works inside ``jax.jit``, so samplers can use it as a hard prior
        boundary.
    """
    prediction = data_obj.model(model_object)
    errors = inflated_errors(data_obj, prediction, vis_error_rel, phi_error)
    logl = _gaussian_loglike(data_obj.residuals(prediction), errors)
    if reject_unphysical:
        logl = np.where(model_object.is_physical(), logl, -np.inf)
    return logl


def joint_prediction(params, observations, model_fn):
    """Concatenate predictions for a parameter pytree and multiple observations.

    ``model_fn(params, index)`` defines which parameters are shared and which
    are specific to each observation.
    """
    return np.concatenate(
        [
            observation.model(model_fn(params, index))
            for index, observation in enumerate(observations)
        ]
    )


def joint_data(observations):
    """Concatenate observed vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.flatten_data()[0] for observation in observations]
    )


def joint_errors(observations):
    """Concatenate uncertainty vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.flatten_data()[1] for observation in observations]
    )


def joint_loglike(params, observations, model_fn, **options):
    """Sum independent Gaussian log likelihoods over multiple observations.

    ``options`` (``vis_error_rel``, ``phi_error``, ``reject_unphysical``) are
    passed to [`model_loglike`][drpangloss.likelihood.model_loglike].
    """
    return sum(
        model_loglike(model_fn(params, index), observation, **options)
        for index, observation in enumerate(observations)
    )


def build_model(model, params, values):
    """Build a model from parameter names and values.

    ``model`` is either a class/callable, called as ``model(**dict(zip(params,
    values)))``, or a [`SourceModel`][drpangloss.models.SourceModel] instance used as a template whose
    leaves at the (dot-separated) paths ``params`` are replaced by ``values``.
    """
    if isinstance(model, SourceModel):
        return model.set(list(params), list(values))
    return model(**dict(zip(params, values)))


def loglike(values, params, data_obj, model, **options):
    """
    Gaussian log-likelihood of a model with the given parameter values, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][drpangloss.likelihood.build_model]).
    **options
        ``vis_error_rel``, ``phi_error`` and ``reject_unphysical``, passed to
        [`model_loglike`][drpangloss.likelihood.model_loglike].

    Returns
    -------
    float
        Log-likelihood value.
    """

    return model_loglike(
        build_model(model, params, values), data_obj, **options
    )


def loglike_nosignal(values, params, data_obj, model):
    """
    Gaussian log-likelihood of the no-signal (unresolved point source) data under a model, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][drpangloss.likelihood.build_model]).

    Returns
    -------
    float
        Log-likelihood value.
    """

    model_data = data_obj.model(build_model(model, params, values))
    _, errors = data_obj.flatten_data()
    unity_cvis = np.ones_like(data_obj.u, dtype=complex)
    null_data = data_obj.standardize_model(unity_cvis)

    return _gaussian_loglike(data_obj.residuals(model_data, null_data), errors)


def _check_positive_flux_prior(name, distribution):
    """Reject flux priors whose support includes negative values."""
    from numpyro.distributions import constraints

    support = distribution.support
    lower = getattr(support, "lower_bound", None)
    if lower is None:
        unbounded = support in (constraints.real, constraints.real_vector)
    else:
        value = concrete(lower)
        unbounded = value is not None and bool(onp.any(value < 0.0))
    if unbounded:
        raise ValueError(
            f"The prior on {name!r} allows negative values, but fluxes must "
            "be non-negative. Use a prior with non-negative support, e.g. "
            "dist.LogUniform or dist.Uniform(0, ...)."
        )


def numpyro_model(model, priors, data_obj, **options):
    """Return a numpyro model sampling the parameters in ``priors``.

    Parameters
    ----------
    model : SourceModel or callable
        Either a template model whose leaves at the paths in ``priors`` are
        sampled, or a function called with the sampled values as keyword
        arguments that returns a [`SourceModel`][drpangloss.models.SourceModel]. A function lets you
        sample parameters that are not leaves of the model, such as a
        separation and position angle, or one inclination shared by two
        components (see [`build_model`][drpangloss.likelihood.build_model]).
    priors : dict[str, numpyro.distributions.Distribution]
        Mapping from parameter path (e.g. ``"comp.flux"``) or function
        argument name to prior; each key is also used as the numpyro
        sample-site name. Priors on fluxes (keys named ``flux`` or ending
        in ``.flux``) must have non-negative support.
    data_obj : OIData or sequence of OIData
        Data whose Gaussian log likelihood is added with ``numpyro.factor``.
    **options
        ``vis_error_rel``, ``phi_error`` and ``reject_unphysical``, passed to
        [`model_loglike`][drpangloss.likelihood.model_loglike].

    Returns
    -------
    callable
        Zero-argument numpyro model, e.g. for ``numpyro.infer.NUTS``.
    """
    import numpyro

    paths = list(priors)
    for path in paths:
        if is_flux_param(path):
            _check_positive_flux_prior(path, priors[path])
    observations = (
        tuple(data_obj) if isinstance(data_obj, (list, tuple)) else (data_obj,)
    )

    def numpyro_fn():
        values = [numpyro.sample(path, priors[path]) for path in paths]
        source = build_model(model, paths, values)
        numpyro.factor(
            "loglike",
            sum(model_loglike(source, obs, **options) for obs in observations),
        )

    return numpyro_fn


def posterior_predictive_summary(samples, model, data_obj, params=None):
    """Mean and spread of the model observables over posterior samples.

    Parameters
    ----------
    samples : dict[str, array-like]
        Posterior samples, as equal-length 1D arrays keyed by parameter
        name or path (e.g. from ``mcmc.get_samples()``).
    model : SourceModel or callable
        Template model or class, as for :func:`loglike`.
    data_obj : OIData
        Data defining the observables.
    params : list[str], optional
        Which keys of ``samples`` to use (default: all of them).

    Returns
    -------
    dict
        ``vis_mean``, ``vis_std``, ``phi_mean`` and ``phi_std``: the mean
        and standard deviation over the samples of each visibility and
        phase observable.
    """
    params = list(samples) if params is None else list(params)
    values = np.stack(
        [np.asarray(samples[param], dtype=float) for param in params], axis=1
    )
    predictions = jax.vmap(
        lambda row: data_obj.model(build_model(model, params, list(row)))
    )(values)
    n_vis = np.asarray(data_obj.vis).size
    vis, phi = predictions[:, :n_vis], predictions[:, n_vis:]
    return {
        "vis_mean": vis.mean(axis=0),
        "vis_std": vis.std(axis=0),
        "phi_mean": phi.mean(axis=0),
        "phi_std": phi.std(axis=0),
    }
