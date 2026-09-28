"""Grid-based fitting and contrast-limit utilities.

Grids are built with ``indexing="ij"``: every output has one axis per grid
key, in the order of ``samples_dict``. For ``{"dra", "ddec", ...}`` axis 0 is
``dra`` (East offset) and axis 1 is ``ddec`` (North offset), so 2D maps need
a transpose to be shown as images with North up; the functions in
[`drpangloss.plotting`][drpangloss.plotting] handle this.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optimistix as optx

from ._utils import renamed_argument
from .models import (
    build_model,
    laplace_parameter_uncertainty,
    loglike,
    nsigma,
)


def _unambiguous_flux_key(samples_dict, params):
    """The flux key if its name alone identifies it, else ``None``."""
    if "flux" in samples_dict:
        return "flux"
    path_fluxes = [key for key in params if key.endswith(".flux")]
    if len(path_fluxes) == 1:
        return path_fluxes[0]
    return None


def _infer_flux_key(samples_dict, params):
    by_name = _unambiguous_flux_key(samples_dict, params)
    if by_name is not None:
        return by_name
    path_fluxes = [key for key in params if key.endswith(".flux")]
    if len(path_fluxes) > 1 and params[-1] not in path_fluxes:
        raise ValueError(
            f"Several flux parameters {path_fluxes}; put the one to optimize "
            "last in samples_dict."
        )
    return params[-1]


def _infer_grid_parameter_keys(samples_dict, params=None):
    """Infer coordinate and flux-like keys from a sample grid.

    Supports any number of coordinate parameters (one or more) plus exactly
    one flux-like parameter, e.g. ``(dra, ddec, flux)`` for a binary
    companion search, ``(sigma, flux)`` for a resolved-source search, or
    ``(comp.dra, comp.ddec, comp.flux)`` for a composed [`System`][drpangloss.models.System].
    """
    if params is None:
        params = tuple(samples_dict.keys())
    if len(params) < 2:
        raise ValueError(
            "Grid-based helpers expect at least two parameters: one or "
            "more coordinates and one flux-like parameter."
        )

    flux_key = _infer_flux_key(samples_dict, params)
    coord_keys = [key for key in params if key != flux_key]
    if not coord_keys:
        raise ValueError(
            "Could not infer any coordinate parameters from samples_dict."
        )
    return coord_keys, flux_key


def _check_flux_axes(samples_dict, flux_param=None):
    """Reject grid axes that would give a flux parameter negative values.

    Axes named ``flux`` or ending in ``.flux`` are checked, as is the
    explicitly selected ``flux_param`` whatever its name.
    """
    for key, values in samples_dict.items():
        is_flux = key == "flux" or key.endswith(".flux")
        if not (is_flux or key == flux_param):
            continue
        try:
            negative = np.any(np.asarray(values) < 0.0)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            continue
        if negative:
            raise ValueError(
                f"The grid axis {key!r} contains negative values, but fluxes "
                "must be non-negative."
            )


def _resolve_flux_param(samples_dict, flux_param, caller):
    """Return ``(params, coord_keys, flux_key)`` for a grid-fitting call.

    Without an explicit ``flux_param``, the flux parameter is inferred as
    before: a key named exactly ``flux``, else the only key ending in
    ``.flux``. When neither applies and the choice falls back on the order of
    ``samples_dict``, a ``DeprecationWarning`` is raised.
    """
    params = tuple(samples_dict.keys())
    _check_flux_axes(samples_dict, flux_param)
    if flux_param is None:
        coord_keys, flux_key = _infer_grid_parameter_keys(samples_dict, params)
        if _unambiguous_flux_key(samples_dict, params) is not None:
            return params, tuple(coord_keys), flux_key
        warnings.warn(
            f"{caller}(): samples_dict has no single key named 'flux' or "
            "ending in '.flux', so the flux parameter was taken to be the "
            "last key. Relying on key order is deprecated; pass "
            f"flux_param={flux_key!r} to keep the current behaviour.",
            DeprecationWarning,
            # user -> renamed_argument wrapper -> public function -> here
            stacklevel=4,
        )
        return params, tuple(coord_keys), flux_key
    if flux_param not in samples_dict:
        raise ValueError(
            f"flux_param {flux_param!r} is not a key of samples_dict "
            f"({list(params)})."
        )
    coord_keys = tuple(key for key in params if key != flux_param)
    if not coord_keys:
        raise ValueError(
            "samples_dict needs at least one coordinate parameter besides "
            f"flux_param {flux_param!r}."
        )
    return params, coord_keys, flux_param


# Grid points are evaluated in batches of this many, which bounds memory on
# large grids (each point holds a full model evaluation). It is read when a
# function is first compiled for a given grid shape.
GRID_BATCH_SIZE = 4096


def _map_points(fn, *xs):
    """Apply ``fn`` to every row of ``xs``, ``GRID_BATCH_SIZE`` rows at a time."""
    return jax.lax.map(lambda args: fn(*args), xs, batch_size=GRID_BATCH_SIZE)


def _meshgrid_vectors(samples_dict, params):
    """Build flattened meshgrid vectors with axis order matching ``params``."""
    samples = [jnp.asarray(samples_dict[param]) for param in params]
    grid_shape = tuple(sample.shape[0] for sample in samples)
    grids = jnp.meshgrid(*samples, indexing="ij")
    vals_vec = jnp.stack([grid.reshape(-1) for grid in grids], axis=1)
    return vals_vec, grid_shape


def _ordered_values_from_flux_and_coords(
    flux, coord_vals, params, coord_keys, flux_key
):
    """Build parameter values in ``params`` order without traced dict objects."""
    flux_value = jnp.asarray(flux).reshape(-1)[0]
    coord_vals = jnp.asarray(coord_vals)
    return [
        flux_value
        if param == flux_key
        else coord_vals[coord_keys.index(param)]
        for param in params
    ]


def _coordinate_points(samples_dict, coord_keys):
    """Flattened ``(n_points, n_coords)`` coordinate grid and its shape.

    The grid uses ``indexing="ij"``: axis ``k`` follows ``coord_keys[k]``,
    so for ``(dra, ddec)`` axis 0 is ``dra``.
    """
    coord_grids = jnp.meshgrid(
        *[jnp.asarray(samples_dict[key]) for key in coord_keys],
        indexing="ij",
    )
    points = jnp.stack([grid.reshape(-1) for grid in coord_grids], axis=1)
    return points, coord_grids[0].shape


def _best_grid_flux(data_obj, model, samples_dict, params, flux_key):
    """Best flux on the grid, and its log likelihood, at every position."""
    vals_vec, grid_shape = _meshgrid_vectors(samples_dict, params)
    loglike_im = _map_points(
        lambda values: loglike(values, params, data_obj, model), vals_vec
    ).reshape(grid_shape)
    flux_axis = params.index(flux_key)
    best_index = jnp.nanargmax(loglike_im, axis=flux_axis)
    best_flux = jnp.asarray(samples_dict[flux_key])[best_index]
    return best_flux, jnp.nanmax(loglike_im, axis=flux_axis)


@eqx.filter_jit
def _optimize_flux_grid(
    data_obj, model, samples_dict, params, coord_keys, flux_key
):
    """Refine the best grid flux at every position with BFGS.

    Returns ``(flux, loglike, converged)``, each with one axis per
    coordinate key; a point has converged when it is within a quarter sigma
    of the likelihood maximum along the flux. The optimizer works in units of the starting flux, and on the log
    likelihood relative to its starting value, so its default tolerances are
    relative to the problem's own scale.
    """
    start_flux, start_loglike = _best_grid_flux(
        data_obj, model, samples_dict, params, flux_key
    )
    coords, shape = _coordinate_points(samples_dict, coord_keys)

    def objective(x, coord_vals, scale, loglike0):
        values = _ordered_values_from_flux_and_coords(
            x * scale, coord_vals, params, coord_keys, flux_key
        )
        return loglike0 - loglike(values, params, data_obj, model)

    def flux_loglike(flux, coord_vals):
        values = _ordered_values_from_flux_and_coords(
            flux, coord_vals, params, coord_keys, flux_key
        )
        return loglike(values, params, data_obj, model)

    def newton_step(flux, coord_vals):
        grad = jax.grad(flux_loglike)(flux, coord_vals)
        curvature = jax.grad(jax.grad(flux_loglike))(flux, coord_vals)
        step = jnp.where(curvature < 0.0, grad / curvature, 0.0)
        # Keep the step only if it improves the fit: far from quadratic
        # regions a Newton step can overshoot.
        trial = flux - step
        improved = flux_loglike(trial, coord_vals) >= flux_loglike(
            flux, coord_vals
        )
        return jnp.where(improved, trial, flux)

    def refine(flux0, coord_vals, loglike0):
        scale = jnp.where(jnp.abs(flux0) > 0.0, jnp.abs(flux0), 1.0)
        result = optx.compat.minimize(
            objective,
            x0=jnp.array([flux0 / scale]),
            args=(coord_vals, scale, loglike0),
            method="BFGS",
            options={"maxiter": 100},
        )
        # BFGS stops once the change in log likelihood is below its
        # tolerance, which in float32 is comparable to rounding noise. Two
        # Newton steps on the analytic gradient pin down the maximum.
        flux = result.x[0] * scale
        for _ in range(2):
            flux = newton_step(flux, coord_vals)
        # Converged if the remaining distance to the maximum, estimated from
        # the gradient and curvature, is under a quarter sigma of the flux.
        # (Float32 rounding alone leaves offsets of up to ~0.1 sigma.)
        grad = jax.grad(flux_loglike)(flux, coord_vals)
        curvature = jax.grad(jax.grad(flux_loglike))(flux, coord_vals)
        converged = (curvature < 0.0) & (
            jnp.abs(grad) < 0.25 * jnp.sqrt(jnp.abs(curvature))
        )
        return flux, flux_loglike(flux, coord_vals), converged

    flux, best_loglike, success = _map_points(
        refine, start_flux.reshape(-1), coords, start_loglike.reshape(-1)
    )
    return (
        flux.reshape(shape),
        best_loglike.reshape(shape),
        success.reshape(shape),
    )


def _warn_unconverged(success, caller):
    """Warn if an optimizer failed to converge at some grid positions."""
    failed = int(np.sum(~np.asarray(success, dtype=bool)))
    if failed:
        warnings.warn(
            f"{caller}(): the optimizer did not converge at {failed} of "
            f"{np.size(success)} grid positions; values there may be "
            "inaccurate.",
            RuntimeWarning,
            # user -> renamed_argument wrapper -> public function -> here
            stacklevel=4,
        )


@renamed_argument("model_class", "model")
def likelihood_grid(data_obj, model, samples_dict):
    """Evaluate the log likelihood at every point of a parameter grid.

    Parameters
    ----------
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model whose parameters at the paths in ``samples_dict`` are
        varied (e.g. a [System][drpangloss.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``samples_dict``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``).
    samples_dict : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds and ``flux`` as a
        companion/primary flux ratio). The output has one axis per key, in
        this order.

    Returns
    -------
    array-like
        Log likelihood with shape
        ``tuple(len(v) for v in samples_dict.values())``. Axis ``k`` follows
        the ``k``-th key (``indexing="ij"``), so for
        ``{"dra", "ddec", "flux"}`` axis 0 is ``dra``: transpose a 2D slice
        before showing it as an image with North up.
    """
    params = tuple(samples_dict.keys())
    _check_flux_axes(samples_dict)
    return _likelihood_grid(data_obj, model, samples_dict, params=params)


@eqx.filter_jit
def _likelihood_grid(data_obj, model, samples_dict, params):
    """Jitted implementation of [`likelihood_grid`][drpangloss.grid_fit.likelihood_grid]."""

    vals_vec, grid_shape = _meshgrid_vectors(samples_dict, params)

    return _map_points(
        lambda values: loglike(values, params, data_obj, model), vals_vec
    ).reshape(grid_shape)


_OPTIMIZED_PARAMS_DOC = """
    Parameters
    ----------
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model whose parameters at the paths in ``samples_dict`` are
        varied (e.g. a [System][drpangloss.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``samples_dict``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``).
    samples_dict : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds and ``flux`` as a
        companion/primary flux ratio). The output has one axis per
        coordinate key (every key except ``flux_param``), in this order;
        the flux axis only sets the optimizer's starting points.
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux (or other brightness)
        parameter that is optimized at each grid position, e.g. ``"flux"`` or
        ``"comp.flux"``. The remaining keys are the grid coordinates. Leaving
        it out uses a key named ``flux`` or the only key ending in
        ``.flux``; falling back on the order of the keys is deprecated.
"""


@renamed_argument("model_class", "model")
def optimized_likelihood_grid(data_obj, model, samples_dict, flux_param=None):
    params, coord_keys, flux_key = _resolve_flux_param(
        samples_dict, flux_param, "optimized_likelihood_grid"
    )
    _, best_loglike, success = _optimize_flux_grid(
        data_obj,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
    )
    _warn_unconverged(success, "optimized_likelihood_grid")
    return best_loglike


optimized_likelihood_grid.__doc__ = (
    """Find the maximum log likelihood over flux at every grid position.

    A grid search over ``flux_param`` gives the starting point, which BFGS
    then refines with the coordinates held fixed. A ``RuntimeWarning`` is
    raised if the optimizer fails to converge anywhere.
"""
    + _OPTIMIZED_PARAMS_DOC
    + """
    Returns
    -------
    array-like
        Log likelihood at the optimized flux, with one axis per coordinate
        key (axis 0 is the first coordinate key, e.g. ``dra``).
    """
)


@renamed_argument("model_class", "model")
def optimized_contrast_grid(data_obj, model, samples_dict, flux_param=None):
    params, coord_keys, flux_key = _resolve_flux_param(
        samples_dict, flux_param, "optimized_contrast_grid"
    )
    best_flux, _, success = _optimize_flux_grid(
        data_obj,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
    )
    _warn_unconverged(success, "optimized_contrast_grid")
    return best_flux


optimized_contrast_grid.__doc__ = (
    """Find the best-fit flux at every grid position.

    A grid search over ``flux_param`` gives the starting point, which BFGS
    then refines with the coordinates held fixed. The flux is not
    constrained to be positive, as [`ruffio_upperlimit`][drpangloss.grid_fit.ruffio_upperlimit] expects. A
    ``RuntimeWarning`` is raised if the optimizer fails to converge
    anywhere.
"""
    + _OPTIMIZED_PARAMS_DOC
    + """
    Returns
    -------
    array-like
        Best-fit value of ``flux_param``, with one axis per coordinate key
        (axis 0 is the first coordinate key, e.g. ``dra``).
    """
)


@renamed_argument("model_class", "model")
def laplace_contrast_uncertainty_grid(
    best_contrast_indices,
    data_obj,
    model,
    samples_dict,
    flux_param=None,
    flux_values=None,
):
    """Laplace uncertainty of the flux at every grid position.

    At each position the coordinates are held fixed and the uncertainty is
    the inverse square root of the curvature of the negative log likelihood
    along ``flux_param``.

    Parameters
    ----------
    best_contrast_indices : array-like of int or None
        Index into ``samples_dict[flux_param]`` of the flux at which to
        evaluate the curvature at each position, e.g.
        ``jnp.argmax(loglike, axis=flux_axis)`` for the output of
        [`likelihood_grid`][drpangloss.grid_fit.likelihood_grid]. Ignored when ``flux_values`` is given.
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model whose parameters at the paths in ``samples_dict`` are
        varied, or a model class, as for [`likelihood_grid`][drpangloss.grid_fit.likelihood_grid].
    samples_dict : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values.
        The output has one axis per coordinate key (every key except
        ``flux_param``), in this order.
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux parameter. Leaving it
        out uses a key named ``flux`` or the only key ending in ``.flux``;
        falling back on the order of the keys is deprecated.
    flux_values : array-like, optional
        Flux at which to evaluate the curvature at each position, with one
        axis per coordinate key. Pass the output of
        [`optimized_contrast_grid`][drpangloss.grid_fit.optimized_contrast_grid] so that the uncertainty is evaluated
        at the same best fit that [`ruffio_upperlimit`][drpangloss.grid_fit.ruffio_upperlimit] uses as the mean.

    Returns
    -------
    array-like
        One-sigma flux uncertainty, with one axis per coordinate key. It is
        NaN where the curvature is not positive (the flux is not at a
        likelihood maximum).
    """
    params, coord_keys, flux_key = _resolve_flux_param(
        samples_dict, flux_param, "laplace_contrast_uncertainty_grid"
    )
    if flux_values is None:
        if best_contrast_indices is None:
            raise ValueError(
                "Pass best_contrast_indices or flux_values to choose where "
                "the curvature is evaluated."
            )
        flux_values = jnp.asarray(samples_dict[flux_key])[
            jnp.asarray(best_contrast_indices)
        ]
    return _laplace_contrast_uncertainty_grid(
        jnp.asarray(flux_values),
        data_obj,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
    )


@eqx.filter_jit
def _laplace_contrast_uncertainty_grid(
    flux_values,
    data_obj,
    model,
    samples_dict,
    params,
    coord_keys,
    flux_key,
):
    """Jitted implementation of [`laplace_contrast_uncertainty_grid`][drpangloss.grid_fit.laplace_contrast_uncertainty_grid]."""
    coords, shape = _coordinate_points(samples_dict, coord_keys)

    def sigma(flux, coord_vals):
        values = jnp.stack(
            _ordered_values_from_flux_and_coords(
                flux, coord_vals, params, coord_keys, flux_key
            )
        )
        return laplace_parameter_uncertainty(
            values=values,
            params=params,
            data_obj=data_obj,
            model=model,
            target_param=flux_key,
        )

    return _map_points(sigma, flux_values.reshape(-1), coords).reshape(shape)


def best_grid_point(loglike_grid, samples_dict):
    """Return the grid point with the highest log likelihood.

    Parameters
    ----------
    loglike_grid : array-like
        Output of [`likelihood_grid`][drpangloss.grid_fit.likelihood_grid] for ``samples_dict``, with one axis
        per key. NaNs are ignored.
    samples_dict : dict[str, array-like]
        The grid axes used to compute ``loglike_grid``.

    Returns
    -------
    dict[str, float]
        ``{name: value}`` at the maximum, in the order of ``samples_dict``.
    """
    shape = jnp.shape(loglike_grid)
    if len(shape) != len(samples_dict):
        raise ValueError(
            f"loglike_grid has {len(shape)} axes but samples_dict has "
            f"{len(samples_dict)} keys; pass the full likelihood_grid output."
        )
    index = np.unravel_index(int(jnp.nanargmax(loglike_grid)), shape)
    return {
        key: float(values[i])
        for (key, values), i in zip(samples_dict.items(), index)
    }


def ruffio_upperlimit(mean, sigma, percentile):
    """
    Percentile of a flux posterior truncated to non-negative values.

    Following Ruffio et al. (2018, eqn 8), the flux posterior at each
    position is the Laplace Gaussian ``N(mean, sigma)`` of the
    unconstrained (possibly negative) best-fit flux, truncated to
    ``flux >= 0`` by the positivity prior. This returns its ``percentile``
    quantile, so ``percentile = norm.cdf(2)`` gives a 2-sigma-equivalent
    upper limit and ``0.16, 0.5, 0.84`` give a median and credible interval.

    Parameters
    ----------
    mean : float or array-like
        Unconstrained best-fit flux, e.g. from [`optimized_contrast_grid`][drpangloss.grid_fit.optimized_contrast_grid].
        It may be negative.
    sigma : float or array-like
        Laplace uncertainty of the flux, e.g. from
        [`laplace_contrast_uncertainty_grid`][drpangloss.grid_fit.laplace_contrast_uncertainty_grid], broadcastable to ``mean``.
    percentile : float or array-like
        Quantile(s) of the truncated posterior to return, between 0 and 1.

    Returns
    -------
    array-like
        Non-negative flux at each percentile, with shape
        ``broadcast(mean, sigma).shape + percentile.shape``.

    Notes
    -----
    The quantile is computed from the upper tail,
    ``mean + sigma * z`` with ``Q(z) = (1 - percentile) Q(-mean / sigma)``
    and ``Q`` the standard normal survival function, evaluated in log space
    so that it stays accurate when the best fit is many sigma below zero.
    Where the tail underflows, the large-deviation limit
    ``z = sqrt(a**2 - 2 log(1 - percentile))`` (``a = -mean / sigma``) is the
    starting point, and two Newton steps on ``log Q(z)`` polish the result.
    """
    mean, sigma = jnp.broadcast_arrays(jnp.asarray(mean), jnp.asarray(sigma))
    percentile = jnp.asarray(percentile)
    expand = (...,) + (None,) * percentile.ndim
    mean, sigma = mean[expand], sigma[expand]

    a = -mean / sigma
    log_tail = jnp.log1p(-percentile) + jsp.special.log_ndtr(-a)
    z_tail = -jsp.special.ndtri(jnp.exp(log_tail))
    z_asymptotic = jnp.sqrt(a**2 - 2.0 * jnp.log1p(-percentile))
    z = jnp.where(jnp.isfinite(z_tail), z_tail, z_asymptotic)

    # Newton steps on log Q(z) = log_tail polish either starting point.
    def newton(z, _):
        log_q = jsp.special.log_ndtr(-z)
        log_pdf = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi)
        return z + (log_q - log_tail) * jnp.exp(log_q - log_pdf), None

    z, _ = jax.lax.scan(newton, z, None, length=2)
    return jnp.maximum(mean + sigma * z, 0.0)


def azimuthalAverage(
    image,
    center=None,
    stddev=False,
    returnradii=False,
    return_nr=False,
    binsize=0.5,
    weights=None,
    steps=False,
    interpnan=False,
    left=None,
    right=None,
    return_max=False,
):
    """
    Calculate an azimuthally averaged radial profile for a 2D image.

    Parameters
    ----------
    image : array-like
        Two-dimensional image.
    center : tuple[float, float], optional
        Pixel coordinates ``(x, y)`` of the radial center. If omitted, the
        geometric image center is used.
    stddev : bool, optional
        If ``True``, return the azimuthal standard deviation instead of the
        weighted mean.
    returnradii : bool, optional
        If ``True``, return ``(radii, profile)``.
    return_nr : bool, optional
        If ``True``, return ``(n_per_bin, radii, profile)``.
    binsize : float, optional
        Radial bin width in pixel units.
    weights : array-like, optional
        Per-pixel weights. Must match ``image.shape``.
    steps : bool, optional
        If ``True``, return step-ready ``(x, y)`` arrays.
    interpnan : bool, optional
        If ``True``, interpolate over bins with ``NaN`` profile values.
    left : float, optional
        Left extrapolation value passed to ``numpy.interp`` when
        ``interpnan=True``.
    right : float, optional
        Right extrapolation value passed to ``numpy.interp`` when
        ``interpnan=True``.
    return_max : bool, optional
        If ``True``, return the maximum of ``image * weights`` per radial
        bin.

    Returns
    -------
    array-like or tuple
        Radial profile array, or a tuple depending on ``returnradii``,
        ``return_nr``, or ``steps``.

    Notes
    -----
    Empty bins are returned as ``NaN`` unless interpolated. Radii and
    ``binsize`` are in *pixels*: for a ``(dra, ddec)`` grid, multiply the
    radii by the grid spacing to get milliarcseconds, and pass ``center`` if
    the grid is not centred on the origin. The image's first axis is taken as
    ``y``.

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array(
            [(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0]
        )

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flatten(), bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin, minlength=nbins + 1)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape

    if stddev:
        radial_prof = np.array(
            [image.flatten()[whichbin == b].std() for b in range(1, nbins + 1)]
        )
    elif return_max:
        radial_prof = np.array(
            [
                np.append(
                    (image * weights).flatten()[whichbin == b], -np.inf
                ).max()
                for b in range(1, nbins + 1)
            ]
        )
    else:
        radial_prof = np.array(
            [
                (image * weights).flatten()[whichbin == b].sum()
                / weights.flatten()[whichbin == b].sum()
                for b in range(1, nbins + 1)
            ]
        )

    if interpnan:
        radial_prof = np.interp(
            bin_centers,
            bin_centers[radial_prof == radial_prof],
            radial_prof[radial_prof == radial_prof],
            left=left,
            right=right,
        )

    if steps:
        xarr = np.column_stack([bins[:-1], bins[1:]]).ravel()
        yarr = np.repeat(radial_prof, 2)
        return xarr, yarr
    elif returnradii:
        return bin_centers, radial_prof
    elif return_nr:
        return nr, bin_centers, radial_prof
    else:
        return radial_prof


@renamed_argument("model_class", "model")
def absil_limits(
    samples_dict,
    data_obj,
    model,
    sigma,
    flux_param=None,
    flux_bounds=(1e-6, 1.0),
):
    """Flux above which a companion is ruled out at ``sigma`` significance.

    Following Absil et al. (2011), at each grid position this finds the flux
    at which the model fits the data worse than the no-companion model by a
    chi-squared ratio corresponding to ``sigma`` (see
    [nsigma][drpangloss.models.nsigma]). Brighter companions at that
    position are excluded at ``sigma``: with ``sigma=3`` the result is a
    3-sigma upper limit on the flux.

    Parameters
    ----------
    samples_dict : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds). The flux axis is only
        used for the starting guess, and must contain at least one positive
        value.
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model or model class, as for [`likelihood_grid`][drpangloss.grid_fit.likelihood_grid]. The
        no-companion model sets every parameter in ``samples_dict`` to zero.
    sigma : float
        Exclusion significance. It must exceed the significance of a
        chi-squared ratio of 1 (about 0.67 for many degrees of freedom).
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux (or other brightness)
        parameter that is optimized at each grid position, e.g. ``"flux"`` or
        ``"comp.flux"``. The remaining keys are the grid coordinates. Leaving
        it out uses a key named ``flux`` or the only key ending in
        ``.flux``; falling back on the order of the keys is deprecated.
    flux_bounds : tuple[float, float] or None, optional
        Limits are clipped to this range (default ``(1e-6, 1.0)``), and a
        ``RuntimeWarning`` reports how many were clipped. Pass ``None`` to
        return them unclipped, e.g. for [`System`][drpangloss.models.System]
        weights that may exceed 1.

    Returns
    -------
    array-like
        Flux limit, with one axis per coordinate key.

    Notes
    -----
    The number of degrees of freedom is the number of data points; the
    fitted parameters are not subtracted.
    """
    params, coord_keys, flux_key = _resolve_flux_param(
        samples_dict, flux_param, "absil_limits"
    )
    if not np.any(np.asarray(samples_dict[flux_key]) > 0.0):
        raise ValueError(
            f"The flux axis {flux_key!r} needs at least one positive value "
            "to start the log-flux optimizer from."
        )
    ndof = int(np.asarray(data_obj.flatten_data()[0]).size)
    floor = float(nsigma(1.0, 1.0, ndof))
    if not float(sigma) > floor:
        raise ValueError(
            f"sigma={sigma} cannot be reached: with {ndof} degrees of "
            f"freedom a chi-squared ratio of 1 is already {floor:.3g} sigma."
        )
    limits, success = _absil_limits(
        samples_dict,
        data_obj,
        model,
        jnp.asarray(sigma, dtype=float),
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
    )
    _warn_unconverged(success, "absil_limits")
    if flux_bounds is None:
        return limits
    low, high = flux_bounds
    clipped = int(
        np.sum((np.asarray(limits) < low) | (np.asarray(limits) > high))
    )
    if clipped:
        warnings.warn(
            f"absil_limits(): {clipped} limits fell outside flux_bounds="
            f"{tuple(flux_bounds)} and were clipped; pass flux_bounds=None "
            "to keep them.",
            RuntimeWarning,
            stacklevel=3,
        )
    return jnp.clip(limits, low, high)


@eqx.filter_jit
def _absil_limits(
    samples_dict,
    data_obj,
    model,
    sigma,
    params,
    coord_keys,
    flux_key,
):
    """Jitted implementation of [`absil_limits`][drpangloss.grid_fit.absil_limits].

    Returns the unclipped limits and whether each reaches ``sigma``.
    """
    data, errors = data_obj.flatten_data()
    ndof = data.size

    def reduced_chi2(values):
        prediction = data_obj.model(build_model(model, params, values))
        residuals = data_obj.residuals(prediction)
        return jnp.sum((residuals / errors) ** 2) / ndof

    null_values = [0.0] * len(params)
    chi2_null = reduced_chi2(null_values)

    def loss(values):
        significance = nsigma(reduced_chi2(values), chi2_null, ndof)
        return (significance - sigma) ** 2

    vals_vec, grid_shape = _meshgrid_vectors(samples_dict, params)
    loss_grid = _map_points(loss, vals_vec).reshape(grid_shape)
    flux_axis = params.index(flux_key)
    best_flux_indices = jnp.nanargmin(loss_grid, axis=flux_axis)

    coords, shape = _coordinate_points(samples_dict, coord_keys)
    # A zero flux would start the log-flux optimizer at -inf; start from the
    # smallest positive flux on the grid instead.
    flux_axis_vals = jnp.asarray(samples_dict[flux_key])
    smallest_positive = jnp.min(
        jnp.where(flux_axis_vals > 0.0, flux_axis_vals, jnp.inf)
    )
    start_flux = flux_axis_vals[best_flux_indices].reshape(-1)
    start_flux = jnp.where(start_flux > 0.0, start_flux, smallest_positive)

    def optimize_log_flux(log_flux, coord_vals):
        flux = 10.0 ** jnp.asarray(log_flux).reshape(-1)[0]
        values = _ordered_values_from_flux_and_coords(
            flux, coord_vals, params, coord_keys, flux_key
        )
        return loss(values)

    def best_flux(flux0, coord_vals):
        solution = optx.compat.minimize(
            optimize_log_flux,
            x0=jnp.array([jnp.log10(flux0)]),
            args=(coord_vals,),
            method="BFGS",
            options={"maxiter": 100},
        )
        limit = 10.0 ** solution.x[0]
        values = _ordered_values_from_flux_and_coords(
            limit, coord_vals, params, coord_keys, flux_key
        )
        # Converged if the target significance is reached to 0.01 sigma.
        reached = jnp.abs(
            nsigma(reduced_chi2(values), chi2_null, ndof) - sigma
        )
        return limit, reached < 1e-2

    limits, success = _map_points(best_flux, start_flux, coords)
    return limits.reshape(shape), success.reshape(shape)
