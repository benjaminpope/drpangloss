from functools import partial

from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from .models import laplace_parameter_uncertainty, loglike, nsigma

import jax.scipy as jsp

"""Grid-based fitting and contrast-limit utilities."""


def _infer_grid_parameter_keys(samples_dict, params=None):
    """Infer coordinate and flux-like keys from a sample grid.

    Supports any number of coordinate parameters (one or more) plus exactly
    one flux-like parameter, e.g. ``(dra, ddec, flux)`` for a binary
    companion search or ``(sigma, flux)`` for a resolved-source search.
    """
    if params is None:
        params = tuple(samples_dict.keys())
    if len(params) < 2:
        raise ValueError(
            "Grid-based helpers expect at least two parameters: one or "
            "more coordinates and one flux-like parameter."
        )

    flux_key = "flux" if "flux" in samples_dict else params[-1]
    coord_keys = [key for key in params if key != flux_key]
    if not coord_keys:
        raise ValueError(
            "Could not infer any coordinate parameters from samples_dict."
        )
    return coord_keys, flux_key


def _meshgrid_vectors(samples_dict, params):
    """Build flattened meshgrid vectors with axis order matching ``params``."""
    samples = [samples_dict[param] for param in params]
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


def likelihood_grid(data_obj, model_class, samples_dict):
    params = tuple(samples_dict.keys())
    return _likelihood_grid(data_obj, model_class, samples_dict, params=params)


@partial(jit, static_argnames=("model_class", "params"))
def _likelihood_grid(data_obj, model_class, samples_dict, params):
    """
    Function to vmap a likelihood function over a grid of parameter values provided in a dictionary.

    Parameters
    ----------
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    samples_dict: dict
        Dictionary of parameter names and values to be fitted to the data.

    Returns
    -------
    array-like
        Log-likelihood values over the grid of parameter values.
    """

    vals_vec, grid_shape = _meshgrid_vectors(samples_dict, params)

    fn = vmap(lambda values: loglike(values, params, data_obj, model_class))

    return fn(vals_vec).reshape(grid_shape)


def optimized_likelihood_grid(data_obj, model_class, samples_dict):
    params = tuple(samples_dict.keys())
    _, flux_key = _infer_grid_parameter_keys(samples_dict, params)
    coord_keys = tuple(param for param in params if param != flux_key)
    return _optimized_likelihood_grid(
        data_obj,
        model_class,
        samples_dict,
        params=params,
        coord_keys=tuple(coord_keys),
        flux_key=flux_key,
    )


@partial(
    jit,
    static_argnames=(
        "model_class",
        "params",
        "coord_keys",
        "flux_key",
    ),
)
def _optimized_likelihood_grid(
    data_obj, model_class, samples_dict, params, coord_keys, flux_key
):
    """
    Function to optimize the contrast of a model over a grid of parameter values provided in a dictionary.

    Parameters
    ----------
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    samples_dict: dict
        Dictionary of parameter names and values to be fitted to the data.

    Returns
    -------
    array-like
        Optimized contrast values over the grid of parameter values.

    """

    # first do a grid search to find a starting point

    vals_vec, grid_shape = _meshgrid_vectors(samples_dict, params)

    fn = vmap(lambda values: loglike(values, params, data_obj, model_class))

    loglike_im = fn(vals_vec).reshape(grid_shape)
    flux_axis = params.index(flux_key)
    best_contrast_indices = jnp.argmax(loglike_im, axis=flux_axis)
    # then do optimization to fine tune the contrast

    coords = [samples_dict[key] for key in coord_keys]
    coord_grids = jnp.meshgrid(*coords, indexing="ij")
    param_grids = {
        flux_key: samples_dict[flux_key][best_contrast_indices],
        **dict(zip(coord_keys, coord_grids)),
    }
    vals = jnp.array([param_grids[param] for param in params])
    vals_vec = vals.reshape((len(vals), -1)).T
    flux_index = params.index(flux_key)
    coord_indices = tuple(params.index(param) for param in coord_keys)

    def to_optimize(flux, coord_vals):
        ordered_values = _ordered_values_from_flux_and_coords(
            flux, coord_vals, params, coord_keys, flux_key
        )
        return -loglike(ordered_values, params, data_obj, model_class)

    bestcon = lambda flux, *coord_vals: optx.compat.minimize(
        to_optimize,
        x0=jnp.array([flux]),
        args=(jnp.asarray(coord_vals),),
        method="BFGS",
        options={"maxiter": 100},
    ).fun

    fn = vmap(
        lambda values: bestcon(
            values[flux_index], *[values[index] for index in coord_indices]
        )
    )

    return -fn(vals_vec).reshape(vals.shape[1:])


def optimized_contrast_grid(data_obj, model_class, samples_dict):
    params = tuple(samples_dict.keys())
    _, flux_key = _infer_grid_parameter_keys(samples_dict, params)
    coord_keys = tuple(param for param in params if param != flux_key)
    return _optimized_contrast_grid(
        data_obj,
        model_class,
        samples_dict,
        params=params,
        coord_keys=tuple(coord_keys),
        flux_key=flux_key,
    )


@partial(
    jit,
    static_argnames=(
        "model_class",
        "params",
        "coord_keys",
        "flux_key",
    ),
)
def _optimized_contrast_grid(
    data_obj, model_class, samples_dict, params, coord_keys, flux_key
):
    """
    Function to optimize the contrast of a model over a grid of parameter values provided in a dictionary.

    Parameters
    ----------
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    samples_dict: dict
        Dictionary of parameter names and values to be fitted to the data.

    Returns
    -------
    array-like
        Optimized contrast values over the grid of parameter values.

    """

    # first do a grid search to find a starting point

    vals_vec, grid_shape = _meshgrid_vectors(samples_dict, params)

    fn = vmap(lambda values: loglike(values, params, data_obj, model_class))

    loglike_im = fn(vals_vec).reshape(grid_shape)

    flux_axis = params.index(flux_key)
    best_contrast_indices = jnp.argmax(loglike_im, axis=flux_axis)

    # then do optimization to fine tune the contrast

    coords = [samples_dict[key] for key in coord_keys]
    coord_grids = jnp.meshgrid(*coords, indexing="ij")
    param_grids = {
        flux_key: samples_dict[flux_key][best_contrast_indices],
        **dict(zip(coord_keys, coord_grids)),
    }
    vals = jnp.array([param_grids[param] for param in params])
    vals_vec = vals.reshape((len(vals), -1)).T
    flux_index = params.index(flux_key)
    coord_indices = tuple(params.index(param) for param in coord_keys)

    def to_optimize(flux, coord_vals):
        ordered_values = _ordered_values_from_flux_and_coords(
            flux, coord_vals, params, coord_keys, flux_key
        )
        return -loglike(ordered_values, params, data_obj, model_class)

    bestcon = lambda flux, *coord_vals: optx.compat.minimize(
        to_optimize,
        x0=jnp.array([flux]),
        args=(jnp.asarray(coord_vals),),
        method="BFGS",
        options={"maxiter": 100},
    ).x[0]

    fn = vmap(
        lambda values: bestcon(
            values[flux_index], *[values[index] for index in coord_indices]
        )
    )

    return fn(vals_vec).reshape(vals.shape[1:])


def laplace_contrast_uncertainty_grid(
    best_contrast_indices, data_obj, model_class, samples_dict
):
    params = tuple(samples_dict.keys())
    _, flux_key = _infer_grid_parameter_keys(samples_dict, params)
    coord_keys = tuple(param for param in params if param != flux_key)
    return _laplace_contrast_uncertainty_grid(
        best_contrast_indices,
        data_obj,
        model_class,
        samples_dict,
        params=params,
        coord_keys=tuple(coord_keys),
        flux_key=flux_key,
    )


@partial(
    jit,
    static_argnames=(
        "model_class",
        "params",
        "coord_keys",
        "flux_key",
    ),
)
def _laplace_contrast_uncertainty_grid(
    best_contrast_indices,
    data_obj,
    model_class,
    samples_dict,
    params,
    coord_keys,
    flux_key,
):
    """
    Calculate the uncertainty with the Laplace method over a grid of parameters, for an optimized fit between a model and data object.

    Parameters
    ----------
    best_contrast_indices : array-like
        Indices of the best contrast values in a grid calculated with likelihood_grid.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    samples_dict: dict
        Dictionary of parameter names and values to be fitted to the data.

    Returns
    -------
    array-like
        Uncertainty in the contrast.
    """

    coords = [samples_dict[key] for key in coord_keys]
    coord_grids = jnp.meshgrid(*coords, indexing="ij")
    param_grids = {
        flux_key: samples_dict[flux_key][best_contrast_indices],
        **dict(zip(coord_keys, coord_grids)),
    }
    vals = jnp.array([param_grids[param] for param in params])
    vals_vec = vals.reshape((len(vals), -1)).T
    sigma = lambda values: laplace_parameter_uncertainty(
        values=values,
        params=params,
        data_obj=data_obj,
        model_class=model_class,
        target_param=flux_key,
    )
    fn = vmap(lambda values: sigma(values))

    return fn(vals_vec).reshape(vals.shape[1:])


@partial(vmap, in_axes=(0, 0, None))
@partial(vmap, in_axes=(None, None, 0))
def ruffio_upperlimit(mean, sigma, percentile):
    """
    Calculate the upper limit of a distribution given the mean and standard deviation.
    This is a vectorized JAX implementation of Ruffio et al. (2018).

    Parameters
    ----------
    mean : array-like
        Mean of the distribution.
    sigma : array-like
        Standard deviation of the distribution.
    percentile : float
        Percentile value for the upper limit.

    Returns
    -------
    array-like
        Upper limit of the distribution.
    """

    # eqn 8 from Ruffio+2018
    limit = jsp.stats.norm.ppf(
        (
            percentile
            + (1 - percentile) * jsp.stats.norm.cdf(0, loc=mean, scale=sigma)
        ),
        loc=mean,
        scale=sigma,
    )

    return limit


# def get_grid(sep_range,
#              step_size,
#              verbose=False):
#     """
#     Parameters
#     ----------
#     sep_range: tuple of float
#         Min. and max. angular separation of grid (mas).
#     step_size: float
#         Step size of grid (mas).
#     verbose: bool
#         True if feedback shall be printed.

#     Returns
#     -------
#     grid_ra_dec: tuple of array
#         grid_ra_dec[0]: array
#             Right ascension offset of grid cells (mas).
#         grid_ra_dec[1]: array
#             Declination offset of grid cells (mas).
#     grid_sep_pa: tuple of array
#         grid_sep_pa[0]: array
#             Angular separation of grid cells (mas).
#         grid_sep_pa[1]: array
#             Position angle of grid cells (deg).
#     """

#     if (verbose == True):
#         print('Computing grid')

#     nc = int(np.ceil(sep_range[1]/step_size))
#     temp = np.linspace(-nc*step_size, nc*step_size, 2*nc+1)
#     grid_ra_dec = np.meshgrid(temp, temp)
#     grid_ra_dec[0] = np.fliplr(grid_ra_dec[0])
#     sep = np.sqrt(grid_ra_dec[0]**2+grid_ra_dec[1]**2)
#     pa = np.rad2deg(np.arctan2(grid_ra_dec[0], grid_ra_dec[1]))
#     grid_sep_pa = np.array([sep, pa])

#     mask = (sep < sep_range[0]-1e-6) | (sep_range[1]+1e-6 < sep)
#     grid_ra_dec[0][mask] = np.nan
#     grid_ra_dec[1][mask] = np.nan
#     grid_sep_pa[0][mask] = np.nan
#     grid_sep_pa[1][mask] = np.nan

#     if (verbose):
#         print('   Min. sep. = %.1f mas' % np.nanmin(grid_sep_pa[0]))
#         print('   Max. sep. = %.1f mas' % np.nanmax(grid_sep_pa[0]))
#         print('   %.0f non-empty grid cells' % np.sum(np.logical_not(np.isnan(grid_sep_pa[0]))))

#     return grid_ra_dec, grid_sep_pa


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
        If ``True``, return the maximum value per radial bin.

    Returns
    -------
    array-like or tuple
        Radial profile array, or a tuple depending on ``returnradii``,
        ``return_nr``, or ``steps``.

    Notes
    -----
    Empty bins are returned as ``NaN`` unless interpolated.

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
    nr = np.bincount(whichbin)[1:]

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

    # import pdb; pdb.set_trace()

    if interpnan:
        radial_prof = np.interp(
            bin_centers,
            bin_centers[radial_prof == radial_prof],
            radial_prof[radial_prof == radial_prof],
            left=left,
            right=right,
        )

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(radial_prof, radial_prof)).ravel()
        return xarr, yarr
    elif returnradii:
        return bin_centers, radial_prof
    elif return_nr:
        return nr, bin_centers, radial_prof
    else:
        return radial_prof


@partial(jit, static_argnames=("model_class"))
def absil_limits(samples_dict, data_obj, model_class, sigma):
    """

    Using Jax for optimization, calculate the detection limits for a given model class and data object.
    This is by finding the contrast at which the detection significance is equal to the sigma value.
    For example, if we set sigma = 3, we optimize in each coordinate cell to find the contrast
    at which we would be 3 sigma confident that the companion is detected.

    Parameters
    ----------
    samples_dict: dict
        Dictionary of parameter names and values to be fitted to the data, eg dra and ddec grids.
    data_obj: object
        Observational data in the format of an OIData object.
    model_class: class
        Model class to be fitted to the data.
    sigma: float
        Detection significance.


    Returns
    -------
    res: float
        Maximum relative flux of companion.
    """

    data, errors = data_obj.flatten_data()
    ndof = data.size
    params = tuple(samples_dict.keys())
    coord_keys, flux_key = _infer_grid_parameter_keys(samples_dict, params)

    def reduced_chi2(values):
        model = data_obj.model(model_class(**dict(zip(params, values))))
        return jnp.sum(((data - model) / errors) ** 2) / ndof

    null_values = [0.0] * len(params)
    chi2_null = reduced_chi2(null_values)

    def loss(values):
        significance = nsigma(reduced_chi2(values), chi2_null, ndof)
        return (significance - sigma) ** 2

    vals_vec, grid_shape = _meshgrid_vectors(samples_dict, params)
    loss_grid = vmap(loss)(vals_vec).reshape(grid_shape)
    flux_axis = params.index(flux_key)
    best_flux_indices = jnp.argmin(loss_grid, axis=flux_axis)

    coords = [samples_dict[key] for key in coord_keys]
    coord_grids = jnp.meshgrid(*coords, indexing="ij")
    start_flux = samples_dict[flux_key][best_flux_indices]
    starts = jnp.stack(
        [jnp.log10(start_flux), *coord_grids], axis=0
    ).reshape((len(coord_keys) + 1, -1)).T

    def optimize_log_flux(log_flux, coord_vals):
        flux = 10.0 ** jnp.asarray(log_flux).reshape(-1)[0]
        values = _ordered_values_from_flux_and_coords(
            flux, coord_vals, params, coord_keys, flux_key
        )
        return loss(values)

    def best_flux(log_flux, *coord_vals):
        solution = optx.compat.minimize(
            optimize_log_flux,
            x0=jnp.array([log_flux]),
            args=(jnp.asarray(coord_vals),),
            method="BFGS",
            options={"maxiter": 100},
        )
        return 10.0 ** solution.x[0]

    limits = vmap(lambda values: best_flux(values[0], *values[1:]))(starts)
    return jnp.clip(limits.reshape(start_flux.shape), 1e-6, 1.0)


# def nsigma_wrap(planet_contrast, u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2, i_cps3, ddec,dra,xs,ppf_arr,ndof,sigma):

#     #constraints
#     planet_contrast = jnp.where(planet_contrast<1e-6,1e-6,planet_contrast)
#     planet_contrast = jnp.where(planet_contrast>1.,1.,planet_contrast)

#     chi2_s = chi2_binary(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, 0.,0.,0.)/ndof
#     chi2_b = chi2_binary(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra,planet_contrast)/ndof

#     q = jsp.stats.chi2.cdf(ndof*chi2_b/chi2_s, ndof)
#     p = 1.-q

#     nsigma = jnp.sqrt(jnp.interp(p,xs,ppf_arr))

#     nsigma_overflow = jnp.sqrt(jnp.interp(1e-15,xs,ppf_arr))

#     nsigmavar = jnp.where(p<1e-15,nsigma_overflow,nsigma)

#     return (sigma-nsigmavar)**2

# def optimize_nsigma(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra,planet_contrast,xs,ppf_arr,ndof,sigma):
#     '''


#     Parameters
#     ----------
#     oidata: object
#         Observational data, including:
#         - u: array
#             Baselines coordinates.
#         - v: array
#             Baselines coordinates.
#         - cp: array
#             Closure phases.
#         - d_cp: array
#             Closure phase uncertainties.
#         - vis2: array
#             Squared visibilities.
#         - d_vis2: array
#             Squared visibility uncertainties.
#         - i_cps1: array
#             Indices of closure phases for triangle 1.
#         - i_cps2: array
#             Indices of closure phases for triangle 2.
#         - i_cps3: array
#             Indices of closure phases for triangle 3.
#     ddec: float
#         Declination offset of companion (mas).
#     dra: float
#         Right ascension offset of companion (mas).
#     planet_contrast: float
#         Relative flux of companion.
#     xs: array
#         x values of PPF.
#     ppf_arr: array
#         PPF values.
#     ndof: int
#         Number of degrees of freedom.
#     sigma: int
#         Confidence level for which the detection limits shall be computed.

#     Returns
#     -------
#     res: float
#         Maximum relative flux of companion.
#     '''

#     sol = optx.compat.minimize(nsigma_wrap,method='BFGS',
#                                 x0=jnp.array([planet_contrast]),
#                                 args=(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra,xs,ppf_arr,ndof,sigma),options={"maxiter":100})

#     res = sol.x

#     return res


# def nsigma(chi2r_test,
#            chi2r_true,
#            ndof):
#     """
#     Parameters
#     ----------
#     chi2r_test: float
#         Reduced chi-squared of test model.
#     chi2r_true: float
#         Reduced chi-squared of true model.
#     ndof: int
#         Number of degrees of freedom.

#     Returns
#     -------
#     nsigma: float
#         Detection significance.
#     """

#     q = stats.chi2.cdf(ndof*chi2r_test/chi2r_true, ndof)
#     p = 1.-q
#     nsigma = np.sqrt(stats.chi2.ppf(1.-p, 1.))
#     if (p < 1e-15):
#         nsigma = np.sqrt(stats.chi2.ppf(1.-1e-15, 1.))

#     return nsigma


# @jit
# def chi2all(cp_modelr,v2_modelr,oidata,
#            const=0.):

#     cp_obsr, vis2_obsr, cp_errr, vis2_errr = oidata.phi, oidata.vis, oidata.d_phi, oidata.d_vis
#     # chi2

#     chi2_closurer = jnp.sum((cp_obsr - cp_modelr.flatten())**2 / cp_errr**2)

#     chi2_v2r = jnp.sum((vis2_obsr - v2_modelr.flatten())**2 / (vis2_errr**2))

#     return ( chi2_closurer+chi2_v2r) + const

# @jit
# def chi2_suball(oidata,cont,vis_in,imsum,ddec,dra):
#     u21, v21 = oidata.u/oidata.wavel, oidata.v/oidata.wavel
#     i_cps121, i_cps221, i_cps321 = oidata.i_cps1, oidata.i_cps2, oidata.i_cps3
#     cont = 10**cont
#     cvis_t211 = vis_binary2(u21, v21, ddec = ddec,dra=dra,
#                       p2=cont/(1.+cont+imsum),p3=1./(1.+cont+imsum))
#     cvis_t211 += vis_in/(1+cont+imsum)
#     cp_model_t211 = closure_phases(cvis_t211,i_cps121,i_cps221,i_cps321)
#     return chi2all(cp_model_t211,jnp.abs(cvis_t211)**2,oidata)

# def lim_absil(f0,
#               oidata,
#               ddec,
#               dra,
#               chi2_true,
#               ndof,
#               sigma=3):
#     """
#     Parameters
#     ----------
#     f0: float
#         Relative flux of companion.
#     func: method
#         Method to compute chi-squared.
#     p0: array
#         p0[0]: float
#             Relative flux of companion.
#         p0[1]: float
#             Right ascension offset of companion.
#         p0[2]: float
#             Declination offset of companion.
#         p0[3]: float
#             Uniform disk diameter (mas).
#     data_list: list of dict
#         List of data whose chi-squared shall be computed. The list
#         contains one data structure for each observation.
#     observables: list of str
#         List of observables which shall be considered.
#     cov: bool
#         True if covariance shall be considered.
#     smear: int
#         Numerical bandwidth smearing which shall be used.
#     chi2r_true: float
#         Reduced chi-squared of true model.
#     ndof: int
#         Number of degrees of freedom.
#     sigma: int
#         Confidence level for which the detection limits shall be computed.

#     Returns
#     -------
#     chi2: float
#         Chi-squared of Absil method.
#     """

#     chi2_test = chi2_suball(oidata,f0,vis_in=0.,imsum=0.,ddec=ddec,dra=dra)
#     nsigmavar = nsigma(chi2r_test=chi2_test/ndof,
#                          chi2r_true=chi2_true/ndof,
#                          ndof=ndof)

#     return np.abs(nsigmavar-sigma)**2
