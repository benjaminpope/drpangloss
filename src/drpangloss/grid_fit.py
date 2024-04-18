import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

from .models import * 

import jax.scipy as jsp
import scipy.stats as stats

'''-------------------------------------------

Functions to fit a model to data and optimize the 
contrast of the model over a grid of parameter values.

-------------------------------------------'''

@partial(jit, static_argnames=("model_class"))
def likelihood_grid(data_obj, model_class, samples_dict):
    ''' 
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
    '''
    
    params = list(samples_dict.keys())
    samples = samples_dict.values()
    vals = np.array(np.meshgrid(*samples))
    vals_vec = vals.reshape((len(vals), -1)).T
    
    fn = vmap(lambda values: loglike(values, params, data_obj,model_class))

    return fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here


@partial(jit, static_argnames=("model_class"))
def optimized_likelihood_grid(data_obj, model_class, samples_dict):
    '''
    Function to optimize the contrast of a model over a grid of parameter values provided in a dictionary.

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
        Optimized contrast values over the grid of parameter values.

    '''
    
    params = list(samples_dict.keys())

    # first do a grid search to find a starting point

    samples = samples_dict.values()
    vals = np.array(np.meshgrid(*samples))
    vals_vec = vals.reshape((len(vals), -1)).T
    
    fn = vmap(lambda values: loglike(values, params, data_obj,model_class))

    loglike_im = fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here

    best_contrast_indices = np.argmax(loglike_im,axis=2)
    
    # then do optimization to fine tune the contrast

    coords = [samples_dict[key] for key in ["dra", "ddec"]]
    ras, decs = np.meshgrid(*coords)
    vals = np.array([samples_dict['flux'][best_contrast_indices].T, ras, decs])
    vals_vec = vals.reshape((len(vals), -1)).T

    to_optimize = lambda flux, dra_inp, ddec_inp: -loglike([dra_inp, ddec_inp, flux], params, data_obj, model_class)
    bestcon = lambda flux, dra, ddec: optx.compat.minimize(to_optimize, x0= np.array([flux]), args=(np.array(dra),np.array(ddec)), 
                            method='BFGS', options={"maxiter":100}).fun

    fn = vmap(lambda values: bestcon(*values))

    return -fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here


@partial(jit, static_argnames=("model_class"))
def optimized_contrast_grid(data_obj, model_class, samples_dict):
    '''
    Function to optimize the contrast of a model over a grid of parameter values provided in a dictionary.

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
        Optimized contrast values over the grid of parameter values.

    '''
    
    params = list(samples_dict.keys())

    # first do a grid search to find a starting point

    samples = samples_dict.values()
    vals = np.array(np.meshgrid(*samples))
    vals_vec = vals.reshape((len(vals), -1)).T
    
    fn = vmap(lambda values: loglike(values, params, data_obj,model_class))

    loglike_im = fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here

    best_contrast_indices = np.argmax(loglike_im,axis=2)
    
    # then do optimization to fine tune the contrast

    coords = [samples_dict[key] for key in ["dra", "ddec"]]
    ras, decs = np.meshgrid(*coords)
    vals = np.array([samples_dict['flux'][best_contrast_indices].T, ras, decs])
    vals_vec = vals.reshape((len(vals), -1)).T

    to_optimize = lambda flux, dra_inp, ddec_inp: -loglike([dra_inp, ddec_inp, flux], params, data_obj, model_class)
    bestcon = lambda flux, dra, ddec: optx.compat.minimize(to_optimize, x0= np.array([flux]), args=(np.array(dra),np.array(ddec)), 
                            method='BFGS', options={"maxiter":100}).x

    fn = vmap(lambda values: bestcon(*values))

    return fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here


@partial(jit, static_argnames=("model_class"))
def laplace_contrast_uncertainty_grid(best_contrast_indices, data_obj, model_class, samples_dict):
    '''
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
    '''

    params = list(samples_dict.keys())
    coords = [samples_dict[key] for key in ["dra", "ddec"]]
    ras, decs = np.meshgrid(*coords)
    vals = np.array([samples_dict['flux'][best_contrast_indices].T, ras, decs])
    vals_vec = vals.reshape((len(vals), -1)).T

    
    sigma = lambda flux, dra, ddec: laplace_contrast_uncertainty(flux, dra, ddec, data_obj, model_class)
    fn = vmap(lambda values: sigma(*values))

    return fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here


@partial(vmap, in_axes=(0,0,None))
@partial(vmap, in_axes=(None,None,0))
def ruffio_upperlimit(mean,sigma,percentile):
    '''
    Calculate the upper limit of a distribution given the mean and standard deviation.
    This is an implementation of the Ruffio method in the old syntax.

    TODO: Update this to the new syntax.

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
    '''


    #eqn 8 from Ruffio+2018
    limit = jsp.stats.norm.ppf((percentile+(1-percentile)*jsp.stats.norm.cdf(0,loc=mean,scale=sigma)),loc=mean,scale=sigma)

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


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None, return_max=False):
    """
    Calculate the azimuthally-averaged radial profile.
    NB: This was found online and should be properly credited! Modified by MJI

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    return_max - (MJI) Return the maximum index.

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flatten(),bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape

    if stddev:
        radial_prof = np.array([image.flatten()[whichbin==b].std() for b in range(1,nbins+1)])
    elif return_max:
        radial_prof = np.array([np.append((image*weights).flatten()[whichbin==b],-np.inf).max() for b in range(1,nbins+1)])
    else:
        radial_prof = np.array([(image*weights).flatten()[whichbin==b].sum() / weights.flatten()[whichbin==b].sum() for b in range(1,nbins+1)])

    #import pdb; pdb.set_trace()

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof
    
@partial(jit, static_argnames=("model_class"))
def absil_limits(samples_dict, data_obj, model_class, sigma):
    '''
    
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
    '''

    ndof = data_obj.vis.size + data_obj.phi.size # number of degrees of freedom

    # unpack the samples dict
    params = list(samples_dict.keys())
    samples = samples_dict.values()
    
    # define chi2 wrappers
    chi2_bin = lambda values: -2*loglike(values, params, data_obj, model_class)/ndof
    chi2_null = chi2_bin([0., 0., 0.])

    # first do a grid search to find the best contrast
    vals = np.array(np.meshgrid(*samples))
    vals_vec = vals.reshape((len(vals), -1)).T

    # define intermediate function: nsigma detection significance, difference from sigma

    loss = lambda values: (nsigma(chi2_bin(values)/ndof, chi2_null/ndof, ndof) - sigma)**2

    loss_im = vmap(loss)(vals_vec).reshape(vals.shape[1:]) # check the shapes output here

    best_contrast_indices = np.argmax(loss_im,axis=2)
     
    # then optimize the contrast at each point
    coords = [samples_dict[key] for key in ["dra", "ddec"]] # TODO: make these arbitrary coordinate labels, eg sep and position angle
    ras, decs = np.meshgrid(*coords)
    vals = np.array([samples_dict['flux'][best_contrast_indices].T, ras, decs])
    vals_vec = vals.reshape((len(vals), -1)).T 

    # define optimization wrapper for the contrast
    to_optimize = lambda flux, dra_inp, ddec_inp: loss([dra_inp, ddec_inp, 10**flux])

    #optimize
    bestcon = lambda flux, dra, ddec: optx.compat.minimize(to_optimize, x0 = np.array([flux]), args=(np.array(dra),np.array(ddec)), 
                            method='BFGS', options={"maxiter":100}).x

    fn = vmap(lambda values: bestcon(*values))
    limits = fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here

    limits_clipped = np.clip(limits, 1e-6, 1) # clip to 1e-6

    return limits_clipped


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
