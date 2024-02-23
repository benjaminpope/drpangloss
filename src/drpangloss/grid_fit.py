import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

from .models import * 

def get_grid(sep_range,
             step_size,
             verbose=False):
    """
    Parameters
    ----------
    sep_range: tuple of float
        Min. and max. angular separation of grid (mas).
    step_size: float
        Step size of grid (mas).
    verbose: bool
        True if feedback shall be printed.
    
    Returns
    -------
    grid_ra_dec: tuple of array
        grid_ra_dec[0]: array
            Right ascension offset of grid cells (mas).
        grid_ra_dec[1]: array
            Declination offset of grid cells (mas).
    grid_sep_pa: tuple of array
        grid_sep_pa[0]: array
            Angular separation of grid cells (mas).
        grid_sep_pa[1]: array
            Position angle of grid cells (deg).
    """
    
    if (verbose == True):
        print('Computing grid')
    
    nc = int(np.ceil(sep_range[1]/step_size))
    temp = np.linspace(-nc*step_size, nc*step_size, 2*nc+1)
    grid_ra_dec = np.meshgrid(temp, temp)
    grid_ra_dec[0] = np.fliplr(grid_ra_dec[0])
    sep = np.sqrt(grid_ra_dec[0]**2+grid_ra_dec[1]**2)
    pa = np.rad2deg(np.arctan2(grid_ra_dec[0], grid_ra_dec[1]))
    grid_sep_pa = np.array([sep, pa])
    
    mask = (sep < sep_range[0]-1e-6) | (sep_range[1]+1e-6 < sep)
    grid_ra_dec[0][mask] = np.nan
    grid_ra_dec[1][mask] = np.nan
    grid_sep_pa[0][mask] = np.nan
    grid_sep_pa[1][mask] = np.nan
    
    if (verbose):
        print('   Min. sep. = %.1f mas' % np.nanmin(grid_sep_pa[0]))
        print('   Max. sep. = %.1f mas' % np.nanmax(grid_sep_pa[0]))
        print('   %.0f non-empty grid cells' % np.sum(np.logical_not(np.isnan(grid_sep_pa[0]))))
    
    return grid_ra_dec, grid_sep_pa


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None, return_max=False):
    """
    Calculate the azimuthally averaged radial profile.
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
    whichbin = np.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape

    if stddev:
        radial_prof = np.array([image.flat[whichbin==b].std() for b in range(1,nbins+1)])
    elif return_max:
        radial_prof = np.array([np.append((image*weights).flat[whichbin==b],-np.inf).max() for b in range(1,nbins+1)])
    else:
        radial_prof = np.array([(image*weights).flat[whichbin==b].sum() / weights.flat[whichbin==b].sum() for b in range(1,nbins+1)])

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