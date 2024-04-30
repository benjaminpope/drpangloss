import jax.numpy as np
from jax import grad, jit, vmap
import jax 

import numpy as onp 

import optimistix as optx
import equinox as eqx
import zodiax as zx
from functools import partial

from numpyro.distributions.util import gammaincinv

"""------------------------------
------------------------------"""


rad2mas = 180./np.pi*3600.*1000. # convert rad to mas
mas2rad = np.pi/180./3600./1000. # convert mas to rad

dtor = np.pi/180.0
i2pi = 1j*2.0*np.pi

'''--------------------------------------------------
Data class
--------------------------------------------------'''

class OIData(zx.Base):
    ''' 
    Class for storing and manipulating data from OIFITS files, and for interfacing with drpangloss Model objects.

    Attributes:

    - u: jax.Array
        u coordinate of baselines (m)
    - v: jax.Array
        v coordinate of baselines (m)
    - wavels: jax.Array
        wavelengths of observations (m)
    - vis: jax.Array
        visibility data: either squared visibilities or visibilities
    - d_vis: jax.Array
        visibility uncertainties in same units as vis
    - phi: jax.Array
        phase data: either absolute phases or closure phases
    - d_phi: jax.Array
        phase uncertainties in same units as phi
    - i_cps1: jax.Array
        indices for closure phases, or None if absolute phases are available
    - i_cps2: jax.Array
        indices for closure phases, or None if absolute phases are available
    - i_cps3: jax.Array
        indices for closure phases, or None if absolute phases are available
    - v2_flag : bool = eqx.field(static=True)
        flag to indicate whether visibilities are squared or not
    - cp_flag: bool = eqx.field(static=True)
        flag to indicate whether phases are closure phases or absolute phases


    Methods:

    - __init__(self, fname)
        Load data from an OIFITS file and store it in the object.

    - __repr__(self)
        Print the object in a readable format.

    - unpack_all(self)
        Convenience function to unpack all data to be used in model functions.

    - flatten_data(self)
        Flatten closure phases and uncertainties.

    - flatten_model(self,cvis)
        Flatten model visibilities and phases.

    - to_vis(self,cvis)
        Convert complex visibilities to visibilities or squared visibilities.

    - to_phases(self,cvis)
        Convert complex visibilities to closure phases or absolute phases. 
    '''

    u: jax.Array
    v: jax.Array
    wavel: jax.Array
    vis: jax.Array
    d_vis: jax.Array
    phi: jax.Array
    d_phi: jax.Array
    i_cps1: jax.Array
    i_cps2: jax.Array
    i_cps3: jax.Array
    v2_flag : bool = eqx.field(static=True)    
    cp_flag: bool = eqx.field(static=True)    

    def __init__(self, data):
        '''
        Object for storing data, including:

        - u,v: baseline coordinates (m)
        - wavels: wavelengths of observations (m)
        - vis: squared visibilities or visibilities
        - d_vis: uncertainties in vis
        - phi: absolute phases or closure phases
        - d_phi: uncertainties in phi
        - i_cps1, i_cps2, i_cps3: indices for closure phases, or None if absolute phases are available

        Args:
        - data: dict
            OIFITS data file opened with pyoifits, or dictionary filling out all the appropriate keywords & values
        '''

        try:
            # assume data is an oifits file opened with pyoifits
            data_names = [d.name for d in data.get_dataHDUs()]
            assert 'OI_VIS' in data_names or 'OI_VIS2' in data_names, "No visibility data found in OIFITS file"
            assert 'OI_T3' in data_names or 'OI_PHI' in data_names, "No phase data found in OIFITS file"

            # get the data from the oifits file
            self.wavel = np.array(data.get_wavelengthHDUs()[0].data['EFF_WAVE']) # note that for AMI this is scalar but for CHARA it is an array

            # if square visibilities are available, get them, otherwise get unsquared visibilities
            if 'OI_VIS2' in data_names:
                
                visdata = data.get_vis2HDUs()[0]
                self.vis = np.array(visdata.data['VIS2DATA'],dtype=float)
                self.d_vis = np.array(visdata.data['VIS2ERR'],dtype=float)
                vis_sta_index = visdata.data['STA_INDEX']

                self.u, self.v = np.array(visdata.data['UCOORD'],dtype=float), np.array(visdata.data['VCOORD'],dtype=float)

                self.v2_flag = True

            elif 'OI_VIS' in data_names:

                visdata = data['OI_VIS']
                self.vis = np.array(visdata.data['VISPHI'],dtype=float)
                self.d_vis = np.array(visdata.data['VISERR'],dtype=float)
                self.u, self.v = np.array(visdata.data['UCOORD'],dtype=float), np.array(visdata.data['VCOORD'],dtype=float)
                vis_sta_index = np.array(visdata.data['STA_INDEX'],dtype=int)

                self.v2_flag = False

            # if absolute phases are available, get them, otherwise get closure phases
            if 'OI_PHI' in data_names:

                phidata = data['OI_PHI']
                self.phi = np.array(phidata.data['VISPHI'],dtype=float)
                self.d_phi = np.array(phidata.data['VISERR'],dtype=float)
                self.i_cps1,self.i_cps2,self.i_cps3 = None, None, None

                self.cp_flag = True

            elif 'OI_T3' in data_names:

                phidata = data['OI_T3']
                self.phi = np.array(phidata.data['T3PHI'],dtype=float)
                self.d_phi = np.array(phidata.data['T3PHIERR'],dtype=float)

                cp_sta_index = np.array(phidata.data['STA_INDEX'],dtype=int)
                self.i_cps1,self.i_cps2,self.i_cps3 = cp_indices(vis_sta_index, cp_sta_index)

                self.cp_flag = True

        except: 
            # assume data is a dict of the form {'u':u,'v':v,'wavel':wavel,'vis':vis,'d_vis':d_vis,
            #'phi':phi,'d_phi':d_phi,'i_cps1':i_cps1,'i_cps2':i_cps2,'i_cps3':i_cps3,'v2_flag':v2_flag,'cp_flag':cp_flag}

            self.u = np.array(data['u'],dtype=float)
            self.v = np.array(data['v'],dtype=float)
            self.wavel = np.array(data['wavel'],dtype=float)

            self.vis = np.array(data['vis'],dtype=float)
            self.d_vis = np.array(data['d_vis'],dtype=float)

            self.phi = np.array(data['phi'],dtype=float)
            self.d_phi = np.array(data['d_phi'],dtype=float)

            self.i_cps1 = np.array(data['i_cps1'],dtype=int)
            self.i_cps2 = np.array(data['i_cps2'],dtype=int)
            self.i_cps3 = np.array(data['i_cps3'],dtype=int)

            self.v2_flag = data['v2_flag']
            self.cp_flag = data['cp_flag']



    def __repr__(self):
        phname = "CP" if self.cp_flag else "Phi"
        visname = "V2" if self.v2_flag else "Vis"
        return (f"OIData(u={self.u}, v={self.v}, {phname}={self.phi}, d_{phname}={self.d_phi}, "
                f"{visname}={self.vis}, d_{visname}={self.d_vis}, " 
                f"i_cps1={self.i_cps1}, i_cps2={self.i_cps2}, i_cps3={self.i_cps3})")
    
    def flatten_data(self):
        '''
        Flatten closure phases and uncertainties.
        '''
        if self.vis.ndim==1:
            return np.concatenate([self.vis, self.phi]), np.concatenate([self.d_vis, self.d_phi])
        else:
            return np.concatenate([self.vis, self.phi], axis=0), np.concatenate([self.d_vis, self.d_phi], axis=0)
    
    def unpack_all(self):
        '''
        Unpack all data to be used in some legacy model functions.
        '''
        if len(self.wavel)==1:
            return self.u/self.wavel, self.v/self.wavel, self.phi, self.d_phi, self.vis, self.d_vis, self.i_cps1, self.i_cps2, self.i_cps3
        else:
            u, v = uv_by_wavel(self.u,self.v,self.wavel)
            return u, v, self.phi, self.d_phi, self.vis, self.d_vis, self.i_cps1, self.i_cps2, self.i_cps3
    
    def flatten_model(self,cvis):
        '''
        cvis: complex visibilities from model

        Flatten model visibilities and phases.
        '''
        if cvis.ndim == 1:
            return np.concatenate([self.to_vis(cvis), self.to_phases(cvis)])
        else:
            return np.concatenate([self.to_vis(cvis), self.to_phases(cvis)], axis=0)
    
    def to_vis(self, cvis):
        '''
        Convert complex visibilities to visibilities or squared visibilities.
        '''
        if self.v2_flag:
            return np.abs(cvis)**2
        else:
            return np.abs(cvis)
            
    def to_phases(self, cvis):
        '''
        Convert complex visibilities to closure phases or absolute phases.
        '''
        if self.cp_flag:
            if cvis.ndim == 1:
                return closure_phases(cvis, self.i_cps1, self.i_cps2, self.i_cps3)  
            else:
                return vmap(closure_phases, in_axes=(1, None, None, None), out_axes=1)(cvis, self.i_cps1, self.i_cps2, self.i_cps3)
        else:
            np.angle(cvis)
    
    def model(self, model_object):
        '''
        Compute the model visibilities and phases for the given model object.
        '''
        cvis = model_object.model(self.u, self.v, self.wavel)
        return self.flatten_model(cvis)
    

'''--------------------------------------------------
Model functions
--------------------------------------------------'''

class BinaryModelAngular(zx.Base):
    ''' 
    Class for a binary star model.
    '''
    sep: jax.Array
    pa: jax.Array
    contrast: jax.Array

    def __init__(self, sep, pa, contrast):

        '''
        Initialize a binary model with separation, position angle, and contrast.

        sep: separation in mas
        pa: position angle in degrees
        contrast: flux ratio between components

        TODO: add flags to denote which coordinates are position and which are flux and use those to pass to 
        plotting and grid functions

        '''

        self.sep = np.asarray(sep,dtype=float)
        self.pa = np.asarray(pa,dtype=float)
        self.contrast = np.asarray(contrast,dtype=float)

    def __repr__(self):
        return f"BinaryModel(sep={self.sep}, pa={self.pa}, contrast={self.contrast})"
    
    def unpack_all(self):
        '''
        Convenience function to unpack all data to be used in model functions.
        '''
        return self.sep, self.pa, self.contrast
    
    def model(self, u, v, wavel):
        '''
        Model for binary star system.
        '''
        uu, vv = uv_by_wavel(u, v, wavel)
        return cvis_binary_angular(uu, vv, self.sep, self.pa, self.contrast)
    
class BinaryModelCartesian(zx.Base):
    ''' 
    Class for a binary star model.
    '''
    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array

    def __init__(self, dra, ddec, flux):

        '''
        Initialize a binary model with separation, position angle, and contrast.

        sep: separation in mas
        pa: position angle in degrees
        contrast: flux ratio between components
        
        TODO: add flags to denote which coordinates are position and which are flux and use those to pass to 
        plotting and grid functions

        '''

        self.dra = np.asarray(dra,dtype=float)
        self.ddec = np.asarray(ddec,dtype=float)
        self.flux = np.asarray(flux,dtype=float)

    def __repr__(self):
        return f"BinaryModelAngular(dra={self.dra}, pa={self.ddec}, flux={self.flux})"
    
    def unpack_all(self):
        '''
        Convenience function to unpack all data to be used in model functions.
        '''
        return self.dra, self.ddec, self.flux
    
    def model(self, u, v, wavel):
        '''
        Model for binary star system.
        '''
        if len(wavel)==1:
            uu, vv = u/wavel, v/wavel
        else:
            uu, vv = uv_by_wavel(u, v, wavel)
        return cvis_binary(uu, vv, self.ddec, self.dra, self.flux)

    
def cvis_binary_angular(u, v, sep, pa, contrast):
    #adapted from pymask
    ''' Calculate the complex visibilities observed by an array on a binary star
    ----------------------------------------------------------------
    - ddec = ddec (mas)
    - dra = dra (mas)
    - planet = planet brightness
    - u,v: baseline coordinates (wavelengths)
    ---------------------------------------------------------------- '''

    #normalize visibilities so total power is 1

    th = pa * dtor

    ddec = mas2rad*(sep * np.cos(th))
    dra = -1*mas2rad*(sep * np.sin(th))

    # decompose into two "luminosity"
    l2 = 1. / (contrast + 1)
    l1 = 1 - l2

    # phase-factor
    phi = np.exp(-i2pi*(u*dra + v*ddec))
    cvis = l1 + l2 * phi

    return cvis

def cvis_binary(u, v, ddec, dra, planet):
    #adapted from pymask
    ''' Calculate the complex visibilities observed by an array on a binary star
    ----------------------------------------------------------------
    - ddec = ddec (mas)
    - dra = dra (mas)
    - planet = planet brightness
    - u,v: baseline coordinates (wavelengths)
    ---------------------------------------------------------------- '''
    
    star = 1 

    #normalize visibilities so total power is 1
    p3 = star/(star+planet)
    p2 = planet/(star+planet)

    # relative locations
    ddec = ddec*np.pi/(180.*3600.*1000.)
    dra =  dra*np.pi/(180.*3600.*1000.)
    phi_r = np.cos(-2*np.pi*(u*dra + v*ddec))
    phi_i = np.sin(-2*np.pi*(u*dra + v*ddec))

    cvis = p3+p2*phi_r+p2*phi_i*1.0j

    return cvis

def loglike(values, params, data_obj, model_class):
    '''
    Abstract log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    '''

    param_dict = dict(zip(params, values))

    model_data = data_obj.model(model_class(**param_dict))
    data, errors = data_obj.flatten_data()

    return -0.5*np.sum((data - model_data)**2/errors**2)

def loglike_nosignal(values, params, data_obj, model_class):
    '''
    Abstract null log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    '''

    param_dict = dict(zip(params, values))

    model_data = data_obj.model(model_class(**param_dict))
    _, errors = data_obj.flatten_data()
    data = np.concatenate([np.ones_like(data_obj.vis), np.zeros_like(data_obj.phi)])

    return -0.5*np.sum((data - model_data)**2/errors**2)

def laplace_cov(values, params, data_obj, model_class):

    '''
    Calculate the uncertainty with the Laplace method from an optimized fit between a model and data object.

    Parameters
    ----------
    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    array-like
        Covariance matrix.
    '''

    hess = jax.hessian(loglike, argnums=0)(values, params, data_obj, model_class)
    return -np.linalg.inv(np.array(hess))

def laplace_contrast_uncertainty(flux, dra, ddec, data_obj, model_class):

    '''
    Calculate the uncertainty with the Laplace method from an optimized fit between a model and data object.

    Parameters
    ----------
    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    array-like
        Uncertainty in the contrast.
    '''

    params = ['dra', 'ddec', 'flux'] # TODO: make this more general

    objective = lambda flux: -loglike([dra, ddec, flux], params, data_obj, model_class)
    hess = jax.hessian(objective)(flux)
    cov = 1/(np.array(hess))
    return np.sqrt(cov)

def chi2ppf(p,df): 
    '''
    tensorflow-probability backend for the percentile function for chi2
    tested - matches scipy.stats.chi2.ppf to machine precision over domain we care about

    Parameters
    ----------
    p : array-like
        Percentile value
    df : array-like
        Degrees of freedom

    Returns
    -------
    array-like
        Corresponding chi2 value to the percentile
    '''
    return gammaincinv(df/2.,p)*2


def nsigma(chi2r_test,
           chi2r_true,
           ndof):
    """
    Parameters
    ----------
    chi2r_test: float
        Reduced chi-squared of test model.
    chi2r_true: float
        Reduced chi-squared of true model.
    ndof: int
        Number of degrees of freedom.
    
    Returns
    -------
    nsigma: float
        Detection significance.
    """
    
    q = jax.scipy.stats.chi2.cdf(ndof*chi2r_test/chi2r_true, ndof)
    p = 1.-q
    nsigma =np.sqrt(chi2ppf(p, 1.))
    
    return nsigma


@jit
def closure_phases(cvis, index_cps1, index_cps2, index_cps3):
    '''
    Calculate closure phases [degrees] from complex visibilities and cp indices

    vis: complex visibilities
    index_cps1, index_cps2, index_cps3: indices for closure phases (e.g. [0,1,2] for 1st 3-baseline closure phase)

    Returns: closure phases [degrees]

    '''
    real = np.real(cvis)
    imag = np.imag(cvis)
    visphiall = np.arctan2(imag,real)
    visphiall = np.mod(visphiall + 10980., 360.)-180.
    visphi = np.reshape(visphiall,(len(cvis),1))
    cp = visphi[np.array(index_cps1)] + visphi[np.array(index_cps2)] - visphi[np.array(index_cps3)]
    out = np.reshape(cp*180/np.pi,len(index_cps1))
    return out

def cp_indices(vis_sta_index, cp_sta_index):
    vis_sta_index, cp_sta_index = onp.array(vis_sta_index,dtype=int), onp.array(cp_sta_index,dtype=int)
    """Extracts indices for calculating closure phase from visibility and closure phase station indices"""
    i_cps1 = onp.zeros(len(onp.array(cp_sta_index)),dtype=int)
    i_cps2 = onp.zeros(len(onp.array(cp_sta_index)),dtype=int)
    i_cps3 = onp.zeros(len(onp.array(cp_sta_index)),dtype=int)

    for i in range(len(cp_sta_index)):
        i_cps1[i] = onp.argwhere((cp_sta_index[i][0]==vis_sta_index[:,0])&(cp_sta_index[i][1]==vis_sta_index[:,1]))[0,0]
        i_cps2[i] = onp.argwhere((cp_sta_index[i][1]==vis_sta_index[:,0])&(cp_sta_index[i][2]==vis_sta_index[:,1]))[0,0]
        i_cps3[i] = onp.argwhere((cp_sta_index[i][0]==vis_sta_index[:,0])&(cp_sta_index[i][2]==vis_sta_index[:,1]))[0,0]
    return np.array(i_cps1,dtype=int),np.array(i_cps2,dtype=int),np.array(i_cps3,dtype=int) 

def uv_by_wavel(u,v,wavel):
    '''Converts u,v coordinates from wavelengths to mas'''
    return u[:,np.newaxis]/wavel, v[:,np.newaxis]/wavel
