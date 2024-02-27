import jax.numpy as jnp
from jax import grad, jit, vmap
import jax 

import numpy as np 

import optimistix as optx
import equinox as eqx


"""------------------------------
------------------------------"""


rad2mas = 180./np.pi*3600.*1000. # convert rad to mas
mas2rad = np.pi/180./3600./1000. # convert mas to rad


'''--------------------------------------------------
Data class
--------------------------------------------------'''

class OIData(eqx.Module):
    u: jax.Array
    v: jax.Array
    cp: jax.Array
    d_cp: jax.Array
    vis2: jax.Array
    d_vis2: jax.Array
    i_cps1: jax.Array
    i_cps2: jax.Array
    i_cps3: jax.Array

    def __init__(self, u, v, cp, d_cp, vis2, d_vis2, i_cps1, i_cps2, i_cps3):
        '''
        Object for storing data, including:

        - u,v: baseline coordinates (wavelengths)
        - cp: closure phases (deg)
        - d_cp: closure phase uncertainty (deg)
        - vis2: squared visibilties
        - d_vis2: squared visibilty uncertainty
        - i_cps1, i_cps2, i_cps3: indices for closure phases
        '''

        self.u = u
        self.v = v
        self.cp = cp
        self.d_cp = d_cp
        self.vis2 = vis2
        self.d_vis2 = d_vis2
        self.i_cps1 = i_cps1
        self.i_cps2 = i_cps2
        self.i_cps3 = i_cps3

    def __repr__(self):
        return f"Data(u={self.u}, v={self.v}, cp={self.cp}, d_cp={self.d_cp}, vis2={self.vis2}, d_vis2={self.d_vis2}, i_cps1={self.i_cps1}, i_cps2={self.i_cps2}, i_cps3={self.i_cps3})"
    
    def unpack_all(self):
        '''
        Convenience function to unpack all data to be used in model functions.
        '''
        return self.u, self.v, self.cp, self.d_cp, self.vis2, self.d_vis2, self.i_cps1, self.i_cps2, self.i_cps3

'''--------------------------------------------------
Model functions
--------------------------------------------------'''

def vis_binary(u, v, ddec,dra,planet,star=1.):
    #adapted from pymask
    ''' Calculate the complex visibilities observed by an array on a binary star
    ----------------------------------------------------------------
    - ddec = ddec (mas)
    - dra = dra (mas)
    - planet = planet brightness
    - star = star brightness
    - u,v: baseline coordinates (wavelengths)
    ---------------------------------------------------------------- '''

    #normalize visibilities so total power is 1
    p3 = star/(star+planet)
    p2 = planet/(star+planet)

    # relative locations
    ddec = ddec*np.pi/(180.*3600.*1000.)
    dra =  dra*np.pi/(180.*3600.*1000.)
    phi_r = jnp.cos(-2*np.pi*(u*dra + v*ddec))
    phi_i = jnp.sin(-2*np.pi*(u*dra + v*ddec))

    cvis = p3+p2*phi_r+p2*phi_i*1.0j

    return cvis

@jit
def vis_binary2(u, v, ddec,dra,p2,p3):
    #adapted from pymask
    ''' Calculate the complex visibilities observed by an array on a binary star
    ----------------------------------------------------------------
    - ddec = ddec (mas)
    - dra = dra (mas)
    - p2 = planet
    - p3 = star


    - u,v: baseline coordinates (wavelengths)
    ---------------------------------------------------------------- '''

    # relative locations
    ddec = (ddec)*np.pi/(180.*3600.*1000.)
    dra =  (dra)*np.pi/(180.*3600.*1000.)
    phi_r = jnp.cos(-2*np.pi*(u*dra + v*ddec))
    phi_i = jnp.sin(-2*np.pi*(u*dra + v*ddec))

    cvis = p3+p2*phi_r+p2*phi_i*1.0j

    return cvis

@jit
def closure_phases(vis, index_cps1, index_cps2, index_cps3):
    '''
    Calculate closure phases [degrees] from complex visibilities and cp indices

    vis: complex visibilities
    index_cps1, index_cps2, index_cps3: indices for closure phases (e.g. [0,1,2] for 1st 3-baseline closure phase)

    Returns: closure phases [degrees]

    '''
    real = jnp.real(vis)
    imag = jnp.imag(vis)
    visphiall = jnp.arctan2(imag,real)
    visphiall = jnp.mod(visphiall + 10980., 360.)-180.
    visphi = jnp.reshape(visphiall,(len(vis),1))
    cp = visphi[jnp.array(index_cps1)] + visphi[jnp.array(index_cps2)] - visphi[jnp.array(index_cps3)]
    out = jnp.reshape(cp*180/np.pi,len(index_cps1))
    return out

'''--------------------------------------------------
Log likelihood functions
--------------------------------------------------'''

def log_like_binary(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra,planet_contrast):
    #adapted from pymask
    ''' Calculate the unnormalized log-likelihood of an unresolved binary star model
    ----------------------------------------------------------------
    - ddec = companion dec offset (mas)
    - dra = companion ra offset (mas)
    - cp = closure phases (deg)
    - d_cp = closure phase uncertainty (deg)
    - vis2 = squared visibilties 
    - d_vis2 = squared visibilty uncertainty 
    - planet_contrast = planet
    - u,v: baseline coordinates (wavelengths)
    ---------------------------------------------------------------- '''

    cvis_model = vis_binary(u, v, ddec,dra,planet_contrast)
    
    #calculate model observables
    cp_obs = closure_phases(cvis_model,i_cps1,i_cps2,i_cps3)
    vis2_obs = jnp.abs(cvis_model)**2
    
    ll_cp = jnp.sum((cp_obs-cp)**2/d_cp**2)
    ll_vis2 = jnp.sum((vis2_obs-vis2)**2/d_vis2**2)
    
    return -0.5*(ll_cp+ll_vis2)

def chi2_binary(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec, dra, planet_contrast):
    #adapted from pymask
    ''' Calculate the unnormalized log-likelihood of an unresolved binary star model
    ----------------------------------------------------------------
    - ddec = companion dec offset (mas)
    - dra = companion ra offset (mas)
    - cp = closure phases (deg)
    - d_cp = closure phase uncertainty (deg)
    - vis2 = squared visibilties 
    - d_vis2 = squared visibilty uncertainty 
    - planet_contrast = planet
    - u,v: baseline coordinates (wavelengths)
    ---------------------------------------------------------------- '''
    
    return -0.5*(log_like_binary(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec, dra, planet_contrast))



def log_like_star(cp, d_cp, vis2, d_vis2):
    ''' Calculate the unnormalized log-likelihood of an unresolved star model
    ----------------------------------------------------------------
    - cp = closure phases (deg)
    - d_cp = closure phase uncertainty (deg)
    - vis2 = squared visibilties 
    - d_vis2 = squared visibilty uncertainty 
    ---------------------------------------------------------------- '''

    #calculate model observables
    cp_obs = 0.
    vis2_obs = 1.
    
    ll_cp = jnp.sum((cp_obs-cp)**2/d_cp**2)
    ll_vis2 = jnp.sum((vis2_obs-vis2)**2/d_vis2**2)
    
    return -0.5*(ll_cp+ll_vis2)


def log_like_wrap(planet_contrast,u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra):
    
    return -log_like_binary(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra,planet_contrast)

def optimize_log_like(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra,planet_contrast):
    
    sol = optx.compat.minimize(log_like_wrap,method='BFGS',
                                x0=jnp.array([planet_contrast]), args=(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec,dra),options={"maxiter":100})
    res = sol.x


    return res

#calc sigma with laplace approximation
def sigma(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3, ddec, dra, planet_contrast):
    hess = jax.hessian(log_like_binary, argnums=[11])(u, v, cp, d_cp, vis2, d_vis2,i_cps1,i_cps2,i_cps3,ddec,dra,planet_contrast)
    cov = -jnp.linalg.inv(jnp.array(hess))

    return jnp.sqrt(cov)