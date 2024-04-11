import jax.numpy as jnp
import jax.random as random
import jax 
jax.config.update("jax_enable_x64", True)

import pyoifits as oifits

import numpy as np

from functools import partial

from drpangloss.models import OIData, BinaryModelAngular, cvis_binary, closure_phases, log_like_binary, chi2_binary, log_like_star, log_like_wrap, optimize_log_like, sigma
# from drpangloss.oifits_implaneia import load_oifits

# load an example dataset

fname = "NuHor_F480M.oifits"
ddir = "./data/"

data = oifits.open(ddir+fname)

data.verify('silentfix')

oidata = OIData(data)
u, v, cp, cp_err, vis2, vis2_err, i_cps1,i_cps2,i_cps3 = oidata.unpack_all()



'''----------------------------------------------------------------'''

ddec, dra, planet = 0.1, 0.1, 10


def test_cvis_binary():
	vis = cvis_binary(u, v, ddec,dra, planet)
	vis2 = np.abs(vis)**2
	assert vis.shape == (u.shape[0],)
	assert np.all(vis2 >= 0.)
	assert np.all(vis2 <= 1.)
	assert np.all(np.isfinite(vis))

def test_closure_phases():
	vis = cvis_binary(u, v, ddec,dra,planet)
	cps = closure_phases(vis, i_cps1, i_cps2, i_cps3)
	assert cps.shape == (35,)
	assert np.all(np.isfinite(cps))

def test_log_like_binary():
	ll = log_like_binary(u, v, cp, cp_err, vis2, vis2_err, i_cps1, i_cps2, i_cps3, ddec, dra, 1/planet)
	assert np.isfinite(ll)

def test_chi2_binary():
	ll = chi2_binary(u, v, cp, cp_err, vis2, vis2_err, i_cps1,i_cps2,i_cps3, ddec, dra, 1/planet)
	assert np.isfinite(ll)

def test_log_like_star():
	ll = log_like_star(cp, cp_err, vis2, vis2_err)
	assert np.isfinite(ll)

def test_log_like_wrap():
	ll = log_like_wrap(100.,u, v, cp, cp_err, vis2, vis2_err,i_cps1,i_cps2,i_cps3, ddec,dra)
	
	assert np.isfinite(ll)

def test_optimize_log_like():

	planet_contrast=10.
	
	res = optimize_log_like(u, v, cp, cp_err, vis2, vis2_err,i_cps1,i_cps2,i_cps3, ddec,dra,planet_contrast)
	assert np.isfinite(res)

def test_sigma():
	planet_contrast=10.
	sig = sigma(u, v, cp, cp_err, vis2, vis2_err,i_cps1,i_cps2,i_cps3, ddec, dra, planet_contrast)
	print(sig)
	assert np.isfinite(sig)

def test_likelihood():
    
    binary = BinaryModelAngular(50,45,10)
    model_data = oidata.model(binary)
    data, errors = oidata.flatten_data()

    like =  -0.5*np.sum((data - model_data)**2/errors**2)
    assert np.all(np.isfinite(like))


def test_BinaryModelAngular():
	binary = BinaryModelAngular(50,45,10)
	model_data = oidata.model(binary)
	assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
	assert np.all(np.isfinite(model_data))