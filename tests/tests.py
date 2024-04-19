import jax.numpy as np
import jax.random as random
import jax
import jax.scipy as jsp
jax.config.update("jax_enable_x64", True)

import pyoifits as oifits

import numpy as onp

from functools import partial

from drpangloss.models import OIData, BinaryModelAngular, cvis_binary, closure_phases, BinaryModelCartesian
from drpangloss.grid_fit import optimized_contrast_grid, likelihood_grid, optimized_likelihood_grid, laplace_contrast_uncertainty_grid, ruffio_upperlimit, absil_limits, azimuthalAverage
from drpangloss.plotting import plot_optimized_and_grid, plot_likelihood_grid, plot_optimized_and_sigma, plot_contrast_limits

from matplotlib import get_backend
import matplotlib.pyplot as plt
import warnings
curr_backend = get_backend()
#switch to non-Gui, preventing plots being displayed
plt.switch_backend("Agg")
#suppress UserWarning that agg cannot show plots
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")


'''
Unit tests for the functions in drpangloss. 
'''

# load an example dataset

fname = "NuHor_F480M.oifits"
ddir = "./data/"

data = oifits.open(ddir+fname)

data.verify('silentfix')

oidata = OIData(data)
u, v, cp, cp_err, vis2, vis2_err, i_cps1,i_cps2,i_cps3 = oidata.unpack_all()

params = ["dra", "ddec", "flux"]

samples_dict = {
    "dra":  np.linspace(600., -600., 100), # left is more RA 
    "ddec": np.linspace(-600., 600., 101), # up is more dec
    "flux": 10**np.linspace(-6, -1, 102)
    }

true_values = [250., 150., 5e-4] # ra, dec, planet flux
binary = BinaryModelCartesian(true_values[0], true_values[1], true_values[2])

cvis_sim = binary.model(oidata.u, oidata.v, oidata.wavel) 

# fill out a new oidata model with simulated values
sim_data = {'u': oidata.u,
            'v': oidata.v,
            'wavel': oidata.wavel,
            'vis': oidata.to_vis(cvis_sim) + onp.random.randn(*oidata.vis.shape)*oidata.d_vis,
            'd_vis': oidata.d_vis,
            'phi': oidata.to_phases(cvis_sim) + onp.random.randn(*oidata.phi.shape)*oidata.d_phi,
            'd_phi': oidata.d_phi,
            'i_cps1': oidata.i_cps1,
            'i_cps2': oidata.i_cps2,
            'i_cps3': oidata.i_cps3,
            'v2_flag': oidata.v2_flag,
            'cp_flag': oidata.cp_flag}

oidata_sim = OIData(sim_data)


'''----------------------------------------------------------------'''

ddec, dra, planet = 0.1, 0.2, 10


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

def test_likelihood():
    
    binary = BinaryModelAngular(50,45,10)
    model_data = oidata.model(binary)
    data, errors = oidata.flatten_data()

    like =  -0.5*np.sum((data - model_data)**2/errors**2)
    assert np.all(np.isfinite(like))


def test_BinaryModelAngular():
	binary = BinaryModelAngular(50,45,0.1)
	model_data = oidata.model(binary)
	assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
	assert np.all(np.isfinite(model_data))

def test_BinaryModelCartesian():
	binary = BinaryModelCartesian(150,150,1e-3)
	model_data = oidata.model(binary)
	assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
	assert np.all(np.isfinite(model_data))

def test_likelihood_grid():
	loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict) # calculate once to jit
	assert np.all(np.isfinite(loglike_im))
	assert loglike_im.shape == (samples_dict['dra'].shape[0], samples_dict['ddec'].shape[0], samples_dict['flux'].shape[0])

	plot_likelihood_grid(loglike_im.max(axis=2).T, samples_dict, truths=true_values);

def test_optimized_likelihood_grid():
	loglike_im = optimized_likelihood_grid(oidata, BinaryModelCartesian, samples_dict) # calculate once to jit
	assert np.all(np.isfinite(loglike_im))
	assert loglike_im.shape == (samples_dict['dra'].shape[0], samples_dict['ddec'].shape[0])
	plot_likelihood_grid(loglike_im, samples_dict, truths=true_values);


def test_optimized():
	loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict) # calculate once to jit
	best_contrast_indices = np.argmax(loglike_im,axis=2)
	best_contrasts = samples_dict['flux'][best_contrast_indices]

	optimized = optimized_contrast_grid(oidata_sim, BinaryModelCartesian, samples_dict)
	assert optimized.shape == (samples_dict['dra'].shape[0], samples_dict['ddec'].shape[0])
	assert np.all(np.isfinite(optimized))
	plot_optimized_and_grid(loglike_im, optimized, samples_dict);

def test_laplace():
	loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict) # calculate once to jit
	best_contrast_indices = np.argmax(loglike_im,axis=2)
	best_contrasts = samples_dict['flux'][best_contrast_indices]

	optimized = optimized_contrast_grid(oidata_sim, BinaryModelCartesian, samples_dict)

	plot_optimized_and_grid(loglike_im, optimized, samples_dict);

	laplace_sigma_grid = laplace_contrast_uncertainty_grid(best_contrast_indices, oidata_sim, BinaryModelCartesian, samples_dict)
	assert laplace_sigma_grid.shape == (samples_dict['dra'].shape[0], samples_dict['ddec'].shape[0])
	assert np.all(np.isfinite(laplace_sigma_grid))
	plot_optimized_and_sigma(optimized, laplace_sigma_grid, samples_dict,snr=False);
	plot_optimized_and_sigma(optimized, laplace_sigma_grid, samples_dict,snr=True);


def test_ruffio():
	loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict) # calculate once to jit
	best_contrast_indices = np.argmax(loglike_im,axis=2)

	# this implementation is in the old syntax
	optimized = optimized_contrast_grid(oidata_sim, BinaryModelCartesian, samples_dict)
	laplace_sigma_grid = laplace_contrast_uncertainty_grid(best_contrast_indices, oidata_sim, BinaryModelCartesian, samples_dict)

	perc = np.array([jsp.stats.norm.cdf(2.)])
	limits = ruffio_upperlimit(optimized.flatten(),laplace_sigma_grid.flatten(),perc)
	limits_rs = limits.reshape(*optimized.shape,perc.shape[0])[:,:,0]

	# TODO: fix this syntax to be more readable
	rad_width_ruffio, avg_width_ruffio  = azimuthalAverage(-2.5*np.log10(limits_rs[:,:]), returnradii=True, binsize=2, stddev=False)
	_, std_width_ruffio  = azimuthalAverage(-2.5*np.log10(limits_rs[:,:]), returnradii=True, binsize=2, stddev=True)
	assert np.all(np.isfinite(limits_rs))
	assert np.all(np.isfinite(rad_width_ruffio))
	assert np.all(np.isfinite(avg_width_ruffio))
	assert np.all(np.isfinite(std_width_ruffio))
	plot_contrast_limits(limits_rs, samples_dict, rad_width_ruffio, avg_width_ruffio, std_width_ruffio,true_values=true_values);

def test_absil():
	
	limits_absil = absil_limits(samples_dict, oidata_sim, BinaryModelCartesian, 5.) # TODO: check this, results look a bit different from Dori implementation

	# TODO: make this syntax more readable
	rad_width_absil, avg_width_absil  = azimuthalAverage(-2.5*np.log10(limits_absil[:,:]), returnradii=True, binsize=2, stddev=False)
	_, std_width_absil  = azimuthalAverage(-2.5*np.log10(limits_absil[:,:]), returnradii=True, binsize=2, stddev=True)
	assert np.all(np.isfinite(limits_absil))
	assert np.all(np.isfinite(rad_width_absil))
	assert np.all(np.isfinite(avg_width_absil))
	assert np.all(np.isfinite(std_width_absil))
	plot_contrast_limits(limits_absil, samples_dict, rad_width_absil, avg_width_absil, std_width_absil,true_values=true_values);