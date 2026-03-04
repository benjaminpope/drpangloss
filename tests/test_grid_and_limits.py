import warnings

import jax.numpy as np
from matplotlib import get_backend
import matplotlib.pyplot as plt

from drpangloss.grid_fit import (
    absil_limits,
    azimuthalAverage,
    laplace_contrast_uncertainty_grid,
    likelihood_grid,
    optimized_contrast_grid,
    optimized_likelihood_grid,
    ruffio_upperlimit,
)
from drpangloss.models import BinaryModelCartesian
from drpangloss.plotting import (
    plot_contrast_limits,
    plot_likelihood_grid,
    plot_optimized_and_grid,
    plot_optimized_and_sigma,
)
from tests._test_data import (
    oidata,
    oidata_sim,
    perc,
    samples_dict,
    true_values,
)

curr_backend = get_backend()
plt.switch_backend("Agg")
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")


def test_likelihood_grid():
    loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict)
    assert np.all(np.isfinite(loglike_im))
    assert loglike_im.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
        samples_dict["flux"].shape[0],
    )

    plot_likelihood_grid(
        loglike_im.max(axis=2).T, samples_dict, truths=true_values
    )


def test_optimized_likelihood_grid():
    loglike_im = optimized_likelihood_grid(
        oidata, BinaryModelCartesian, samples_dict
    )
    assert np.all(np.isfinite(loglike_im))
    assert loglike_im.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
    )
    plot_likelihood_grid(loglike_im, samples_dict, truths=true_values)


def test_optimized():
    loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict)

    optimized = optimized_contrast_grid(
        oidata_sim, BinaryModelCartesian, samples_dict
    )
    assert optimized.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
    )
    assert np.all(np.isfinite(optimized))
    plot_optimized_and_grid(loglike_im, optimized, samples_dict)


def test_laplace():
    loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict)
    best_contrast_indices = np.argmax(loglike_im, axis=2)

    optimized = optimized_contrast_grid(
        oidata_sim, BinaryModelCartesian, samples_dict
    )

    plot_optimized_and_grid(loglike_im, optimized, samples_dict)

    laplace_sigma_grid = laplace_contrast_uncertainty_grid(
        best_contrast_indices, oidata_sim, BinaryModelCartesian, samples_dict
    )
    assert laplace_sigma_grid.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
    )
    assert np.all(np.isfinite(laplace_sigma_grid))
    plot_optimized_and_sigma(
        optimized, laplace_sigma_grid, samples_dict, snr=False
    )
    plot_optimized_and_sigma(
        optimized, laplace_sigma_grid, samples_dict, snr=True
    )


def test_ruffio():
    loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict)
    best_contrast_indices = np.argmax(loglike_im, axis=2)

    optimized = optimized_contrast_grid(
        oidata_sim, BinaryModelCartesian, samples_dict
    )
    laplace_sigma_grid = laplace_contrast_uncertainty_grid(
        best_contrast_indices, oidata_sim, BinaryModelCartesian, samples_dict
    )

    limits = ruffio_upperlimit(
        optimized.flatten(), laplace_sigma_grid.flatten(), perc
    )
    limits_rs = limits.reshape(*optimized.shape, perc.shape[0])[:, :, 0]

    rad_width_ruffio, avg_width_ruffio = azimuthalAverage(
        -2.5 * np.log10(limits_rs[:, :]),
        returnradii=True,
        binsize=2,
        stddev=False,
    )
    _, std_width_ruffio = azimuthalAverage(
        -2.5 * np.log10(limits_rs[:, :]),
        returnradii=True,
        binsize=2,
        stddev=True,
    )
    assert np.all(np.isfinite(limits_rs))
    assert np.all(np.isfinite(rad_width_ruffio))
    assert np.all(np.isfinite(avg_width_ruffio))
    assert np.all(np.isfinite(std_width_ruffio))
    plot_contrast_limits(
        limits_rs,
        samples_dict,
        rad_width_ruffio,
        avg_width_ruffio,
        std_width_ruffio,
        true_values=true_values,
    )


def test_absil():
    limits_absil = absil_limits(
        samples_dict, oidata_sim, BinaryModelCartesian, 5.0
    )

    rad_width_absil, avg_width_absil = azimuthalAverage(
        -2.5 * np.log10(limits_absil[:, :]),
        returnradii=True,
        binsize=2,
        stddev=False,
    )
    _, std_width_absil = azimuthalAverage(
        -2.5 * np.log10(limits_absil[:, :]),
        returnradii=True,
        binsize=2,
        stddev=True,
    )
    assert np.all(np.isfinite(limits_absil))
    assert np.all(np.isfinite(rad_width_absil))
    assert np.all(np.isfinite(avg_width_absil))
    assert np.all(np.isfinite(std_width_absil))
    plot_contrast_limits(
        limits_absil,
        samples_dict,
        rad_width_absil,
        avg_width_absil,
        std_width_absil,
        true_values=true_values,
    )
