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
from drpangloss.models import BinaryModelCartesian, nsigma
from drpangloss.oidata import OIData
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


def test_likelihood_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::40],
        "ddec": samples_dict["ddec"][::40],
        "flux": samples_dict["flux"][::40],
    }
    ordered = likelihood_grid(oidata, BinaryModelCartesian, reduced_samples)

    permuted_samples = {
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
        "ddec": reduced_samples["ddec"],
    }
    permuted = likelihood_grid(oidata, BinaryModelCartesian, permuted_samples)

    assert ordered.shape == (
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
        reduced_samples["flux"].shape[0],
    )
    assert permuted.shape == (
        reduced_samples["flux"].shape[0],
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 2, 0)))


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


def test_optimized_likelihood_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::40],
        "ddec": samples_dict["ddec"][::40],
        "flux": samples_dict["flux"][::40],
    }
    ordered = optimized_likelihood_grid(
        oidata, BinaryModelCartesian, reduced_samples
    )

    permuted_samples = {
        "ddec": reduced_samples["ddec"],
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
    }
    permuted = optimized_likelihood_grid(
        oidata, BinaryModelCartesian, permuted_samples
    )

    assert ordered.shape == (
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
    )
    assert permuted.shape == (
        reduced_samples["ddec"].shape[0],
        reduced_samples["dra"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 0)))


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


def test_optimized_contrast_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::40],
        "ddec": samples_dict["ddec"][::40],
        "flux": samples_dict["flux"][::40],
    }
    ordered = optimized_contrast_grid(
        oidata_sim, BinaryModelCartesian, reduced_samples
    )

    permuted_samples = {
        "ddec": reduced_samples["ddec"],
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
    }
    permuted = optimized_contrast_grid(
        oidata_sim, BinaryModelCartesian, permuted_samples
    )

    assert ordered.shape == (
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
    )
    assert permuted.shape == (
        reduced_samples["ddec"].shape[0],
        reduced_samples["dra"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 0)))


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


def test_laplace_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::40],
        "ddec": samples_dict["ddec"][::40],
        "flux": samples_dict["flux"][::40],
    }
    ordered_loglike = likelihood_grid(
        oidata, BinaryModelCartesian, reduced_samples
    )
    ordered_best = np.argmax(ordered_loglike, axis=2)
    ordered = laplace_contrast_uncertainty_grid(
        ordered_best, oidata_sim, BinaryModelCartesian, reduced_samples
    )

    permuted_samples = {
        "ddec": reduced_samples["ddec"],
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
    }
    permuted_loglike = likelihood_grid(
        oidata, BinaryModelCartesian, permuted_samples
    )
    permuted_best = np.argmax(
        permuted_loglike, axis=list(permuted_samples.keys()).index("flux")
    )
    permuted = laplace_contrast_uncertainty_grid(
        permuted_best, oidata_sim, BinaryModelCartesian, permuted_samples
    )

    assert ordered.shape == (
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
    )
    assert permuted.shape == (
        reduced_samples["ddec"].shape[0],
        reduced_samples["dra"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 0)))


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
        percentile=perc,
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
        sigma=5.0,
    )


def test_nsigma_increases_with_chi2_ratio():
    significances = nsigma(np.array([1.0, 2.0, 4.0]), 1.0, 56)

    assert np.all(np.diff(significances) > 0.0)


def test_absil_limit_responds_to_smaller_uncertainties():
    samples = {
        "dra": np.array([100.0]),
        "ddec": np.array([100.0]),
        "flux": 10 ** np.linspace(-6.0, -1.0, 30),
    }
    null_cvis = np.ones_like(oidata_sim.u, dtype=complex)
    null_vis = oidata_sim.to_vis(null_cvis)
    null_phi = oidata_sim.to_phases(null_cvis)
    vis_noise = np.linspace(-1.0, 1.0, oidata_sim.vis.size)
    phi_noise = np.linspace(1.0, -1.0, oidata_sim.phi.size)

    def noisy_null(error_scale):
        return OIData(
            {
                "u": oidata_sim.u,
                "v": oidata_sim.v,
                "wavel": oidata_sim.wavel,
                "vis": null_vis
                + vis_noise * oidata_sim.d_vis * error_scale,
                "d_vis": oidata_sim.d_vis * error_scale,
                "phi": null_phi
                + phi_noise * oidata_sim.d_phi * error_scale,
                "d_phi": oidata_sim.d_phi * error_scale,
                "i_cps1": oidata_sim.i_cps1,
                "i_cps2": oidata_sim.i_cps2,
                "i_cps3": oidata_sim.i_cps3,
                "v2_flag": oidata_sim.v2_flag,
                "cp_flag": oidata_sim.cp_flag,
            }
        )

    nominal = absil_limits(
        samples, noisy_null(1.0), BinaryModelCartesian, 2.0
    )
    improved = absil_limits(
        samples, noisy_null(0.1), BinaryModelCartesian, 2.0
    )

    assert improved.item() < nominal.item()


def test_plot_contrast_limits_percentile_label():
    plt.close("all")

    plot_contrast_limits(
        np.ones((2, 2)) * 1e-4,
        {"dra": np.array([1.0, 0.0]), "ddec": np.array([0.0, 1.0])},
        np.array([0.0, 1.0]),
        np.array([10.0, 9.0]),
        np.array([0.2, 0.3]),
        percentile=perc,
    )

    fig = plt.gcf()
    map_axis = next(ax for ax in fig.axes if ax.get_title())
    legend_axis = next(ax for ax in fig.axes if ax.get_legend() is not None)
    assert map_axis.get_title() == "97.7% Upper Limit Map ($\\Delta$mag)"
    assert legend_axis.get_legend().texts[0].get_text() == "97.7% Upper Limit"


def test_plot_contrast_limits_sigma_label():
    plt.close("all")

    plot_contrast_limits(
        np.ones((2, 2)) * 1e-4,
        {"dra": np.array([1.0, 0.0]), "ddec": np.array([0.0, 1.0])},
        np.array([0.0, 1.0]),
        np.array([10.0, 9.0]),
        np.array([0.2, 0.3]),
        sigma=5.0,
    )

    fig = plt.gcf()
    map_axis = next(ax for ax in fig.axes if ax.get_title())
    legend_axis = next(ax for ax in fig.axes if ax.get_legend() is not None)
    assert (
        map_axis.get_title() == "5$\\sigma$ Contrast Limit Map ($\\Delta$mag)"
    )
    assert (
        legend_axis.get_legend().texts[0].get_text()
        == "5$\\sigma$ Contrast Limit"
    )
