import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from matplotlib.ticker import FuncFormatter


matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["font.family"] = ["serif"]
plt.rcParams.update({"font.size": 14})

"""
Plotting functions to automatically render outputs from the model fitting functions.
"""


def _range_aware_float_formatter(
    vmin, vmax, min_sigfigs=3, max_sigfigs=6, scale=1.0
):
    span = abs(float(vmax) - float(vmin)) * abs(float(scale))
    if not np.isfinite(span) or span <= 0:
        sigfigs = min_sigfigs + 1
    else:
        sigfigs = int(
            np.clip(
                np.ceil(-np.log10(span)) + 2,
                min_sigfigs,
                max_sigfigs,
            )
        )
    return FuncFormatter(lambda x, _: f"{(x * scale):.{sigfigs}g}")


def posterior_predictive_summary(
    dra_samples, ddec_samples, flux_samples, oidata, model_class
):
    """
    Compute posterior predictive means and standard deviations for visibilities and phases.

    Parameters
    ----------
    dra_samples, ddec_samples, flux_samples : array-like
        Posterior samples for Cartesian offset and flux ratio.
    oidata : OIData
        Data object defining observable conventions and geometry.
    model_class : type
        Model class with signature ``model_class(dra=..., ddec=..., flux=...)``.

    Returns
    -------
    dict
        Dictionary containing ``vis_mean``, ``vis_std``, ``phi_mean``, ``phi_std`` arrays.
    """
    vis_pred = []
    phi_pred = []

    for dra_i, ddec_i, flux_i in zip(dra_samples, ddec_samples, flux_samples):
        model_i = model_class(
            dra=float(dra_i), ddec=float(ddec_i), flux=float(flux_i)
        )
        cvis_i = model_i.model(oidata.u, oidata.v, oidata.wavel)
        vis_pred.append(np.asarray(oidata.to_vis(cvis_i)).reshape(-1))
        phi_pred.append(np.asarray(oidata.to_phases(cvis_i)).reshape(-1))

    vis_pred = np.stack(vis_pred, axis=0)
    phi_pred = np.stack(phi_pred, axis=0)

    return {
        "vis_mean": vis_pred.mean(axis=0),
        "vis_std": vis_pred.std(axis=0),
        "phi_mean": phi_pred.mean(axis=0),
        "phi_std": phi_pred.std(axis=0),
    }


def plot_data_model_correlation(
    oidata,
    predictions_by_label,
    colors=None,
    figsize=(12, 4),
    phase_title="Phase correlation",
):
    """
    Plot data-vs-model correlation panels for visibility and phase observables.

    Parameters
    ----------
    oidata : OIData
        Observed data container.
    predictions_by_label : dict
        Mapping ``label -> prediction summary`` where each summary contains
        ``vis_mean``, ``vis_std``, ``phi_mean``, ``phi_std`` arrays.
    colors : list, optional
        Matplotlib color list. If omitted, cycle ``C0``, ``C1``, ...
    figsize : tuple, optional
        Figure size.
    phase_title : str, optional
        Title for the phase panel.

    Returns
    -------
    tuple
        ``(fig, (ax1, ax2))`` for visibility and phase axes.
    """
    vis_data = np.asarray(oidata.vis).reshape(-1)
    phi_data = np.asarray(oidata.phi).reshape(-1)
    d_vis_data = np.asarray(oidata.d_vis).reshape(-1)
    d_phi_data = np.asarray(oidata.d_phi).reshape(-1)

    if colors is None:
        colors = [f"C{i}" for i in range(max(1, len(predictions_by_label)))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    vis_low = [vis_data.min()]
    vis_high = [vis_data.max()]
    phi_low = [phi_data.min()]
    phi_high = [phi_data.max()]

    for idx, (label, pred) in enumerate(predictions_by_label.items()):
        color = colors[idx % len(colors)]
        ax1.errorbar(
            vis_data,
            np.asarray(pred["vis_mean"]).reshape(-1),
            xerr=d_vis_data,
            yerr=np.asarray(pred["vis_std"]).reshape(-1),
            fmt="o",
            alpha=0.45,
            color=color,
            label=label,
        )
        ax2.errorbar(
            phi_data,
            np.asarray(pred["phi_mean"]).reshape(-1),
            xerr=d_phi_data,
            yerr=np.asarray(pred["phi_std"]).reshape(-1),
            fmt="o",
            alpha=0.45,
            color=color,
            label=label,
        )
        vis_low.append(np.asarray(pred["vis_mean"]).min())
        vis_high.append(np.asarray(pred["vis_mean"]).max())
        phi_low.append(np.asarray(pred["phi_mean"]).min())
        phi_high.append(np.asarray(pred["phi_mean"]).max())

    vis_pad = float(np.median(d_vis_data)) if d_vis_data.size else 0.0
    phi_pad = float(np.median(d_phi_data)) if d_phi_data.size else 0.0

    vis_min = min(vis_low) - vis_pad
    vis_max = max(vis_high) + vis_pad
    phi_min = min(phi_low) - phi_pad
    phi_max = max(phi_high) + phi_pad

    vis_line = np.linspace(vis_min, vis_max, 200)
    phi_line = np.linspace(phi_min, phi_max, 200)
    vis_formatter = _range_aware_float_formatter(vis_min, vis_max, scale=100.0)
    phi_formatter = _range_aware_float_formatter(phi_min, phi_max)

    ax1.plot(vis_line, vis_line, "k--", lw=1)
    ax1.set_xlim(vis_min, vis_max)
    ax1.set_ylim(vis_min, vis_max)
    ax1.xaxis.set_major_formatter(vis_formatter)
    ax1.yaxis.set_major_formatter(vis_formatter)
    ax1.set_xlabel("Data (V2, %)")
    ax1.set_ylabel("Model (V2, %)")
    ax1.set_title("Visibility correlation")
    ax1.legend(loc="best")

    ax2.plot(phi_line, phi_line, "k--", lw=1)
    ax2.set_xlim(phi_min, phi_max)
    ax2.set_ylim(phi_min, phi_max)
    ax2.xaxis.set_major_formatter(phi_formatter)
    ax2.yaxis.set_major_formatter(phi_formatter)
    ax2.set_xlabel("Data (deg)")
    ax2.set_ylabel("Model (deg)")
    ax2.set_title(phase_title)
    ax2.legend(loc="best")

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_trace_panels(samples_dict, keys, title, color="C0", figsize=(10, 6)):
    """
    Plot simple one-dimensional trace panels for selected sample keys.

    Parameters
    ----------
    samples_dict : dict
        Mapping from key name to one-dimensional sample arrays.
    keys : list
        Ordered list of keys to plot.
    title : str
        Figure title.
    color : str, optional
        Line color.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ax.plot(
            np.asarray(samples_dict[key]).reshape(-1),
            lw=0.8,
            alpha=0.9,
            color=color,
        )
        ax.set_ylabel(key)
    axes[-1].set_xlabel("Sample")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_likelihood_grid(
    loglike_im,
    samples_dict,
    truths=None,
    best_point=None,
    truth_label="Truth",
    best_label="Grid max",
    colorbar_label="Log likelihood",
    cmap="inferno",
    figsize=(12, 6),
):
    """
    Plot the results of a likelihood_grid calculation.

    Parameters
    ----------
    loglike_im : array
        The likelihood grid, output of likelihood_grid
    samples_dict : dict
        Dictionary of samples used in the grid calculation
    truths : list, optional
        List of true values for the parameters, default None
    """

    dra_axis = np.asarray(samples_dict["dra"])
    ddec_axis = np.asarray(samples_dict["ddec"])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        np.asarray(loglike_im).T,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        extent=[
            float(dra_axis.min()),
            float(dra_axis.max()),
            float(ddec_axis.min()),
            float(ddec_axis.max()),
        ],
    )
    fig.colorbar(im, ax=ax, shrink=0.9, label=colorbar_label, pad=0.01)

    if truths is not None:
        if isinstance(truths, dict):
            dra_inp = float(truths["dra"])
            ddec_inp = float(truths["ddec"])
        else:
            dra_inp = float(truths[0])
            ddec_inp = float(truths[1])
        ax.scatter(
            [dra_inp],
            [ddec_inp],
            marker="x",
            s=80,
            c="white",
            linewidths=2,
            label=truth_label,
        )

    if best_point is not None:
        if isinstance(best_point, dict):
            dra_best = float(best_point["dra"])
            ddec_best = float(best_point["ddec"])
        else:
            dra_best = float(best_point[0])
            ddec_best = float(best_point[1])
        ax.scatter(
            [dra_best],
            [ddec_best],
            marker="o",
            s=40,
            facecolors="none",
            edgecolors="cyan",
            label=best_label,
        )

    ax.set_xlabel("ΔRA [mas]")
    ax.set_ylabel("ΔDec [mas]")
    ax.set_title("Likelihood grid")
    if truths is not None or best_point is not None:
        ax.legend(loc="best")
    ax.invert_yaxis()
    return fig, ax


def plot_chainconsumer_diagnostics(
    chains_by_label,
    columns,
    truth,
    colors=None,
    walk_columns=None,
):
    """
    Plot comparison diagnostics using ChainConsumer for multiple posterior chains.

    Parameters
    ----------
    chains_by_label : dict
        Mapping ``label -> pandas.DataFrame`` containing chain samples.
    columns : list[str]
        Columns used for contour/corner plotting.
    truth : dict
        Truth mapping for columns displayed.
    colors : list[str], optional
        Per-chain colors.
    walk_columns : list[str], optional
        Columns to show in walk plots. Defaults to ``columns``.

    Returns
    -------
    ChainConsumer
        Configured ChainConsumer instance.
    """
    from chainconsumer import ChainConsumer, Chain, Truth

    if colors is None:
        colors = [f"C{i}" for i in range(max(1, len(chains_by_label)))]
    if walk_columns is None:
        walk_columns = columns

    consumer = ChainConsumer()
    for idx, (label, samples) in enumerate(chains_by_label.items()):
        consumer.add_chain(
            Chain(
                samples=samples[columns],
                name=label,
                color=colors[idx % len(colors)],
                plot_point=False,
                plot_cloud=False,
            )
        )
    consumer.add_truth(Truth(location=truth))
    consumer.plotter.plot()
    consumer.plotter.plot_walks(
        columns=walk_columns, plot_weights=False, plot_posterior=False
    )
    return consumer


def diagnostics_table_from_samples(
    samples,
    dra_key="dra",
    ddec_key="ddec",
    flux_key="flux",
    log10_flux=False,
):
    """
    Build a standardized diagnostics table from posterior sample arrays.

    Parameters
    ----------
    samples : dict-like
        Mapping of sample arrays.
    dra_key : str, optional
        Key for right-ascension offsets.
    ddec_key : str, optional
        Key for declination offsets.
    flux_key : str, optional
        Key for flux or log10-flux samples.
    log10_flux : bool, optional
        If ``True``, exponentiate ``flux_key`` values as base-10.

    Returns
    -------
    pandas.DataFrame
        Table with ``dra``, ``ddec``, ``flux``, ``sep``, and ``pa`` columns.
    """
    dra = np.asarray(samples[dra_key], dtype=float)
    ddec = np.asarray(samples[ddec_key], dtype=float)
    flux_raw = np.asarray(samples[flux_key], dtype=float)
    flux = np.power(10.0, flux_raw) if log10_flux else flux_raw

    df = pd.DataFrame({"dra": dra, "ddec": ddec, "flux": flux})
    df["sep"] = np.sqrt(df["dra"] ** 2 + df["ddec"] ** 2)
    df["pa"] = (np.degrees(np.arctan2(df["ddec"], df["dra"])) + 360.0) % 360.0
    return df


def truth_cartesian_and_polar(truth):
    """
    Return truth mappings for shared ChainConsumer Cartesian and polar interfaces.

    Parameters
    ----------
    truth : dict
        Mapping with ``dra``, ``ddec``, and ``flux`` values.

    Returns
    -------
    tuple[dict, dict]
        Cartesian and polar truth dictionaries.
    """
    truth_cart = {
        "dra": float(truth["dra"]),
        "ddec": float(truth["ddec"]),
        "flux": float(truth["flux"]),
    }
    truth_polar = {
        "sep": float(
            np.sqrt(truth_cart["dra"] ** 2 + truth_cart["ddec"] ** 2)
        ),
        "pa": float(
            (
                np.degrees(np.arctan2(truth_cart["ddec"], truth_cart["dra"]))
                + 360.0
            )
            % 360.0
        ),
        "flux": truth_cart["flux"],
    }
    return truth_cart, truth_polar


def plot_hmc_fisher_chainconsumer(
    hmc_table,
    fisher_table,
    truth_cartesian,
    colors=("#1f77b4", "#ff7f0e"),
):
    """
    Plot paired Cartesian and polar ChainConsumer diagnostics for HMC and Fisher-HMC.

    Parameters
    ----------
    hmc_table : pandas.DataFrame
        Posterior table for vanilla HMC.
    fisher_table : pandas.DataFrame
        Posterior table for Fisher-reparameterized HMC.
    truth_cartesian : dict
        Truth mapping in Cartesian coordinates.
    colors : tuple[str, str], optional
        Colors used for HMC and Fisher-HMC chains.

    Returns
    -------
    dict
        Mapping containing configured ChainConsumer objects and truth mappings.
    """
    truth_cart, truth_polar = truth_cartesian_and_polar(truth_cartesian)

    cartesian_consumer = plot_chainconsumer_diagnostics(
        {
            "HMC Cartesian": hmc_table,
            "Fisher-HMC Cartesian": fisher_table,
        },
        columns=["dra", "ddec", "flux"],
        truth=truth_cart,
        colors=list(colors),
    )

    polar_consumer = plot_chainconsumer_diagnostics(
        {
            "HMC Polar": hmc_table,
            "Fisher-HMC Polar": fisher_table,
        },
        columns=["sep", "pa", "flux"],
        truth=truth_polar,
        colors=list(colors),
    )

    return {
        "cartesian": cartesian_consumer,
        "polar": polar_consumer,
        "truth_cartesian": truth_cart,
        "truth_polar": truth_polar,
    }


def plot_recovery_residuals(
    params,
    truth,
    estimates_by_label,
    std_by_label,
    figsize=(8, 4),
):
    """
    Plot parameter recovery and normalized residuals for multiple estimators.

    Parameters
    ----------
    params : list[str]
        Parameter names.
    truth : array-like
        Truth values in the same order as ``params``.
    estimates_by_label : dict
        Mapping of label to posterior median arrays.
    std_by_label : dict
        Mapping of label to posterior standard-deviation arrays.
    figsize : tuple, optional
        Base figure size.

    Returns
    -------
    tuple
        ``((fig1, ax1), (fig2, ax2))`` for recovery and residual panels.
    """
    x = np.arange(len(params))
    labels = list(estimates_by_label.keys())
    n_labels = max(1, len(labels))
    offsets = np.linspace(-0.3, 0.3, n_labels)

    fig1, ax1 = plt.subplots(figsize=figsize)
    for idx, label in enumerate(labels):
        ax1.errorbar(
            x + offsets[idx],
            np.asarray(estimates_by_label[label], dtype=float),
            yerr=np.asarray(std_by_label[label], dtype=float),
            fmt="o",
            capsize=4,
            label=label,
        )
    ax1.scatter(
        x,
        np.asarray(truth, dtype=float),
        marker="x",
        s=80,
        linewidths=2,
        label="Truth",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(params)
    ax1.set_title("Synthetic recovery: truth vs posterior medians")
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(figsize[0], 3.5))
    ax2.axhline(0.0, color="k", lw=1)
    ax2.axhline(2.0, color="gray", lw=1, ls="--")
    ax2.axhline(-2.0, color="gray", lw=1, ls="--")
    for label in labels:
        residual = (
            np.asarray(estimates_by_label[label], dtype=float)
            - np.asarray(truth, dtype=float)
        ) / np.maximum(np.asarray(std_by_label[label], dtype=float), 1e-12)
        ax2.plot(x, residual, "o-", label=f"{label} z-residual")
    ax2.set_xticks(x)
    ax2.set_xticklabels(params)
    ax2.set_ylabel("(estimate - truth) / σ")
    ax2.set_title("Normalized recovery residuals")
    ax2.legend()
    fig2.tight_layout()
    return (fig1, ax1), (fig2, ax2)


def radial_limit_summary(
    limit_map,
    dra_axis,
    ddec_axis,
    center=(0.0, 0.0),
    r_max=350.0,
    n_bins=20,
):
    """
    Compute radial median and percentile bands for a 2D contrast-limit map.

    Parameters
    ----------
    limit_map : array-like
        Two-dimensional map of contrast limits.
    dra_axis : array-like
        Right-ascension axis in milliarcseconds.
    ddec_axis : array-like
        Declination axis in milliarcseconds.
    center : tuple[float, float], optional
        Radial center in milliarcseconds.
    r_max : float, optional
        Maximum radial separation to summarize.
    n_bins : int, optional
        Number of radial edges (bins are ``n_bins - 1``).

    Returns
    -------
    dict
        Mapping with ``r_centers``, ``median``, ``q16``, and ``q84`` arrays.
    """
    yy, xx = np.meshgrid(
        np.asarray(ddec_axis), np.asarray(dra_axis), indexing="xy"
    )
    rr = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)

    r_edges = np.linspace(0.0, float(r_max), int(n_bins))
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    med = []
    q16 = []
    q84 = []
    limit_np = np.asarray(limit_map)

    for lo, hi in zip(r_edges[:-1], r_edges[1:]):
        mask = (rr >= lo) & (rr < hi)
        vals = limit_np[mask]
        if vals.size == 0:
            med.append(np.nan)
            q16.append(np.nan)
            q84.append(np.nan)
        else:
            med.append(np.nanmedian(vals))
            q16.append(np.nanpercentile(vals, 16))
            q84.append(np.nanpercentile(vals, 84))

    return {
        "r_centers": np.asarray(r_centers),
        "median": np.asarray(med),
        "q16": np.asarray(q16),
        "q84": np.asarray(q84),
    }


def plot_contrast_limit_map(
    limit_map,
    dra_axis,
    ddec_axis,
    truth=None,
    unit_mode="flux_ratio",
    title="Contrast-limit map",
    cmap="inferno",
    figsize=(8, 6),
):
    """
    Plot a 2D contrast-limit map in flux-ratio or Δmag units.

    Parameters
    ----------
    limit_map : array-like
        Two-dimensional contrast-limit map.
    dra_axis : array-like
        Right-ascension axis in milliarcseconds.
    ddec_axis : array-like
        Declination axis in milliarcseconds.
    truth : dict or tuple, optional
        Truth location to overplot.
    unit_mode : {"flux_ratio", "delta_mag"}, optional
        Display units for the map.
    title : str, optional
        Axes title.
    cmap : str, optional
        Matplotlib colormap name.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    limit_np = np.asarray(limit_map)
    cmap_to_use = cmap
    if unit_mode == "delta_mag":
        map_to_plot = -2.5 * np.log10(np.maximum(limit_np, 1e-30))
        cbar_label = "Contrast limit (Δmag)"
        if not str(cmap_to_use).endswith("_r"):
            cmap_to_use = f"{cmap_to_use}_r"
    else:
        map_to_plot = limit_np
        cbar_label = "Contrast limit (flux ratio)"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        map_to_plot.T,
        extent=(
            float(np.asarray(dra_axis).min()),
            float(np.asarray(dra_axis).max()),
            float(np.asarray(ddec_axis).min()),
            float(np.asarray(ddec_axis).max()),
        ),
        origin="lower",
        aspect="equal",
        cmap=cmap_to_use,
    )
    fig.colorbar(im, ax=ax, label=cbar_label)

    if truth is not None:
        if isinstance(truth, dict):
            dra_truth = float(truth["dra"])
            ddec_truth = float(truth["ddec"])
        else:
            dra_truth = float(truth[0])
            ddec_truth = float(truth[1])
        ax.scatter(
            [dra_truth],
            [ddec_truth],
            marker="x",
            s=80,
            c="white",
            linewidths=2,
            label="Truth",
        )
        ax.legend(loc="upper right")
    ax.set_xlabel("ΔRA (mas)")
    ax.set_ylabel("ΔDec (mas)")
    ax.set_title(title)
    return fig, ax


def plot_radial_limit_summary(
    radial_summary,
    unit_mode="flux_ratio",
    title="Radial limit summary",
    figsize=(8, 4),
    ax=None,
):
    """
    Plot radial median and percentile spread from ``radial_limit_summary`` output.

    Parameters
    ----------
    radial_summary : dict
        Output mapping from ``radial_limit_summary``.
    unit_mode : {"flux_ratio", "delta_mag"}, optional
        Display units for the y-axis.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If omitted, create a new figure and axes.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    r_centers = np.asarray(radial_summary["r_centers"])
    med = np.asarray(radial_summary["median"])
    q16 = np.asarray(radial_summary["q16"])
    q84 = np.asarray(radial_summary["q84"])

    if unit_mode == "delta_mag":
        med_plot = -2.5 * np.log10(np.maximum(med, 1e-30))
        q16_plot = -2.5 * np.log10(np.maximum(q16, 1e-30))
        q84_plot = -2.5 * np.log10(np.maximum(q84, 1e-30))
        ylabel = "Contrast limit (Δmag)"
    else:
        med_plot = med
        q16_plot = q16
        q84_plot = q84
        ylabel = "Contrast limit (flux ratio)"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot(r_centers, med_plot, lw=2, label="Median")
    ax.fill_between(r_centers, q16_plot, q84_plot, alpha=0.3, label="16–84%")
    if unit_mode == "flux_ratio":
        ax.set_yscale("log")
    ax.set_xlabel("Separation (mas)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_optimized_and_grid(loglike_im, optimized, samples_dict):
    """
    Plot optimized contrast results alongside the brute-force grid maximum.

    Parameters
    ----------
    loglike_im : array
        Full 3D log-likelihood cube from ``likelihood_grid``.
    optimized : array
        2D optimized contrast map from ``optimized_contrast_grid``.
    samples_dict : dict
        Sampling dictionary containing ``dra``, ``ddec``, and ``flux`` axes.
    """

    best_contrast_indices = np.argmax(loglike_im, axis=2)
    best_contrasts = samples_dict["flux"][best_contrast_indices]

    plt.figure(figsize=(14, 5))
    matplotlib.rcParams["figure.dpi"] = 100
    matplotlib.rcParams["font.family"] = ["serif"]
    plt.rcParams.update({"font.size": 14})
    plt.subplot(1, 2, 1)
    plt.imshow(
        optimized.T,
        cmap="inferno",
        norm=matplotlib.colors.LogNorm(),
        extent=[
            samples_dict["dra"].max(),
            samples_dict[
                "dra"
            ].min(),  # this may seem weird, but left is more RA and up is more Dec
            samples_dict["ddec"].max(),
            samples_dict["ddec"].min(),
        ],
    )  # this took me far too long to get the sign right for
    plt.colorbar(shrink=1, label="Contrast", pad=0.01)
    plt.scatter(0, 0, s=140, c="black", marker="*")
    plt.xlabel("$\\Delta$RA [mas]")
    plt.ylabel("$\\Delta$DEC [mas]")
    plt.title("Optimization")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.imshow(
        best_contrasts.T,
        cmap="inferno",
        norm=matplotlib.colors.LogNorm(),
        extent=[
            samples_dict["dra"].max(),
            samples_dict[
                "dra"
            ].min(),  # this may seem weird, but left is more RA and up is more Dec
            samples_dict["ddec"].max(),
            samples_dict["ddec"].min(),
        ],
    )  # this took me far too long to get the sign right for
    plt.colorbar(shrink=1, label="Contrast", pad=0.01)
    plt.scatter(0, 0, s=140, c="black", marker="*")
    plt.xlabel("$\\Delta$RA [mas]")
    plt.ylabel("$\\Delta$DEC [mas]")
    plt.title("Grid Search")
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0.0)
    plt.show()


def plot_optimized_and_sigma(contrast, sigma_grid, samples_dict, snr=False):
    """
    Plot the results of an optimized contrast grid calculation and the corresponding uncertainty grid.

    Parameters
    ----------
    contrast : array
        The optimized contrast grid, output of optimized_contrast_grid
    sigma_grid : array
        The uncertainty grid, output of laplace_contrast_uncertainty_grid
    samples_dict : dict
        Dictionary of samples used in the grid calculation
    snr : bool, optional
        If True, plot the SNR instead of the uncertainty, default False

    """

    plt.figure(figsize=(14, 5))
    matplotlib.rcParams["figure.dpi"] = 100
    matplotlib.rcParams["font.family"] = ["serif"]
    plt.rcParams.update({"font.size": 14})
    plt.subplot(1, 2, 1)
    plt.imshow(
        contrast.T,
        cmap="inferno",
        norm=matplotlib.colors.LogNorm(),
        extent=[
            samples_dict["dra"].max(),
            samples_dict[
                "dra"
            ].min(),  # this may seem weird, but left is more RA and up is more Dec
            samples_dict["ddec"].max(),
            samples_dict["ddec"].min(),
        ],
    )  # this took me far too long to get the sign right for
    plt.colorbar(shrink=1, label="Contrast", pad=0.01)
    plt.scatter(0, 0, s=140, c="y", marker="*")
    plt.xlabel("$\\Delta$RA [mas]")
    plt.ylabel("$\\Delta$DEC [mas]")
    plt.title("Contrast")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    if snr:
        plt.imshow(
            contrast.T / sigma_grid.T,
            cmap="inferno",
            norm=matplotlib.colors.PowerNorm(1),
            extent=[
                samples_dict["dra"].max(),
                samples_dict[
                    "dra"
                ].min(),  # this may seem weird, but left is more RA and up is more Dec
                samples_dict["ddec"].max(),
                samples_dict["ddec"].min(),
            ],
        )  # this took me far too long to get the sign right for
        plt.colorbar(shrink=1, label="SNR", pad=0.01)
        plt.scatter(0, 0, s=140, c="y", marker="*")  # mark star at origin
        plt.title("SNR")

    else:
        plt.imshow(
            sigma_grid.T,
            cmap="inferno",
            norm=matplotlib.colors.LogNorm(),
            extent=[
                samples_dict["dra"].max(),
                samples_dict[
                    "dra"
                ].min(),  # this may seem weird, but left is more RA and up is more Dec
                samples_dict["ddec"].max(),
                samples_dict["ddec"].min(),
            ],
        )  # this took me far too long to get the sign right for
        plt.colorbar(shrink=1, label="σ(Contrast)", pad=0.01)
        plt.scatter(0, 0, s=140, c="y", marker="*")  # mark star at origin
        plt.title("σ(Contrast)")

    plt.xlabel("$\\Delta$RA [mas]")
    plt.ylabel("$\\Delta$DEC [mas]")
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0.0)
    plt.show()


def plot_contrast_limits(
    contrast_limits,
    samples_dict,
    rad_width,
    avg_width,
    std_width,
    true_values=None,
    limit_label="98% Upper Limit",
):
    """
    Plot the contrast limits calculated with the Ruffio or Absil methods.

    Parameters
    ----------
    contrast_limits : array
        The contrast limits calculated with the Ruffio or Absil methods.
    samples_dict : dict
        Dictionary of samples used in the grid calculation
    rad_width : array
        Radial width of the contrast limits.
    avg_width : array
        Average width of the contrast limits.
    std_width : array
        Standard deviation of the contrast limits.
    true_values : list, optional
        List of true values for the parameters, default None
    limit_label : str, optional
        Label used for the map title and curve legend.

    """

    plt.figure(figsize=(20, 5))
    matplotlib.rcParams["figure.dpi"] = 150
    matplotlib.rcParams["font.family"] = ["serif"]
    plt.rcParams.update({"font.size": 16})

    # first show x% upper limit map

    plt.subplot(1, 2, 1)
    plt.imshow(
        -2.5 * np.log10(contrast_limits[:, :].T),
        cmap=matplotlib.colormaps["magma_r"],
        extent=[
            samples_dict["dra"].max(),
            samples_dict[
                "dra"
            ].min(),  # this may seem weird, but left is more RA and up is more Dec
            samples_dict["ddec"].max(),
            samples_dict["ddec"].min(),
        ],
    )  # this took me far too long to get the sign right for
    plt.colorbar(shrink=1, pad=0.01)
    plt.scatter(0, 0, marker="*", s=100, c="black", alpha=0.5)
    plt.gca().invert_yaxis()
    plt.title(f"{limit_label} Map ($\\Delta$mag)")
    plt.xlabel("$\\Delta$RA [mas]")
    plt.ylabel("$\\Delta$DEC [mas]")

    # then show contrast curve including detected target
    plt.subplot(1, 2, 2)
    dx = np.abs(np.median(np.diff(samples_dict["dra"])))
    plt.plot(rad_width * dx, avg_width, "-k", label=limit_label)
    plt.fill_between(
        rad_width * dx,
        avg_width - std_width,
        avg_width + std_width,
        color=(0.6, 0.4, 0.9),
        alpha=0.3,
    )
    plt.ylabel("Contrast ($\\Delta$mag)")
    plt.xlabel("Separation [mas]")
    plt.gca().invert_yaxis()
    plt.xlim(
        np.nanmin(rad_width * dx + avg_width * 0.0),
        np.nanmax(rad_width * dx + avg_width * 0.0),
    )
    plt.grid(color="black", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout(pad=0.0)

    if true_values is not None:
        true_dra, true_ddec, true_contrast = true_values
        plt.plot(
            np.sqrt(true_dra**2 + true_ddec**2),
            -2.5 * np.log10(true_contrast),
            marker="*",
            c="k",
            markersize=15,
        )  # detected value
