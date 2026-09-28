"""Plotting functions to render outputs from the model fitting functions.

Sky images follow the package convention: East (positive ``dra``) to the
left and North (positive ``ddec``) up. Importing this module does not change
matplotlib's global settings; call [`set_style`][drpangloss.plotting.set_style] to opt in to the
drpangloss look for every figure.
"""

import contextlib
import functools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


STYLE = {
    "figure.dpi": 100,
    "font.family": ["serif"],
    "font.size": 14,
}


def set_style():
    """Apply the drpangloss matplotlib style globally.

    The plotting functions in this module already use this style for the
    figures they create, without touching global settings. Call this to use
    it for your own figures too.
    """
    matplotlib.rcParams.update(STYLE)


@contextlib.contextmanager
def _style_context(style=None):
    """Temporarily apply ``style`` (default ``STYLE``) to ``rcParams``.

    Only the keys in ``style`` are saved and restored. ``plt.rc_context``
    restores every key, including ``backend``, and in Jupyter that can
    switch backends on exit, which closes the open figures before they are
    shown.
    """
    style = STYLE if style is None else style
    saved = {key: matplotlib.rcParams[key] for key in style}
    matplotlib.rcParams.update(style)
    try:
        yield
    finally:
        matplotlib.rcParams.update(saved)


def _styled(fn):
    """Run ``fn`` with the drpangloss style applied."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _style_context():
            return fn(*args, **kwargs)

    return wrapper


def _centres_to_extent(x_axis, y_axis):
    """Return an imshow ``extent`` whose pixels are centred on the samples.

    ``imshow`` places ``extent`` at the outer pixel *edges*, so grid sample
    positions (pixel centres) are padded by half a sample spacing.
    """

    def _edges(axis):
        axis = np.asarray(axis, dtype=float).reshape(-1)
        half = 0.5 * (axis[1] - axis[0]) if axis.size > 1 else 0.5
        return float(axis[0] - half), float(axis[-1] + half)

    return [*_edges(x_axis), *_edges(y_axis)]


def _flux_key(samples_dict, flux_param=None):
    """Key of ``samples_dict`` that is optimized out rather than plotted."""
    if flux_param is not None:
        if flux_param not in samples_dict:
            raise KeyError(
                f"flux_param {flux_param!r} is not a key of samples_dict."
            )
        return flux_param
    for key in ("contrast", "flux"):
        if key in samples_dict:
            return key
    return list(samples_dict)[-1]


def _is_sky_key(key, name):
    return key == name or key.endswith("." + name)


def _image_axes(grid, coord_keys):
    """Pick the plotted x/y keys and orient ``grid`` as ``image[y, x]``.

    ``grid`` has one axis per entry of ``coord_keys``, in that order. If the
    keys include a ``dra`` and a ``ddec`` (possibly as zodiax paths such as
    ``"comp.dra"``), those are used as x and y whatever their order.
    Otherwise the first two keys are used. Returns
    ``(x_key, y_key, image, is_sky)``.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or len(coord_keys) != 2:
        raise ValueError(
            f"Expected a 2D grid with two coordinate keys; got shape "
            f"{grid.shape} for keys {list(coord_keys)}. Reduce the flux axis "
            "first, e.g. loglike_im.max(axis=...)."
        )
    dra = [k for k in coord_keys if _is_sky_key(k, "dra")]
    ddec = [k for k in coord_keys if _is_sky_key(k, "ddec")]
    if len(dra) == 1 and len(ddec) == 1:
        x_key, y_key = dra[0], ddec[0]
        is_sky = True
    else:
        x_key, y_key = coord_keys
        is_sky = False
    if list(coord_keys).index(x_key) == 0:
        image = grid.T
    else:
        image = grid
    return x_key, y_key, image, is_sky


def _to_delta_mag(flux_ratio):
    """Convert a flux ratio to Δmag, flooring non-positive values."""
    return -2.5 * np.log10(np.maximum(np.asarray(flux_ratio), 1e-30))


def _range_aware_float_formatter(
    vmin, vmax, min_sigfigs=3, max_sigfigs=6, scale=1.0
):
    """Tick formatter whose significant figures adapt to the axis span."""
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


def _enforce_sky_orientation(ax):
    """Ensure the displayed x-axis increases toward the left (East) and
    the y-axis increases toward the top (North), matching drpangloss's
    image coordinate convention. Works regardless of
    whether the plotted array's ``dra``/``ddec`` axis was built ascending
    or descending, since it corrects the axes' final displayed limits
    rather than assuming any particular extent/origin construction.
    """
    if ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()
    if ax.get_ylim()[0] > ax.get_ylim()[1]:
        ax.invert_yaxis()


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


@_styled
def plot_data_model_correlation(
    oidata,
    predictions_by_label,
    colors=None,
    figsize=(10, 5),
    phase_title="Phase correlation",
    square_axes=True,
    vis_label=None,
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
    square_axes : bool, optional
        If ``True``, enforce square panel boxes for both subplots.
    vis_label : str, optional
        Axis label for the visibility observable. By default it follows
        ``oidata.vis_mode``: V² and amplitudes are shown in percent,
        log-amplitudes and projected (e.g. DISCO) observables as plain
        values.

    Returns
    -------
    tuple
        ``(fig, (ax1, ax2))`` for visibility and phase axes.
    """
    vis_mode = getattr(oidata, "vis_mode", "v2")
    projected = getattr(oidata, "vis_mat", None) is not None
    if projected or vis_mode == "logamp":
        vis_scale = 1.0
        default_label = "projected" if projected else "log amplitude"
    else:
        vis_scale = 100.0
        default_label = "V2, %" if vis_mode == "v2" else "amplitude, %"
    if vis_label is None:
        vis_label = default_label
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
            markersize=4,
            alpha=0.45,
            elinewidth=1.0,
            capsize=2,
            color=color,
            label=label,
        )
        ax2.errorbar(
            phi_data,
            np.asarray(pred["phi_mean"]).reshape(-1),
            xerr=d_phi_data,
            yerr=np.asarray(pred["phi_std"]).reshape(-1),
            fmt="o",
            markersize=4,
            alpha=0.45,
            elinewidth=1.0,
            capsize=2,
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
    vis_formatter = _range_aware_float_formatter(
        vis_min, vis_max, scale=vis_scale
    )
    phi_formatter = _range_aware_float_formatter(phi_min, phi_max)

    ax1.plot(vis_line, vis_line, "k--", lw=1)
    ax1.set_xlim(vis_min, vis_max)
    ax1.set_ylim(vis_min, vis_max)
    ax1.xaxis.set_major_formatter(vis_formatter)
    ax1.yaxis.set_major_formatter(vis_formatter)
    ax1.set_xlabel(f"Data ({vis_label})")
    ax1.set_ylabel(f"Model ({vis_label})")
    ax1.set_title("Visibility correlation")
    if square_axes:
        ax1.set_box_aspect(1)
    ax1.legend(loc="best")

    ax2.plot(phi_line, phi_line, "k--", lw=1)
    ax2.set_xlim(phi_min, phi_max)
    ax2.set_ylim(phi_min, phi_max)
    ax2.xaxis.set_major_formatter(phi_formatter)
    ax2.yaxis.set_major_formatter(phi_formatter)
    ax2.set_xlabel("Data (rad)")
    ax2.set_ylabel("Model (rad)")
    ax2.set_title(phase_title)
    if square_axes:
        ax2.set_box_aspect(1)
    ax2.legend(loc="best")

    fig.tight_layout()
    return fig, (ax1, ax2)


@_styled
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
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


@_styled
def plot_model(
    model,
    fov_mas,
    npix=256,
    ax=None,
    title=None,
    saturate=None,
    cmap="magma",
):
    """Show a source model's rendered image with sky axes.

    East is to the left and North is up, matching the orientation of
    [`render`][drpangloss.models.SourceModel.render].

    Parameters
    ----------
    model : SourceModel
        Model to render.
    fov_mas : float
        Width of the field of view in milliarcseconds.
    npix : int, optional
        Number of pixels on a side.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    title : str, optional
        Axes title.
    saturate : float, optional
        Quantile (e.g. ``0.99``) at which to saturate the colour scale, so
        that faint structure next to a bright star is visible.
    cmap : str, optional
        Matplotlib colour map.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    image = np.asarray(model.render(npix=npix, fov_mas=fov_mas))
    vmax = None if saturate is None else np.quantile(image, saturate)
    half = float(fov_mas) / 2.0
    ax.imshow(
        image,
        extent=[half, -half, -half, half],
        origin="upper",
        cmap=cmap,
        vmax=vmax,
    )
    ax.set(xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
    if title is not None:
        ax.set_title(title)
    _enforce_sky_orientation(ax)
    return ax


def _coord_value(values, key, coord_keys):
    """Look up ``key`` in a dict, or by position in ``coord_keys`` order."""
    if isinstance(values, dict):
        return float(values[key])
    return float(values[list(coord_keys).index(key)])


@_styled
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
    flux_param=None,
):
    """
    Plot a likelihood map (or profile) over the coordinate axes of a grid.

    Parameters
    ----------
    loglike_im : array
        Log likelihood with one axis per coordinate key of ``samples_dict``
        (every key except the flux key), in key order. Pass the output of
        [`drpangloss.grid_fit.optimized_likelihood_grid`][drpangloss.grid_fit.optimized_likelihood_grid], or reduce the
        flux axis of [`drpangloss.grid_fit.likelihood_grid`][drpangloss.grid_fit.likelihood_grid] first, e.g.
        ``loglike_im.max(axis=2)``.
    samples_dict : dict
        Dictionary of samples used in the grid calculation.
    truths : dict or sequence, optional
        True coordinate values to mark. A dict is looked up by key; a
        sequence is read in the coordinate-key order of ``samples_dict``.
    best_point : dict or sequence, optional
        Best-fit coordinates to mark, in the same form as ``truths``.
    truth_label, best_label : str, optional
        Legend labels for the two markers.
    colorbar_label : str, optional
        Label of the colour bar (or of the y-axis for a 1D profile).
    cmap : str, optional
        Matplotlib colour map.
    figsize : tuple, optional
        Figure size.
    flux_param : str, optional
        Key of ``samples_dict`` that is not a plotted coordinate (e.g.
        ``"comp.flux"``). By default ``"contrast"`` or ``"flux"`` if present,
        otherwise the last key.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Notes
    -----
    When the coordinates include ``dra`` and ``ddec`` (also as paths such as
    ``"comp.dra"``) they are drawn as x and y, East-left and North-up,
    whatever their order in ``samples_dict``.
    """

    params = list(samples_dict.keys())
    opt_key = _flux_key(samples_dict, flux_param)
    coord_keys = [k for k in params if k != opt_key]

    grid = np.asarray(loglike_im)

    if grid.ndim == 1 or len(coord_keys) == 1:
        x_key = coord_keys[0] if coord_keys else params[0]
        x_axis = np.asarray(samples_dict[x_key])
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_axis, grid.reshape(-1), color="C0", lw=2)

        if truths is not None:
            x_truth = _coord_value(truths, x_key, [x_key])
            ax.axvline(x_truth, color="k", lw=1.5, ls="--", label=truth_label)

        if best_point is not None:
            x_best = _coord_value(best_point, x_key, [x_key])
            ax.axvline(x_best, color="C1", lw=1.2, ls=":", label=best_label)

        ax.set_xlabel(x_key)
        ax.set_ylabel(colorbar_label)
        ax.set_title("Likelihood profile")
        if truths is not None or best_point is not None:
            ax.legend(loc="best")
        return fig, ax

    x_key, y_key, image, is_sky = _image_axes(grid, coord_keys)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        image,
        cmap=cmap,
        origin="lower",
        aspect="equal" if is_sky else "auto",
        extent=_centres_to_extent(samples_dict[x_key], samples_dict[y_key]),
    )
    fig.colorbar(im, ax=ax, shrink=0.9, label=colorbar_label, pad=0.01)

    if truths is not None:
        ax.scatter(
            [_coord_value(truths, x_key, coord_keys)],
            [_coord_value(truths, y_key, coord_keys)],
            marker="x",
            s=80,
            c="white",
            linewidths=2,
            label=truth_label,
        )

    if best_point is not None:
        ax.scatter(
            [_coord_value(best_point, x_key, coord_keys)],
            [_coord_value(best_point, y_key, coord_keys)],
            marker="o",
            s=40,
            facecolors="none",
            edgecolors="cyan",
            label=best_label,
        )

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title("Likelihood grid")
    if truths is not None or best_point is not None:
        ax.legend(loc="best")
    if is_sky:
        _enforce_sky_orientation(ax)
    return fig, ax


@_styled
def plot_chainconsumer_diagnostics(
    chains_by_label,
    columns,
    truth=None,
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
    truth : dict, optional
        Truth mapping for columns displayed. When omitted, no truth overlay is
        drawn.
    colors : list[str], optional
        Per-chain colors.
    walk_columns : list[str], optional
        Columns to show in walk plots. Defaults to ``columns``.

    Returns
    -------
    tuple
        ``(consumer, corner_fig, walks_fig)``: the configured ChainConsumer
        instance and the two figures it drew.
    """
    from chainconsumer import ChainConsumer, Chain, Truth

    if colors is None:
        colors = matplotlib.colormaps["tab10"].colors
        colors = [matplotlib.colors.to_hex(color) for color in colors]
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
    if truth is not None:
        consumer.add_truth(Truth(location=truth))
    corner_fig = consumer.plotter.plot()
    walks_fig = consumer.plotter.plot_walks(
        columns=walk_columns, plot_weights=False, plot_posterior=False
    )
    return consumer, corner_fig, walks_fig


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
    df["pa"] = (np.degrees(np.arctan2(df["dra"], df["ddec"])) + 360.0) % 360.0
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
                np.degrees(np.arctan2(truth_cart["dra"], truth_cart["ddec"]))
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
        Mapping with the ChainConsumer objects (``"cartesian"``,
        ``"polar"``), their figures (``"cartesian_figures"``,
        ``"polar_figures"``, each ``(corner, walks)``) and the truth
        mappings.
    """
    truth_cart, truth_polar = truth_cartesian_and_polar(truth_cartesian)

    cartesian_consumer, *cartesian_figures = plot_chainconsumer_diagnostics(
        {
            "HMC Cartesian": hmc_table,
            "Fisher-HMC Cartesian": fisher_table,
        },
        columns=["dra", "ddec", "flux"],
        truth=truth_cart,
        colors=list(colors),
    )

    polar_consumer, *polar_figures = plot_chainconsumer_diagnostics(
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
        "cartesian_figures": tuple(cartesian_figures),
        "polar_figures": tuple(polar_figures),
        "truth_cartesian": truth_cart,
        "truth_polar": truth_polar,
    }


@_styled
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
    offsets = np.linspace(-0.3, 0.3, n_labels) if n_labels > 1 else np.zeros(1)

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
        Two-dimensional map of contrast limits with shape
        ``(len(dra_axis), len(ddec_axis))``, i.e. axis 0 is ``dra``, as
        returned by the [`drpangloss.grid_fit`][drpangloss.grid_fit] functions.
    dra_axis : array-like
        Right-ascension axis in milliarcseconds.
    ddec_axis : array-like
        Declination axis in milliarcseconds.
    center : tuple[float, float], optional
        Radial centre ``(dra, ddec)`` in milliarcseconds.
    r_max : float, optional
        Maximum radial separation to summarize, in milliarcseconds.
    n_bins : int, optional
        Number of radial bin *edges* (there are ``n_bins - 1`` bins).

    Returns
    -------
    dict
        Mapping with ``r_centers`` (mas), ``median``, ``q16``, and ``q84``
        arrays. Empty bins are NaN.
    """
    xx, yy = np.meshgrid(
        np.asarray(dra_axis), np.asarray(ddec_axis), indexing="ij"
    )
    limit_np = np.asarray(limit_map)
    if limit_np.shape != xx.shape:
        raise ValueError(
            f"limit_map has shape {limit_np.shape}; expected "
            f"(len(dra_axis), len(ddec_axis)) = {xx.shape}."
        )
    rr = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)

    r_edges = np.linspace(0.0, float(r_max), int(n_bins))
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    med = []
    q16 = []
    q84 = []

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


def _reversed_cmap(cmap):
    """Reverse a colour map given by name or as a ``Colormap``.

    Maps whose name already ends in ``_r`` are taken as deliberately
    reversed and kept.
    """
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    if cmap.name.endswith("_r"):
        return cmap
    return cmap.reversed()


def _sky_map(ax, grid, samples_dict, coord_keys, **imshow_kwargs):
    """``imshow`` a coordinate grid with centred pixels and sky orientation.

    Returns the image and whether the axes are sky offsets.
    """
    x_key, y_key, image, is_sky = _image_axes(grid, coord_keys)
    im = ax.imshow(
        image,
        origin="lower",
        extent=_centres_to_extent(samples_dict[x_key], samples_dict[y_key]),
        **imshow_kwargs,
    )
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    if is_sky:
        _enforce_sky_orientation(ax)
    return im, is_sky


@_styled
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
        Two-dimensional contrast-limit map (flux ratios) with shape
        ``(len(dra_axis), len(ddec_axis))``.
    dra_axis : array-like
        Right-ascension axis in milliarcseconds.
    ddec_axis : array-like
        Declination axis in milliarcseconds.
    truth : dict or tuple, optional
        Truth location ``(dra, ddec)`` in milliarcseconds to overplot.
    unit_mode : {"flux_ratio", "delta_mag"}, optional
        Display units for the map.
    title : str, optional
        Axes title.
    cmap : str or matplotlib.colors.Colormap, optional
        Colour map. It is reversed for ``"delta_mag"``, so that deeper
        limits keep the same colour in both modes.
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
        map_to_plot = _to_delta_mag(limit_np)
        cbar_label = "Contrast limit (Δmag)"
        cmap_to_use = _reversed_cmap(cmap)
    else:
        map_to_plot = limit_np
        cbar_label = "Contrast limit (flux ratio)"

    fig, ax = plt.subplots(figsize=figsize)
    im, _ = _sky_map(
        ax,
        map_to_plot,
        {"dra": dra_axis, "ddec": ddec_axis},
        ["dra", "ddec"],
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


@_styled
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
        Display units for the y-axis. In both modes deeper (fainter) limits
        are drawn lower down: flux ratios on a log axis, Δmag on an inverted
        axis.
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
        med_plot = _to_delta_mag(med)
        q16_plot = _to_delta_mag(q16)
        q84_plot = _to_delta_mag(q84)
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
    elif not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("Separation (mas)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig, ax


@_styled
def plot_optimized_and_grid(
    loglike_im, optimized, samples_dict, flux_param=None
):
    """
    Plot optimized contrast results alongside the brute-force grid maximum.

    Parameters
    ----------
    loglike_im : array
        Full log-likelihood cube from ``likelihood_grid``, with one axis per
        key of ``samples_dict``.
    optimized : array
        Optimized flux map from ``optimized_contrast_grid``, with one axis
        per coordinate key.
    samples_dict : dict
        Sampling dictionary used for both grids, e.g. with ``dra``, ``ddec``
        and ``flux`` axes.
    flux_param : str, optional
        Key of the optimized flux axis. By default ``"contrast"`` or
        ``"flux"`` if present, otherwise the last key.

    Returns
    -------
    tuple
        ``(fig, ax)`` for a 1D profile, or ``(fig, (ax1, ax2))`` for 2D maps.
    """

    params = list(samples_dict.keys())
    opt_key = _flux_key(samples_dict, flux_param)
    coord_keys = [k for k in params if k != opt_key]
    flux_values = np.asarray(samples_dict[opt_key])
    best_idx = np.nanargmax(np.asarray(loglike_im), axis=params.index(opt_key))
    best_grid = flux_values[best_idx]

    if len(coord_keys) == 1:
        x_key = coord_keys[0]
        x = np.asarray(samples_dict[x_key])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            x,
            np.asarray(optimized).reshape(-1),
            lw=2,
            label=f"Optimized {opt_key}",
        )
        ax.plot(
            x,
            np.asarray(best_grid).reshape(-1),
            lw=1.5,
            ls="--",
            label=f"Grid {opt_key}",
        )
        ax.set_xlabel(x_key)
        ax.set_ylabel(opt_key)
        ax.set_title("Optimization vs grid")
        ax.legend(loc="best")
        return fig, ax

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, grid, title in (
        (ax1, optimized, "Optimization"),
        (ax2, best_grid, "Grid Search"),
    ):
        im, is_sky = _sky_map(
            ax,
            grid,
            samples_dict,
            coord_keys,
            cmap="inferno",
            norm=matplotlib.colors.LogNorm(),
        )
        fig.colorbar(im, ax=ax, shrink=1, label="Contrast", pad=0.01)
        if is_sky:
            ax.scatter(0, 0, s=140, c="black", marker="*")
        ax.set_title(title)
    fig.tight_layout(pad=0.0)
    return fig, (ax1, ax2)


@_styled
def plot_optimized_and_sigma(
    contrast, sigma_grid, samples_dict, snr=False, flux_param=None
):
    """
    Plot an optimized contrast grid and the corresponding uncertainty grid.

    Parameters
    ----------
    contrast : array
        The optimized contrast grid, output of optimized_contrast_grid.
    sigma_grid : array
        The uncertainty grid, output of laplace_contrast_uncertainty_grid.
    samples_dict : dict
        Dictionary of samples used in the grid calculation.
    snr : bool, optional
        If True, plot the SNR instead of the uncertainty, default False.
    flux_param : str, optional
        Key of the optimized flux axis. By default ``"contrast"`` or
        ``"flux"`` if present, otherwise the last key.

    Returns
    -------
    tuple
        ``(fig, ax)`` for a 1D profile, or ``(fig, (ax1, ax2))`` for 2D maps.
    """

    params = list(samples_dict.keys())
    opt_key = _flux_key(samples_dict, flux_param)
    coord_keys = [k for k in params if k != opt_key]
    contrast = np.asarray(contrast)
    sigma_grid = np.asarray(sigma_grid)

    if len(coord_keys) == 1:
        x_key = coord_keys[0]
        x = np.asarray(samples_dict[x_key])
        fig, ax = plt.subplots(figsize=(8, 4))
        if snr:
            ax.plot(x, (contrast / sigma_grid).reshape(-1), lw=2)
            ax.set_ylabel("SNR")
            ax.set_title("SNR profile")
        else:
            ax.plot(x, sigma_grid.reshape(-1), lw=2)
            ax.set_ylabel(f"σ({opt_key})")
            ax.set_title("Uncertainty profile")
        ax.set_xlabel(x_key)
        return fig, ax

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im, is_sky = _sky_map(
        ax1,
        contrast,
        samples_dict,
        coord_keys,
        cmap="inferno",
        norm=matplotlib.colors.LogNorm(),
    )
    fig.colorbar(im, ax=ax1, shrink=1, label="Contrast", pad=0.01)
    ax1.set_title("Contrast")

    if snr:
        right, norm, label = contrast / sigma_grid, None, "SNR"
    else:
        right, norm = sigma_grid, matplotlib.colors.LogNorm()
        label = "σ(Contrast)"
    im, _ = _sky_map(
        ax2, right, samples_dict, coord_keys, cmap="inferno", norm=norm
    )
    fig.colorbar(im, ax=ax2, shrink=1, label=label, pad=0.01)
    ax2.set_title(label)

    if is_sky:
        for ax in (ax1, ax2):
            ax.scatter(0, 0, s=140, c="y", marker="*")  # star at origin
    fig.tight_layout(pad=0.0)
    return fig, (ax1, ax2)


def _format_sigma_or_percent_value(value):
    """Format a confidence level compactly, e.g. ``97.7`` or ``5``."""
    formatted = f"{float(value):.3g}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def _resolve_contrast_limit_label(
    limit_label=None, percentile=None, sigma=None
):
    """Legend/title label for a contrast limit at a given confidence."""
    if limit_label is not None:
        return limit_label

    if percentile is not None and sigma is not None:
        raise ValueError("Provide only one of percentile or sigma.")

    if percentile is not None:
        percentile = np.asarray(percentile, dtype=float).reshape(-1)
        if percentile.size != 1:
            raise ValueError("percentile must be a scalar or length-1 array.")
        return f"{_format_sigma_or_percent_value(percentile[0] * 100)}% Upper Limit"

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        if sigma.size != 1:
            raise ValueError("sigma must be a scalar or length-1 array.")
        return f"{_format_sigma_or_percent_value(sigma[0])}$\\sigma$ Contrast Limit"

    return "98% Upper Limit"


def plot_contrast_limits(
    contrast_limits,
    samples_dict,
    rad_width,
    avg_width,
    std_width,
    true_values=None,
    limit_label=None,
    percentile=None,
    sigma=None,
):
    """
    Plot a contrast-limit map and its azimuthally averaged contrast curve.

    Parameters
    ----------
    contrast_limits : array
        Contrast limits (flux ratios) from the Ruffio or Absil methods, with
        shape ``(len(samples_dict["dra"]), len(samples_dict["ddec"]))``.
        They are shown in Δmag.
    samples_dict : dict
        Dictionary of samples used in the grid calculation; must contain
        ``"dra"`` and ``"ddec"`` axes in milliarcseconds.
    rad_width : array
        Radii of the contrast curve in *pixels* of the ``dra`` grid, as
        returned by [`drpangloss.grid_fit.azimuthalAverage`][drpangloss.grid_fit.azimuthalAverage]. They are
        converted to mas using the median ``dra`` spacing.
    avg_width : array
        Azimuthal mean of the limit map at each radius, already in Δmag.
    std_width : array
        Azimuthal standard deviation at each radius, in Δmag.
    true_values : tuple, optional
        ``(dra, ddec, flux_ratio)`` of a detected companion to mark on the
        contrast curve.
    limit_label : str, optional
        Label used for the map title and curve legend. If not provided, the
        label is inferred from ``percentile`` or ``sigma``, falling back to
        ``"98% Upper Limit"``.
    percentile : float or array-like, optional
        Percentile confidence level for Ruffio-style upper limits.
    sigma : float or array-like, optional
        Sigma confidence level for Absil-style detection limits.

    Returns
    -------
    tuple
        ``(fig, (ax_map, ax_curve))``.
    """
    limit_label = _resolve_contrast_limit_label(
        limit_label=limit_label, percentile=percentile, sigma=sigma
    )

    with _style_context({**STYLE, "figure.dpi": 150, "font.size": 16}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

        # First show the upper-limit map.
        im, _ = _sky_map(
            ax1,
            _to_delta_mag(contrast_limits),
            samples_dict,
            ["dra", "ddec"],
            cmap=matplotlib.colormaps["magma_r"],
        )
        fig.colorbar(im, ax=ax1, shrink=1, pad=0.01, label="$\\Delta$mag")
        ax1.scatter(0, 0, marker="*", s=100, c="black", alpha=0.5)
        ax1.set_title(f"{limit_label} Map ($\\Delta$mag)")
        ax1.set_xlabel("$\\Delta$RA [mas]")
        ax1.set_ylabel("$\\Delta$DEC [mas]")

        # Then show the contrast curve, including any detected target.
        dx = np.abs(np.median(np.diff(np.asarray(samples_dict["dra"]))))
        sep = np.asarray(rad_width) * dx
        avg_width = np.asarray(avg_width)
        std_width = np.asarray(std_width)
        ax2.plot(sep, avg_width, "-k", label=limit_label)
        ax2.fill_between(
            sep,
            avg_width - std_width,
            avg_width + std_width,
            color=(0.6, 0.4, 0.9),
            alpha=0.3,
        )
        ax2.set_ylabel("Contrast ($\\Delta$mag)")
        ax2.set_xlabel("Separation [mas]")
        ax2.invert_yaxis()
        finite = np.isfinite(sep) & np.isfinite(avg_width)
        if np.any(finite):
            ax2.set_xlim(sep[finite].min(), sep[finite].max())
        ax2.grid(color="black", alpha=0.3)
        ax2.legend(loc="best")

        if true_values is not None:
            true_dra, true_ddec, true_contrast = true_values
            ax2.plot(
                np.sqrt(true_dra**2 + true_ddec**2),
                _to_delta_mag(true_contrast),
                marker="*",
                c="k",
                markersize=15,
            )  # detected value
        fig.tight_layout(pad=0.0)
    return fig, (ax1, ax2)
