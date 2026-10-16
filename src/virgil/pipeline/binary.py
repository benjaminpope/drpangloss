"""`BinaryPipeline`: companion search, contrast limits, fit and posterior."""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as onp

from . import _checks, _io
from ._common import (
    _SAMPLER_SETTINGS,
    _diagnostics,
    _load_report,
    _run_chains,
    _noise_quantiles,
    _sample_stats,
    _bound_samples,
    _chi2,
    _geometry,
    _moments,
    _quantiles,
    _save,
)
from .._geometry import separation_pa
from ._core import Stage, _Pipeline
from .._deprecate import external_stacklevel

GLOBAL_NSIGMA_METHOD = "Sidak estimate, not a simulated FAP"

# Default grid step, as a fraction of the finest λ/B (Nyquist is one half).
_STEP_FRACTION = 0.25
# Relative slack before a step counts as coarsened (rounding of the odd n).
_STEP_TOLERANCE = 1e-9
# The fit starts at least this factor inside the ends of the flux prior.
_START_MARGIN = 2.0
_AXIS_ORDER = ("dra", "ddec", "flux")
_AXIS_UNITS = {"dra": "mas", "ddec": "mas", "flux": "companion/primary"}


class _Axis(NamedTuple):
    """The position axes of the search grid.

    ``max_sep`` is the half-width (mas), ``n`` the points per axis, ``step``
    the effective spacing and ``wanted`` the requested one.
    """

    max_sep: float
    n: int
    step: float
    wanted: float

    @property
    def coarsened(self):
        """Whether ``max_grid`` made the step coarser than requested."""
        return self.step > self.wanted * (1 + _STEP_TOLERANCE)


def _global_nsigma(local_nsigma, n_trials):
    """Look-elsewhere (Sidak) significance of ``local_nsigma`` in sigma.

    The global p-value of the best of ``n_trials`` independent trials is
    ``1 - (1 - p)**n_trials``; its Gaussian sigma is the quantile of the
    complement ``(1 - p)**n_trials``, which is kept in log space so that it
    stays finite when ``n_trials * p`` is large (then the p-value rounds to
    1 and the sigma is negative: no significance). NaN if ``local_nsigma``
    is not finite.
    """
    from scipy.special import ndtri_exp
    from scipy.stats import norm

    if not math.isfinite(local_nsigma):
        return float("nan")
    p_local = float(norm.sf(local_nsigma))
    sigma = float(ndtri_exp(n_trials * math.log1p(-p_local)))
    return min(sigma, local_nsigma)  # not above the local value


class BinaryPipeline(_Pipeline):
    """Search for a companion, set contrast limits, fit and sample a binary.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Stages: ``load → overview → search → limits → fit → posterior →
    quicklook``.

    * ``load`` applies ``wavel_range`` and ``error_floor`` and writes the
      data that are fitted to ``data/processed.oifits``.
    * ``overview`` plots the data and their uv coverage.
    * ``search`` runs
      [`detection_statistics`][virgil.detection.detection_statistics]
      (Δχ², log Bayes factor, best SNR) on a point-companion grid
      (``dra``, ``ddec`` in mas, log-spaced ``flux``), with maps of Δχ²,
      best flux and SNR, and the significance at the peak
      ([`local_nsigma`][virgil.detection.local_nsigma]) before and after a
      look-elsewhere correction for the number of resolution elements
      searched.
    * ``limits`` runs [`absil_limits`][virgil.limits.absil_limits] at
      ``sigma`` and its [`radial_profile`][virgil.limits.radial_profile].
    * ``fit`` runs [`fit`][virgil.fitting.fit] on the model template from
      the best grid point.
    * ``posterior`` samples
      [`numpyro_model`][virgil.likelihood.numpyro_model] with NUTS from
      the fit.
    * ``quicklook`` writes and executes ``quicklook.ipynb``.

    The fit is on the quoted errors unless ``error_scale="fit"``, and χ²/N
    is always reported on the quoted errors. Both templates sample the
    same priors: uniform in position over the search box (``dra``, ``ddec``;
    ``sep`` and ``pa`` of an angular template are derived from them, so
    there is no wrap at 0°/360°) and log-uniform in flux over
    ``flux_range``.

    Parameters
    ----------
    data : OIData
        The data.
    model : BinaryModelAngular or BinaryModelCartesian, optional
        Model template; its class sets the fitted parameters (``sep``,
        ``pa``, ``flux`` or ``dra``, ``ddec``, ``flux``). Default
        ``BinaryModelAngular``. The search and limits always use a point
        companion in Cartesian offsets.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of the defaults (see ``BinaryPipeline.defaults()``):

        - ``sigma`` (3.0): significance of the contrast limits.
        - ``detection_sigma`` (3.0): threshold of the detection check, on
          the look-elsewhere-corrected significance.
        - ``max_sep_mas`` (None): half-width of the search grid; by
          default ``fov_fraction`` × the coarsest λ/B of the samples (each
          sample's own wavelength over its own baseline, not λ_max/B_min).
        - ``fov_fraction`` (1.0): see ``max_sep_mas``.
        - ``grid_step_mas`` (None): grid spacing; by default a quarter of
          the finest λ/B of the samples, coarsened so that no axis exceeds
          ``max_grid`` points (the search report then records the effective
          step and warns).
        - ``max_grid`` (101): most points per position axis; the grid has
          the largest odd number of points not above it (at least 3).
        - ``flux_range`` ([1e-5, 0.5]): flux axis of the grid and bounds
          of the log-uniform flux prior (companion/primary).
        - ``n_flux`` (40): points on the flux axis.
        - ``wavel_range`` (None): ``[min, max]`` wavelengths to keep (m).
        - ``error_floor`` (None): absolute error floors by observable,
          e.g. ``{"vis": 0.01, "phi": 0.005}`` (phases in radians), as
          [`OIData.with_error_floor`][virgil.oidata.OIData.with_error_floor].
        - ``error_scale`` ("quoted"): ``"fit"`` also fits log-uniform
          error scales ``vis_scale`` and ``phi_scale`` on [0.1, 10].
        - ``num_warmup``, ``num_samples``, ``num_chains`` (1000, 1000, 4),
          ``chain_method`` ("vectorized") and ``seed`` (0): NUTS.
        - ``batch_size`` (None): grid batch size.
        - ``limit_bins`` (20): annuli of the contrast curve.
        - ``quicklook_execute`` (True): execute the quicklook notebook.

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import BinaryPipeline, load  # doctest: +SKIP
    >>> res = BinaryPipeline(vg.OIData("hd1234.oifits"), output="runs/hd1234").run()  # doctest: +SKIP
    >>> res.summary["companion"]["sep_mas"]  # doctest: +SKIP
    """

    NAME = "binary"
    STABILITY = "provisional"
    STAGES = (
        "load",
        "overview",
        "search",
        "limits",
        "fit",
        "posterior",
        "quicklook",
    )
    _DEFAULTS = {
        "sigma": 3.0,
        "detection_sigma": 3.0,
        "max_sep_mas": None,
        "fov_fraction": 1.0,
        "grid_step_mas": None,
        "max_grid": 101,
        "flux_range": [1e-5, 0.5],
        "n_flux": 40,
        "wavel_range": None,
        "error_floor": None,
        "error_scale": "quoted",
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "chain_method": "vectorized",
        "seed": 0,
        "batch_size": None,
        "limit_bins": 20,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        super().__init__(data, model, output=output, **settings)
        from ..models import BinaryModelAngular, BinaryModelCartesian

        if not isinstance(
            self.model, (BinaryModelAngular, BinaryModelCartesian)
        ):
            raise TypeError(
                "BinaryPipeline fits a BinaryModelAngular or "
                f"BinaryModelCartesian template; got {type(self.model).__name__}."
            )

    def _default_model(self):
        from ..models import BinaryModelAngular

        return BinaryModelAngular(sep=100.0, pa=0.0, flux=1e-3)

    @classmethod
    def _validate(cls, s):
        def positive(name):
            if not (isinstance(s[name], (int, float)) and s[name] > 0):
                raise ValueError(f"{name} must be positive; got {s[name]!r}.")

        for name in ("sigma", "detection_sigma", "fov_fraction"):
            positive(name)
        for name in ("max_sep_mas", "grid_step_mas", "batch_size"):
            if s[name] is not None:
                positive(name)
        for name in ("num_samples", "num_chains", "limit_bins"):
            if not (isinstance(s[name], int) and s[name] >= 1):
                raise ValueError(f"{name} must be an integer >= 1.")
        if not (isinstance(s["max_grid"], int) and s["max_grid"] >= 3):
            raise ValueError("max_grid must be an integer >= 3.")
        if not (isinstance(s["num_warmup"], int) and s["num_warmup"] >= 0):
            raise ValueError("num_warmup must be an integer >= 0.")
        if not (isinstance(s["n_flux"], int) and s["n_flux"] >= 2):
            raise ValueError("n_flux must be an integer >= 2.")
        lo, hi = s["flux_range"]
        if not 0 < lo < hi:
            raise ValueError("flux_range must satisfy 0 < min < max.")
        if s["error_scale"] not in ("quoted", "fit"):
            raise ValueError('error_scale must be "quoted" or "fit".')
        if s["chain_method"] not in ("vectorized", "sequential", "parallel"):
            raise ValueError(
                'chain_method must be "vectorized", "sequential" or "parallel".'
            )
        if s["wavel_range"] is not None and len(s["wavel_range"]) != 2:
            raise ValueError("wavel_range must be [min, max] in metres.")
        if s["error_floor"] is not None and not isinstance(
            s["error_floor"], dict
        ):
            raise ValueError("error_floor must be a dict of absolute floors.")

    # --- shared state, rebuilt on demand so that resume works ------------

    def _grid_axis(self):
        """The position axes as an ``_Axis``.

        ``wanted`` is the requested step (``grid_step_mas``, or a fraction
        of the finest λ/B); ``n`` points (odd, so that the star is on the
        grid, and at most ``max_grid``) give the effective ``step``.
        """
        s = self.settings
        geo = _geometry(self.processed)
        max_sep = s["max_sep_mas"] or s["fov_fraction"] * geo["fov_mas"]
        wanted = (
            s["grid_step_mas"] or _STEP_FRACTION * geo["lambda_over_b_mas"]
        )
        most = s["max_grid"] - (1 - s["max_grid"] % 2)  # the odd number
        n = min(2 * math.ceil(max_sep / wanted) + 1, most)
        return _Axis(max_sep, n, 2.0 * max_sep / (n - 1), wanted)

    def _grid(self):
        import jax.numpy as np

        s = self.settings
        max_sep, n, _, _ = self._grid_axis()
        axis = np.linspace(-max_sep, max_sep, n)
        lo, hi = s["flux_range"]
        return {
            "dra": axis,
            "ddec": axis,
            "flux": np.logspace(math.log10(lo), math.log10(hi), s["n_flux"]),
        }, max_sep

    def _priors(self, max_sep):
        """Priors on ``dra``, ``ddec`` (uniform on the box) and ``flux``.

        Both templates sample these, so they share one prior on the sky:
        uniform in position is the translation-invariant choice, and a
        uniform ``sep`` would put p(dra, ddec) proportional to 1/sep.
        """
        import numpyro.distributions as dist

        lo, hi = self.settings["flux_range"]
        return {
            "dra": dist.Uniform(-max_sep, max_sep),
            "ddec": dist.Uniform(-max_sep, max_sep),
            "flux": dist.LogUniform(lo, hi),
        }

    def _sampled_model(self):
        """The template as a function of the sampled ``dra``, ``ddec``, ``flux``.

        A ``BinaryModelAngular`` template gets its ``sep`` and ``pa`` from
        the offsets, so that its fit and posterior have the same prior as a
        ``BinaryModelCartesian`` one.
        """
        from ..models import BinaryModelAngular, BinaryModelCartesian

        if not isinstance(self.model, BinaryModelAngular):
            return BinaryModelCartesian(0.0, 0.0, 0.0)

        def model(dra, ddec, flux):
            sep, pa = separation_pa(dra, ddec)
            return BinaryModelAngular(sep, pa, flux)

        return model

    def _stages(self):
        return (
            Stage("load", _load_report, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage(
                "search",
                _search,
                (
                    "grids.h5",
                    "plots/search_delta_chi2.png",
                    "plots/search_snr.png",
                ),
            ),
            Stage(
                "limits",
                _limits,
                ("plots/limits_map.png", "plots/limits_contrast_curve.png"),
            ),
            Stage(
                "fit",
                _fit,
                (
                    "models/best/manifest.json",
                    "models/best/values.npz",
                    "models/best/info.json",
                    "plots/fit_correlation.png",
                ),
            ),
            Stage(
                "posterior",
                _posterior,
                (
                    "samples.h5",
                    "plots/posterior_corner.png",
                    "plots/posterior_trace.png",
                ),
            ),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        return _summarize(self.settings, reports)


# === STAGES ===


def _overview(p):
    from ..plotting import plot_oidata_overview, plot_uv_coverage

    fig, _ = plot_oidata_overview(p.processed)
    _save(fig, p.output / "plots" / "overview_data.png")
    fig, _ = plot_uv_coverage(p.processed)
    _save(fig, p.output / "plots" / "overview_uv.png")
    return {}


# --- search ---


def _search(p):
    from ..detection import detection_statistics
    from ..models import BinaryModelCartesian

    data, s = p.processed, p.settings
    grid, max_sep = p._grid()
    axis = p._grid_axis()
    if axis.coarsened:
        warnings.warn(
            f"max_grid={s['max_grid']} coarsens the grid step from "
            f"{axis.wanted:.3g} to {axis.step:.3g} mas; raise max_grid (or "
            "grid_step_mas) for a finer search.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    # detection_statistics evaluates the full likelihood grid, refines the
    # flux and takes the Laplace width internally, but returns only scalars.
    # Its log Bayes factor needs the full likelihood grid and its Δχ² falls
    # back on the grid flux where the refinement fails, so neither follows
    # from the maps of _search_maps: it stays one call beside them.
    detection = detection_statistics(
        BinaryModelCartesian, data, grid, batch_size=s["batch_size"]
    )
    maps = _search_maps(data, grid, s["batch_size"])
    _write_search_grids(p.output, grid, maps)
    best = {"dra": float(detection["dra"]), "ddec": float(detection["ddec"])}
    _plot_search(p.output, grid, maps, best)
    return _search_report(data, grid, axis, detection)


def _refined_flux(data, grid, batch_size):
    """``(flux, loglike)`` of the point companion refined at every position.

    This is ``optimized_flux_grid`` and ``optimized_likelihood_grid`` from
    one optimizer pass (each public function runs it and keeps one half).
    """
    from .._grid import (
        batch_size_or_default,
        resolve_grid_keys,
        warn_unconverged,
    )
    from ..grid_fit import _optimize_flux_grid
    from ..models import BinaryModelCartesian

    params, coord_keys, flux_key = resolve_grid_keys(grid)
    flux, loglike, converged = _optimize_flux_grid(
        data,
        BinaryModelCartesian,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
    )
    warn_unconverged(converged, "optimized_flux_grid")
    return flux, loglike


def _search_maps(data, grid, batch_size):
    """The maps of the search over the grid: ``search`` group of grids.h5."""
    import jax.numpy as np

    from ..grid_fit import laplace_flux_uncertainty_grid
    from ..likelihood import model_loglike
    from ..models import BinaryModelCartesian

    flux, loglike = (
        onp.asarray(x) for x in _refined_flux(data, grid, batch_size)
    )
    sigma = onp.asarray(
        laplace_flux_uncertainty_grid(
            BinaryModelCartesian,
            data,
            grid,
            flux=np.asarray(flux),
            batch_size=batch_size,
        )
    )
    loglike_null = float(model_loglike(_null_model(), data))
    delta_chi2 = onp.where(
        flux > 0, onp.maximum(2.0 * (loglike - loglike_null), 0.0), 0.0
    )
    return {
        "delta_chi2": delta_chi2,
        "loglike": loglike,
        "flux_best": flux,
        "flux_sigma": sigma,
        "snr": flux / sigma,
    }


def _null_model():
    """The star alone: a point companion of zero flux."""
    from ..models import BinaryModelCartesian

    return BinaryModelCartesian(0.0, 0.0, 0.0)


def _grid_attrs():
    return {"/": {"axis_order": list(_AXIS_ORDER), "schema": _io.SCHEMA}}


def _axes_group(axes):
    """The ``axes`` group of grids.h5: each axis with its units."""
    return {
        k: (onp.asarray(axes[k]), {"units": _AXIS_UNITS[k]})
        for k in _AXIS_ORDER
    }


def _write_search_grids(output, grid, maps):
    _io.write_h5(
        output / "grids.h5",
        {"axes": _axes_group(grid), "search": maps},
        attrs=_grid_attrs(),
    )


def _plot_search(output, grid, maps, best):
    from ..plotting import plot_grid_map

    fig, _ = plot_grid_map(
        maps["delta_chi2"],
        grid,
        kind="loglike",
        label="Δχ²",
        title="Δχ² (point companion)",
        best=best,
    )
    _save(fig, output / "plots" / "search_delta_chi2.png")
    fig, _ = plot_grid_map(maps["snr"], grid, kind="snr", best=best)
    _save(fig, output / "plots" / "search_snr.png")


def _peak_index(grid, best):
    """Grid index of ``best`` (the detection statistics' peak) on each axis.

    The position indices and the flux index come from the same source, the
    point where ``delta_chi2`` is reached; the flux is matched in log.
    """
    index = []
    for key in _AXIS_ORDER:
        axis = onp.asarray(grid[key], dtype=float)
        value = float(best[key])
        if key == "flux":
            axis, value = onp.log(axis), math.log(max(value, axis[0]))
        index.append(int(onp.argmin(onp.abs(axis - value))))
    return index


def _n_trials(max_sep, lambda_over_b):
    """Resolution elements in the search box, at least one."""
    return max(1.0, (2.0 * max_sep / lambda_over_b) ** 2)


def _search_report(data, grid, axis, detection):
    from ..detection import local_nsigma

    local = float(local_nsigma(float(detection["delta_chi2"])))
    n_trials = _n_trials(axis.max_sep, _geometry(data)["lambda_over_b_mas"])
    chi2_null, _ = _chi2(_null_model(), data)
    return {
        "delta_chi2": float(detection["delta_chi2"]),
        "log_bayes_factor": float(detection["log_bayes_factor"]),
        "max_snr": float(detection["max_snr"]),
        "dra_mas": float(detection["dra"]),
        "ddec_mas": float(detection["ddec"]),
        "flux": float(detection["flux"]),
        "local_nsigma": local,
        "global_nsigma": _global_nsigma(local, n_trials),
        "global_nsigma_method": GLOBAL_NSIGMA_METHOD,
        "n_trials": n_trials,
        "max_sep_mas": axis.max_sep,
        "grid_step_mas": axis.step,
        "grid_step_wanted_mas": axis.wanted,
        "grid_shape": [len(onp.asarray(grid[k])) for k in _AXIS_ORDER],
        "peak_index": _peak_index(grid, detection),
        "chi2_null": chi2_null,
        "n_independent": int(data.n_independent),
    }


# --- limits ---


def _limits(p):
    from ..limits import radial_profile

    s = p.settings
    grid, max_sep = p._grid()
    limit = _limit_map(p, grid)
    dra, ddec = onp.asarray(grid["dra"]), onp.asarray(grid["ddec"])
    profile = radial_profile(
        limit, dra, ddec, r_max=max_sep, bins=s["limit_bins"]
    )
    _write_limits(p, grid, limit)
    _plot_limits(p, grid, limit, profile)
    return _limits_report(s["sigma"], profile)


def _limit_map(p, grid):
    """Absil limits on the flux over the grid, NaN inside the resolution."""
    from ..limits import absil_limits
    from ..models import BinaryModelCartesian

    s = p.settings
    limit = onp.asarray(
        absil_limits(
            BinaryModelCartesian,
            p.processed,
            grid,
            s["sigma"],
            batch_size=s["batch_size"],
        )
    )
    r = onp.hypot(
        *onp.meshgrid(
            onp.asarray(grid["dra"]), onp.asarray(grid["ddec"]), indexing="ij"
        )
    )
    # Inside the resolution limit separation and flux are degenerate.
    resolution = _geometry(p.processed)["resolution_mas"]
    return onp.where(r < resolution, onp.nan, limit)


def _write_limits(p, grid, limit):
    """Add the ``limits`` group to the search's grids.h5."""
    path = p.output / "grids.h5"
    meta = {"sigma": p.settings["sigma"], "method": "absil"}
    _io.write_h5(
        path,
        {
            "axes": _axes_group(grid),
            "search": _io.read_h5(path, "search"),
            "limits": {"limit_flux": (limit, meta)},
        },
        attrs=_grid_attrs(),
    )


def _plot_limits(p, grid, limit, profile):
    from ..plotting import plot_contrast_curve, plot_grid_map

    sigma = p.settings["sigma"]
    fig, _ = plot_grid_map(
        limit, grid, kind="limit", units="delta_mag", sigma=sigma
    )
    _save(fig, p.output / "plots" / "limits_map.png")
    fig, _ = plot_contrast_curve(profile, sigma=sigma)
    _save(fig, p.output / "plots" / "limits_contrast_curve.png")


def _limits_report(sigma, profile):
    from ..limits import flux_to_delta_mag

    median = onp.asarray(profile["median"], dtype=float)
    delta_mag = onp.asarray(flux_to_delta_mag(median), dtype=float)
    return {
        "sigma": sigma,
        "method": "absil",
        "r_mas": profile["r"],
        "median_flux": median,
        "median_delta_mag": delta_mag,
        "deepest_delta_mag": float(onp.nanmax(delta_mag)),
    }


# --- fit ---


def _is_angular(p):
    from ..models import BinaryModelAngular

    return isinstance(p.model, BinaryModelAngular)


def _fit(p):
    from ..fitting import fit

    data = p.processed
    _, max_sep = p._grid()
    priors = p._priors(max_sep)
    angular = _is_angular(p)
    template, init = _fit_start(p, p._report("search"), priors, angular)
    result = fit(template, priors, data, noise=p._noise(), init=init)
    values = _fitted_values(result, angular)
    reported = ["sep", "pa", "flux"] if angular else list(priors)
    _io.save_model(
        p.output / "models" / "best", result.model, values, reported
    )
    _io.write_json(
        p.output / "models" / "best" / "info.json",
        _io.clean_json(dict(result.info)),
    )
    _plot_fit(p, result, values, list(priors), angular)
    return _fit_report(data, result, values, reported, priors)


def _fit_start(p, search, priors, angular):
    """The template to fit, and the starting values (``init``) if needed.

    The fit starts at the search's best point, with the flux kept inside
    the prior. An angular template takes ``init``; a Cartesian one is
    started by setting its parameters.
    """
    lo, hi = p.settings["flux_range"]
    start = {
        "dra": search["dra_mas"],
        "ddec": search["ddec_mas"],
        "flux": min(
            max(search["flux"], _START_MARGIN * lo), hi / _START_MARGIN
        ),
    }
    start = {k: onp.asarray(start[k], dtype=float) for k in priors}
    template = p._sampled_model()
    if angular:
        return template, start
    return template.set(list(priors), list(start.values())), None


def _fitted_values(result, angular):
    """The fitted parameters, with the angular model's own ``sep``, ``pa``."""
    values = {k: onp.asarray(v) for k, v in result.values.items()}
    if angular:
        model = result.model
        values.update(sep=onp.asarray(model.sep), pa=onp.asarray(model.pa))
    return values


def _plot_fit(p, result, values, params, angular):
    from ..likelihood import posterior_predictive_summary
    from ..plotting import plot_data_model_correlation

    one = {k: onp.asarray([float(values[k])]) for k in params}
    model = p._sampled_model() if angular else result.model
    pred = posterior_predictive_summary(one, model, p.processed, params)
    fig = plot_data_model_correlation(p.processed, {"MAP fit": pred})
    fig = fig[0] if isinstance(fig, tuple) else fig
    _save(fig, p.output / "plots" / "fit_correlation.png")


def _fit_report(data, result, values, reported, priors):
    chi2, resid = _chi2(result.model, data)
    # The periodic penalty terms of correlated closure phases are not
    # meant to be normal: test only the independent whitened residuals.
    resid = resid[: int(data.n_independent)]
    skew, kurt = _moments(resid)
    scales = {
        k.split(".", 1)[1]: float(v)
        for k, v in values.items()
        if k.startswith("noise.")
    }
    return {
        "params": {k: float(values[k]) for k in reported},
        "converged": result.info.get("converged"),
        "chi2": chi2,
        "n_independent": int(data.n_independent),
        "chi2_reduced": chi2 / data.n_independent,
        "error_scales": scales or None,
        "residual_skew": skew,
        "residual_excess_kurtosis": kurt,
        "n_residuals": int(resid.size),
        "at_bound": list(result.info.get("at_bound", []) or []),
        "priors": {k: repr(v) for k, v in priors.items()},
    }


# --- posterior ---


def _posterior(p):
    s = p.settings
    fit_report = p._report("fit")
    _, max_sep = p._grid()
    priors = p._priors(max_sep)
    reported = list(fit_report["params"])
    angular = _is_angular(p)

    mcmc = _run_nuts(p, priors, angular)
    sampled = {
        k: onp.asarray(v)
        for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    samples = dict(sampled)
    if angular:
        # The model's own parameters, derived from the sampled offsets.
        samples["sep"], samples["pa"] = separation_pa(
            sampled["dra"], sampled["ddec"]
        )
    sample_stats = _sample_stats(mcmc)
    _write_samples(p, samples, sample_stats, reported, fit_report["priors"])

    diagnostics = {k: _diagnostics([x]) for k, x in sampled.items()}
    derived = _derived_diagnostics(samples) if angular else {}
    summary = {
        k: {
            **_quantiles(samples[k], angle=(k == "pa")),
            **derived.get(k, diagnostics.get(k, {})),
        }
        for k in reported
    }
    _plot_posterior(p.output, samples, reported)
    return {
        "params": summary,
        "noise": _noise_quantiles(samples),
        "num_chains": s["num_chains"],
        "num_samples": s["num_samples"],
        "num_warmup": s["num_warmup"],
        "r_hat_max": max(d["r_hat"] for d in diagnostics.values()),
        "ess_bulk_min": min(d["ess_bulk"] for d in diagnostics.values()),
        "divergence_fraction": float(onp.mean(sample_stats["diverging"])),
        "prior_bound_fraction": _prior_bounds(priors, samples),
    }


def _run_nuts(p, priors, angular):
    """NUTS from the fit, one start per chain; returns the finished MCMC."""
    from ..likelihood import numpyro_model

    s = p.settings
    best = p.output / "models" / "best"
    model = p._sampled_model() if angular else _io.load_model(best)
    start = {
        k: onp.asarray(v)
        for k, v in _io.load_model_values(best).items()
        if k in priors
    }
    posterior = numpyro_model(model, priors, p.processed, noise=p._noise())
    return _run_chains(posterior, s, start, dense_mass=False, priors=priors)


def _write_samples(p, samples, sample_stats, reported, priors):
    s = p.settings
    _io.write_h5(
        p.output / "samples.h5",
        {"posterior": samples, "sample_stats": sample_stats},
        attrs={
            "/": {
                "schema": _io.SCHEMA,
                "params": reported,
                "priors": priors,
                "sampler": {k: s[k] for k in _SAMPLER_SETTINGS},
            }
        },
    )


def _derived_diagnostics(samples):
    """Diagnostics of ``sep`` and ``pa``; an angle through its sine and cosine."""
    pa = onp.radians(samples["pa"])
    return {
        "sep": _diagnostics([samples["sep"]]),
        "pa": _diagnostics([onp.sin(pa), onp.cos(pa)]),
    }


def _prior_bounds(priors, samples):
    """Fraction of samples at a prior bound, for the priors that have edges."""
    fractions = {k: _bound_samples(priors[k], samples[k]) for k in priors}
    return {k: f for k, f in fractions.items() if f is not None}


def _plot_posterior(output, samples, reported):
    import pandas as pd

    from ..plotting import plot_chainconsumer_diagnostics

    flat = {k: onp.asarray(samples[k], dtype=float).ravel() for k in reported}
    _, fig_corner, fig_walk = plot_chainconsumer_diagnostics(
        {"posterior": pd.DataFrame(flat)}, columns=reported
    )
    _save(fig_corner, output / "plots" / "posterior_corner.png")
    _save(fig_walk, output / "plots" / "posterior_trace.png")


def _quicklook(p):
    from . import _quicklook as ql

    ql.write_quicklook(p.output, execute=p.settings["quicklook_execute"])
    return {"executed": bool(p.settings["quicklook_execute"])}


# === SUMMARY AND CHECKS ===

_DATA_KEYS = (
    "n_vis",
    "n_phi",
    "n_independent",
    "closure_phases",
    "wavel_min_m",
    "wavel_max_m",
    "baseline_min_m",
    "baseline_max_m",
    "resolution_mas",
    "fov_mas",
)
_SEARCH_KEYS = (
    "delta_chi2",
    "log_bayes_factor",
    "max_snr",
    "local_nsigma",
    "global_nsigma",
    "global_nsigma_method",
    "n_trials",
    "dra_mas",
    "ddec_mas",
    "flux",
    "max_sep_mas",
)
_LIMITS_KEYS = (
    "sigma",
    "method",
    "r_mas",
    "median_flux",
    "median_delta_mag",
    "deepest_delta_mag",
)
_FIT_KEYS = ("params", "converged", "chi2_reduced", "at_bound")
_POSTERIOR_KEYS = (
    "params",
    "noise",
    "num_chains",
    "num_samples",
    "num_warmup",
    "r_hat_max",
    "ess_bulk_min",
    "divergence_fraction",
)


def _pick(report, keys):
    return {k: report[k] for k in keys}


def _companion(params, source):
    """Companion position and flux (median and 1σ) from fit or posterior."""
    from ..limits import flux_to_contrast, flux_to_delta_mag
    from ..models import BinaryModelAngular

    def q(name):
        x = params[name]
        if isinstance(x, dict):
            return x["median"], 0.5 * (x["q84"] - x["q16"])
        return x, None

    flux, flux_err = q("flux")
    out = {"source": source}
    if "sep" in params:
        (sep, sep_err), (pa, pa_err) = q("sep"), q("pa")
        cartesian = BinaryModelAngular(sep, pa, flux).to_cartesian()
        dra, ddec = float(cartesian.dra), float(cartesian.ddec)
    else:
        (dra, _), (ddec, _) = q("dra"), q("ddec")
        sep, pa = separation_pa(dra, ddec)
        sep, pa, sep_err, pa_err = float(sep), float(pa), None, None
    out.update(
        sep_mas=sep,
        sep_err_mas=sep_err,
        pa_deg=pa,
        pa_err_deg=pa_err,
        dra_mas=dra,
        ddec_mas=ddec,
        flux=flux,
        flux_err=flux_err,
        contrast=float(flux_to_contrast(flux)),
        delta_mag=float(flux_to_delta_mag(flux)),
    )
    return out


def _chi2_section(settings, search, fit):
    return {
        "n_independent": search["n_independent"],
        "null_reduced": search["chi2_null"] / search["n_independent"],
        "companion_reduced": fit["chi2_reduced"] if fit else None,
        "error_scale": settings["error_scale"],
        "error_scales": fit["error_scales"] if fit else None,
    }


def _chi2_checks(search, fit):
    """The χ²/N check: of the best-fit binary, else of the star alone."""
    null = search["chi2_null"] / search["n_independent"] if search else None
    if fit:
        label = "the best-fit binary" + (
            f" (star alone: {null:.3g})" if null is not None else ""
        )
        return [_checks.chi2_reduced("chi2", fit["chi2_reduced"], label=label)]
    if search:
        return [_checks.chi2_reduced("chi2", null, label="the star alone")]
    return []


def _search_checks(settings, search):
    return [
        _checks.detection(
            search["local_nsigma"],
            search["global_nsigma"],
            threshold=settings["detection_sigma"],
        ),
        # The flux axis only seeds the optimizer: not a search boundary.
        _checks.grid_edge(
            search["peak_index"][:2], search["grid_shape"][:2], ("dra", "ddec")
        ),
    ]


def _fit_checks(fit, companion, load):
    checks = []
    if fit["error_scales"]:
        checks.append(_checks.error_scale(fit["error_scales"]))
    if fit["at_bound"]:
        checks.append(_checks.start_at_bound(fit["at_bound"]))
    checks.append(
        _checks.residual_normality(
            fit["residual_skew"],
            fit["residual_excess_kurtosis"],
            fit["n_residuals"],
        )
    )
    if load:
        checks.append(
            _checks.field_of_view(
                companion["sep_mas"], load["resolution_mas"], load["fov_mas"]
            )
        )
    return checks


def _posterior_checks(post):
    return [
        _checks.prior_bound(post["prior_bound_fraction"]),
        _checks.r_hat(post["r_hat_max"]),
        _checks.ess(post["ess_bulk_min"]),
        _checks.divergences(post["divergence_fraction"]),
    ]


def _summarize(settings, reports):
    """``(sections, checks)`` of the summary from the stage reports so far."""
    load, search, limits, fit, post = (
        reports.get(k)
        for k in ("load", "search", "limits", "fit", "posterior")
    )
    sections, checks = {}, _chi2_checks(search, fit)
    if load:
        sections["data"] = _pick(load, _DATA_KEYS)
    if search:
        sections["chi2"] = _chi2_section(settings, search, fit)
        sections["search"] = _pick(search, _SEARCH_KEYS)
        checks += _search_checks(settings, search)
    if limits:
        sections["limits"] = _pick(limits, _LIMITS_KEYS)
    if fit:
        sections["fit"] = _pick(fit, _FIT_KEYS)
        source = post["params"] if post else fit["params"]
        sections["companion"] = _companion(
            source, "posterior" if post else "fit"
        )
        checks += _fit_checks(fit, sections["companion"], load)
    if post:
        sections["posterior"] = _pick(post, _POSTERIOR_KEYS)
        checks += _posterior_checks(post)
    return sections, checks
