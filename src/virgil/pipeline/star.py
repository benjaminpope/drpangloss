"""`StarPipeline`: stellar angular diameter and limb darkening."""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Callable

import numpy as onp

from .._precision import in_float64
from .._utils import rad2mas
from . import _checks, _io
from ._common import (
    _overview,
    _quicklook,
    _scales_of,
    _chi2_64,
    _m2loglike_64,
    _chain_health,
    _corner_trace,
    _diagnostics,
    _draws,
    _integer,
    _load_report,
    _load_start,
    _noise_quantiles,
    _pair,
    _pick,
    _prior_bounds,
    _run_chains,
    _sampler_checks,
    _sampler_report,
    _save_fit,
    _site_summary,
    _validate_common,
    _write_samples,
    _chi2,
    _geometry,
    _m2loglike_grid,
    _refine_diameter,
    _save,
    _spatial_frequencies,
)
from ._core import Stage, _Pipeline

# Points of the grid that refines a lobe's diameter, ±1 coarse step.
_REFINE_POINTS = 201
# Points and span (fractions of the lobe's diameter) of the grid whose best
# diameter starts a second fit, and on which a uniform disk's error scales
# are profiled.
_START_POINTS = 201
_START_SPAN = (0.75, 1.25)
# Fractional separation within which a scan point must be the lowest to be
# a lobe's minimum.
_LOBE_SEPARATION = 0.05
# The default diameter prior spans these multiples of the finest and the
# coarsest λ/B: well inside the unresolved regime, beyond the first null of
# the shortest baseline.
_DIAM_LOW_FACTOR = 0.1
_DIAM_HIGH_FACTOR = 4.0
# Δ(−2 log L) of a closure-phase flip at the data's precision.
_WALL_JUMP = 1000.0
# Posterior draws probed for hard walls, and the step in posterior sds.
_WALL_DRAWS = 40
_WALL_RADIUS = 2.0
# Points of the grid of θf (θ the diameter, f = B/λ) searched for a null,
# and its extent: the uniform disk's first null is at 1.22.
_NULL_POINTS = 4000
_NULL_REACH = 4.0


# === THE MODEL REGISTRY ===


@dataclasses.dataclass(frozen=True)
class StarModel:
    """A model ``StarPipeline`` can fit, registered under a short name.

    Adding a model to the pipeline is adding an entry to ``MODELS``: the
    stages, summary and checks are written against this description.

    Attributes
    ----------
    name : str
        The short name accepted by ``StarPipeline(model=...)``; also the
        name of the model's folder under ``models/``.
    label : str
        Human-readable name, for plots and messages.
    classes : tuple of type
        Virgil classes recognized as templates of this entry, so that a
        model instance can be passed instead of the name.
    build : callable
        ``build(values) -> model``: the model for a dict of parameter
        values (every path in ``params``).
    params : tuple of str
        The fitted parameter paths; the first is the angular diameter.
    start : dict
        Starting values of the parameters other than the diameter.
    priors : callable
        ``priors(diam_lo, diam_hi) -> dict`` of group-invariant numpyro
        priors on ``params`` (log-uniform diameter).
    scan : bool
        Whether the model is fitted by a scan of the diameter alone rather
        than a gradient-based fit (a uniform disk's closure phases flip
        between 0 and 180 degrees at each null, so its χ² is not smooth).
    shape : tuple of str
        Parameters, other than the diameter, that shape the star (e.g. the
        limb-darkening coefficients); the pipeline compares their posterior
        widths with their prior widths.
    derived : callable or None
        ``derived(samples) -> dict`` of extra reported quantities computed
        from sampled values (for example ``u1`` and ``u2`` from ``q1`` and
        ``q2``).
    parent : str or None
        The name of the simpler model this one contains, for the model
        comparison (None if it contains none).
    check_stem : str or None
        Stem of the names of this model's checks: ``<stem>_gain`` (its gain
        over its parent) and ``<stem>_constrained`` (its shape parameters
        against their priors). Defaults to ``name``.
    """

    name: str
    label: str
    classes: tuple
    build: Callable
    params: tuple
    start: dict
    priors: Callable
    scan: bool = False
    shape: tuple = ()
    derived: Callable | None = None
    parent: str | None = None
    check_stem: str | None = None

    @property
    def n_params(self):
        return len(self.params)

    @property
    def gain_check(self):
        return f"{self.check_stem or self.name}_gain"

    @property
    def constrained_check(self):
        return f"{self.check_stem or self.name}_constrained"


def _uniform_disk(values):
    from ..models import UniformDisk

    return UniformDisk(values["diam"])


def _quadratic_disk(values):
    from ..models import QuadraticLimbDarkenedDisk

    return QuadraticLimbDarkenedDisk(
        values["diam"], q1=values["q1"], q2=values["q2"]
    )


def _quadratic_derived(samples):
    from ..models import QuadraticLimbDarkenedDisk

    star = QuadraticLimbDarkenedDisk(
        samples["diam"], q1=samples["q1"], q2=samples["q2"]
    )
    return {"u1": onp.asarray(star.u1), "u2": onp.asarray(star.u2)}


def _diam_prior(lo, hi):
    import numpyro.distributions as dist

    return dist.LogUniform(lo, hi)


def _uniform_priors(lo, hi):
    return {"diam": _diam_prior(lo, hi)}


def _quadratic_priors(lo, hi):
    import numpyro.distributions as dist

    return {
        "diam": _diam_prior(lo, hi),
        "q1": dist.Uniform(0.0, 1.0),
        "q2": dist.Uniform(0.0, 1.0),
    }


def _registry():
    from ..models import QuadraticLimbDarkenedDisk, UniformDisk

    entries = (
        StarModel(
            name="uniform",
            label="uniform disk",
            classes=(UniformDisk,),
            build=_uniform_disk,
            params=("diam",),
            start={},
            priors=_uniform_priors,
            scan=True,
        ),
        StarModel(
            name="limb_darkened",
            label="limb-darkened disk",
            classes=(QuadraticLimbDarkenedDisk,),
            build=_quadratic_disk,
            params=("diam", "q1", "q2"),
            start={"q1": 0.5, "q2": 0.5},
            priors=_quadratic_priors,
            shape=("q1", "q2"),
            derived=_quadratic_derived,
            parent="uniform",
            check_stem="limb_darkening",
        ),
    )
    return {e.name: e for e in entries}


_MODELS = None


def models():
    """The registry of short names to [`StarModel`][virgil.pipeline.star.StarModel]."""
    global _MODELS
    if _MODELS is None:
        _MODELS = _registry()
    return _MODELS


# The models fitted and compared when ``model`` is not given.
DEFAULT_MODELS = ("uniform", "limb_darkened")


def _entry_for(instance):
    for entry in models().values():
        if type(instance) in entry.classes:
            return entry
    raise TypeError(
        f"StarPipeline has no model for a {type(instance).__name__}; pass "
        f"one of the names {sorted(models())} or a template of the classes "
        + ", ".join(
            sorted(c.__name__ for e in models().values() for c in e.classes)
        )
        + "."
    )


# === THE PIPELINE ===


class StarPipeline(_Pipeline):
    """Measure a star's angular diameter, with and without limb darkening.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Stages: ``load → overview → fit → posterior → quicklook``.

    * ``load`` applies ``wavel_range`` and ``error_floor`` and writes the
      data that are fitted to ``data/processed.oifits``.
    * ``overview`` plots the data and their uv coverage.
    * ``fit`` scans the uniform-disk diameter (a uniform disk's closure
      phases flip at every null, so its χ² is not smooth and a scan is
      the reliable fit; with ``error_scale="fit"`` its error scales are
      profiled along the scan), then fits each other model with
      [`fit`][virgil.fitting.fit] from the scanned diameter. With the
      default ``model`` it compares the limb-darkened and uniform fits by
      their Δ(−2 log L) against a BIC penalty. Lobes of the diameter are
      compared the same way.
    * ``posterior`` samples every fitted model with NUTS
      ([`numpyro_model`][virgil.likelihood.numpyro_model]).
    * ``quicklook`` writes and executes ``quicklook.ipynb``: the main
      result is V² against spatial frequency with the model curves.

    The fit is on the quoted errors unless ``error_scale="fit"``, and χ²/N
    is always reported on the quoted errors. (With fitted error scales,
    χ² at each model's own scales is N by construction, so lobes and
    models are compared on −2 log L, whose normalization penalizes inflated
    errors; on the quoted errors that is χ² up to a constant.) Priors are group-invariant:
    log-uniform in the diameter over ``diam_range_mas`` and uniform on
    [0, 1] in Kipping's limb-darkening coefficients ``q1``, ``q2``, which
    cover exactly the physical quadratic laws.

    ``res.model()`` is the preferred model (the limb-darkened one only if
    its Δ(−2 log L) exceeds the BIC penalty of its two extra parameters);
    ``res.model("uniform")`` and ``res.model("limb_darkened")`` give each.

    Parameters
    ----------
    data : OIData
        The data.
    model : str or SourceModel, optional
        What to fit. A short name, ``"uniform"`` (a
        [`UniformDisk`][virgil.models.UniformDisk]) or ``"limb_darkened"``
        (a [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk]
        in Kipping's q1, q2), fits that model alone. An instance of one of
        those classes fits that model, with its ``q1`` and ``q2`` as
        starting values. By default both are fitted and compared.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of the defaults (see ``StarPipeline.defaults()``):

        - ``diam_range_mas`` (None): ``[min, max]`` of the log-uniform
          diameter prior; by default from 0.1 times the finest λ/B (well
          inside the unresolved regime) to 4 times the coarsest (beyond
          the first null on every baseline), where λ/B is taken over the
          observed samples (see ``fringe_scales``).
        - ``n_scan`` (2000): points of the log-spaced diameter scan; each
          lobe's minimum is then refined on 201 points within one scan step
          either side.
        - ``max_lobes`` (5): the most diameter lobes (separate minima of
          the scan) fitted for each model. The best lobe gives the fit
          and bounds the posterior's diameter prior; the table of lobes
          is in ``summary["fit"]["models"][name]["lobes"]``.
        - ``wavel_range`` (None): ``[min, max]`` wavelengths to keep (m).
        - ``error_floor`` (None): absolute error floors by observable,
          e.g. ``{"vis": 0.01, "phi": 0.005}``, as
          [`OIData.with_error_floor`][virgil.oidata.OIData.with_error_floor].
        - ``error_scale`` ("quoted"): ``"fit"`` also fits log-uniform
          error scales ``vis_scale`` and ``phi_scale`` on [0.1, 10].
        - ``num_warmup``, ``num_samples``, ``num_chains`` (1000, 1000, 4),
          ``chain_method`` ("vectorized") and ``seed`` (0): NUTS.
        - ``batch_size`` (None): scan batch size.
        - ``quicklook_execute`` (True): execute the quicklook notebook.

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import StarPipeline  # doctest: +SKIP
    >>> res = StarPipeline(vg.OIData("star.oifits"), output="runs/star").run()  # doctest: +SKIP
    >>> res.summary["star"]["diam_mas"]  # doctest: +SKIP
    """

    NAME = "star"
    STABILITY = "provisional"
    STAGES = ("load", "overview", "fit", "posterior", "quicklook")
    _DEFAULTS = {
        "diam_range_mas": None,
        "n_scan": 2000,
        "max_lobes": 5,
        "wavel_range": None,
        "error_floor": None,
        "error_scale": "quoted",
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "chain_method": "vectorized",
        "seed": 0,
        "batch_size": None,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        registry = models()
        starts = {}
        if model is None:
            names = list(DEFAULT_MODELS)
            template = None
        elif isinstance(model, str):
            if model not in registry:
                raise ValueError(
                    f"model must be one of {sorted(registry)} or a model "
                    f"instance; got {model!r}."
                )
            names = [model]
            template = None
        else:
            entry = _entry_for(model)
            names = [entry.name]
            template = model
            starts[entry.name] = {
                k: float(onp.asarray(getattr(model, k)))
                for k in entry.params
                if k != "diam"
            }
        self.names = names
        self._starts = {
            n: {**registry[n].start, **starts.get(n, {})} for n in names
        }
        if template is None:
            template = registry[names[-1]].build(
                {"diam": 1.0, **self._starts[names[-1]]}
            )
        super().__init__(data, template, output=output, **settings)

    @classmethod
    def _validate(cls, s):
        if s["diam_range_mas"] is not None:
            _pair(s, "diam_range_mas")
        _integer(s, "n_scan", 20)
        _integer(s, "max_lobes", 1)
        if s["error_scale"] not in ("quoted", "fit"):
            raise ValueError('error_scale must be "quoted" or "fit".')
        _validate_common(s)

    def _model_digest(self):
        record = super()._model_digest()
        record["models"] = list(self.names)
        return record

    def _diam_range(self):
        """Bounds of the log-uniform diameter prior (mas)."""
        given = self.settings["diam_range_mas"]
        if given is not None:
            return float(given[0]), float(given[1])
        geo = _geometry(self.processed)
        return (
            _DIAM_LOW_FACTOR * geo["lambda_over_b_mas"],
            _DIAM_HIGH_FACTOR * geo["fov_mas"],
        )

    def _stages(self):
        fit_outputs = [
            "grids.h5",
            "models/best/manifest.json",
            "models/best/values.npz",
            "models/best/info.json",
            "plots/fit_correlation.png",
            "plots/fit_v2.png",
        ]
        for name in self.names:
            fit_outputs += [
                f"models/{name}/manifest.json",
                f"models/{name}/values.npz",
                f"models/{name}/info.json",
            ]
        return (
            Stage("load", _load_star, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage("fit", _fit, tuple(fit_outputs)),
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


# === PLOTS ===


def plot_v2_models(data, fitted, ax=None, curves=400):
    """The squared visibilities against spatial frequency, with model curves.

    Parameters
    ----------
    data : OIData
        The data (its ``vis`` are plotted with their errors).
    fitted : dict[str, SourceModel]
        Models to draw, by label, as curves over the data's range of
        spatial frequency.
    ax : matplotlib axis, optional
    curves : int, optional
        Points of each curve.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    freq = _spatial_frequencies(data)
    power = 2 if data.v2_flag else 1
    ax.errorbar(
        freq / 1e6,
        onp.asarray(data.vis),
        onp.asarray(data.d_vis),
        fmt=".",
        color="0.4",
        label="data",
    )
    grid = onp.linspace(1e-3, 1.05 * freq.max(), curves)
    for label, model in fitted.items():
        vis = onp.abs(onp.asarray(model.model(grid, 0.0 * grid, 1.0)))
        ax.plot(grid / 1e6, vis**power, label=label)
    positive = onp.asarray(data.vis)[onp.asarray(data.vis) > 0]
    floor = min(1e-4, float(positive.min()) / 3.0) if positive.size else 1e-4
    ax.set(
        yscale="log",
        ylim=(floor, 1.5),
        xlabel="Spatial frequency (Mλ)",
        ylabel="$V^2$" if data.v2_flag else "$|V|$",
    )
    ax.legend()
    return ax.figure


# === STAGES ===


def _freq_max(data):
    """The longest spatial frequency B/λ in the data (cycles per radian)."""
    from .._geometry import fringe_scales

    return float(rad2mas) / fringe_scales(data)[0]


def _first_null(model, diam_mas, n=_NULL_POINTS):
    """θB/λ at the first sign change of the model visibility.

    The visibility is evaluated on a one-dimensional grid of B/λ out to
    θf = 4 (about 3.3 times the uniform-disk null). Falls back to the
    uniform-disk value when the model has no null in range.
    """
    scale = float(rad2mas) / float(diam_mas)  # f for θf = 1
    grid = onp.linspace(1e-3, _NULL_REACH, n) * scale
    vis = onp.real(onp.asarray(model.model(grid, 0.0 * grid, 1.0)))
    flips = onp.flatnonzero(onp.sign(vis[1:]) * onp.sign(vis[:-1]) < 0)
    if not flips.size:
        return _checks.FIRST_NULL
    i = int(flips[0])
    frac = vis[i] / (vis[i] - vis[i + 1])
    return float((grid[i] + frac * (grid[i + 1] - grid[i])) / scale)


def _load_star(p):
    return _load_report(p, freq_max_per_rad=_freq_max(p.processed))


# --- the diameter scan and its lobes ---

# A closure phase flips by 180 degrees where a model's visibility changes
# sign, so a χ² evaluated in float32 near a null can jump by orders of
# magnitude; ``fit`` and the grid tools are float64 inside, and so are the
# numbers this pipeline reports.


@in_float64
def _scan(data, lo, hi, n_scan, batch_size):
    """Uniform-disk Δ(−2 log L) on a log-spaced diameter grid.

    Returns
    -------
    tuple
        ``(axis, delta)``: the diameters and the excess of −2 log L over
        its minimum on the grid (Δχ² on the quoted errors).
    """
    import jax.numpy as np

    from ..grid_fit import likelihood_grid
    from ..models import UniformDisk

    axis = onp.geomspace(lo, hi, n_scan)
    ll = onp.asarray(
        likelihood_grid(
            UniformDisk,
            data=data,
            grid={"diam": np.asarray(axis)},
            batch_size=batch_size,
        )
    )
    return axis, 2.0 * (onp.nanmax(ll) - ll)


def _find_lobes(axis, delta, lo, hi, cap, *, separation=_LOBE_SEPARATION):
    """The scan's lobes: separate minima of the diameter scan.

    Past the first null the squared visibility repeats in lobes (and
    closure phases flip at its zeros), so the scan has several separate
    minima. A point is a minimum if it is the lowest within ``separation``
    (fractionally) either side; the ``cap`` lowest are kept, and the lobe
    boundaries are the geometric midpoints between neighbouring minima
    (``lo`` and ``hi`` at the ends).

    Returns
    -------
    list[dict]
        By increasing diameter: ``{"diam": scan minimum, "scan_delta_chi2":
        its excess over the best, "bounds": (low, high)}``.

    Raises
    ------
    RuntimeError
        If the scan is not finite anywhere.
    """
    if not onp.isfinite(delta).any():
        raise RuntimeError(
            "The uniform-disk scan is not finite at any diameter in "
            f"[{lo:.4g}, {hi:.4g}] mas; check the data and diam_range_mas."
        )
    n = axis.size
    dlog = math.log(axis[-1] / axis[0]) / (n - 1)
    half = max(1, int(round(separation / dlog)))
    delta = onp.where(onp.isfinite(delta), delta, onp.inf)
    minima = []
    for i in range(n):
        window = delta[max(0, i - half) : i + half + 1]
        if onp.isfinite(delta[i]) and delta[i] <= window.min():
            if not minima or i - minima[-1] > half:
                minima.append(i)
            elif delta[i] < delta[minima[-1]]:
                minima[-1] = i
    keep = sorted(sorted(minima, key=lambda i: delta[i])[:cap])
    centres = [float(axis[i]) for i in keep]
    edges = [lo] + [math.sqrt(a * b) for a, b in zip(centres, centres[1:])]
    edges.append(hi)
    return [
        {
            "diam": c,
            "scan_delta_chi2": float(delta[i]),
            "bounds": (float(edges[k]), float(edges[k + 1])),
        }
        for k, (c, i) in enumerate(zip(centres, keep))
    ]


@in_float64
def _refine(data, centre, lo, hi, step, batch_size):
    """The uniform-disk diameter of best likelihood within one coarse step."""
    import jax.numpy as np

    from ..grid_fit import likelihood_grid
    from ..models import UniformDisk

    fine = onp.geomspace(
        max(lo, centre / step), min(hi, centre * step), _REFINE_POINTS
    )
    ll = onp.asarray(
        likelihood_grid(
            UniformDisk,
            data=data,
            grid={"diam": np.asarray(fine)},
            batch_size=batch_size,
        )
    )
    return float(fine[int(onp.nanargmax(ll))])


def _scan_lobes(data, lo, hi, n_scan, max_lobes, batch_size):
    """Scan, find and refine: ``(axis, delta, lobes)``.

    The lobes are those of ``_find_lobes``, each with its diameter refined
    on the uniform disk's likelihood.
    """
    axis, delta = _scan(data, lo, hi, n_scan, batch_size)
    lobes = _find_lobes(axis, delta, lo, hi, max_lobes)
    step = (hi / lo) ** (1.0 / (n_scan - 1))
    for lobe in lobes:
        lobe["diam"] = _refine(data, lobe["diam"], lo, hi, step, batch_size)
    return axis, delta, lobes


# --- fitting one model in one lobe ---


@dataclasses.dataclass(frozen=True)
class LobeFit:
    """A model fitted in one diameter lobe.

    Attributes
    ----------
    model : SourceModel
        The fitted model.
    values : dict
        Fitted values by path (and ``noise.<name>`` for error scales).
    info : dict
        The fit's diagnostics.
    chi2 : float
        χ² on the quoted errors.
    chi2_fitted : float
        χ² at the fitted error scales (N by construction at the MAP; a
        diagnostic only, not a score).
    m2loglike : float
        −2 log L at the fitted error scales: the score that compares lobes
        and models.
    residuals : ndarray
        Whitened residuals on the quoted errors.
    scales : dict
        The fitted error scales by name (empty on the quoted errors).
    priors : dict
        The priors the lobe was fitted with.
    lobe : dict
        The lobe (``diam``, ``bounds``) it was fitted in.
    warnings : list
        Warnings the fit raised.
    """

    model: object
    values: dict
    info: dict
    chi2: float
    chi2_fitted: float
    m2loglike: float
    residuals: onp.ndarray
    scales: dict
    priors: dict
    lobe: dict
    warnings: list


def _diameter_grid(centre, priors):
    """The grid of diameters around ``centre``, within the prior."""
    low, high = float(priors["diam"].low), float(priors["diam"].high)
    return onp.geomspace(
        max(low, _START_SPAN[0] * centre),
        min(high, _START_SPAN[1] * centre),
        _START_POINTS,
    )


def _fit_scanned(p, entry, start, grid, noise):
    """A scan model's fit: the lobe's diameter, error scales profiled.

    Without ``noise`` the lobe's refined diameter is the fit. With fitted
    error scales the scales are profiled analytically at each diameter of
    ``grid`` (see ``_common._profiled_scales``); the diameter of least
    −2 log L is then refined between its grid neighbours, with the scales
    re-profiled at each trial, so the fit is not limited to the grid
    spacing (a gradient fit of the non-smooth χ² would stall).
    """
    info = {
        "method": "scan",
        "converged": True,
        "n_scan": p.settings["n_scan"],
    }
    if noise is None:
        return entry.build(start), {"diam": onp.asarray(start["diam"])}, info
    data = p.processed
    m2loglike, grid_scales = _m2loglike_grid(
        entry.build, start, data, grid, noise, p.settings["batch_size"]
    )
    best = int(onp.nanargmin(m2loglike))
    low = grid[max(best - 1, 0)]
    high = grid[min(best + 1, grid.size - 1)]
    diam, m2, scales = _refine_diameter(
        entry.build, start, data, low, high, noise
    )
    if not m2 <= m2loglike[best]:  # not unimodal between the neighbours
        diam = float(grid[best])
        scales = {k: float(v[best]) for k, v in grid_scales.items()}
    values = {"diam": onp.asarray(diam)}
    values.update({f"noise.{k}": v for k, v in scales.items()})
    model = entry.build({**start, "diam": diam})
    return model, values, {**info, "scales": "profiled"}


def _fit_from_starts(p, entry, priors, start, grid, noise):
    """A gradient fit from the lobe's diameter and from the best of a grid.

    A limb-darkened star seen past its first null has closure phases that
    flip at visibility nulls, so χ² is a union of thin slivers and a local
    fit started on the wrong one stalls at a huge χ². Start from the scan
    diameter and from the best of a small grid of diameters with the
    starting shape, and keep the better fit.
    """
    from ..fitting import fit

    data = p.processed
    m2loglike, _ = _m2loglike_grid(
        entry.build, start, data, grid, None, p.settings["batch_size"]
    )
    starts = [start["diam"], float(grid[int(onp.nanargmin(m2loglike))])]
    best = None
    for diam in dict.fromkeys(starts):
        template = entry.build({**start, "diam": diam})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit(template, priors, data, noise=noise)
        score = _m2loglike_64(result.model, data, **_scales_of(result.values))
        if best is None or score < best[0]:
            best = (score, result, caught)
    _, result, caught = best
    _rewarn(caught)
    values = {k: onp.asarray(v) for k, v in result.values.items()}
    return result.model, values, _io.clean_json(dict(result.info))


def _fit_model(p, entry, priors, best_diam, noise):
    """MAP fit of one registered model: ``(model, values, info)``."""
    start = {"diam": best_diam, **p._starts[entry.name]}
    grid = _diameter_grid(best_diam, priors)
    if entry.scan:
        return _fit_scanned(p, entry, start, grid, noise)
    return _fit_from_starts(p, entry, priors, start, grid, noise)


def _fit_lobe(p, entry, lobe, noise):
    """Fit ``entry`` in ``lobe`` and score it: a ``LobeFit``."""
    data = p.processed
    priors = entry.priors(*lobe["bounds"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, values, info = _fit_model(p, entry, priors, lobe["diam"], noise)
    chi2, residuals = _chi2_64(model, data)
    scales = _scales_of(values)
    chi2_fitted = _chi2_64(model, data, **scales)[0] if scales else chi2
    return LobeFit(
        model=model,
        values=values,
        info=info,
        chi2=float(chi2),
        chi2_fitted=float(chi2_fitted),
        m2loglike=float(_m2loglike_64(model, data, **scales)),
        residuals=residuals,
        scales=scales,
        priors=priors,
        lobe=lobe,
        warnings=caught,
    )


def _finite_or(value, default):
    return value if math.isfinite(value) else default


def _choose_lobe(fits):
    """The lobe fit of least −2 log L."""
    return min(fits, key=lambda f: _finite_or(f.m2loglike, math.inf))


def _lobe_table(fits, n):
    """One row per lobe, with its −2 log L above the best lobe's.

    ``delta_chi2`` is that difference (the Δχ² on the quoted errors, up to
    the constant that cancels), kept under its older name.
    """
    top = _choose_lobe(fits).m2loglike
    return [
        {
            "diam_bounds_mas": list(f.lobe["bounds"]),
            "diam_mas": float(onp.asarray(f.values["diam"])),
            "chi2": f.chi2,
            "chi2_reduced": f.chi2 / n,
            "m2loglike": f.m2loglike,
            "delta_m2loglike": f.m2loglike - top,
            "delta_chi2": f.m2loglike - top,
        }
        for f in fits
    ]


def _model_report(entry, fit, table, n):
    """The stage report of one model's best lobe."""
    params = {k: float(onp.asarray(fit.values[k])) for k in entry.params}
    if entry.derived:
        derived = entry.derived(
            {k: onp.asarray(fit.values[k]) for k in entry.params}
        )
        params.update({k: float(v) for k, v in derived.items()})
    return {
        "params": params,
        "converged": fit.info.get("converged"),
        "chi2": fit.chi2,
        "chi2_fitted": fit.chi2_fitted,
        "m2loglike": fit.m2loglike,
        "first_null": _first_null(fit.model, float(fit.values["diam"])),
        "diam_bounds_mas": list(fit.lobe["bounds"]),
        "lobes": table,
        "chi2_reduced": fit.chi2 / n,
        "error_scales": fit.scales or None,
        "at_bound": list(fit.info.get("at_bound", []) or []),
        "priors": {k: repr(v) for k, v in fit.priors.items()},
        "prior_std": {
            k: float(onp.sqrt(fit.priors[k].variance)) for k in entry.shape
        },
    }


def _fit_in_lobes(p, entry, lobes, noise):
    """Fit ``entry`` in every lobe: ``(best LobeFit, report)``."""
    n = int(p.processed.n_independent)
    fits = [_fit_lobe(p, entry, lobe, noise) for lobe in lobes]
    best = _choose_lobe(fits)
    # Only the best lobe's fit warns: a fit that stalls in an alias lobe,
    # whose χ² is thousands above the best, is not a problem.
    _rewarn(best.warnings)
    return best, _model_report(entry, best, _lobe_table(fits, n), n)


# --- choosing between models ---


def _compare(registry, reports, child, n):
    """The comparison of a model with its parent: gain against BIC penalty."""
    parent = registry[child].parent
    gain = reports[parent]["m2loglike"] - reports[child]["m2loglike"]
    n_extra = registry[child].n_params - registry[parent].n_params
    penalty = _checks.bic_penalty(n_extra, n)
    return {
        "simple": parent,
        "complex": child,
        "delta_m2loglike": gain,
        "delta_chi2": gain,
        "n_extra_params": n_extra,
        "bic_penalty": penalty,
        "preferred": child if gain > penalty else parent,
    }


def _choose_model(reports, registry, n):
    """The preferred model, by walking down the chain of parents.

    Every fitted model with a fitted parent is compared with it (the gain
    in −2 log L against ``n_extra ln N``). The walk starts at the fitted
    model that has no fitted parent (the one of least −2 log L if there
    are several) and moves to a child while that child is preferred over
    the model it contains, taking the largest gain among children.

    With more than two nested models, a deeper model C is reached only
    through a preferred intermediate B: C is never compared with the root
    A directly, so a C that would beat A but not B is not chosen.

    Returns
    -------
    tuple
        ``(best, decisive, comparisons)``: the name of the preferred model,
        the comparison that decided it (the last one made on the walk; None
        if no model has a fitted parent) and all of them.
    """
    comparisons = [
        _compare(registry, reports, name, n)
        for name in reports
        if registry[name].parent in reports
    ]
    roots = [k for k in reports if registry[k].parent not in reports]
    best = min(
        roots, key=lambda k: _finite_or(reports[k]["m2loglike"], math.inf)
    )
    decisive = None
    while True:
        children = [c for c in comparisons if c["simple"] == best]
        if not children:
            return best, decisive, comparisons
        decisive = max(
            children, key=lambda c: _finite_or(c["delta_m2loglike"], -math.inf)
        )
        if decisive["preferred"] == best:
            return best, decisive, comparisons
        best = decisive["complex"]


# --- the fit stage ---


def _fit_plots(p, registry, fits, best):
    """The MAP fit's data-model correlation and the V² curves."""
    from ..likelihood import posterior_predictive_summary
    from ..plotting import plot_data_model_correlation

    data, params = p.processed, list(registry[best].params)
    one = {
        k: onp.asarray([float(onp.asarray(fits[best].values[k]))])
        for k in params
    }
    pred = posterior_predictive_summary(one, fits[best].model, data, params)
    fig = plot_data_model_correlation(data, {"MAP fit": pred})
    fig = fig[0] if isinstance(fig, tuple) else fig
    _save(fig, p.output / "plots" / "fit_correlation.png")
    fig = plot_v2_models(
        data, {registry[n].label: fits[n].model for n in p.names}
    )
    _save(fig, p.output / "plots" / "fit_v2.png")


def _fit(p):
    data, s = p.processed, p.settings
    lo, hi = p._diam_range()
    registry = models()
    axis, delta, lobes = _scan_lobes(
        data, lo, hi, s["n_scan"], s["max_lobes"], s["batch_size"]
    )
    _io.write_h5(
        p.output / "grids.h5",
        {
            "axes": {"diam": (axis, {"units": "mas"})},
            "scan": {"delta_chi2": delta},
        },
        attrs={"/": {"axis_order": ["diam"], "schema": _io.SCHEMA}},
    )

    # Each lobe is fitted for each model; the best lobe gives the model's
    # fit and (in the posterior) its bounded, group-invariant log-uniform
    # diameter prior. The lobe table is recorded in the summary.
    noise = p._noise()
    fits, reports = {}, {}

    for name in p.names:
        fits[name], reports[name] = _fit_in_lobes(
            p, registry[name], lobes, noise
        )
        _save_fit(
            p.output / "models" / name, fits[name], list(registry[name].params)
        )
    n = int(data.n_independent)
    best, decisive, comparisons = _choose_model(reports, registry, n)
    _save_fit(
        p.output / "models" / "best", fits[best], list(registry[best].params)
    )
    _fit_plots(p, registry, fits, best)
    scan_best = min(lobes, key=lambda lobe: lobe["scan_delta_chi2"])
    return {
        "names": list(p.names),
        "best": best,
        "n_independent": n,
        "diam_range_mas": [lo, hi],
        "diam_bounds_mas": reports[best]["diam_bounds_mas"],
        "rivals": [
            [row["diam_mas"], row["delta_chi2"]]
            for row in reports[best]["lobes"]
            if 0 < row["delta_chi2"] <= _checks.RIVAL_DELTA_CHI2
        ],
        "scan_diam_mas": scan_best["diam"],
        "models": reports,
        "comparison": decisive,
        "comparisons": comparisons,
    }


def _bound_fractions(entry, priors, noise, samples):
    """Fraction of samples within 1% of the range of either prior bound.

    Covers the sampled priors except the entry's shape parameters (whose
    priors are meant to matter and are judged by the ``constrained`` check
    instead), and every fitted error scale ``noise.<term>``. Parameters
    whose prior has no finite edges are left out.
    """
    sampled = {k: v for k, v in priors.items() if k not in entry.shape}
    sampled.update({f"noise.{k}": v for k, v in (noise or {}).items()})
    return _prior_bounds(sampled, samples)


def _rewarn(caught):
    """Re-raise the recorded warnings of the fit that was kept."""
    for w in caught:
        warnings.warn_explicit(
            w.message, w.category, w.filename, w.lineno, source=w.source
        )


def _wall_probe(
    entry, data, samples, priors, n=_WALL_DRAWS, radius=_WALL_RADIUS, seed=0
):
    """How much of the posterior's surroundings is a hard wall.

    A closure phase of a centrosymmetric star flips by π wherever a model
    visibility changes sign, which costs about 4/σ² in χ² (5e4 at 0.5°).
    The likelihood is then exact-zero outside a cell of the parameters, and
    NUTS integrators that cross its wall are rejected and flagged
    divergent. This probes it: from ``n`` posterior draws it steps
    ``radius`` posterior standard deviations in a random direction, and
    counts the steps that raise χ² by more than ``_WALL_JUMP``, which in a
    smooth posterior would raise it by a few.

    Returns
    -------
    dict
        ``{"radius_sd", "n", "exposed"}``, with ``exposed`` the fraction of
        the ``n`` valid steps that hit a wall.
    """
    names = list(entry.params)
    flat = {k: onp.asarray(samples[k], dtype=float).ravel() for k in names}
    sd = onp.array([flat[k].std() for k in names])
    rng = onp.random.default_rng(seed)
    draws = rng.choice(flat[names[0]].size, size=n, replace=True)
    hits = valid = 0
    for i in draws:
        base = onp.array([flat[k][i] for k in names])
        direction = rng.normal(size=len(names))
        trial = base + radius * sd * direction / onp.linalg.norm(direction)
        inside = all(
            float(priors[k].low) < v < float(priors[k].high)
            for k, v in zip(names, trial)
            if hasattr(priors.get(k), "low")
        )
        if not inside:
            continue
        chi0 = _chi2(entry.build(dict(zip(names, base))), data)[0]
        chi1 = _chi2(entry.build(dict(zip(names, trial))), data)[0]
        valid += 1
        hits += bool(float(chi1 - chi0) > _WALL_JUMP)
    return {
        "radius_sd": radius,
        "n": valid,
        "exposed": hits / valid if valid else 0.0,
    }


@in_float64
def _nuts(model, data, p, name, priors, start):
    """NUTS for one fitted model: ``(samples, stats, summary)``."""
    from ..likelihood import numpyro_model

    entry = models()[name]
    noise = p._noise()
    posterior = numpyro_model(model, priors, data, noise=noise)
    mcmc = _run_chains(
        posterior, p.settings, start, dense_mass=True, priors=priors
    )
    samples, stats = _draws(mcmc)
    diagnostics = {site: _diagnostics([x]) for site, x in samples.items()}
    if entry.derived:
        samples.update(entry.derived(samples))
    reported = list(entry.params) + list(
        k
        for k in samples
        if k not in entry.params and not k.startswith("noise.")
    )
    summary = {
        k: _site_summary(samples[k], diagnostics.get(k, {})) for k in reported
    }
    return (
        samples,
        stats,
        {
            "params": summary,
            "noise": _noise_quantiles(samples),
            **_chain_health(diagnostics, stats),
            "wall_probe": _wall_probe(entry, data, samples, priors),
            "prior_bound_fraction": _bound_fractions(
                entry, priors, noise, samples
            ),
        },
    )


def _posterior(p):
    fit_report = p._report("fit")
    best = fit_report["best"]
    groups, per_model, params_attr = {}, {}, {}
    for name in p.names:
        folder = p.output / "models" / name
        priors = models()[name].priors(
            *fit_report["models"][name]["diam_bounds_mas"]
        )
        samples, stats, summary = _nuts(
            _io.load_model(folder),
            p.processed,
            p,
            name,
            priors,
            _load_start(folder, priors),
        )
        groups[f"posterior_{name}"] = samples
        groups[f"sample_stats_{name}"] = stats
        per_model[name] = summary
        params_attr[name] = list(summary["params"])
        if name == best:
            groups["posterior"], groups["sample_stats"] = samples, stats
    _write_samples(
        p,
        groups,
        best=best,
        params=params_attr,
        priors={n: fit_report["models"][n]["priors"] for n in p.names},
    )
    sampled = list(models()[best].params)
    _corner_trace(p.output, groups["posterior"], sampled)
    return {
        "models": per_model,
        "best": best,
        "sampled": sampled,
        **_sampler_report(p.settings, per_model.values()),
    }


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
    "freq_max_per_rad",
)


_FIT_KEYS = (
    "params",
    "converged",
    "chi2",
    "chi2_fitted",
    "m2loglike",
    "chi2_reduced",
    "diam_bounds_mas",
    "lobes",
    "at_bound",
)
_POSTERIOR_KEYS = (
    "models",
    "best",
    "sampled",
    "num_chains",
    "num_samples",
    "num_warmup",
    "r_hat_max",
    "ess_bulk_min",
    "divergence_fraction",
)


def _chi2_section(settings, fit):
    names = fit["names"]
    return {
        "n_independent": fit["n_independent"],
        "error_scale": settings["error_scale"],
        "reduced": {k: fit["models"][k]["chi2_reduced"] for k in names},
        "error_scales": {k: fit["models"][k]["error_scales"] for k in names},
    }


def _fit_section(fit):
    return {
        "best": fit["best"],
        "scan_diam_mas": fit["scan_diam_mas"],
        "diam_bounds_mas": fit["diam_bounds_mas"],
        "rivals": fit["rivals"],
        "models": {
            k: {f: fit["models"][k][f] for f in _FIT_KEYS}
            for k in fit["names"]
        },
    }


def _fit_checks(fit, registry):
    """Checks on the fits: χ²/N, model gains and the fitted error scales."""
    reports = fit["models"]
    checks = [
        _checks.chi2_reduced(
            f"chi2_{k}",
            reports[k]["chi2_reduced"],
            label=f"the {registry[k].label}",
        )
        for k in fit["names"]
    ]
    for comparison in fit["comparisons"]:
        simple, complex_ = comparison["simple"], comparison["complex"]
        checks.append(
            _checks.model_gain(
                registry[complex_].gain_check,
                reports[simple]["m2loglike"],
                reports[complex_]["m2loglike"],
                fit["n_independent"],
                n_extra=comparison["n_extra_params"],
                simple=registry[simple].label,
                complex_=registry[complex_].label,
            )
        )
    scales = {
        f"{k}.{kk}": v
        for k in fit["names"]
        for kk, v in (reports[k]["error_scales"] or {}).items()
    }
    if scales:
        checks.append(_checks.error_scale(scales))
    at_bound = [
        f"{k}.{site}" for k in fit["names"] for site in reports[k]["at_bound"]
    ]
    if at_bound:
        checks.append(_checks.start_at_bound(at_bound))
    return checks


def _star_section(fit, post):
    """The star's diameter: from the posterior if there is one."""
    best = fit["best"]
    if post:
        row = post["models"][best]["params"]["diam"]
        diam, diam_err = row["median"], 0.5 * (row["q84"] - row["q16"])
    else:
        diam, diam_err = fit["models"][best]["params"]["diam"], None
    return {
        "model": best,
        "source": "posterior" if post else "fit",
        "diam_mas": diam,
        "diam_err_mas": diam_err,
    }


def _resolution(fit, load, diam):
    """The resolution section and check: ``(section, check)``."""
    null = fit["models"][fit["best"]]["first_null"]
    check = _checks.resolution_regime(
        diam, load["freq_max_per_rad"], first_null=null
    )
    section = {
        "diam_mas": diam,
        "freq_max_per_rad": load["freq_max_per_rad"],
        "first_null": null,
        "first_null_fraction": check.value,
    }
    return section, check


def _shape(fit, post, entry):
    """The shape section and check: the posterior's width against the
    prior's for the best model's shape parameters."""
    best = entry.name
    rows = {k: post["models"][best]["params"][k] for k in entry.shape}
    prior_std = fit["models"][best]["prior_std"]
    ratios = {k: rows[k]["std"] / prior_std[k] for k in entry.shape}
    section = {
        "model": best,
        "params": {
            k: {
                "median": rows[k]["median"],
                "posterior_std": rows[k]["std"],
                "prior_std": prior_std[k],
                "ratio": ratios[k],
            }
            for k in entry.shape
        },
    }
    return section, _checks.prior_constrained(entry.constrained_check, ratios)


def _posterior_checks(post):
    """Checks on the sampler: bounds, R-hat, ESS and divergences."""
    worst = max(
        post["models"].values(), key=lambda m: m["divergence_fraction"]
    )
    bounds = {
        f"{n}.{k}": v
        for n, m in post["models"].items()
        for k, v in m["prior_bound_fraction"].items()
    }
    return _sampler_checks(post, bounds=bounds, walls=worst.get("wall_probe"))


def _summarize(settings, reports):
    registry = models()
    sections, checks = {}, []
    load, fit, post = (
        reports.get("load"),
        reports.get("fit"),
        reports.get("posterior"),
    )
    if load:
        sections["data"] = _pick(load, _DATA_KEYS)
    if not fit:
        return sections, checks

    sections["chi2"] = _chi2_section(settings, fit)
    sections["fit"] = _fit_section(fit)
    if fit["comparison"]:
        sections["comparison"] = dict(fit["comparison"])
        sections["comparisons"] = [dict(c) for c in fit["comparisons"]]
    checks += _fit_checks(fit, registry)
    sections["star"] = _star_section(fit, post)
    if load:
        sections["resolution"], check = _resolution(
            fit, load, sections["star"]["diam_mas"]
        )
        checks.append(check)
    checks.append(_checks.multimodal(fit["models"][fit["best"]]["lobes"]))
    if post:
        entry = registry[fit["best"]]
        if entry.shape:
            sections["shape"], check = _shape(fit, post, entry)
            checks.append(check)
        sections["posterior"] = _pick(post, _POSTERIOR_KEYS)
        checks += _posterior_checks(post)
    return sections, checks
