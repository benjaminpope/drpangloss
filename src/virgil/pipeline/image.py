"""`ImagingPipeline`: a data-consistent image of the source (provisional).

An automatic imaging pipeline can promise much less than ``StarPipeline`` or
``BinaryPipeline``. It reports an image that fits the data on the quoted
errors, from stated starts, at hyperparameters chosen by a stated rule. It
cannot promise that the morphology is right: every data-fidelity check passed
on contest reconstructions that were judged unusable. The design is in
``design/pipeline.md``.
"""

from __future__ import annotations

import time

import jax
import numpy as onp

from . import _checks, _io
from ._common import (
    _integer,
    _load_report,
    _overview_plots,
    _positive,
    _quicklook,
    _save,
    _save_fit,
    _spatial_frequencies,
)
from ._core import Stage, _Pipeline

# Every check of the pipeline, in the order they are reported. Checks 12, 13
# and 15 (the ensemble) and 14 (chromatic) arrive with the ensemble and
# channel stages; ``chi2_reduced_<kind>`` of extra observables follow these.
CHECKS = (
    "chi2_reduced",
    "chi2_reduced_vis",
    "chi2_reduced_phi",
    "quoted_errors",
    "error_scale",
    "effective_parameters",
    "short_baselines",
    "hyperparameter_edge",
    "star",
    "orientation",
    "edge_flux",
)

_FAMILIES = ("tv", "tsv", "maxent", "starlet")
_CLEAN_UNITS = {True: "relative to the star's flux", False: "relative flux"}


class ImagingPipeline(_Pipeline):
    """Image the source from V², closure phases or complex visibilities.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    **What it can and cannot promise.** The image fits the data on the
    quoted errors, to a χ²/N per kind of observable that the summary states,
    from a stated start and at hyperparameters chosen by a stated rule. It
    does **not** promise that the morphology is right. On contest data,
    reconstructions that passed every data-fidelity check (χ²/N near 1) were
    judged "terrible", "blobby" or "a chain of dots": no check here measures
    whether an image looks like the source. The evidence also assumes the
    errors and the model family are right; with biased data it still returns
    a best point, and the checks only detect the symptoms. A grey image of
    chromatic data is not a fit.

    Stages: ``load → overview → start → star → hyperparameters → members →
    combine → quicklook``.

    * ``load`` applies ``wavel_range`` and ``error_floor``, writes
      ``data/processed.oifits``, and records the number of data, the
      flagged fraction and the spread of the quoted errors of each
      observable block.
    * ``overview`` plots the data and sets the grid: pixels ``oversample``
      times finer than the Nyquist scale, over the field that
      [`image_field`][virgil.imaging.image_field] chooses (or ``field_mas``),
      coarsened to ``max_npix`` pixels with the field kept.
    * ``start`` fits the envelope
      ([`starting_image`][virgil.imaging.starting_image]) and runs gradient
      [`clean`][virgil.imaging.clean] on a grid capped at ``clean_max_npix``;
      CLEAN's restored image is ``images/clean.fits``.
    * ``star`` decides whether the scene has an unresolved star. With
      ``star="auto"``, Gaussian-process fits with and without a star at the
      grid's central amplitude and length are compared by log evidence on the
      same grid: ΔlogZ of at least ``star_threshold`` (5 nats) decides, and
      otherwise the decision is ``"both"`` and the member with the larger
      evidence is kept. ``star=True`` or ``False`` skips the comparison.
    * ``hyperparameters`` chooses the prior. By default it is a Gaussian
      process on the log-brightness
      ([`evidence_grid`][virgil.ensemble.evidence_grid]): amplitude σ and
      correlation length ℓ (in beam minor axes) are chosen by the Laplace
      log evidence on a grid that grows while its optimum is on a movable
      edge, with an anisotropic variant if the data are elongated. With
      ``regularizers=("tv",)`` (or ``"tsv"``, ``"maxent"``, ``"starlet"``) a
      classical regularizer is used instead, and its weight is chosen at the
      corner of an L-curve. The grid is **not** repeated on rescaled errors;
      MacKay's error scale of each observable at the winner is reported and
      checked.
    * ``members`` writes the member (one in this version: the evidence
      grid's winner) to ``members/0``.
    * ``combine`` takes the mean of the members (the member itself) and
      writes ``images/mean.fits``, ``models/best`` and the diagnostics:
      raw χ²/N per observable, the short-baseline residual, the residual
      image and the 180° flip.
    * ``quicklook`` writes and executes ``quicklook.ipynb``; its first cell
      says what the run can and cannot claim.

    All χ²/N values are on the **quoted** errors; the summary stores a
    rescaled χ² next to them, labelled, and no check uses it.

    Not in this version (the settings and checks exist in the design):
    several ensemble members, ``run_member``, further families, the
    ensemble checks, per-channel and SPARCO imaging (with several channels
    and a star the image is grey), ``star_model="ud"`` and the support hole
    under the star. ``anneal`` is a stub that raises ``NotImplementedError``.

    Parameters
    ----------
    data : OIData
        The data.
    model : {"gp"}, optional
        The image prior (default and only value ``"gp"``); a classical
        regularizer is chosen with ``regularizers``.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of the defaults (see ``ImagingPipeline.defaults()``):

        - ``oversample`` (3), ``field_fwhms`` (6), ``field_mas`` (None),
          ``max_npix`` (128): the grid.
        - ``clean`` (True), ``clean_max_npix`` (65), ``clean_scales_beam``
          ([0, 0.5, 1]), ``clean_iterations`` (3000): the CLEAN start.
        - ``star`` ("auto"), ``star_model`` ("point"), ``star_threshold``
          (5 nats): the star decision.
        - ``sigmas`` ([1, 2, 4]), ``lengths_beam`` ([0.5, 1, 2]),
          ``anisotropic`` ("auto": when an elliptical Gaussian's axis ratio
          is below 0.9; or True, False), ``grid_growths`` (2): the evidence
          grid.
        - ``regularizers`` ([]), ``n_weights`` (8): classical regularizers
          and the weights of their L-curve (at most one in this version).
        - ``anneal`` (None): reserved; not implemented.
        - ``method`` (None), ``max_steps`` (None): the optimizer of
          [`fit`][virgil.fitting.fit] (tests set ``max_steps`` small).
        - ``seed`` (0), ``wavel_range`` (None), ``error_floor`` (None),
          ``quicklook_execute`` (True).

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import ImagingPipeline  # doctest: +SKIP
    >>> pipeline = ImagingPipeline(vg.OIData("star.oifits"), output="runs/image")  # doctest: +SKIP
    >>> res = pipeline.run()  # doctest: +SKIP
    >>> res.image().pixels.shape  # doctest: +SKIP
    """

    NAME = "image"
    STABILITY = "provisional"
    STAGES = (
        "load",
        "overview",
        "start",
        "star",
        "hyperparameters",
        "members",
        "combine",
        "quicklook",
    )
    _DEFAULTS = {
        "oversample": 3.0,
        "field_fwhms": 6.0,
        "field_mas": None,
        "max_npix": 128,
        "clean": True,
        "clean_max_npix": 65,
        "clean_scales_beam": [0.0, 0.5, 1.0],
        "clean_iterations": 3000,
        "star": "auto",
        "star_model": "point",
        "star_threshold": 5.0,
        "sigmas": [1.0, 2.0, 4.0],
        "lengths_beam": [0.5, 1.0, 2.0],
        "anisotropic": "auto",
        "grid_growths": 2,
        "regularizers": [],
        "n_weights": 8,
        "anneal": None,
        "method": None,
        "max_steps": None,
        "seed": 0,
        "wavel_range": None,
        "error_floor": None,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        if model not in (None, "gp"):
            raise ValueError(
                f'model must be "gp" (choose a classical regularizer with '
                f"regularizers=...); got {model!r}."
            )
        super().__init__(data, "gp", output=output, **settings)

    def _default_model(self):
        return "gp"

    def __repr__(self):
        return f"ImagingPipeline(output={str(self.output)!r})"

    @classmethod
    def _validate(cls, s):
        for name in ("oversample", "field_fwhms", "star_threshold"):
            _positive(s, name)
        if s["field_mas"] is not None:
            _positive(s, "field_mas")
        _integer(s, "max_npix", 4)
        _integer(s, "clean_max_npix", 4)
        _integer(s, "clean_iterations", 1)
        _integer(s, "grid_growths", 0)
        _integer(s, "n_weights", 3)
        _integer(s, "seed", 0)
        if not isinstance(s["clean"], bool):
            raise ValueError("clean must be True or False.")
        if not s["clean_scales_beam"] or any(
            v < 0 for v in s["clean_scales_beam"]
        ):
            raise ValueError("clean_scales_beam must be non-negative numbers.")
        for name in ("sigmas", "lengths_beam"):
            if not s[name] or any(not v > 0 for v in s[name]):
                raise ValueError(f"{name} must be positive numbers.")
        if s["star"] not in ("auto", True, False):
            raise ValueError('star must be "auto", True or False.')
        if s["star_model"] != "point":
            raise NotImplementedError(
                f"star_model={s['star_model']!r} is not available yet: "
                '"point" is the only star model, because the evidence grid '
                "builds a point-source star ("
                '"ud", a fitted uniform disk, needs support in '
                "virgil.ensemble.evidence_grid)."
            )
        if s["anisotropic"] not in ("auto", True, False):
            raise ValueError('anisotropic must be "auto", True or False.')
        unknown = sorted(set(s["regularizers"]) - set(_FAMILIES))
        if unknown:
            raise ValueError(
                f"regularizers {unknown} are unknown; choose from "
                f"{sorted(_FAMILIES)}."
            )
        if len(s["regularizers"]) > 1:
            raise NotImplementedError(
                "One regularizer at a time in this version: several "
                "families arrive with the ensemble stage."
            )
        if s["regularizers"] and s["star"] == "auto":
            raise ValueError(
                'Set star=True or star=False with regularizers: the "auto" '
                "star decision compares Gaussian-process evidences."
            )
        if s["anneal"] is not None:
            raise NotImplementedError(
                "anneal (convolving the image with a sub-beam Gaussian at "
                "each optimizer step, so that a chain of dots anneals out) "
                "is not implemented yet: it needs a design check against "
                "the L-BFGS stall problem. Leave it None."
            )
        if s["wavel_range"] is not None and len(s["wavel_range"]) != 2:
            raise ValueError("wavel_range must be [min, max] in metres.")
        if s["error_floor"] is not None and not isinstance(
            s["error_floor"], dict
        ):
            raise ValueError("error_floor must be a dict of absolute floors.")

    def _model_digest(self):
        return {"class": "Image", "digest": None, "models": ["gp"]}

    def _stages(self):
        clean = ("images/clean.fits", *_model_files("clean"))
        return (
            Stage("load", _load, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage(
                "start",
                _start,
                clean
                if self.settings["clean"]
                else ("stages/start/report.json",),
            ),
            Stage("star", _star, ("stages/star/report.json",)),
            Stage(
                "hyperparameters",
                _hyperparameters,
                (
                    "grids.h5",
                    "plots/hyperparameters.png",
                    *_model_files("hyper_best"),
                ),
            ),
            Stage(
                "members",
                _members,
                tuple(f"members/0/{f}" for f in _MODEL_FILES),
            ),
            Stage(
                "combine",
                _combine,
                (
                    "images/mean.fits",
                    "plots/image_mean.png",
                    "plots/image_residual.png",
                    "plots/residual_frequency.png",
                    *_model_files("best"),
                ),
            ),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        return _summarize(self.settings, reports)


_MODEL_FILES = ("manifest.json", "values.npz", "info.json")


def _model_files(name):
    """The files ``_save_fit`` writes for the model ``models/<name>``."""
    return tuple(f"models/{name}/{f}" for f in _MODEL_FILES)


# === SHARED HELPERS ===


def _fit_options(s):
    """The keywords of ``fit`` that the settings set."""
    return {k: s[k] for k in ("method", "max_steps") if s[k] is not None}


def _grid_spec(s, field_mas, *, family="gp", central=False):
    """The ensemble spec of the one group of this run.

    With ``central`` the grid is the single point at the middle amplitude
    and length (the star comparison): no growth, isotropic.
    """
    from ..ensemble import EnsembleSpec

    sigmas, lengths = sorted(s["sigmas"]), sorted(s["lengths_beam"])
    axis_ratio = {"auto": 0.9, True: 1.0 + 1e-9, False: None}[s["anisotropic"]]
    if central:
        sigmas = [sigmas[len(sigmas) // 2]]
        lengths = [lengths[len(lengths) // 2]]
        axis_ratio = None
    return EnsembleSpec(
        families=(family,),
        oversample=(float(s["oversample"]),),
        field_factors=(1.0,),
        starts=("moments",),
        max_npix=int(s["max_npix"]),
        field_mas=field_mas,
        n_weights=int(s["n_weights"]),
        gp_sigmas=tuple(sigmas),
        gp_lengths=tuple(lengths),
        gp_growths=0 if central else int(s["grid_growths"]),
        gp_axis_ratio=axis_ratio,
    )


def _draw(p):
    """The group's ``Draw`` and the ensemble spec, from the overview report."""
    from ..ensemble import Draw

    ov = p._report("overview")
    family = (
        p.settings["regularizers"][0] if p.settings["regularizers"] else "gp"
    )
    d = ov["draw"]
    draw = Draw(
        0,
        family,
        int(d["npix"]),
        float(d["pixel_scale_mas"]),
        "moments",
        tuple(d["weights"]),
    )
    return draw, _grid_spec(
        p.settings, ov["field_requested_mas"], family=family
    )


def _star_on(p):
    """Whether the member has an analytic star: the star stage's choice."""
    return bool(p._report("star")["member_star"])


def _starts(data, star):
    from ..ensemble import reference_starts

    return reference_starts(data, star, ("moments",))


def _block_stats(values):
    values = onp.asarray(values, dtype=float).ravel()
    ok = onp.isfinite(values) & (values > 0)
    good = values[ok]
    spread = float(good.max() - good.min()) if good.size else 0.0
    median = float(onp.median(good)) if good.size else float("nan")
    return {
        "min": float(good.min()) if good.size else None,
        "median": median if good.size else None,
        "max": float(good.max()) if good.size else None,
        "all_equal": bool(good.size and spread <= 1e-6 * median),
        "n_bad": int(values.size - good.size),
    }


# === STAGES: LOAD AND OVERVIEW ===


def _load(p):
    from ..imaging import _observable_blocks

    data = p.processed
    _, counts = _observable_blocks([data])
    errors = {}
    if onp.size(data.vis):
        errors["vis"] = _block_stats(data.d_vis)
    if onp.size(data.phi):
        errors["phi"] = _block_stats(data.d_phi)
    wavel = onp.unique(onp.round(onp.asarray(data.wavel, float), 12))
    return _load_report(
        p,
        n_by_observable={k: int(v) for k, v in counts.items()},
        flagged_fraction=float(1.0 - onp.mean(onp.asarray(data.unflagged))),
        n_channels=int(wavel.size),
        quoted_errors=errors,
    )


def _overview(p):
    from ..ensemble import draw_groups
    from ..imaging import (
        beam,
        envelope_fwhm,
        field_of_view,
        image_field,
        nyquist_pixel_scale,
        second_moment_fwhm,
    )

    s, data = p.settings, p.processed
    _overview_plots(p, data)
    candidate = s["star"] is not False
    b = beam(data)
    envelope = float(envelope_fwhm(data, candidate))
    if s["field_mas"] is None:
        requested = float(
            image_field(data, fwhm_mas=envelope, field_fwhms=s["field_fwhms"])
        )
        source = "image_field"
    else:
        requested, source = float(s["field_mas"]), "setting"
    family = s["regularizers"][0] if s["regularizers"] else "gp"
    spec = _grid_spec(s, requested, family=family)
    draw = draw_groups(
        data,
        1,
        jax.random.PRNGKey(int(s["seed"])),
        spec,
        star=candidate,
    )[0]
    try:
        moment = second_moment_fwhm(data)  # None when no V² is above 0.3
    except (ValueError, FloatingPointError):
        moment = None
    fwhm = None if moment is None else float(moment)
    nyquist = float(nyquist_pixel_scale(data))
    return {
        "field_mas": float(draw.npix * draw.pixel_scale_mas),
        "field_requested_mas": requested,
        "field_source": source,
        "pixel_scale_mas": float(draw.pixel_scale_mas),
        "npix": int(draw.npix),
        "nyquist_mas": nyquist,
        "oversample_actual": nyquist / float(draw.pixel_scale_mas),
        "beam": {
            "major_mas": float(b.major_mas),
            "minor_mas": float(b.minor_mas),
            "pa_deg": float(b.pa_deg),
        },
        "fwhm_mas": fwhm,
        "envelope_fwhm_mas": envelope,
        "ceiling_mas": float(field_of_view(data)),
        "draw": {
            "family": draw.family,
            "npix": int(draw.npix),
            "pixel_scale_mas": float(draw.pixel_scale_mas),
            "weights": [float(w) for w in draw.weights],
        },
    }


# === STAGE: START (ENVELOPE AND CLEAN) ===


def _start(p):
    from ..imaging import Beam, clean, starting_image
    from ..models import PointSource

    s, data = p.settings, p.processed
    ov = p._report("overview")
    candidate = s["star"] is not False
    b = Beam(**ov["beam"])
    start = starting_image(data, star=candidate)
    env = start.env if candidate else start
    report = {
        "envelope_fwhm_mas": ov["envelope_fwhm_mas"],
        "envelope_flux": float(env.flux) if candidate else None,
        "clean": bool(s["clean"]),
    }
    if not s["clean"]:
        return report
    n = int(min(ov["npix"], s["clean_max_npix"]))
    scale = ov["field_mas"] / n
    scales = tuple(f * b.minor_mas for f in s["clean_scales_beam"])
    result = clean(
        data,
        n,
        scale,
        PointSource() if candidate else None,
        max_iterations=int(s["clean_iterations"]),
        scales_mas=scales,
    )
    components = onp.asarray(result.components, dtype=float)
    total = float(components.sum())
    restored = onp.asarray(result.restored(b), dtype=float)
    _io.write_fits_image(
        p.output / "images" / "clean.fits",
        restored,
        scale,
        beam=ov["beam"],
        header={"BUNIT": _CLEAN_UNITS[candidate]},
        extensions={"COMPONENTS": components},
    )
    folder = p.output / "models" / "clean"
    _io.save_model(folder, result.model, {}, [])
    _io.write_json(
        folder / "info.json",
        {
            "chi2_red": float(result.chi2_red[-1]),
            "stop": result.stop,
            "scales_mas": list(scales),
        },
    )
    report.update(
        clean_chi2_red=float(result.chi2_red[-1]),
        clean_stop=result.stop,
        clean_npix=n,
        clean_pixel_scale_mas=float(scale),
        clean_flux_fraction=(total / (1.0 + total) if candidate else 1.0),
    )
    return report


# === STAGE: STAR ===


def _star(p):
    from ..ensemble import draw_groups, evidence_grid

    s, data = p.settings, p.processed
    mode = s["star"]
    if mode != "auto":
        return {
            "mode": "on" if mode else "off",
            "decision": "on" if mode else "off",
            "member_star": bool(mode),
            "delta_log_z": None,
            "star_model": s["star_model"],
        }
    ov = p._report("overview")
    spec = _grid_spec(s, ov["field_requested_mas"], central=True)
    draw = draw_groups(
        data, 1, jax.random.PRNGKey(int(s["seed"])), spec, star=True
    )[0]
    log_z = {}
    for star in (True, False):
        grid = evidence_grid(
            data,
            draw,
            star=star,
            starts=_starts(data, star),
            spec=spec,
            **_fit_options(s),
        )
        log_z[star] = float(grid.best.log_evidence)
    delta = log_z[True] - log_z[False]
    threshold = float(s["star_threshold"])
    decision = (
        "on"
        if delta >= threshold
        else "off"
        if delta <= -threshold
        else "both"
    )
    return {
        "mode": "auto",
        "decision": decision,
        "member_star": bool(delta >= 0.0),
        "delta_log_z": float(delta),
        "log_z_star": log_z[True],
        "log_z_no_star": log_z[False],
        "star_model": s["star_model"],
        "central_sigma": float(spec.gp_sigmas[0]),
        "central_length_beam": float(spec.gp_lengths[0]),
    }


# === STAGE: HYPERPARAMETERS ===


def _edges(grid):
    """Which edges of its own grid the evidence optimum is on."""
    best = grid.best
    lengths, sigmas = grid.lengths_mas[best.variant], grid.sigmas[best.variant]
    i, j = lengths.index(best.length_mas), sigmas.index(best.sigma)
    edges = []
    for name, k, n in (("length", i, len(lengths)), ("sigma", j, len(sigmas))):
        if k == 0:
            edges.append(f"{name} min")
        if k == n - 1:
            edges.append(f"{name} max")
    return edges


def _plot_grid(p, grid):
    import matplotlib.pyplot as plt

    variants = sorted(grid.log_evidence)
    fig, axes = plt.subplots(
        1, len(variants), figsize=(4.5 * len(variants), 3.8), squeeze=False
    )
    for ax, name in zip(axes[0], variants):
        z = onp.asarray(grid.log_evidence[name])
        shown = onp.where(onp.isfinite(z), z, onp.nan)
        im = ax.imshow(
            shown - onp.nanmax(shown), origin="lower", cmap="viridis"
        )
        ax.set_xticks(range(len(grid.sigmas[name])))
        ax.set_xticklabels([f"{v:.3g}" for v in grid.sigmas[name]])
        ax.set_yticks(range(len(grid.lengths_mas[name])))
        ax.set_yticklabels([f"{v:.3g}" for v in grid.lengths_mas[name]])
        ax.set_xlabel("σ")
        ax.set_ylabel("ℓ (mas)")
        ax.set_title(f"{name}: log Z − max")
        fig.colorbar(im, ax=ax)
    _save(fig, p.output / "plots" / "hyperparameters.png")


def _plot_lcurve(p, curve, index):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.loglog(curve.chi2, curve.penalty, "o-")
    ax.loglog(curve.chi2[index], curve.penalty[index], "o", ms=10, mfc="none")
    ax.set_xlabel("χ²")
    ax.set_ylabel("regularizer")
    ax.set_title("L-curve (circle: the weight used)")
    _save(fig, p.output / "plots" / "hyperparameters.png")


def _hyperparameters(p):
    from ..ensemble import evidence_grid, run_group
    from ..imaging import error_scale

    s, data = p.settings, p.processed
    draw, spec = _draw(p)
    star = _star_on(p)
    options = _fit_options(s)
    starts = _starts(data, star)
    folder = p.output / "models" / "hyper_best"
    if draw.family == "gp":
        grid = evidence_grid(
            data, draw, star=star, starts=starts, spec=spec, **options
        )
        best = grid.best
        _save_fit(folder, best.result, list(best.result.values))
        groups = {}
        for name in sorted(grid.log_evidence):
            groups[f"evidence_{name}"] = {
                "lengths_mas": onp.asarray(grid.lengths_mas[name]),
                "sigmas": onp.asarray(grid.sigmas[name]),
                "log_evidence": grid.log_evidence[name],
            }
        _io.write_h5(p.output / "grids.h5", groups)
        _plot_grid(p, grid)
        path = "env" if star else None
        scales = error_scale(best.result, data, path, by_observable=True)
        gamma = _gamma(best.result.model, data, star)
        ls = grid.lengths_mas[best.variant]
        minor = p._report("overview")["beam"]["minor_mas"]
        return {
            "method": "evidence",
            "sigma": float(best.sigma),
            "length_mas": float(best.length_mas),
            "length_beam": float(best.length_mas / minor),
            "variant": best.variant,
            "axis_ratio": grid.axis_ratio,
            "pa_deg": grid.pa_deg,
            "log_z": float(best.log_evidence),
            "grid_shape": [len(ls), len(grid.sigmas[best.variant])],
            "interior": bool(grid.interior),
            "edge": _edges(grid),
            "scale_by_observable": {k: float(v) for k, v in scales.items()},
            "gamma": float(gamma),
            "converged": bool(best.result.info.get("converged", True)),
        }
    group = run_group(
        data, draw, star=star, starts=starts, spec=spec, **options
    )
    curve = group.curve
    note = None
    try:
        corner = curve.corner()
    except ValueError as err:
        weights = onp.sort(onp.asarray(curve.weights))
        corner = float(weights[len(weights) // 2])
        note = f"no L-curve corner ({err}); the median weight was used"
    index = int(
        onp.argmin(
            onp.abs(onp.log(onp.asarray(curve.weights)) - onp.log(corner))
        )
    )
    result = curve.results[index]
    _save_fit(folder, result, list(result.values))
    _io.write_h5(
        p.output / "grids.h5",
        {
            "lcurve": {
                "weights": onp.asarray(curve.weights),
                "chi2": onp.asarray(curve.chi2),
                "penalty": onp.asarray(curve.penalty),
            }
        },
    )
    _plot_lcurve(p, curve, index)
    return {
        "method": "lcurve",
        "regularizer": draw.family,
        "weight": float(curve.weights[index]),
        "corner": float(corner),
        "note": note,
        "interior": True,
        "edge": None,
        "scale_by_observable": None,
        "gamma": None,
        "converged": bool(result.info.get("converged", True)),
    }


def _gamma(model, data, star):
    """MacKay's number of well-determined latent parameters of ``model``.

    A star-free model is the Image itself, which has no path; its γ is that
    of the same Image as the only component of a ``System``.
    """
    from ..imaging import chi2_by_observable
    from ..models import System

    if star:
        return chi2_by_observable(model, data, "env")["n_gamma"]
    return chi2_by_observable(System(env=model), data, "env")["n_gamma"]


# === STAGE: MEMBERS (ONE) ===


def _load_fit(folder):
    """A saved fit's model, values and info."""
    return (
        _io.load_model(folder),
        _io.load_model_values(folder),
        _io.read_json(folder / "info.json"),
    )


def _members(p):
    """The single member: the evidence grid's (or L-curve's) winner.

    TODO (ensemble PR): ``n_members`` CLEAN-started fits at the chosen
    hyperparameters, each with its own seeded draw, resumable on its own
    (``run_member(k)``); each is written to ``members/<k>/``.
    """
    from ..imaging import chi2_by_observable

    t0 = time.perf_counter()
    data = p.processed
    star = _star_on(p)
    model, values, info = _load_fit(p.output / "models" / "hyper_best")
    folder = p.output / "members" / "0"
    _io.save_model(folder, model, values, list(values))
    _io.write_json(folder / "info.json", info)
    by_observable = chi2_by_observable(model, data)
    report = {
        "k": 0,
        "source": "hyperparameters",
        "star": bool(star),
        "chi2": by_observable["chi2"],
        "chi2_red": by_observable["chi2_red"],
        "converged": bool(info.get("converged", True)),
        "steps": info.get("steps"),
        "seconds": time.perf_counter() - t0,
    }
    _io.write_json(folder / "report.json", report)
    return {
        "n_members": 1,
        "n_kept": 1,
        "chi2_red_by_observable": by_observable["chi2_red"],
    }


# === STAGE: COMBINE (PASS-THROUGH) ===


def _sky_image(model, star):
    """``(pixels, pixel_scale_mas, dra, ddec, star_fraction)`` of the mean.

    The pixels are the extended emission on the sky grid, carrying its share
    of the total flux (the star's flux is in ``star_fraction``).
    """
    from ..ensemble import _on_common_grid, _parts

    env, fraction = _parts(model, star)
    n = int(env.eta.shape[0])
    scale = float(env.pixel_scale_mas)
    pixels = _on_common_grid(env, n, scale)
    pixels = onp.asarray(pixels, dtype=float)
    pixels = pixels / pixels.sum() * fraction
    return (
        pixels,
        scale,
        float(env.dra),
        float(env.ddec),
        1.0 - fraction,
    )


def _plot_image(p, pixels, scale, dra, ddec, beam, star_fraction):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    n = pixels.shape[0]
    half = 0.5 * n * scale
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(
        pixels,
        origin="upper",
        extent=(dra + half, dra - half, ddec - half, ddec + half),
        cmap="inferno",
    )
    fig.colorbar(im, ax=ax, label="relative flux per pixel")
    ax.add_patch(
        Ellipse(
            (dra + 0.8 * half, ddec - 0.8 * half),
            beam["minor_mas"],
            beam["major_mas"],
            angle=-beam["pa_deg"],
            fill=False,
            ec="w",
        )
    )
    ax.set_xlabel("ΔRA (mas, East left)")
    ax.set_ylabel("ΔDec (mas)")
    ax.set_title(f"mean image (star: {100 * star_fraction:.3g}% of the flux)")
    _save(fig, p.output / "plots" / "image_mean.png")


def _plot_residual_image(p, residual, field):
    import matplotlib.pyplot as plt

    half = 0.5 * field
    limit = max(float(onp.abs(residual).max()), 1e-30)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(
        residual,
        origin="upper",
        extent=(half, -half, -half, half),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    fig.colorbar(im, ax=ax, label="normalized χ² gradient")
    ax.set_xlabel("ΔRA (mas, East left)")
    ax.set_ylabel("ΔDec (mas)")
    ax.set_title("residual image (positive: needs more flux)")
    _save(fig, p.output / "plots" / "image_residual.png")


def _plot_residual_frequency(p, model, data):
    import matplotlib.pyplot as plt

    from ..imaging import _observable_blocks
    from ..likelihood import whitened_residuals

    rows, _ = _observable_blocks([data])
    r = onp.asarray(whitened_residuals(model, data), dtype=float)
    rho = onp.ravel(_spatial_frequencies(data))
    fig, axes = plt.subplots(
        1, len(rows), figsize=(4.5 * len(rows), 3.4), squeeze=False
    )
    for ax, (kind, idx) in zip(axes[0], rows.items()):
        idx = onp.asarray(idx)
        x = rho[idx] if idx.size == rho.size or kind == "vis" else None
        if x is None or x.size != idx.size:
            x = onp.arange(idx.size)
            ax.set_xlabel("sample")
        else:
            ax.axvline(onp.quantile(x, 0.2), color="C1", ls=":")
            ax.set_xlabel("B/λ (cycles per radian); dotted: shortest 20%")
        ax.plot(x, r[idx], ".", ms=4)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_title(f"{kind}: (model − data)/σ")
    _save(fig, p.output / "plots" / "residual_frequency.png")


def _combine(p):
    """Pass-through: the mean of one member is that member.

    TODO (ensemble PR): the generalized ``virgil.ensemble.combine`` (χ²
    filter, MAD cut, alignment, running-mean acceptance), ``images/mean``
    extensions ``STD`` and ``NMEMBERS``, and the best single member.
    """
    from ..imaging import (
        Centroid,
        chi2_by_observable,
        diagnose,
        residual_image,
        short_baseline_residual,
    )

    s, data = p.settings, p.processed
    ov = p._report("overview")
    star = _star_on(p)
    model, values, info = _load_fit(p.output / "members" / "0")
    folder = p.output / "models" / "best"
    _io.save_model(folder, model, values, list(values))
    _io.write_json(folder / "info.json", info)

    evidence = p._report("hyperparameters")["method"] == "evidence"
    by_observable = chi2_by_observable(model, data)
    gamma = _gamma(model, data, star) if evidence else None
    try:
        sb_mean, sb_error = short_baseline_residual(model, data)
    except ValueError:
        sb_mean = sb_error = None
    regularizers = () if star else (Centroid(ov["pixel_scale_mas"]),)
    diagnosis = diagnose(model, data, regularizers)
    n = int(min(ov["npix"], s["clean_max_npix"]))
    field = ov["field_mas"]
    residual = residual_image(model, data, n, field / n)

    pixels, scale, dra, ddec, star_fraction = _sky_image(model, star)
    _io.write_fits_image(
        p.output / "images" / "mean.fits",
        pixels,
        scale,
        dra=dra,
        ddec=ddec,
        beam=ov["beam"],
        header={"STARFRAC": star_fraction, "NMEMBERS": 1},
    )
    _plot_image(p, pixels, scale, dra, ddec, ov["beam"], star_fraction)
    _plot_residual_image(p, residual, field)
    _plot_residual_frequency(p, model, data)

    scales = p._report("hyperparameters")["scale_by_observable"]
    rescaled = None
    if scales:
        rescaled = {
            k: by_observable["chi2"][k]
            / (scales[k] ** 2 * by_observable["n"][k])
            for k in scales
            if k in by_observable["chi2"]
        }
    n_total = by_observable["n"]["total"]
    checks = diagnosis.checks
    return {
        "chi2": by_observable["chi2"]["total"],
        "chi2_reduced": by_observable["chi2_red"]["total"],
        "chi2_by_observable": by_observable["chi2"],
        "n_by_observable": by_observable["n"],
        "chi2_red_by_observable": by_observable["chi2_red"],
        "best_member_chi2_reduced": by_observable["chi2_red"]["total"],
        "chi2_rescaled_by_observable": rescaled,
        "short_baseline_mean": sb_mean,
        "short_baseline_error": sb_error,
        "short_baseline_mean_sigma": (
            None if sb_mean is None else sb_mean / sb_error
        ),
        "flip_dchi2": float(checks["flip_dchi2"]),
        "edge_flux": float(max(checks["edge_flux"])),
        "anchored": bool(checks["anchored"]),
        "gamma": gamma,
        "n_minus_gamma": None if gamma is None else n_total - gamma,
        "star_fraction": float(star_fraction),
        "flux_ratio": (
            float(model.env.flux / model.star.flux) if star else None
        ),
        "pixel_scale_mas": scale,
        "npix": int(pixels.shape[0]),
        "diagnose_warnings": list(diagnosis.warnings),
    }


# === SUMMARY AND CHECKS ===


def _section(reports, name, keys):
    report = reports.get(name)
    if report is None:
        return None
    return {k: report.get(k) for k in keys}


_LOAD_KEYS = (
    "n_independent",
    "n_by_observable",
    "flagged_fraction",
    "n_channels",
    "wavel_min_m",
    "wavel_max_m",
    "quoted_errors",
)
_FIELD_KEYS = (
    "field_mas",
    "field_source",
    "pixel_scale_mas",
    "npix",
    "nyquist_mas",
    "beam",
    "fwhm_mas",
    "ceiling_mas",
)
_START_KEYS = (
    "envelope_fwhm_mas",
    "envelope_flux",
    "clean",
    "clean_chi2_red",
    "clean_stop",
    "clean_flux_fraction",
)
_STAR_KEYS = ("mode", "decision", "delta_log_z", "star_model")
_HYPER_KEYS = (
    "method",
    "sigma",
    "length_mas",
    "length_beam",
    "variant",
    "axis_ratio",
    "pa_deg",
    "log_z",
    "grid_shape",
    "interior",
    "edge",
    "regularizer",
    "weight",
    "corner",
    "note",
)
_FIT_KEYS = (
    "chi2",
    "chi2_reduced",
    "chi2_red_by_observable",
    "best_member_chi2_reduced",
    "short_baseline_mean_sigma",
    "flip_dchi2",
    "edge_flux",
)


def _run_checks(settings, reports):
    """The checks of the stages that ran, in the order of ``CHECKS``."""
    load, star = reports.get("load"), reports.get("star")
    hyper, combine = reports.get("hyperparameters"), reports.get("combine")
    checks = []
    if combine is not None:
        red = combine["chi2_red_by_observable"]
        checks.append(_checks_chi2("chi2_reduced", red["total"], "the image"))
        for kind in ("vis", "phi"):
            if kind in red:
                checks.append(
                    _checks_chi2(
                        f"chi2_reduced_{kind}",
                        red[kind],
                        f"the {kind} block of the image",
                    )
                )
    if load is not None:
        checks.append(_checks.quoted_errors(load["quoted_errors"]))
    biased = False
    if hyper is not None:
        scales = hyper["scale_by_observable"]
        scale_check = _checks.error_scale_both_ways(scales)
        checks.append(scale_check)
        biased = scale_check.status == "fail"
    if combine is not None and load is not None:
        checks.append(
            _checks.effective_parameters(
                load["n_independent"], combine["gamma"]
            )
        )
        short = _checks.short_baselines(
            combine["short_baseline_mean"], combine["short_baseline_error"]
        )
        checks.append(short)
        biased = biased or short.status == "fail"
    if hyper is not None:
        checks.append(
            _checks.hyperparameter_edge(hyper["edge"], biased=biased)
        )
    if star is not None:
        checks.append(
            _checks.star_decision(
                star["delta_log_z"],
                threshold=settings["star_threshold"],
                decision=star["decision"],
                fixed=settings["star"] is True,
            )
        )
    if combine is not None and load is not None:
        checks.append(
            _checks.orientation(combine["flip_dchi2"], load["closure_phases"])
        )
        checks.append(_checks.edge_flux(combine["edge_flux"]))
        red = combine["chi2_red_by_observable"]
        checks += [
            _checks_chi2(f"chi2_reduced_{kind}", value, f"the {kind} block")
            for kind, value in red.items()
            if kind not in ("total", "vis", "phi")
        ]
    return checks, biased


def _checks_chi2(name, value, label):
    return _checks.chi2_reduced(name, value, label=label)


def _summarize(settings, reports):
    checks, biased = _run_checks(settings, reports)
    sections = {}
    for key, report, keys in (
        ("data", reports.get("load"), _LOAD_KEYS),
        ("field", reports.get("overview"), _FIELD_KEYS),
        ("start", reports.get("start"), _START_KEYS),
        ("star", reports.get("star"), _STAR_KEYS),
    ):
        if report is not None:
            sections[key] = {k: report.get(k) for k in keys}
    hyper, combine = reports.get("hyperparameters"), reports.get("combine")
    load = reports.get("load")
    if load is not None:
        sections["data"]["has_closure_phases"] = load["closure_phases"]
        sections["data"]["chromatic"] = (
            "grey: with several channels the image is one grey image "
            "(per-channel and SPARCO imaging arrive with the channels stage)"
            if load["n_channels"] > 1
            else "single channel"
        )
    if hyper is not None:
        section = {k: hyper.get(k) for k in _HYPER_KEYS}
        section["trusted"] = not biased
        sections["hyperparameters"] = section
        sections["errors"] = {
            "scale_by_observable": hyper["scale_by_observable"],
            "gamma": hyper["gamma"] if combine is None else combine["gamma"],
            "n_minus_gamma": None
            if combine is None
            else combine["n_minus_gamma"],
            "chi2_rescaled_by_observable": (
                None
                if combine is None
                else combine["chi2_rescaled_by_observable"]
            ),
            "note": "rescaled χ² is labelled and never checked; all checks "
            "use the quoted errors",
        }
    if combine is not None:
        sections["fit"] = {k: combine.get(k) for k in _FIT_KEYS}
        sections["fit"]["n_by_observable"] = combine["n_by_observable"]
        sections["ensemble"] = {
            "n_members": 1,
            "n_kept": 1,
            "note": "one member in this version: the sigma map and the "
            "ensemble checks arrive with the ensemble stage, and the "
            "hyperparameters are shared by every member",
        }
    return sections, checks
