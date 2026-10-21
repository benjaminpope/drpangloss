"""`ImagingPipeline`: a data-consistent image of the source (provisional).

It reports an image that fits the data on the quoted errors, from stated
starts, at hyperparameters chosen by a stated rule, and cannot promise that
the morphology is right: every data-fidelity check passed on contest
reconstructions that were judged unusable. Design: ``design/imaging_pipeline.md``.
"""

from __future__ import annotations

import jax
import numpy as onp

from . import _checks, _io
from ._common import (
    _integer,
    _load_report,
    _overview_plots,
    _positive,
    _quicklook,
    _save_fit,
)
from ._core import Stage, _Pipeline

# The checks, in reporting order. The ensemble (PR C) and channel (PR D)
# checks append to this.
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
_MODEL_FILES = ("manifest.json", "values.npz", "info.json")
_ANISOTROPIC_BELOW = (
    0.9  # axis ratio of the beam that adds the elliptical grid
)


class ImagingPipeline(_Pipeline):
    """Image the source from V², closure phases or complex visibilities.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The image fits the data on the quoted errors, to a χ²/N per observable
    that the summary states. It does **not** promise that the morphology is
    right: on contest data, reconstructions that passed every
    data-fidelity check were judged "terrible" or "a chain of dots". With
    several channels the image is one grey image.

    Stages: ``load → overview → start → star → hyperparameters → members →
    combine → quicklook``. ``overview`` sets the grid (pixels ``oversample``
    times finer than Nyquist, over the field of
    [`image_field`][virgil.imaging.image_field] or ``field_mas``, at most
    ``max_npix`` across). ``star`` decides whether the scene has an
    unresolved star: with ``star="auto"`` Gaussian-process fits with and
    without one are compared by log evidence, and |ΔlogZ| of at least
    ``star_threshold`` nats decides (else the larger evidence is kept,
    decision ``"both"``). ``hyperparameters`` chooses the Gaussian-process
    amplitude σ and length ℓ by the Laplace evidence
    ([`evidence_grid`][virgil.ensemble.evidence_grid]); the grid is not
    repeated on rescaled errors. There is one member in this version;
    ``combine`` writes ``images/mean.fits`` and ``models/best``.

    All χ²/N values are on the **quoted** errors; a rescaled χ² is stored
    beside them, labelled, and no check uses it.

    Parameters
    ----------
    data : OIData
        The data.
    model : {"gp"}, optional
        The image prior (the only one in this version).
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of ``ImagingPipeline.defaults()``: ``oversample``,
        ``field_fwhms``, ``field_mas``, ``max_npix`` (the grid); ``star``
        (``"auto"``, True, False) and ``star_threshold``; ``sigmas``,
        ``lengths_beam`` and ``grid_growths`` (the evidence grid);
        ``method`` and ``max_steps`` (the optimizer of
        [`fit`][virgil.fitting.fit]); ``seed``, ``wavel_range``,
        ``error_floor`` and ``quicklook_execute``.

    Examples
    --------
    >>> from virgil.pipeline import ImagingPipeline  # doctest: +SKIP
    >>> res = ImagingPipeline(data, output="runs/image").run()  # doctest: +SKIP
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
        "star": "auto",
        "star_threshold": 5.0,
        "sigmas": [1.0, 2.0, 4.0],
        "lengths_beam": [0.5, 1.0, 2.0],
        "grid_growths": 2,
        "method": None,
        "max_steps": None,
        "seed": 0,
        "wavel_range": None,
        "error_floor": None,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        if model not in (None, "gp"):
            raise ValueError(f'model must be "gp"; got {model!r}.')
        super().__init__(data, "gp", output=output, **settings)

    def _default_model(self):
        return "gp"

    @classmethod
    def _validate(cls, s):
        for name in ("oversample", "field_fwhms", "star_threshold"):
            _positive(s, name)
        if s["field_mas"] is not None:
            _positive(s, "field_mas")
        _integer(s, "max_npix", 4)
        _integer(s, "grid_growths", 0)
        _integer(s, "seed", 0)
        for name in ("sigmas", "lengths_beam"):
            if not s[name] or any(not v > 0 for v in s[name]):
                raise ValueError(f"{name} must be positive numbers.")
        if s["star"] not in ("auto", True, False):
            raise ValueError('star must be "auto", True or False.')
        if s["wavel_range"] is not None and len(s["wavel_range"]) != 2:
            raise ValueError("wavel_range must be [min, max] in metres.")
        if s["error_floor"] is not None and not isinstance(
            s["error_floor"], dict
        ):
            raise ValueError("error_floor must be a dict of absolute floors.")

    def _model_digest(self):
        return {"class": "Image", "digest": None, "models": ["gp"]}

    def _stages(self):
        def models(name):
            return tuple(f"models/{name}/{f}" for f in _MODEL_FILES)

        return (
            Stage("load", _load, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage("start", _start, ("stages/start/report.json",)),
            Stage("star", _star, ("stages/star/report.json",)),
            Stage(
                "hyperparameters",
                _hyperparameters,
                ("grids.h5", *models("hyper_best")),
            ),
            Stage(
                "members",
                _members,
                tuple(f"members/0/{f}" for f in _MODEL_FILES),
            ),
            Stage(
                "combine",
                _combine,
                ("images/mean.fits", *models("best")),
            ),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        return _summarize(self.settings, reports)


# === SHARED HELPERS ===


def _fit_options(s):
    """The keywords of ``fit`` that the settings set."""
    return {k: s[k] for k in ("method", "max_steps") if s[k] is not None}


def _grid_spec(s, field_mas, *, central=False):
    """The ensemble spec of the one group of this run.

    With ``central`` the grid is the single point at the middle amplitude
    and length (the star comparison): no growth, isotropic.
    """
    from ..ensemble import EnsembleSpec

    sigmas, lengths = sorted(s["sigmas"]), sorted(s["lengths_beam"])
    axis_ratio = _ANISOTROPIC_BELOW
    if central:
        sigmas = [sigmas[len(sigmas) // 2]]
        lengths = [lengths[len(lengths) // 2]]
        axis_ratio = None
    return EnsembleSpec(
        families=("gp",),
        oversample=(float(s["oversample"]),),
        field_factors=(1.0,),
        starts=("moments",),
        max_npix=int(s["max_npix"]),
        field_mas=field_mas,
        gp_sigmas=tuple(sigmas),
        gp_lengths=tuple(lengths),
        gp_growths=0 if central else int(s["grid_growths"]),
        gp_axis_ratio=axis_ratio,
    )


def _draw(p):
    """The group's ``Draw`` and the ensemble spec, from the overview report."""
    from ..ensemble import Draw

    ov = p._report("overview")
    d = ov["draw"]
    draw = Draw(
        0,
        "gp",
        int(d["npix"]),
        float(d["pixel_scale_mas"]),
        "moments",
        tuple(d["weights"]),
    )
    return draw, _grid_spec(p.settings, ov["field_requested_mas"])


def _star_on(p):
    """Whether the member has an analytic star: the star stage's choice."""
    return bool(p._report("star")["member_star"])


def _starts(data, star):
    from ..ensemble import reference_starts

    return reference_starts(data, star, ("moments",))


def _block_stats(values):
    values = onp.asarray(values, dtype=float).ravel()
    good = values[onp.isfinite(values) & (values > 0)]
    spread = float(good.max() - good.min()) if good.size else 0.0
    median = float(onp.median(good)) if good.size else None
    return {
        "min": float(good.min()) if good.size else None,
        "median": median,
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
        image_field,
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
    else:
        requested = float(s["field_mas"])
    draw = draw_groups(
        data,
        1,
        jax.random.PRNGKey(int(s["seed"])),
        _grid_spec(s, requested),
        star=candidate,
    )[0]
    return {
        "field_mas": float(draw.npix * draw.pixel_scale_mas),
        "field_requested_mas": requested,
        "pixel_scale_mas": float(draw.pixel_scale_mas),
        "npix": int(draw.npix),
        "beam": {
            "major_mas": float(b.major_mas),
            "minor_mas": float(b.minor_mas),
            "pa_deg": float(b.pa_deg),
        },
        "envelope_fwhm_mas": envelope,
        "draw": {
            "npix": int(draw.npix),
            "pixel_scale_mas": float(draw.pixel_scale_mas),
            "weights": [float(w) for w in draw.weights],
        },
    }


# === STAGE: START (ENVELOPE AND CLEAN) ===


def _start(p):
    """The envelope fit that sets the field and the starts.

    TODO (ensemble PR): gradient CLEAN here, writing ``images/clean.fits``
    and the starts of the ensemble's members.
    """
    from ..imaging import starting_image

    candidate = p.settings["star"] is not False
    start = starting_image(p.processed, star=candidate)
    env = start.env if candidate else start
    return {
        "envelope_fwhm_mas": p._report("overview")["envelope_fwhm_mas"],
        "envelope_flux": float(env.flux) if candidate else None,
    }


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


def _hyperparameters(p):
    from ..ensemble import evidence_grid
    from ..imaging import error_scale

    data = p.processed
    draw, spec = _draw(p)
    star = _star_on(p)
    grid = evidence_grid(
        data,
        draw,
        star=star,
        starts=_starts(data, star),
        spec=spec,
        **_fit_options(p.settings),
    )
    best = grid.best
    _save_fit(
        p.output / "models" / "hyper_best",
        best.result,
        list(best.result.values),
    )
    _io.write_h5(
        p.output / "grids.h5",
        {
            f"evidence_{name}": {
                "lengths_mas": onp.asarray(grid.lengths_mas[name]),
                "sigmas": onp.asarray(grid.sigmas[name]),
                "log_evidence": grid.log_evidence[name],
            }
            for name in sorted(grid.log_evidence)
        },
    )
    scales = error_scale(
        best.result, data, "env" if star else None, by_observable=True
    )
    minor = p._report("overview")["beam"]["minor_mas"]
    return {
        "sigma": float(best.sigma),
        "length_mas": float(best.length_mas),
        "length_beam": float(best.length_mas / minor),
        "variant": best.variant,
        "axis_ratio": grid.axis_ratio,
        "pa_deg": grid.pa_deg,
        "log_z": float(best.log_evidence),
        "grid_shape": [
            len(grid.lengths_mas[best.variant]),
            len(grid.sigmas[best.variant]),
        ],
        "interior": bool(grid.interior),
        "edge": _edges(grid),
        "scale_by_observable": {k: float(v) for k, v in scales.items()},
        "gamma": float(_gamma(best.result.model, data, star)),
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


def _members(p):
    """The single member: the evidence grid's winner.

    TODO (ensemble PR): ``n_members`` CLEAN-started fits, each resumable.
    """
    return _copy_fit(
        p.output / "models" / "hyper_best", p.output / "members" / "0"
    )


def _copy_fit(source, folder):
    model = _io.load_model(source)
    values = _io.load_model_values(source)
    _io.save_model(folder, model, values, list(values))
    _io.write_json(folder / "info.json", _io.read_json(source / "info.json"))
    return {"n_members": 1}


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


def _combine(p):
    """Pass-through: the mean of one member is that member.

    TODO (ensemble PR): the generalized ``virgil.ensemble.combine`` and the
    ``STD`` extension of ``images/mean.fits``.
    """
    from ..imaging import (
        Centroid,
        chi2_by_observable,
        diagnose,
        short_baseline_residual,
    )

    data = p.processed
    ov = p._report("overview")
    star = _star_on(p)
    _copy_fit(p.output / "members" / "0", p.output / "models" / "best")
    model = _io.load_model(p.output / "models" / "best")
    by_observable = chi2_by_observable(model, data)
    try:
        sb_mean, sb_error = short_baseline_residual(model, data)
    except ValueError:  # no baseline-ordered visibilities
        sb_mean = sb_error = None
    regularizers = () if star else (Centroid(ov["pixel_scale_mas"]),)
    diagnosis = diagnose(model, data, regularizers)

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

    hyper = p._report("hyperparameters")
    scales = hyper["scale_by_observable"]
    rescaled = {
        k: by_observable["chi2"][k] / (scales[k] ** 2 * by_observable["n"][k])
        for k in scales
        if k in by_observable["chi2"]
    }
    return {
        "chi2": by_observable["chi2"]["total"],
        "chi2_reduced": by_observable["chi2_red"]["total"],
        "n_by_observable": by_observable["n"],
        "chi2_red_by_observable": by_observable["chi2_red"],
        "chi2_rescaled_by_observable": rescaled,
        "short_baseline_mean": sb_mean,
        "short_baseline_error": sb_error,
        "flip_dchi2": float(diagnosis.checks["flip_dchi2"]),
        "edge_flux": float(max(diagnosis.checks["edge_flux"])),
        "gamma": hyper["gamma"],
        "n_minus_gamma": by_observable["n"]["total"] - hyper["gamma"],
        "star_fraction": float(star_fraction),
    }


# === SUMMARY AND CHECKS ===


def _run_checks(settings, reports):
    """The checks of the stages that ran, in the order of ``CHECKS``."""
    load, star = reports.get("load"), reports.get("star")
    hyper, combine = reports.get("hyperparameters"), reports.get("combine")
    checks = []
    if combine is not None:
        red = combine["chi2_red_by_observable"]
        checks.append(
            _checks.chi2_reduced(
                "chi2_reduced", red["total"], label="the image"
            )
        )
        for kind in ("vis", "phi"):
            if kind in red:
                checks.append(
                    _checks.chi2_reduced(
                        f"chi2_reduced_{kind}",
                        red[kind],
                        label=f"the {kind} block of the image",
                    )
                )
    if load is not None:
        checks.append(_checks.quoted_errors(load["quoted_errors"]))
    if hyper is not None:
        checks.append(
            _checks.error_scale(hyper["scale_by_observable"], kind="estimated")
        )
    if combine is not None and load is not None:
        checks.append(
            _checks.effective_parameters(
                load["n_independent"], combine["gamma"]
            )
        )
        checks.append(
            _checks.short_baselines(
                combine["short_baseline_mean"], combine["short_baseline_error"]
            )
        )
    if hyper is not None:
        checks.append(_checks.hyperparameter_edge(hyper["edge"]))
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
    return checks


def _summarize(settings, reports):
    checks = _run_checks(settings, reports)
    sections = {
        key: dict(reports[name])
        for key, name in (
            ("data", "load"),
            ("field", "overview"),
            ("start", "start"),
            ("star", "star"),
            ("hyperparameters", "hyperparameters"),
            ("fit", "combine"),
        )
        if name in reports
    }
    if "fit" in sections:
        sections["fit"]["rescaled_note"] = (
            "the rescaled χ² is labelled and never checked; all checks use "
            "the quoted errors"
        )
        sections["ensemble"] = {
            k: reports["members"][k]
            for k in ("n_members",)
            if "members" in reports
        }
    return sections, checks
