"""`ImagingPipeline`: a data-consistent image of the source (provisional).

It reports an image that fits the data on the quoted errors, at
hyperparameters chosen by a stated rule, and cannot promise that the
morphology is right: every data-fidelity check passed on contest
reconstructions that were judged unusable.
"""

from __future__ import annotations

import dataclasses

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

# The checks, in reporting order.
CHECKS = (
    "chi2_reduced",
    "chi2_reduced_vis",
    "chi2_reduced_phi",
    "quoted_errors",
    "error_scale",
    "effective_parameters",
    "short_baselines",
    "star",
    "orientation",
    "edges",
)
_ANISOTROPIC_BELOW = 0.9  # beam axis ratio that adds the elliptical GP grid


class ImagingPipeline(_Pipeline):
    """Image the source from V², closure phases or complex visibilities.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The image fits the data on the quoted errors; it does **not** promise
    that the morphology is right (contest reconstructions that passed every
    data-fidelity check were judged unusable). Multi-channel data give one
    grey image.

    Stages: ``load → overview → star → hyperparameters → combine →
    quicklook``. ``overview`` sets the grid; ``star="auto"`` compares fits
    with and without a star by log evidence (|ΔlogZ| ≥ ``star_threshold``
    decides); ``hyperparameters`` picks the Gaussian-process σ and ℓ by the
    Laplace evidence and saves ``models/best``; ``combine`` writes
    ``images/mean.fits`` and the diagnostics. Every χ²/N is on the quoted
    errors.

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
    _OWNED_EXTRA = ("images",)
    STABILITY = "provisional"
    STAGES = (
        "load",
        "overview",
        "star",
        "hyperparameters",
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
            if not s[name] or min(s[name]) <= 0:
                raise ValueError(f"{name} must be positive numbers.")
        if s["star"] not in ("auto", True, False):
            raise ValueError('star must be "auto", True or False.')

    def _model_digest(self):
        return {"class": "Image", "digest": None, "models": ["gp"]}

    def _stages(self):
        model = tuple(
            f"models/best/{f}"
            for f in ("manifest.json", "values.npz", "info.json")
        )
        return (
            Stage("load", _load, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage("star", _star, ("stages/star/report.json",)),
            Stage("hyperparameters", _hyperparameters, ("grids.h5", *model)),
            Stage("combine", _combine, ("images/mean.fits",)),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        checks = _run_checks(self.settings, reports)
        names = {"load": "data", "overview": "field", "combine": "fit"}
        sections = {names.get(k, k): dict(v) for k, v in reports.items()}
        return sections, checks


# === HELPERS ===


def _grid_spec(s, field_mas, *, central=False):
    """The ensemble spec of this run's one group; ``central`` is the single
    middle-σ, middle-ℓ isotropic point of the star comparison."""
    from ..ensemble import EnsembleSpec

    sigmas, lengths = sorted(s["sigmas"]), sorted(s["lengths_beam"])
    if central:
        sigmas, lengths = (
            sigmas[len(sigmas) // 2 :][:1],
            lengths[len(lengths) // 2 :][:1],
        )
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
        gp_axis_ratio=None if central else _ANISOTROPIC_BELOW,
    )


def _draw(p, spec, star):
    from ..ensemble import draw_groups

    return draw_groups(
        p.processed,
        1,
        jax.random.PRNGKey(int(p.settings["seed"])),
        spec,
        star=star,
    )[0]


def _evidence_grid(p, spec, star):
    from ..ensemble import evidence_grid, reference_starts

    s, data = p.settings, p.processed
    return evidence_grid(
        data,
        _draw(p, spec, star),
        star=star,
        starts=reference_starts(data, star, ("moments",)),
        spec=spec,
        **{k: s[k] for k in ("method", "max_steps") if s[k] is not None},
    )


# === STAGES ===


def _block_stats(values):
    values = onp.asarray(values, dtype=float).ravel()
    good = values[onp.isfinite(values) & (values > 0)]
    equal = good.size and onp.ptp(good) <= 1e-6 * good.max()
    return {"all_equal": bool(equal), "n_bad": int(values.size - good.size)}


def _load(p):
    from ..imaging import _observable_blocks

    data = p.processed
    _, counts = _observable_blocks([data])
    blocks = {"vis": data.d_vis, "phi": data.d_phi}
    return _load_report(
        p,
        n_by_observable={k: int(v) for k, v in counts.items()},
        quoted_errors={
            k: _block_stats(v) for k, v in blocks.items() if onp.size(v)
        },
    )


def _overview(p):
    from ..imaging import beam, envelope_fwhm, image_field

    s, data = p.settings, p.processed
    _overview_plots(p, data)
    candidate = s["star"] is not False
    field = s["field_mas"] or float(
        image_field(
            data,
            fwhm_mas=float(envelope_fwhm(data, candidate)),
            field_fwhms=s["field_fwhms"],
        )
    )
    draw = _draw(p, _grid_spec(s, field), candidate)
    return {
        "field_requested_mas": float(field),
        "pixel_scale_mas": float(draw.pixel_scale_mas),
        "npix": int(draw.npix),
        "beam": dataclasses.asdict(beam(data)),
    }


def _star(p):
    s = p.settings
    if s["star"] != "auto":
        return {
            "decision": "on" if s["star"] else "off",
            "member_star": bool(s["star"]),
            "delta_log_z": None,
        }
    spec = _grid_spec(
        s, p._report("overview")["field_requested_mas"], central=True
    )
    log_z = {
        star: float(_evidence_grid(p, spec, star).best.log_evidence)
        for star in (True, False)
    }
    delta = log_z[True] - log_z[False]
    threshold = float(s["star_threshold"])
    return {
        "decision": "on"
        if delta >= threshold
        else "off"
        if delta <= -threshold
        else "both",
        "member_star": bool(delta >= 0.0),
        "delta_log_z": float(delta),
        "log_z_star": log_z[True],
        "log_z_no_star": log_z[False],
    }


def _edges(grid):
    """Which edges of its own grid the evidence optimum is on."""
    best = grid.best
    axes = (
        ("length", grid.lengths_mas[best.variant], best.length_mas),
        ("sigma", grid.sigmas[best.variant], best.sigma),
    )
    return [
        f"{name} {side}"
        for name, values, at in axes
        for side, end in (("min", values[0]), ("max", values[-1]))
        if at == end
    ]


def _hyperparameters(p):
    from ..imaging import chi2_by_observable, error_scale
    from ..models import System

    data, star = p.processed, p._report("star")["member_star"]
    ov = p._report("overview")
    grid = _evidence_grid(
        p, _grid_spec(p.settings, ov["field_requested_mas"]), star
    )
    best = grid.best
    _save_fit(
        p.output / "models" / "best", best.result, list(best.result.values)
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
    return {
        "sigma": float(best.sigma),
        "length_mas": float(best.length_mas),
        "length_beam": float(best.length_mas / ov["beam"]["minor_mas"]),
        "variant": best.variant,
        "log_z": float(best.log_evidence),
        "edge": _edges(grid),
        "scale_by_observable": {k: float(v) for k, v in scales.items()},
        # MacKay's well-determined parameters; a star-free model is a bare
        # Image, so it is wrapped to give chi2_by_observable a path.
        "gamma": float(
            chi2_by_observable(
                best.result.model if star else System(env=best.result.model),
                data,
                "env",
            )["n_gamma"]
        ),
    }


def _combine(p):
    """The image and the diagnostics of the one fit."""
    from ..ensemble import _on_common_grid, _parts
    from ..imaging import (
        Centroid,
        chi2_by_observable,
        diagnose,
        short_baseline_residual,
    )

    data, star = p.processed, p._report("star")["member_star"]
    ov = p._report("overview")
    model = _io.load_model(p.output / "models" / "best")
    by_observable = chi2_by_observable(model, data)
    try:
        sb_mean, sb_error = short_baseline_residual(model, data)
    except ValueError:  # no baseline-ordered visibilities
        sb_mean = sb_error = None
    diagnosis = diagnose(
        model, data, () if star else (Centroid(ov["pixel_scale_mas"]),)
    )

    env, fraction = _parts(model, star)
    scale = float(env.pixel_scale_mas)
    pixels = onp.asarray(
        _on_common_grid(env, int(env.eta.shape[0]), scale), dtype=float
    )
    pixels *= fraction / pixels.sum()
    # A rotated image is rendered on the sky grid with its offset in the
    # pixels; an unrotated one is resampled about its own centre.
    offset = (0.0, 0.0) if float(env.rotation_deg) else (env.dra, env.ddec)
    _io.write_fits_image(
        p.output / "images" / "mean.fits",
        pixels,
        scale,
        dra=float(offset[0]),
        ddec=float(offset[1]),
        header={"STARFRAC": 1.0 - fraction},
    )
    return {
        "chi2": by_observable["chi2"]["total"],
        "chi2_reduced": by_observable["chi2_red"]["total"],
        "n_by_observable": by_observable["n"],
        "chi2_red_by_observable": by_observable["chi2_red"],
        "short_baseline_mean": sb_mean,
        "short_baseline_error": sb_error,
        "flip_dchi2": float(diagnosis.checks["flip_dchi2"]),
        "edge_flux": float(max(diagnosis.checks["edge_flux"])),
        "star_fraction": 1.0 - fraction,
    }


def _run_checks(settings, reports):
    """The checks of the stages that ran, in the order of ``CHECKS``."""
    load, star = reports.get("load"), reports.get("star")
    hyper, fit = reports.get("hyperparameters"), reports.get("combine")
    checks = []
    if fit:
        red = fit["chi2_red_by_observable"]
        checks += [
            _checks.chi2_reduced(
                name, red[kind], label=f"the {kind} of the image"
            )
            for name, kind in (
                ("chi2_reduced", "total"),
                ("chi2_reduced_vis", "vis"),
                ("chi2_reduced_phi", "phi"),
            )
            if kind in red
        ]
    if load:
        checks.append(_checks.quoted_errors(load["quoted_errors"]))
    if hyper:
        checks.append(
            _checks.error_scale(hyper["scale_by_observable"], kind="estimated")
        )
    if fit and load:
        checks += [
            _checks.effective_parameters(
                load["n_independent"], hyper["gamma"]
            ),
            _checks.short_baselines(
                fit["short_baseline_mean"], fit["short_baseline_error"]
            ),
        ]
    if star:
        checks.append(
            _checks.star_decision(
                star["delta_log_z"],
                threshold=settings["star_threshold"],
                decision=star["decision"],
                fixed=settings["star"] is True,
            )
        )
    if fit and load:
        checks += [
            _checks.orientation(fit["flip_dchi2"], load["closure_phases"]),
            _checks.edges(hyper["edge"], fit["edge_flux"]),
        ]
    return checks
