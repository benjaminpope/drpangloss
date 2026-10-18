"""Helpers shared by the pipelines: χ² and −2 log L (with error scales
profiled over a grid), residual moments, quantiles, geometry of the data,
prior-bound fractions and figure saving."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from scipy import stats

from .._grid import batch_size_or_default
from .._precision import cast_tree, in_float64, run_in
from ..likelihood import model_loglike
from . import _io

# Fraction of a prior's range, at either end, counted as "at the bound".
_BOUND_FRACTION = 0.01


def _bound_samples(prior, samples):
    """Fraction of ``samples`` in the outer ``_BOUND_FRACTION`` of the
    prior's finite range at either end, or ``None`` if it has no edges.

    The range is the one ``fit`` reports (``fitting._bound_fraction``): a
    LogUniform's in log x, an interval support's, and none for tied terms,
    angle vectors, or priors with a flat coordinate (their ends are poles).
    """
    from ..fitting import _bound_fraction

    x = onp.asarray(samples, dtype=float).ravel()
    fraction = _bound_fraction(prior, x)
    if fraction is None:
        return None
    return float(
        onp.mean(
            (fraction < _BOUND_FRACTION) | (fraction > 1 - _BOUND_FRACTION)
        )
    )


def _close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def _save(fig, path):
    with _io.atomic_path(path) as tmp:
        fig.savefig(tmp, dpi=110, bbox_inches="tight")
    _close(fig)


def _geometry(data):
    """Baseline and resolution figures of ``data`` (metres, mas).

    Only the unflagged samples count (``OIData.unflagged``): a flagged
    sample keeps its baseline in the data but measures nothing.
    """
    from .._geometry import _unflagged_samples, fringe_scales

    baseline, wavel, _ = _unflagged_samples(data)
    finest, coarsest = fringe_scales(data)
    return {
        "baseline_min_m": float(baseline.min()),
        "baseline_max_m": float(baseline.max()),
        "wavel_min_m": float(wavel.min()),
        "wavel_max_m": float(wavel.max()),
        # Half of the finest λ/B: inside it, separation and flux are
        # degenerate.
        "resolution_mas": float(0.5 * finest),
        # The coarsest λ/B: the field over which the coverage is
        # unambiguous.
        "fov_mas": float(coarsest),
        "lambda_over_b_mas": float(finest),
    }


def _spatial_frequencies(data):
    """The spatial frequency B/λ of every sample of ``data`` (cycles per
    radian), in the shape of ``data.u``."""
    u = onp.asarray(data.u, dtype=float)
    v = onp.asarray(data.v, dtype=float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, dtype=float), u.shape)
    return onp.hypot(u, v) / wavel


def _load_report(p, **extra):
    """The ``load`` stage: the data's summary report, and its OIFITS copy.

    ``extra`` entries are added to the report after the geometry. A stale
    ``processed.oifits`` of an earlier run is removed when the data cannot
    be written (``_io.oidata_tables`` returns no tables).
    """
    from ..oifits import write_oifits

    data = p.processed
    tables, reason = _io.oidata_tables(data)
    report = {
        "n_vis": int(onp.asarray(data.vis).size),
        "n_phi": int(onp.asarray(data.phi).size),
        "n_independent": int(data.n_independent),
        "closure_phases": bool(data.cp_flag),
        **_geometry(data),
        **extra,
        "data_sha256": _io.data_fingerprint(data),
        "processed_oifits_skipped": reason,
    }
    path = p.output / "data" / "processed.oifits"
    if tables is None:
        path.unlink(missing_ok=True)  # not an earlier run's data
    else:
        with _io.atomic_path(path) as tmp:
            write_oifits(tables, tmp)
    return report


def _chi2(model, data, **noise):
    from ..likelihood import whitened_residuals

    r = onp.asarray(whitened_residuals(model, data, **noise), dtype=float)
    return float(onp.sum(r**2)), r


def _m2loglike(model, data, **noise):
    """−2 log L of ``model``, including the error terms' normalization.

    This is the score for comparing models and lobes when error scales are
    fitted: at each candidate's own scales χ²/N is 1 by construction, but
    −2 log L keeps the −N log s term that penalizes inflated errors. On
    the quoted errors it differs from χ² by a constant.
    """
    return -2.0 * float(model_loglike(model, data, **noise))


def _overview_plots(p, data):
    """Save the overview of ``data`` and its (u, v) coverage."""
    from ..plotting import plot_oidata_overview, plot_uv_coverage

    fig, _ = plot_oidata_overview(data)
    _save(fig, p.output / "plots" / "overview_data.png")
    fig, _ = plot_uv_coverage(data)
    _save(fig, p.output / "plots" / "overview_uv.png")


def _overview(p):
    _overview_plots(p, p.processed)
    return {}


def _quicklook(p):
    from . import _quicklook as ql

    ql.write_quicklook(p.output, execute=p.settings["quicklook_execute"])
    return {"executed": bool(p.settings["quicklook_execute"])}


def _scales_of(values):
    """The fitted error scales among a fit's ``values``, by name."""
    return {
        k.split(".", 1)[1]: float(v)
        for k, v in values.items()
        if k.startswith("noise.")
    }


@in_float64
def _chi2_64(model, data, **noise):
    return _chi2(model, data, **noise)


@in_float64
def _m2loglike_64(model, data, **noise):
    return _m2loglike(model, data, **noise)


def _scale_blocks(data, noise):
    """Where each fitted error scale applies: ``((key, start, n, low, high),
    ...)``, with the scale's name, the first row and number of independent
    rows of its block of ``whitened_residuals``, and its prior range.

    Concrete (hashable, so a jitted function can take it as static).
    """
    from ..epochs import _blocks

    ranges = {
        k: (float(d.low), float(d.high)) for k, d in (noise or {}).items()
    }
    return tuple(
        (f"{name}_scale", start, nu, *ranges[f"{name}_scale"])
        for name, start, _, nu, _ in _blocks(data)
        if f"{name}_scale" in ranges
    )


_GOLDEN_STEPS = (
    48  # golden-section steps in ln s: a bracket of 1e-10 of ln(100)
)


def _golden_max(f, lo, hi, steps=_GOLDEN_STEPS):
    """The maximum of a unimodal ``f`` on ``[lo, hi]`` by golden section
    (traceable; ``steps`` fixed)."""
    g = 0.5 * (np.sqrt(5.0) - 1.0)

    def step(_, bracket):
        a, b = bracket
        c, d = b - g * (b - a), a + g * (b - a)
        left = f(c) >= f(d)
        return np.where(left, a, c), np.where(left, d, b)

    a, b = jax.lax.fori_loop(0, steps, step, (lo, hi))
    return 0.5 * (a + b)


def _profiled_scales(model, data, blocks):
    """The error scales that maximize the likelihood of ``model``.

    Each scale multiplies the errors of one block of ``_scale_blocks`` and
    enters only that block's terms of ``model_loglike``, so each is
    maximized on its own, in ln s over its prior range. The block's log
    likelihood is unimodal in ln s: ``-χ²/(2s²) - N ln s`` for Gaussian
    blocks, and concave in the concentration κ = 1/(sσ)² for von Mises
    phases, whose maximum is not at ``sqrt(χ²/N)``. Traceable.
    """
    from ..likelihood import whitened_residuals

    r = whitened_residuals(model, data)
    scales = {
        key: np.clip(np.sqrt(np.sum(r[start : start + n] ** 2) / n), low, high)
        for key, start, n, low, high in blocks
    }
    for key, _, _, low, high in blocks:

        def loglike(log_s, key=key):
            return model_loglike(model, data, **{**scales, key: np.exp(log_s)})

        bounds = np.log(np.asarray([low, high], r.dtype))
        scales[key] = np.exp(_golden_max(loglike, bounds[0], bounds[1]))
    return scales


def _profiled_m2loglike(build, start, data, diam, blocks):
    """``(−2 log L, scales)`` of ``build({**start, "diam": diam})`` at its
    profiled error scales. Traceable."""
    model = build({**start, "diam": diam})
    scales = _profiled_scales(model, data, blocks)
    return -2.0 * model_loglike(model, data, **scales), scales


@eqx.filter_jit
def _m2loglike_grid_jit(build, start, data, diams, blocks, batch_size):
    def one(diam):
        return _profiled_m2loglike(build, start, data, diam, blocks)

    return jax.lax.map(one, diams, batch_size=batch_size)


@eqx.filter_jit
def _refine_diameter_jit(build, start, data, low, high, blocks):
    def loglike(log_diam):
        m2, _ = _profiled_m2loglike(
            build, start, data, np.exp(log_diam), blocks
        )
        return -0.5 * m2

    diam = np.exp(_golden_max(loglike, np.log(low), np.log(high)))
    return diam, *_profiled_m2loglike(build, start, data, diam, blocks)


def _m2loglike_grid(build, start, data, diams, noise=None, batch_size=None):
    """−2 log L of ``build({**start, "diam": d})`` over the diameters ``d``.

    One batched, jitted evaluation in float64. With ``noise`` (the
    pipeline's error-scale priors) the scales are profiled at every
    diameter (``_profiled_scales``) and the likelihood is evaluated at
    them. ``build`` should be a module-level function, so that the compiled
    function is reused between calls; ``start`` holds the other parameters.

    Returns
    -------
    tuple
        ``(m2loglike, scales)``: an array over the diameters and a dict of
        arrays by scale name (empty without ``noise``).
    """
    with run_in("float64"):
        data = cast_tree(data, "float64")
        m2, scales = _m2loglike_grid_jit(
            build,
            {k: np.asarray(v) for k, v in start.items()},
            data,
            np.asarray(diams),
            _scale_blocks(data, noise),
            batch_size_or_default(batch_size, data),
        )
        return onp.asarray(m2), {k: onp.asarray(v) for k, v in scales.items()}


def _refine_diameter(build, start, data, low, high, noise):
    """The diameter in ``[low, high]`` of least profiled −2 log L.

    Golden-section search in ln θ (the profiled likelihood is unimodal on
    a bracket this narrow), with the error scales of ``noise`` re-profiled
    at every trial diameter. ``build`` and ``start`` are as for
    ``_m2loglike_grid``.

    Returns
    -------
    tuple
        ``(diam, m2loglike, scales)``: floats and a dict of floats.
    """
    with run_in("float64"):
        data = cast_tree(data, "float64")
        diam, m2, scales = _refine_diameter_jit(
            build,
            {k: np.asarray(v) for k, v in start.items()},
            data,
            np.asarray(low),
            np.asarray(high),
            _scale_blocks(data, noise),
        )
        return (
            float(diam),
            float(m2),
            {k: float(v) for k, v in scales.items()},
        )


# Per-transition NUTS diagnostics kept in samples.h5.
_NUTS_FIELDS = (
    "diverging",
    "accept_prob",
    "adapt_state.step_size",
    "num_steps",
    "potential_energy",
)
# Settings recorded as attributes of samples.h5.
_SAMPLER_SETTINGS = (
    "num_warmup",
    "num_samples",
    "num_chains",
    "chain_method",
    "seed",
)


def _mcmc(posterior, settings, *, dense_mass):
    """A NUTS ``MCMC`` with the pipeline's sampler settings (not yet run)."""
    from numpyro.infer import MCMC, NUTS

    return MCMC(
        NUTS(posterior, dense_mass=dense_mass),
        num_warmup=settings["num_warmup"],
        num_samples=settings["num_samples"],
        num_chains=settings["num_chains"],
        chain_method=settings["chain_method"],
        progress_bar=False,
    )


# How far inside its prior a NUTS start is pulled, as a fraction of the
# prior's width, measured as ``fit`` measures ``at_bound`` (in log x for a
# LogUniform). A start exactly on a bound has an infinite unconstrained
# value. Inside, it stays finite in float32 and float64 but its size depends
# on the sampler's coordinate: about -7 (``logit(1e-3)``) in the flat
# coordinates, and about ``logit(inset ln(R) / R)`` for a LogUniform with
# R = high / low in numpyro's linear interval (about -15 for R = 5e4).
_START_INSET = 1e-3


def _interior_start(priors, start, inset=_START_INSET):
    """``start`` with every value pulled at least ``inset`` of its prior's
    width inside the prior's finite bounds, as ``fit`` measures them.

    The fit's best values sit on a bound when the data cannot constrain a
    parameter (a companion's flux in a non-detection, the diameter of an
    unresolved star), and numpyro cannot start there: the unconstrained
    value is infinite. Values already inside, and parameters without
    finite bounds, are returned as they were.
    """
    from .._flat import _base, _flat_coordinate
    from ..fitting import _bound_fraction

    out = dict(start)
    for name, value in start.items():
        prior = priors.get(name)
        fraction = None if prior is None else _bound_fraction(prior, value)
        if fraction is None or not onp.any(
            (fraction < inset) | (fraction > 1.0 - inset)
        ):
            continue
        flat = _flat_coordinate(prior)
        if flat is not None:
            _, from_flat, low, high = flat
        else:
            support = _base(prior).support
            low, high = support.lower_bound, support.upper_bound
            from_flat = lambda x: x  # noqa: E731
        low, high = onp.asarray(low, float), onp.asarray(high, float)
        inside = onp.clip(fraction, inset, 1.0 - inset)
        out[name] = onp.asarray(
            from_flat(low + inside * (high - low)),
            dtype=onp.asarray(value).dtype,
        )
    return out


def _run_chains(posterior, settings, start, *, dense_mass, priors=None):
    """Run NUTS from ``start`` (one jittered start per chain); the finished MCMC.

    With ``priors`` (by sample site), the start is first pulled just inside
    any bound it sits on (``_interior_start``). If numpyro still finds no
    finite log-density at the start, a ``ValueError`` says which values sit
    on a bound, instead of numpyro's own message.
    """
    import jax

    from ..likelihood import chain_init_params

    if priors is not None:
        start = _interior_start(priors, start)
    init_key, run_key = jax.random.split(jax.random.PRNGKey(settings["seed"]))
    try:
        init = chain_init_params(
            posterior, [start] * settings["num_chains"], key=init_key
        )
    except RuntimeError as error:
        if "valid initial parameters" not in str(error):
            raise
        raise ValueError(
            "NUTS cannot start from the best fit: the log-posterior or its "
            "gradient is not finite there. Start values: "
            + ", ".join(
                f"{k}={onp.asarray(v).tolist()}" for k, v in start.items()
            )
            + ". Check that the fitted model and its priors agree."
        ) from error
    mcmc = _mcmc(posterior, settings, dense_mass=dense_mass)
    mcmc.run(run_key, init_params=init, extra_fields=_NUTS_FIELDS)
    return mcmc


def _sample_stats(mcmc):
    """NUTS diagnostics per transition, ``(chain, draw)``, by clean name."""
    extra = mcmc.get_extra_fields(group_by_chain=True)
    return {
        ("step_size" if k == "adapt_state.step_size" else k): onp.asarray(v)
        for k, v in extra.items()
    }


def _diagnostics(arrays):
    """Worst split R-hat and bulk ESS of ``arrays`` (each ``(chain, draw)``)."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    return {
        "r_hat": max(
            float(onp.nanmax(onp.asarray(split_gelman_rubin(x))))
            for x in arrays
        ),
        "ess_bulk": min(
            float(onp.nanmin(onp.asarray(effective_sample_size(x))))
            for x in arrays
        ),
    }


def _noise_quantiles(samples):
    noise = {
        k.split(".", 1)[1]: _quantiles(samples[k])
        for k in samples
        if k.startswith("noise.")
    }
    return noise or None


def _moments(r):
    r = onp.asarray(r, dtype=float)
    if r.std() <= 1e-12 * max(abs(r.mean()), 1.0):
        return 0.0, -3.0  # scipy gives NaN for (near-)constant residuals
    return float(stats.skew(r)), float(stats.kurtosis(r))


def _wrap(angle):
    return (onp.asarray(angle) + 360.0) % 360.0


def _quantiles(x, angle=False, period=360.0):
    """Median, 16th and 84th percentiles; angles about their mean direction.

    ``period`` is the angle's period in degrees (180 for an axis, a position
    angle without a sense).
    """
    x = onp.asarray(x, dtype=float).ravel()
    if angle:
        scale = 360.0 / period
        x = x * scale
        mean = math.degrees(
            math.atan2(
                onp.mean(onp.sin(onp.radians(x))),
                onp.mean(onp.cos(onp.radians(x))),
            )
        )
        dev = (x - mean + 180.0) % 360.0 - 180.0
        q = onp.percentile(dev, [50, 16, 84])
        median = float(_wrap(q[0] + mean))
        q = (
            onp.array([median, median + q[1] - q[0], median + q[2] - q[0]])
            / scale
        )
    else:
        q = onp.percentile(x, [50, 16, 84])
    return {"median": float(q[0]), "q16": float(q[1]), "q84": float(q[2])}
