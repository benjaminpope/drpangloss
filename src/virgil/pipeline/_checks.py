"""Deterministic quality checks for pipeline results.

Every check is a pure function of plain numbers that returns a
[`Check`][virgil.pipeline.Check]: the same inputs always give the same
status and message. Thresholds are keyword arguments with the defaults
used by the pipelines, so the tests can reach every branch.

Status meanings:

* ``"pass"``: nothing to report.
* ``"warn"``: the result is usable but needs a human look.
* ``"fail"``: the result should not be used as it stands.
"""

from __future__ import annotations

import dataclasses
import math
import re

from .._utils import mas2rad

STATUSES = ("pass", "warn", "fail")


@dataclasses.dataclass(frozen=True)
class Check:
    """One quality check of a pipeline run.

    Attributes
    ----------
    name : str
        Stable identifier, e.g. ``"chi2_companion"``.
    status : {"pass", "warn", "fail"}
        Outcome.
    value : float, list or None
        The quantity tested.
    threshold : float, list or None
        The threshold it was compared with.
    message : str
        One templated sentence saying what the outcome means.
    """

    name: str
    status: str
    value: object
    threshold: object
    message: str

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(
                f"status must be one of {STATUSES}; got {self.status!r}."
            )

    def to_dict(self):
        """The check as a JSON-ready dict."""
        return {
            "name": self.name,
            "status": self.status,
            "value": _jsonable(self.value),
            "threshold": _jsonable(self.threshold),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, record):
        """Rebuild a check from :meth:`to_dict`."""
        return cls(**{f.name: record[f.name] for f in dataclasses.fields(cls)})


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    value = float(value)
    return value if math.isfinite(value) else None


def _finite(value):
    return value is not None and math.isfinite(float(value))


def format_sigma(value):
    """``value`` as ``"3.21σ"``, or ``"n/a"`` if it is missing or not finite."""
    return f"{float(value):.2f}σ" if _finite(value) else "n/a"


def chi2_reduced(name, chi2_red, *, label, warn=2.0, fail=5.0):
    """χ²/N on the quoted errors, even when error scales are fitted.

    Parameters
    ----------
    name : str
        Check name.
    chi2_red : float
        χ² per independent observable, on the errors as quoted.
    label : str
        What was fitted, e.g. ``"the best-fit binary"``.
    warn, fail : float, optional
        χ²/N above ``warn`` warns; above ``fail`` fails.
    """
    if not _finite(chi2_red):
        return Check(
            name, "fail", chi2_red, warn, f"χ²/N of {label} is not finite."
        )
    if chi2_red > fail:
        status, meaning = (
            "fail",
            "the model does not describe the data, or the errors are "
            "much too small",
        )
    elif chi2_red > warn:
        status, meaning = "warn", "errors likely underestimated"
    elif chi2_red < 1.0 / warn:
        status, meaning = "warn", "errors likely overestimated"
    else:
        status, meaning = "pass", "consistent with the quoted errors"
    return Check(
        name,
        status,
        chi2_red,
        warn,
        f"χ²/N = {chi2_red:.3g} for {label} on quoted errors: {meaning}.",
    )


def error_scale(scales, *, warn=2.0, fail=5.0):
    """Fitted error-scale factors ``s`` (only with ``error_scale="fit"``).

    A factor much greater than 1 means the quoted errors are too small or
    the model is wrong; rescaling then makes χ² ≈ 1 by construction, so the
    raw χ²/N on quoted errors is the number to quote.
    """
    worst = max((float(s) for s in scales.values()), default=1.0)
    names = ", ".join(f"{k} = {float(v):.3g}" for k, v in scales.items())
    if worst > fail:
        status = "fail"
        meaning = "the fit failed to describe the data on its quoted errors"
    elif worst > warn:
        status = "warn"
        meaning = "the quoted errors are too small or the model is wrong"
    else:
        status, meaning = "pass", "close to the quoted errors"
    return Check(
        "error_scale",
        status,
        worst,
        warn,
        f"Fitted error scales {names}: {meaning}.",
    )


def detection(local_sigma, global_sigma, *, threshold=3.0):
    """Significance of the best companion against a threshold.

    ``global_sigma`` corrects ``local_sigma`` for the look-elsewhere effect
    of searching a grid. A clear detection passes; one significant only
    locally warns; no detection warns too, since the fit and posterior
    stages then describe the highest noise peak.

    ``global_sigma`` is an approximate Šidák estimate, not a simulated
    false-alarm probability: for a calibrated threshold use
    [`injection_recovery`][virgil.detection.injection_recovery] or
    [`gaussian_null`][virgil.detection.gaussian_null].
    """
    value = [local_sigma, global_sigma]
    if _finite(global_sigma) and global_sigma >= threshold:
        return Check(
            "detection",
            "pass",
            value,
            threshold,
            f"Companion detected at {format_sigma(global_sigma)} after an "
            f"approximate look-elsewhere correction "
            f"({format_sigma(local_sigma)} local); for a "
            "calibrated threshold use injection_recovery or gaussian_null.",
        )
    if _finite(local_sigma) and local_sigma >= threshold:
        return Check(
            "detection",
            "warn",
            value,
            threshold,
            f"Marginal: {format_sigma(local_sigma)} local but "
            f"{format_sigma(global_sigma)} after an approximate "
            "look-elsewhere correction; confirm it "
            "with simulated nulls (injection_recovery or gaussian_null).",
        )
    return Check(
        "detection",
        "warn",
        value,
        threshold,
        f"No companion above {threshold:g}σ "
        f"({format_sigma(local_sigma)} local): "
        "the fit and posterior describe the highest noise peak; quote the "
        "contrast limits instead.",
    )


def grid_edge(index, shape, names):
    """Whether the grid peak lies on an edge of the searched grid.

    Parameters
    ----------
    index : sequence of int
        Index of the peak along each axis.
    shape : sequence of int
        Length of each axis.
    names : sequence of str
        Axis names, for the message.
    """
    edges = [
        name
        for name, i, n in zip(names, index, shape)
        if n > 1 and int(i) in (0, int(n) - 1)
    ]
    if edges:
        return Check(
            "grid_edge",
            "warn",
            list(int(i) for i in index),
            None,
            f"The grid peak lies on the edge of the {', '.join(edges)} "
            "axis: the best companion may lie outside the searched grid.",
        )
    return Check(
        "grid_edge",
        "pass",
        list(int(i) for i in index),
        None,
        "The grid peak lies inside the searched grid.",
    )


def prior_bound(fractions, *, warn=0.05):
    """Posterior mass piled up at a prior bound.

    Parameters
    ----------
    fractions : dict[str, float]
        For each bounded parameter, the fraction of posterior samples
        within 1% of the prior range of either bound.
    warn : float, optional
        Fraction above which to warn.
    """
    worst = max(fractions.values(), default=0.0)
    piled = [k for k, f in fractions.items() if f > warn]
    if piled:
        return Check(
            "prior_bound",
            "warn",
            worst,
            warn,
            f"Posterior mass piles up at a prior bound for "
            f"{', '.join(piled)}: widen the prior or check the model.",
        )
    return Check(
        "prior_bound",
        "pass",
        worst,
        warn,
        "No posterior mass piles up at a prior bound.",
    )


def r_hat(value, *, warn=1.01, fail=1.05):
    """Largest split R-hat over the sampled parameters."""
    if not _finite(value):
        return Check("r_hat", "fail", value, warn, "R-hat is not finite.")
    if value > fail:
        status, meaning = "fail", "the chains have not converged"
    elif value > warn:
        status, meaning = "warn", "run longer chains to be sure"
    else:
        status, meaning = "pass", "the chains agree"
    return Check(
        "r_hat", status, value, warn, f"Largest R-hat {value:.4f}: {meaning}."
    )


def ess(value, *, warn=400.0, fail=100.0):
    """Smallest bulk effective sample size over the sampled parameters."""
    if not _finite(value):
        return Check("ess", "fail", value, warn, "ESS is not finite.")
    if value < fail:
        status, meaning = "fail", "too few independent samples"
    elif value < warn:
        status, meaning = "warn", "quantiles in the tails are noisy"
    else:
        status, meaning = "pass", "enough independent samples"
    return Check(
        "ess",
        status,
        value,
        warn,
        f"Smallest bulk ESS {value:.0f}: {meaning}.",
    )


def divergences(fraction, *, walls=None, warn=0.0, fail=0.01, exposed=0.05):
    """Fraction of divergent NUTS transitions.

    ``walls`` is the posterior's wall probe (``{"radius_sd", "n",
    "exposed"}``, see ``StarPipeline``): when at least ``exposed`` of the
    steps of a few posterior standard deviations hit a hard wall of the
    likelihood (a closure-phase flip, which costs thousands in χ²), the
    divergences are NUTS trajectories rejected at that wall, so the check
    warns instead of failing and says so. The ESS and R-hat checks still
    judge the draws.
    """
    if fraction > fail:
        status, meaning = "fail", "the posterior is not reliable"
    elif fraction > warn:
        status, meaning = "warn", "inspect where they occur"
    else:
        status, meaning = "pass", "none"
    if status == "fail" and walls and walls.get("exposed", 0.0) >= exposed:
        status = "warn"
        meaning = (
            f"{100 * walls['exposed']:.2g} per cent of steps of "
            f"{walls['radius_sd']:g} posterior standard deviations hit a "
            "hard wall of the likelihood (a closure-phase sign flip), so "
            "NUTS rejects the trajectories that cross it; the draws are "
            "expected to follow the truncated posterior, with reduced "
            "efficiency (see the ESS). A bispectrum likelihood would "
            "remove the walls (virgil#309)"
        )
    return Check(
        "divergences",
        status,
        fraction,
        fail,
        f"Divergent transitions: {100 * fraction:.2g} per cent: {meaning}.",
    )


def residual_normality(skew, excess_kurtosis, n, *, n_se=3.0):
    """Skew and excess kurtosis of the whitened residuals.

    Each is compared with ``n_se`` of its standard error under a normal
    distribution, √(6/n) and √(24/n).
    """
    if n < 8:
        return Check(
            "residual_normality",
            "pass",
            [skew, excess_kurtosis],
            None,
            "Too few residuals to test their normality.",
        )
    limits = [n_se * math.sqrt(6.0 / n), n_se * math.sqrt(24.0 / n)]
    bad = [
        name
        for name, value, limit in zip(
            ("skew", "kurtosis"), (skew, excess_kurtosis), limits
        )
        if not _finite(value) or abs(value) > limit
    ]
    if bad:
        return Check(
            "residual_normality",
            "warn",
            [skew, excess_kurtosis],
            limits,
            f"Whitened residuals are not normal (skew {skew:.2f}, excess "
            f"kurtosis {excess_kurtosis:.2f}): outliers or a missing model "
            "component.",
        )
    return Check(
        "residual_normality",
        "pass",
        [skew, excess_kurtosis],
        limits,
        f"Whitened residuals look normal (skew {skew:.2f}, excess kurtosis "
        f"{excess_kurtosis:.2f}).",
    )


def field_of_view(sep, resolution, fov):
    """Whether a companion lies inside the field and outside the core.

    Parameters
    ----------
    sep : float
        Separation (mas).
    resolution : float
        Resolution limit (mas), half of the finest λ/B of the samples.
    fov : float
        Field-of-view radius (mas), the coarsest λ/B of the samples.
    """
    if sep < resolution:
        return Check(
            "field_of_view",
            "warn",
            sep,
            [resolution, fov],
            f"Separation {sep:.3g} mas is inside the resolution limit "
            f"({resolution:.3g} mas): separation and flux are degenerate.",
        )
    if sep > fov:
        return Check(
            "field_of_view",
            "warn",
            sep,
            [resolution, fov],
            f"Separation {sep:.3g} mas is outside the field of view "
            f"({fov:.3g} mas): the position may be an alias.",
        )
    return Check(
        "field_of_view",
        "pass",
        sep,
        [resolution, fov],
        f"Separation {sep:.3g} mas lies between the resolution limit and "
        "the field of view.",
    )


# θB/λ of the first zero of a uniform disk's visibility, j_{1,1} / π.
FIRST_NULL = 1.2196699


def resolution_regime(
    diam_mas, freq_max, *, unresolved=0.15, first_null=FIRST_NULL
):
    """Whether the baselines reach the first null of the visibility.

    A uniform disk of angular diameter θ has its first null at
    θB/λ = 1.22 (more for a limb-darkened star), so the
    fraction ``x = θ f_max / first_null`` of the way there, with ``f_max`` the
    longest spatial frequency B/λ, says what the data can measure.

    Parameters
    ----------
    diam_mas : float
        Fitted angular diameter (mas).
    freq_max : float
        Longest spatial frequency B/λ in the data (cycles per radian).
    unresolved : float, optional
        Fraction of the first null below which the star is unresolved and
        its diameter is an upper limit.
    first_null : float, optional
        θB/λ of the first null of the fitted model's visibility; the
        uniform-disk value by default.
    """
    ratio = float(diam_mas) * float(mas2rad) * float(freq_max)
    ratio /= float(first_null)
    if not _finite(ratio):
        return Check(
            "resolution", "fail", ratio, 1.0, "The resolution is not finite."
        )
    if ratio < unresolved:
        status, meaning = (
            "fail",
            "the star is unresolved and its diameter is an upper limit",
        )
    elif ratio < 1.0:
        status, meaning = (
            "warn",
            "the baselines stop short of the first null, so the diameter "
            "is measured but limb darkening is degenerate with it",
        )
    else:
        status, meaning = (
            "pass",
            "the baselines reach the first null, so the diameter is "
            "well measured and limb darkening can be constrained",
        )
    return Check(
        "resolution",
        status,
        ratio,
        1.0,
        f"The longest baseline reaches {ratio:.3g} of the first null: "
        f"{meaning}.",
    )


# Δ(−2 log L) within which another diameter lobe counts as a rival.
RIVAL_DELTA_CHI2 = 25.0


def bic_penalty(n_extra, n):
    """The BIC penalty ``n_extra ln N`` of ``n_extra`` extra parameters fitted
    to ``n`` independent observables."""
    return n_extra * math.log(max(int(n), 2))


def multimodal(lobes, *, threshold=RIVAL_DELTA_CHI2):
    """Whether another lobe of the diameter fits nearly as well as the best.

    Parameters
    ----------
    lobes : list[dict]
        The per-lobe table of the preferred model: each row has
        ``diam_mas``, ``diam_bounds_mas`` and ``delta_chi2``, the lobe's
        −2 log L above the best lobe's (zero for the best lobe; on the
        quoted errors, its Δχ²).
    threshold : float, optional
        The Δ(−2 log L) of the next-best lobe above which the best lobe is
        unambiguous.
    """
    rows = sorted(lobes, key=lambda r: r["delta_chi2"])
    if len(rows) < 2:
        return Check(
            "multimodal",
            "pass",
            None,
            threshold,
            "Only one diameter lobe was found.",
        )
    best, other = rows[0], rows[1]
    gap = float(other["delta_chi2"])
    if gap > threshold:
        return Check(
            "multimodal",
            "pass",
            gap,
            threshold,
            f"The best lobe ({best['diam_mas']:.4g} mas) beats the next "
            f"({other['diam_mas']:.4g} mas) by Δχ² = {gap:.3g}.",
        )
    return Check(
        "multimodal",
        "warn",
        gap,
        threshold,
        f"Another lobe fits nearly as well: {other['diam_mas']:.4g} mas "
        f"against the best {best['diam_mas']:.4g} mas, Δχ² = {gap:.3g} < "
        f"{threshold:g}. The posterior is bounded to the best lobe; the "
        "alias is not excluded by the data.",
    )


def model_gain(
    name, m2loglike_simple, m2loglike_complex, n, *, n_extra, simple, complex_
):
    """Whether a richer model improves the fit by more than its BIC penalty.

    The gain is the drop in −2 log L from the simpler to the richer model,
    which is their Δχ² when the errors are quoted. The richer model
    contains the simpler one, so a negative gain means that one of the fits
    did not converge.

    Parameters
    ----------
    name : str
        Check name.
    m2loglike_simple, m2loglike_complex : float
        −2 log L of the simpler and richer model, each at its own fitted
        error scales if any.
    n : int
        Number of independent observables.
    n_extra : int
        Extra parameters of the richer model.
    simple, complex_ : str
        Their names, for the message.
    """
    gain = float(m2loglike_simple) - float(m2loglike_complex)
    penalty = bic_penalty(n_extra, n)
    if not _finite(gain):
        return Check(
            name, "fail", gain, penalty, "The −2 log L gain is not finite."
        )
    if gain < -1.0:
        status = "warn"
        message = (
            f"The {complex_} model fits worse than the {simple} model it "
            f"contains (Δ(−2 log L) = {gain:.3g}): a fit did not converge."
        )
    elif gain > penalty:
        status = "pass"
        message = (
            f"Δ(−2 log L) = {gain:.3g} for {n_extra} extra parameters "
            f"exceeds the BIC penalty {penalty:.3g}: the {complex_} model "
            "is preferred."
        )
    else:
        status = "pass"
        message = (
            f"Δ(−2 log L) = {gain:.3g} for {n_extra} extra parameters does "
            f"not exceed the BIC penalty {penalty:.3g}: the {simple} model "
            "is adequate."
        )
    return Check(name, status, gain, penalty, message)


def prior_constrained(name, ratios, *, warn=0.8):
    """Whether parameters are constrained by the data or by their priors.

    Parameters
    ----------
    name : str
        Check name.
    ratios : dict[str, float]
        For each parameter, the posterior standard deviation divided by the
        standard deviation of its prior.
    warn : float, optional
        Ratio at or above which the prior dominates.
    """
    bad = [k for k, v in ratios.items() if not _finite(v) or v >= warn]
    worst = max(ratios.values(), default=0.0)
    listing = ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())
    if bad:
        return Check(
            name,
            "warn",
            worst,
            warn,
            f"The posterior is as wide as the prior for {', '.join(bad)} "
            f"(σ_posterior/σ_prior: {listing}): the data do not constrain "
            "them.",
        )
    return Check(
        name,
        "pass",
        worst,
        warn,
        f"The data constrain every parameter (σ_posterior/σ_prior: "
        f"{listing}).",
    )


# --- checks of DiskPipeline ---------------------------------------------------


def _upper_first(text):
    """``text`` with its first letter capitalized (``str.capitalize`` would
    lowercase the rest)."""
    return text[:1].upper() + text[1:]


def length_scale(evidences, ells, halffringe, *, nats=3.0):
    """Whether the data measure the Gaussian-process length scale ℓ.

    The Laplace evidences of the profile fitted at each ℓ of a grid differ
    by little when the data are silent about ℓ: it is then a prior choice
    and not a result, and the profile's fine structure is the prior
    wherever the data say nothing.

    Parameters
    ----------
    evidences : sequence of float
        Laplace log evidence at each ℓ.
    ells : sequence of float
        The Matérn ℓ (mas) of each, in the same order.
    halffringe : float
        λ/2B of the longest baseline (mas).
    nats : float, optional
        Spread of the evidences (max − min) below which ℓ is not measured.

    Notes
    -----
    A Matérn length is not a feature width. The message quotes the
    squared-exponential equivalent of the best ℓ, which matches the
    curvature at zero lag: ℓ·√((ν − 1)/ν) = ℓ/√3 for the pipeline's ν = 3/2.
    """
    z = [float(v) for v in evidences]
    if len(z) < 2 or not all(_finite(v) for v in z):
        return Check(
            "length_scale",
            "warn",
            None,
            nats,
            "The evidence of the length scale could not be computed at "
            "every ℓ of the grid.",
        )
    spread = max(z) - min(z)
    best = max(range(len(z)), key=z.__getitem__)
    equivalent = float(ells[best]) / math.sqrt(3.0)
    where = (
        f"best ℓ = {float(ells[best]):.3g} mas, a squared-exponential "
        f"length of {equivalent:.3g} mas, {equivalent / halffringe:.2g}× "
        "the half-fringe"
    )
    if spread < nats:
        return Check(
            "length_scale",
            "warn",
            spread,
            nats,
            f"The evidences across ℓ differ by only {spread:.2g} nats "
            f"({where}): the data do not measure ℓ, so it is a prior "
            "choice, and the profile's fine structure is the prior wherever "
            "the data are silent.",
        )
    return Check(
        "length_scale",
        "pass",
        spread,
        nats,
        f"The evidence prefers one length scale by {spread:.2g} nats "
        f"({where}).",
    )


def ring_count(n_best, rows, halffringe, *, nats=3.0):
    """Whether the data clearly prefer the chosen number of rings.

    Parameters
    ----------
    n_best : int
        The chosen number of rings.
    rows : sequence of dict
        One per fitted ring count: ``n``, ``bic``, ``log_evidence`` (None if
        not available) and ``widths`` (the rings' Gaussian σ, mas).
    halffringe : float
        λ/2B of the longest baseline (mas).
    nats : float, optional
        A neighbouring count is clearly worse when it loses by at least
        this many nats of log evidence and 2 × this many of BIC.

    Notes
    -----
    A ring narrower than the half-fringe leaves almost no signature in the
    visibilities, so its width is not measured.
    """
    by_n = {r["n"]: r for r in rows}
    best = by_n[n_best]
    margins = {}
    for n in (n_best - 1, n_best + 1):
        if n not in by_n:
            continue
        other = by_n[n]
        gaps = [0.5 * (other["bic"] - best["bic"])]
        if _finite(best.get("log_evidence")) and _finite(
            other.get("log_evidence")
        ):
            gaps.append(best["log_evidence"] - other["log_evidence"])
        margins[n] = min(gaps)
    narrow = [
        i + 1
        for i, w in enumerate(best["widths"])
        if not _finite(w) or w < halffringe
    ]
    worst = min(margins.values(), default=None)
    problems = []
    unclear = [n for n, m in margins.items() if m < nats]
    if unclear:
        problems.append(
            "the data do not clearly prefer "
            f"{n_best} to {' or '.join(str(n) for n in unclear)} "
            f"ring{'s' if unclear[-1] != 1 else ''} "
            f"(margin {', '.join(f'{margins[n]:.2g}' for n in unclear)} "
            "nats)"
        )
    if narrow:
        problems.append(
            f"ring {', '.join(str(i) for i in narrow)} is narrower than "
            "the half-fringe, so its width is not resolved"
        )
    if problems:
        return Check(
            "ring_count",
            "warn",
            worst,
            nats,
            f"With {n_best} ring{'s' if n_best != 1 else ''}: "
            + "; ".join(problems)
            + ".",
        )
    return Check(
        "ring_count",
        "pass",
        worst,
        nats,
        f"{n_best} ring{'s' if n_best != 1 else ''} preferred over its "
        f"neighbours by at least {worst if worst is not None else 0:.2g} "
        "nats, and every ring is resolved in width.",
    )


def inner_edge(
    fraction, correlation, halffringe, *, max_fraction=0.1, max_correlation=0.9
):
    """Light that the profile puts in an unresolved core at its inner edge.

    Such light trades against the star's flux, which the data cannot tell
    apart. ``fraction`` is the fraction of the disk's light within one
    half-fringe of the inner edge; ``correlation`` is the Laplace
    correlation between the disk's flux and the light in the innermost
    nodes.
    """
    if not _finite(fraction):
        return Check(
            "inner_edge",
            "warn",
            fraction,
            max_fraction,
            "The light near the inner edge could not be computed.",
        )
    corr = float(correlation) if _finite(correlation) else None
    reasons = []
    if fraction > max_fraction:
        reasons.append(
            f"{100 * fraction:.2g} per cent of the disk's light lies within "
            f"one half-fringe ({halffringe:.3g} mas) of the inner edge"
        )
    if corr is not None and abs(corr) > max_correlation:
        reasons.append(
            f"the light there is correlated with the disk flux "
            f"(|ρ| = {abs(corr):.2f})"
        )
    shown = "n/a" if corr is None else f"{corr:.2f}"
    if reasons:
        return Check(
            "inner_edge",
            "warn",
            fraction,
            [max_fraction, max_correlation],
            _upper_first("; ".join(reasons))
            + ": an unresolved core trades against the star's flux; "
            "refit with a larger inner radius to test it.",
        )
    return Check(
        "inner_edge",
        "pass",
        fraction,
        [max_fraction, max_correlation],
        f"{100 * fraction:.2g} per cent of the light is within one "
        f"half-fringe of the inner edge (correlation with the flux {shown}).",
    )


def outer_edge(fraction, halffringe, *, warn=0.01):
    """Light within one half-fringe of the outer edge of the model.

    A proxy for light beyond the edge that the model cannot place; the
    remedy is a larger outer radius.
    """
    if not _finite(fraction):
        return Check(
            "outer_edge",
            "warn",
            fraction,
            warn,
            "The light near the outer edge could not be computed.",
        )
    if fraction > warn:
        return Check(
            "outer_edge",
            "warn",
            fraction,
            warn,
            f"{100 * fraction:.2g} per cent of the model's light lies "
            f"within one half-fringe ({halffringe:.3g} mas) of its outer "
            "edge: light beyond it cannot be placed. Refit with a larger "
            "r_out_mas.",
        )
    return Check(
        "outer_edge",
        "pass",
        fraction,
        warn,
        f"{100 * fraction:.2g} per cent of the light is within one "
        "half-fringe of the outer edge.",
    )


def inclination(inc, pa_sd, n_minima, *, minimum=15.0, max_pa_sd=45.0):
    """Whether inclination and position angle are well determined.

    Near face-on the position angle is undefined; it is then not quoted
    with an uncertainty. A geometry grid with another comparable minimum
    means the geometry is multimodal.

    Parameters
    ----------
    inc : float
        Inclination (degrees).
    pa_sd : float or None
        Laplace standard deviation of the position angle (degrees).
    n_minima : int
        Comparable minima of the geometry grid, the best included.
    minimum : float, optional
        Inclination (degrees) below which the position angle is
        unconstrained.
    max_pa_sd : float, optional
        Standard deviation (degrees) of the position angle above which it
        is unconstrained.
    """
    reasons = []
    if not _finite(inc) or inc < minimum:
        reasons.append(
            f"the inclination is {float(inc):.3g}°, below {minimum:g}°, so "
            "the position angle is unconstrained and is reported without an "
            "uncertainty"
        )
    elif pa_sd is not None and (not _finite(pa_sd) or pa_sd > max_pa_sd):
        reasons.append(
            "the position angle's Laplace standard deviation is "
            f"{float(pa_sd):.3g}°, so it is unconstrained"
        )
    if n_minima > 1:
        reasons.append(
            f"the geometry grid has {n_minima} comparable minima, so the "
            "inclination and position angle are multimodal"
        )
    if reasons:
        return Check(
            "inclination",
            "warn",
            inc,
            minimum,
            _upper_first("; ".join(reasons)) + ".",
        )
    return Check(
        "inclination",
        "pass",
        inc,
        minimum,
        f"Inclination {float(inc):.3g}° with a single minimum of the "
        "geometry grid.",
    )


def absorbed_companion(
    delta_chi2, delta_chi2_raised, degree, *, threshold=25.0
):
    """A compact companion that the disk may have absorbed.

    ``delta_chi2`` is the improvement in χ² from adding a point source to
    the best model, at the best position of a grid search on the residuals.
    A significant point source remaining means the model does not hold it.
    If a refit at harmonic ``degree`` + 1 removes it (``delta_chi2_raised``)
    the source is ambiguous: it may be disk asymmetry that the current degree
    cannot represent, or a disk of the higher degree may be flexible enough
    to absorb a real compact source. Either is ``None`` when it was not
    computed.
    """
    if delta_chi2 is None or not _finite(delta_chi2):
        return Check(
            "absorbed_companion",
            "pass",
            None,
            threshold,
            "The residuals were not searched for a point source.",
        )
    if delta_chi2 < threshold:
        return Check(
            "absorbed_companion",
            "pass",
            delta_chi2,
            threshold,
            f"No significant point source remains in the residuals "
            f"(Δχ² = {delta_chi2:.3g}).",
        )
    removed = (
        delta_chi2_raised is not None
        and _finite(delta_chi2_raised)
        and delta_chi2_raised < threshold
    )
    detail = (
        f" Raising the harmonic degree from {degree} to {degree + 1} removes "
        f"it (Δχ² = {delta_chi2_raised:.3g}): it may be asymmetry of the "
        f"disk rather than a companion, or a disk of degree {degree + 1} can "
        "absorb a compact source."
        if removed
        else ""
    )
    return Check(
        "absorbed_companion",
        "warn",
        delta_chi2,
        threshold,
        f"A point source remains in the residuals (Δχ² = {delta_chi2:.3g})"
        f".{detail}",
    )


def worst_status(checks):
    """The worst status among ``checks`` (``"pass"`` if there are none)."""
    rank = {status: i for i, status in enumerate(STATUSES)}
    return max((c.status for c in checks), key=rank.get, default="pass")


# --- checks from the warnings raised inside stages ----------------------------

# Warnings of these categories are recorded but raise no check: they say
# something about the software environment, not about the result.
# The match is by suffix: libraries subclass these, such as pyparsing's
# ``PyparsingDeprecationWarning``.
_BENIGN_CATEGORIES = (
    "DeprecationWarning",
    "ImportWarning",
    "ResourceWarning",
)


def _numbers(pattern, text):
    match = re.search(pattern, text)
    return [float(g) for g in match.groups()] if match else None


def convergence(records):
    """Optimizer non-convergence reported by the grid functions.

    Parameters
    ----------
    records : list of dict
        The recorded warnings that mention non-convergence ("the optimizer
        did not converge at N of M grid positions"). The one with the worst
        fraction N/M is reported; grids of different sizes can appear.
    """
    worst, total = 0.0, 0.0
    for record in records:
        counts = _numbers(r"(\d+)\s+of\s+(\d+)", record["message"])
        if (
            counts
            and counts[1]
            and (not total or counts[0] / counts[1] > worst / total)
        ):
            worst, total = counts
    if not total:
        return Check(
            "convergence",
            "warn",
            None,
            None,
            "The optimizer did not converge at some grid positions; values "
            "there may be inaccurate.",
        )
    return Check(
        "convergence",
        "warn",
        worst / total,
        0.0,
        f"The optimizer did not converge at {worst:.0f} of {total:.0f} grid "
        f"positions ({100 * worst / total:.2g} per cent): values there may "
        "be inaccurate.",
    )


def flux_axis_resolution(records, *, minimum=2.0):
    """The flux axis does not resolve the likelihood peak."""
    steps = []
    for r in records:
        found = _numbers(r"([\d.]+(?:e[-+]?\d+)?)\s+steps", r["message"])
        if found:
            steps.append(found[0])
    value = min(steps) if steps else None
    detail = f" ({value:.2g} steps across its FWHM)" if steps else ""
    return Check(
        "flux_axis_resolution",
        "warn",
        value,
        minimum,
        f"The flux axis does not resolve the likelihood peak{detail}: "
        "log_bayes_factor is inaccurate; use a finer flux axis (n_flux).",
    )


def limits_clipped(records):
    """Contrast limits clipped to the flux bounds."""
    counts = []
    for r in records:
        found = _numbers(r"(\d+)\s+limits", r["message"])
        if found:
            counts.append(found[0])
    value = max(counts) if counts else None
    which = f"{value:.0f} limits" if counts else "Some limits"
    return Check(
        "limits_clipped",
        "warn",
        value,
        0,
        f"{which} fell outside the flux bounds and were clipped to the "
        "nearer bound: those limits are not the true limits; widen "
        "flux_range.",
    )


def stage_warnings(records):
    """Warnings no other rule recognizes, summarized in one check."""
    shown = "; ".join(r["message"] for r in records[:3])
    more = f" (and {len(records) - 3} more)" if len(records) > 3 else ""
    return Check(
        "stage_warnings",
        "warn",
        len(records),
        0,
        f"{len(records)} warning(s) were raised inside the stages: "
        f"{shown}{more}. See the 'warnings' list of summary.json.",
    )


def _classify(message):
    text = message.lower()
    if "converge" in text and "optimizer" in text:
        return "convergence"
    if "resolve" in text and "peak" in text:
        return "flux_axis_resolution"
    if "clipped" in text or "flux_bounds" in text:
        return "limits_clipped"
    return None


_RULES = {
    "convergence": convergence,
    "flux_axis_resolution": flux_axis_resolution,
    "limits_clipped": limits_clipped,
}


def stage_warning_checks(warning_records):
    """Checks for the warnings recorded inside the stages.

    Known messages map to specific checks (``convergence``,
    ``flux_axis_resolution``, ``limits_clipped``), matched by keywords so
    that rewording a message does not break the mapping; any other
    warning, except deprecation notices, goes to one ``stage_warnings``
    check. No warnings give no checks.

    Parameters
    ----------
    warning_records : list of dict
        Each with ``stage``, ``category`` and ``message``.
    """
    grouped, other = {}, []
    for record in warning_records:
        rule = _classify(record["message"])
        if rule:
            grouped.setdefault(rule, []).append(record)
        elif not record["category"].endswith(_BENIGN_CATEGORIES):
            other.append(record)
    checks = [
        fn(grouped[name]) for name, fn in _RULES.items() if name in grouped
    ]
    if other:
        checks.append(stage_warnings(other))
    return checks
