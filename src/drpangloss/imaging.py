"""Regularisers and helpers for image reconstruction.

An image fit is a [`Problem`][drpangloss.fitting.Problem] whose model contains an
[`Image`][drpangloss.models.Image], with a prior on its ``log_brightness``
(see :func:`image_priors`) and usually a regulariser, which adds a penalty
on the pixel fluxes ``b`` to the loss:

| Regulariser | Penalty | Least squares (LM)? | A prior density? |
| --- | --- | --- | --- |
| [`TSV`][drpangloss.imaging.TSV] | ``w Σ (Δx b)² + (Δy b)²`` | yes | no |
| [`TV`][drpangloss.imaging.TV] | ``w Σ √((Δx b)² + (Δy b)² + ε²)`` | no | no |
| [`MaxEntropy`][drpangloss.imaging.MaxEntropy] | ``w Σ b log(b / q)`` | no | no |
| [`Centroid`][drpangloss.imaging.Centroid] | ``½ |centroid / σ|²`` | yes | yes |

Differences ``Δ`` are between neighbouring pixels, with zeros beyond the
edges, so edge pixels are penalised too. TSV (total squared variation)
favours smooth images, TV (total variation) piecewise-flat ones, and maximum
entropy images close to a default ``q``. The weight ``w`` depends on the
scene and the data; :func:`l_curve` sweeps it.

Closure, kernel and DISCO phases do not fix an image's position. Something
must: an analytic star at the origin, a [`Centroid`][drpangloss.imaging.Centroid]
prior, or a centred prior mean. [`diagnose`][drpangloss.imaging.diagnose]
checks a fit for this and other common pitfalls.
"""

import dataclasses

import equinox as eqx
import jax.numpy as np
from jax.scipy.special import xlogy

from ._geometry import pixel_offsets, rotate
from ._utils import mas2rad
from .fitting import fit
from .likelihood import whitened_residuals
from .models import Image, PointSource, System


class _ImageRegulariser(eqx.Module):
    """A penalty on the pixels of one Image, found at ``path`` in the model."""

    weight: float
    path: str | None = eqx.field(static=True)

    def image(self, model):
        """The regularised [`Image`][drpangloss.models.Image] in ``model``."""
        image = model if self.path is None else model.get(self.path)
        if not isinstance(image, Image):
            raise TypeError(
                f"{type(self).__name__}(path={self.path!r}) found a "
                f"{type(image).__name__}, not an Image."
            )
        return image

    probabilistic = False


def _differences(b):
    """Steps between neighbouring pixels along rows and columns.

    The image is padded with a ring of zeros, so the steps into and out of
    the image at every edge count. Both arrays have shape ``(N + 1, M + 1)``
    and line up, pixel by pixel, for total variation.
    """
    padded = np.pad(b, 1)
    dx = padded[:-1, 1:] - padded[:-1, :-1]
    dy = padded[1:, :-1] - padded[:-1, :-1]
    return dx, dy


class TSV(_ImageRegulariser):
    """Total squared variation: ``weight * Σ (Δx b)² + (Δy b)²``.

    A quadratic smoothness penalty, so it can be fitted by
    Levenberg–Marquardt.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    path : str, optional
        Path of the Image in the model (e.g. ``"env"``); ``None`` if the
        model is the Image itself.
    """

    def __init__(self, weight, path=None):
        self.weight = float(weight)
        self.path = path

    def residuals(self, model):
        dx, dy = _differences(self.image(model).brightness)
        return np.sqrt(2.0 * self.weight) * np.concatenate(
            [np.ravel(dx), np.ravel(dy)]
        )

    def value(self, model):
        return 0.5 * np.sum(self.residuals(model) ** 2)


class TV(_ImageRegulariser):
    """Total variation: ``weight * Σ √((Δx b)² + (Δy b)² + ε²)``.

    Favours piecewise-flat images with sharp edges. ``ε`` smooths the
    penalty where the image is flat, so that it is differentiable.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    epsilon : float, optional
        Smoothing scale, as a fraction of the mean pixel flux (default
        1e-2).
    path : str, optional
        Path of the Image in the model.
    """

    epsilon: float

    def __init__(self, weight, epsilon=1e-2, path=None):
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        self.path = path

    def value(self, model):
        b = self.image(model).brightness
        dx, dy = _differences(b)
        eps = self.epsilon / b.size
        return self.weight * np.sum(np.sqrt(dx**2 + dy**2 + eps**2))


class MaxEntropy(_ImageRegulariser):
    """Maximum entropy: ``weight * Σ b log(b / q)``, with ``b`` and ``q`` unit-sum.

    The relative entropy of the image with respect to a default image ``q``.
    It is zero for ``b = q`` and positive otherwise.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    prior : array-like, optional
        The default image ``q``, with the image's shape and positive where
        the image's support is; normalised here. Default: flat over the
        support.
    path : str, optional
        Path of the Image in the model.
    """

    prior: object

    def __init__(self, weight, prior=None, path=None):
        self.weight = float(weight)
        self.prior = None if prior is None else np.asarray(prior)
        self.path = path

    def value(self, model):
        image = self.image(model)
        b = image.brightness
        if self.prior is None:
            inside = (
                np.ones(b.shape, bool)
                if image.support is None
                else image.support
            )
            q = inside / np.sum(inside)
        else:
            q = np.asarray(self.prior) / np.sum(self.prior)
        q = np.where(b > 0.0, q, 1.0)  # 0 log 0 = 0 outside the support
        return self.weight * np.sum(xlogy(b, b) - xlogy(b, q))


class Centroid(_ImageRegulariser):
    """A Gaussian prior centring an Image's flux on the origin.

    The penalty is ``½ |c / sigma_mas|²``, where ``c`` is the flux-weighted
    centroid of the Image on the sky, including its ``dra``/``ddec``. This
    is a genuine prior density, so it may be used when sampling.

    Parameters
    ----------
    sigma_mas : float
        Standard deviation of the centroid, in milliarcseconds.
    path : str, optional
        Path of the Image in the model.
    """

    sigma_mas: float
    probabilistic = True

    def __init__(self, sigma_mas, path=None):
        self.weight = 1.0
        self.sigma_mas = float(sigma_mas)
        self.path = path

    def centroid(self, model):
        """The Image's flux-weighted centroid ``(dra, ddec)`` in mas."""
        image = self.image(model)
        b = image.brightness
        nrow, ncol = b.shape
        x = np.sum(b, axis=0) @ pixel_offsets(ncol, image.pixel_scale_mas)
        y = np.sum(b, axis=1) @ pixel_offsets(nrow, image.pixel_scale_mas)
        x, y = rotate(x, y, image.rotation_deg)
        return np.stack([x + image.dra, y + image.ddec])

    def residuals(self, model):
        return self.centroid(model) / self.sigma_mas

    def value(self, model):
        return 0.5 * np.sum(self.residuals(model) ** 2)


def image_priors(scene):
    """Flat priors on the log-brightness of every Image in a scene.

    Returns a dict for [`Problem`][drpangloss.fitting.Problem], e.g.
    ``{"env.log_brightness": ImproperUniform(...)}``. The pixels are then
    constrained only by the data and the regularisers. Add priors for any
    other free parameters (fluxes, offsets) to the dict.
    """
    import numpyro.distributions as dist

    priors = {}

    def visit(model, prefix):
        if isinstance(model, Image):
            shape = model.log_brightness.shape
            priors[prefix + "log_brightness"] = dist.ImproperUniform(
                dist.constraints.real, (), event_shape=shape
            )
        elif isinstance(model, System):
            for name, part in model.components.items():
                visit(part, prefix + name + ".")

    visit(scene, "")
    if not priors:
        raise ValueError("The scene contains no Image.")
    return priors


def nyquist_pixel_scale(data):
    """The largest pixel scale (mas) that samples the data's finest fringes.

    That is λ / (2 B) for the longest baseline B in units of wavelength,
    over all samples of ``data`` (an OIData or a sequence of them).
    Reconstructions normally use pixels 2–4 times smaller.
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    longest = max(
        float(np.max(np.hypot(d.u, d.v) / d.wavel)) for d in observations
    )
    return 1.0 / (2.0 * longest * mas2rad)


@dataclasses.dataclass(frozen=True)
class LCurve:
    """The result of :func:`l_curve`.

    Attributes
    ----------
    weights : array
        The regularisation weights, in the order fitted (largest first).
    chi2 : array
        Total χ² of each fit.
    chi2_red : array
        Total χ² per data point.
    penalty : array
        The unweighted regulariser, ``value / weight``, of each fit.
    results : list of FitResult
        The fits.
    """

    weights: object
    chi2: object
    chi2_red: object
    penalty: object
    results: list

    def corner(self):
        """The weight at the L-curve's corner, where it bends most sharply.

        The curve is ``(log χ², log penalty)`` parametrised by ``log w``;
        the corner is the interior point of largest curvature (Hansen &
        O'Leary 1993). Check it by eye: the curvature of a sparse or noisy
        sweep is itself noisy, and a smooth curve has no clear corner.
        """
        t = np.log(self.weights)
        x, y = np.log(self.chi2), np.log(self.penalty)
        # Drop fits that barely moved from the previous (warm) start: at weak
        # regularisation the optimiser can stall, which would add a flat,
        # spuriously curved tail.
        moved = np.hypot(np.diff(x), np.diff(y)) > 1e-3 * np.ptp(y)
        keep = np.concatenate([np.array([True]), moved])
        t, x, y, weights = t[keep], x[keep], y[keep], self.weights[keep]
        if t.size < 3:
            raise ValueError("The sweep needs at least three distinct fits.")
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
        curvature = (dx * ddy - ddx * dy) / (dx**2 + dy**2) ** 1.5
        # The end points have one-sided derivatives; leave them out.
        return float(weights[1 + int(np.argmax(np.abs(curvature[1:-1])))])

    def discrepancy(self, target=1.0):
        """The weight at which χ² per data point reaches ``target``.

        This is Morozov's discrepancy principle: regularise as strongly as
        the data allow. It interpolates linearly in ``log w`` between the
        fitted weights, and returns ``None`` if the sweep never crosses the
        target. It relies on correct error bars; with underestimated errors
        it over-regularises.
        """
        above = self.chi2_red > target
        crossings = np.nonzero(above[:-1] & ~above[1:])[0]
        if crossings.size == 0:
            return None
        i = int(crossings[0])
        t = np.log(self.weights[i : i + 2])
        c = self.chi2_red[i : i + 2]
        fraction = (c[0] - target) / (c[0] - c[1])
        return float(np.exp(t[0] + fraction * (t[1] - t[0])))


def l_curve(make_problem, weights, **fit_options):
    """Fit a problem over a range of regularisation weights.

    The weights are fitted from largest to smallest, each starting from the
    previous solution, which is faster and more stable than starting every
    fit afresh. Plot ``penalty`` against ``chi2`` (both on log axes) to see
    the trade-off between fitting the data and regularising the image, and
    compare [`LCurve.corner`][drpangloss.imaging.LCurve.corner] and
    [`LCurve.discrepancy`][drpangloss.imaging.LCurve.discrepancy] with
    the images either side: there is usually a wide range of good weights.
    Other ways of choosing the weight are compared in
    ``design/regulariser_weight_selection.md``.

    Parameters
    ----------
    make_problem : callable
        ``make_problem(weight)`` returns the
        [`Problem`][drpangloss.fitting.Problem] for one weight, with exactly
        one weighted regulariser.
    weights : sequence of float
        The weights to try.
    **fit_options
        Passed to [`fit`][drpangloss.fitting.fit].

    Returns
    -------
    LCurve
    """
    weights = sorted((float(w) for w in weights), reverse=True)
    results, chi2, chi2_red, penalty = [], [], [], []
    start = None
    for weight in weights:
        problem = make_problem(weight)
        weighted = [r for r in problem.regularisers if not r.probabilistic]
        if len(weighted) != 1:
            raise ValueError(
                "l_curve needs exactly one weighted regulariser per problem."
            )
        if start is not None:
            problem = eqx.tree_at(lambda p: p.model, problem, start)
        result = fit(problem, **fit_options)
        start = result.model
        results.append(result)
        chi2.append(sum(result.info["chi2"]))
        chi2_red.append(result.info["chi2_red"])
        penalty.append(float(weighted[0].value(result.model)) / weight)
    return LCurve(
        np.asarray(weights),
        np.asarray(chi2),
        np.asarray(chi2_red),
        np.asarray(penalty),
        results,
    )


@dataclasses.dataclass(frozen=True)
class Diagnosis:
    """The result of :func:`diagnose`.

    Attributes
    ----------
    checks : dict
        Check name to value. Checks made per dataset or per Image are lists,
        in the order of the data or of the Images in the model.
    warnings : list of str
        One sentence for each problem found, saying what to do about it.
        Empty if none.
    """

    checks: dict
    warnings: list

    def __str__(self):
        def show(value):
            if isinstance(value, (list, tuple)):
                return "[" + ", ".join(show(v) for v in value) + "]"
            return f"{value:.3g}" if isinstance(value, float) else str(value)

        width = max(map(len, self.checks))
        lines = [f"{k:<{width}}  {show(v)}" for k, v in self.checks.items()]
        if self.warnings:
            lines += ["", *(f"Warning: {w}" for w in self.warnings)]
        else:
            lines += ["", "No warnings."]
        return "\n".join(lines)


def _parts(model, path=None):
    """``(path, part)`` for ``model`` and everything nested in it.

    The path is ``None`` for the model itself, else e.g. ``"env"``.
    """
    found = [(path, model)]
    if isinstance(model, System):
        prefix = "" if path is None else path + "."
        for name, part in model.components.items():
            found += _parts(part, prefix + name)
    return found


def _map_images(model, fn):
    """``model`` with ``fn`` applied to each of its Images."""
    if isinstance(model, Image):
        return fn(model)
    if not isinstance(model, System):
        return model
    parts = tuple(_map_images(c, fn) for c in model.components.values())
    return eqx.tree_at(lambda s: tuple(s.components.values()), model, parts)


def _rotated_180(image):
    """The Image turned by 180 degrees about the origin of the sky."""
    support = image.support
    return dataclasses.replace(
        image,
        log_brightness=image.log_brightness[::-1, ::-1],
        support=None if support is None else support[::-1, ::-1],
        dra=-image.dra,
        ddec=-image.ddec,
    )


def _chi2(model, observations):
    return [
        float(np.sum(whitened_residuals(model, d) ** 2)) for d in observations
    ]


def diagnose(model, data, regularisers=()):
    """Check a model and its data for common imaging pitfalls.

    Nothing is printed or warned: print the returned
    [`Diagnosis`][drpangloss.imaging.Diagnosis] to read it. Run it on the
    fitted model, with the regularisers used in the fit.

    Parameters
    ----------
    model : SourceModel
        The model, normally containing at least one
        [`Image`][drpangloss.models.Image] (the Image checks are skipped
        otherwise).
    data : OIData or sequence of OIData
        The data.
    regularisers : sequence, optional
        The regularisers of the fit; a
        [`Centroid`][drpangloss.imaging.Centroid] fixes the position.

    Returns
    -------
    Diagnosis
        ``checks`` holds, per dataset, ``chi2_red`` (χ² per data point) and
        ``phase_regime`` (fraction of model visibilities with |arg V| > 0.8π
        or |V| < 0.05, for projected phases only); per Image,
        ``pixel_scale_mas``, ``edge_flux``
        (fraction of the flux in the outer 2 pixels) and ``centroid_mas``
        (East, North offset in mas); and ``anchored`` (whether something
        fixes the position), ``flip_dchi2`` (Δχ² when every Image is
        rotated by 180° about the origin) and ``backend_error_sigma``
        (largest difference between an Image's backend and the DFT, in units
        of the data σ; only when an Image is not a DFT).
    """
    observations = list(data) if isinstance(data, (list, tuple)) else [data]
    parts = _parts(model)
    images = [(path, part) for path, part in parts if isinstance(part, Image)]
    checks, warns = {}, []

    chi2 = _chi2(model, observations)
    checks["chi2_red"] = [
        c / whitened_residuals(model, d).size
        for c, d in zip(chi2, observations)
    ]
    for i, red in enumerate(checks["chi2_red"]):
        if red > 2.0:
            warns.append(
                f"Dataset {i} has chi2 per point {red:.2f} > 2: the model "
                "under-fits or the errors are underestimated; refit, or "
                "check the error bars."
            )
        elif red < 0.5:
            warns.append(
                f"Dataset {i} has chi2 per point {red:.2f} < 0.5: the model "
                "over-fits or the errors are overestimated; regularise more "
                "or check the error bars."
            )

    checks["anchored"] = (
        any(isinstance(part, PointSource) for _, part in parts)
        or any(isinstance(r, Centroid) for r in regularisers)
        or any(
            d.observable_kind == "split"
            and not d.cp_flag
            and d.phi_mat is None
            for d in observations
        )
    )
    if images and not checks["anchored"]:
        warns.append(
            "Nothing fixes the position of the Image: closure, kernel and "
            "DISCO phases are blind to a shift. Add a PointSource star or a "
            "Centroid prior."
        )

    if images:
        nyquist = nyquist_pixel_scale(observations)
        checks["pixel_scale_mas"] = [im.pixel_scale_mas for _, im in images]
        checks["edge_flux"], checks["centroid_mas"] = [], []
        for (path, im), scale in zip(images, checks["pixel_scale_mas"]):
            label = path or "the Image"
            if scale > nyquist:
                warns.append(
                    f"{label} has pixels of {scale:.3g} mas, coarser than "
                    f"the Nyquist scale {nyquist:.3g} mas of the longest "
                    "baseline; use smaller pixels."
                )
            edge = float(1.0 - np.sum(im.brightness[2:-2, 2:-2]))
            checks["edge_flux"].append(edge)
            if edge > 0.05:
                warns.append(
                    f"{label} has {edge:.0%} of its flux in the outer 2 "
                    "pixels; enlarge the field or add a support."
                )
            x, y = Centroid(1.0, path).centroid(model)
            checks["centroid_mas"].append((float(x), float(y)))

        flipped = _map_images(model, _rotated_180)
        checks["flip_dchi2"] = sum(_chi2(flipped, observations)) - sum(chi2)
        if checks["flip_dchi2"] < 1.0:
            warns.append(
                "Rotating the Images by 180 degrees changes chi2 by "
                f"{checks['flip_dchi2']:.2g} < 1: the data do not constrain "
                "the orientation (V^2-only data cannot); expect a mirror "
                "ambiguity."
            )

        if any(im.backend != "dft" for _, im in images):
            exact = _map_images(
                model, lambda im: dataclasses.replace(im, backend="dft")
            )
            checks["backend_error_sigma"] = max(
                float(
                    np.max(
                        np.abs(d.model(model) - d.model(exact))
                        / d.flatten_data()[1]
                    )
                )
                for d in observations
            )
            if checks["backend_error_sigma"] > 0.1:
                warns.append(
                    "The Image backend differs from the DFT by "
                    f"{checks['backend_error_sigma']:.2g} sigma > 0.1; "
                    "use backend='dft'."
                )

    regimes = []
    for d in observations:
        if d.phi_mat is None and d.observable_kind != "mixed_log_complex":
            continue
        cvis = model.model(d.u, d.v, d.wavel)
        bad = (np.abs(np.angle(cvis)) > 0.8 * np.pi) | (np.abs(cvis) < 0.05)
        regimes.append(float(np.mean(bad)))
    if regimes:
        checks["phase_regime"] = regimes
        if any(regimes):
            warns.append(
                "Some model visibilities have |phase| > 0.8 pi or |V| < "
                "0.05, where projected (kernel or DISCO) phases, built from "
                "wrapped phases, are unreliable; use a smaller field or "
                "fainter extended flux."
            )

    return Diagnosis(checks, warns)
