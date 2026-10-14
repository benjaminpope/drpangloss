"""Source models: sky-brightness distributions and their visibilities.

* Components ([`PointSource`][virgil.models.PointSource],
  [`GaussianDisk`][virgil.models.GaussianDisk],
  [`UniformDisk`][virgil.models.UniformDisk], the limb-darkened disks such
  as [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk],
  [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim],
  [`PolarDisk`][virgil.models.PolarDisk] (a radial profile, with its
  profiles such as [`GaussianRing`][virgil.models.GaussianRing]), and
  the flared scattered-light disks such as
  [`FlaredDiskPowerLaw`][virgil.models.FlaredDiskPowerLaw]) are
  single shapes, combined with flux weights in a
  [`System`][virgil.models.System].
* [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian] and
  [`BinaryModelAngular`][virgil.models.BinaryModelAngular] are fast
  forms of a primary plus a point-source companion.
* The ``cvis_*`` functions are the analytic visibilities behind them.

Every flux is relative: for a companion it is the companion/primary flux
ratio. Likelihoods of these models are in
[`virgil.likelihood`][virgil.likelihood].
"""

import dataclasses
import math
import textwrap
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zdx
from jax.scipy.ndimage import map_coordinates
from jax.scipy.signal import fftconvolve
from jaxbessel import bessel_jn, bessel_jv_over_xv

from ._geometry import (
    apply_elliptical_transf_coord,
    check_az_prof_nonnegative,
    grid_visibilities,
    image_coordinates,
    image_visibilities,
    offset_phase,
    pixel_offsets,
    separation_pa,
    rotate,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)
from . import _elr
from ._deprecate import renamed
from ._utils import FWHM_PER_SIGMA, check_part_name, concrete, dtor, mas2rad
from .orbits import _days_since, _warn_if_mjd_without_t_ref
from .spectra import Spectrum, _planck_ratio, flux_at


def _normalize_image(image):
    """Return a finite unit-sum image for render outputs.

    Empty images are rejected when the flux is known; under ``jax.jit``
    (e.g. rendering many posterior samples) the check is skipped.
    """
    image = np.nan_to_num(np.asarray(image), nan=0.0, posinf=0.0, neginf=0.0)
    total = np.sum(image)
    value = concrete(total)
    if value is not None and not (onp.isfinite(value) and value > 0.0):
        raise ValueError("Rendered image must contain positive finite flux.")
    return image / total


def _unit_flux(component):
    """Scale an image component to unit sum; components off the grid stay zero."""
    total = np.sum(component)
    return np.where(
        total > 0.0, component / np.where(total > 0.0, total, 1.0), 0.0
    )


def _format_leaf(x):
    """Short human-readable form of a parameter value for ``repr``."""
    value = concrete(x)
    if value is None or value.dtype.kind not in "fiu":
        return repr(x)
    if value.ndim == 0:
        return f"{float(value):g}"
    return "[" + ", ".join(f"{float(v):g}" for v in value.ravel()) + "]"


def _as_flux(flux):
    """A component flux: a spectrum as given, or a number as a float array."""
    if isinstance(flux, Spectrum):
        return flux
    return np.asarray(flux, dtype=float)


def _flux_is_non_negative(flux):
    """Traced check that a number or spectrum is a valid (non-negative) flux."""
    if isinstance(flux, Spectrum):
        return flux.is_physical()
    return np.all(np.asarray(flux) >= 0.0)


def _check_non_negative_flux(flux, owner):
    """Raise if a concrete ``flux`` is negative; traced values are not checked.

    A spectrum is checked at its characteristic wavelengths (``wavel0``,
    every node and line centre), where its total must be non-negative.
    """
    if isinstance(flux, Spectrum):
        flux = flux(flux._check_wavel())
    value = concrete(flux)
    if value is not None and onp.any(value < 0.0):
        raise ValueError(
            f"{owner} has flux {value.tolist()}; fluxes must be non-negative."
        )


def _check_shape_params(owner, checks):
    """Reject concrete shape parameters outside their valid ranges.

    ``checks`` holds ``(name, value, valid, requirement)`` with ``valid`` a
    function of the concrete array. NaN and infinite values fail every
    check, since ``valid`` uses comparisons. Traced values are skipped; the
    model's ``is_physical`` covers them.
    """
    for name, value, valid, requirement in checks:
        value = concrete(value)
        if value is not None and not (
            onp.all(onp.isfinite(value)) and onp.all(valid(value))
        ):
            raise ValueError(
                f"{owner} has {name} {value.tolist()}; it must be {requirement}."
            )


def _check_non_negative_modulation(az_amps, az_pas):
    """Raise if concrete azimuthal modulations make the brightness negative."""
    amps, pas = concrete(az_amps), concrete(az_pas)
    if amps is None or pas is None:
        return
    if not bool(check_az_prof_nonnegative(np.asarray(amps), np.asarray(pas))):
        raise ValueError(
            f"Azimuthal modulations az_amps={amps.tolist()}, "
            f"az_pas={pas.tolist()} make the rim brightness negative "
            "somewhere; sum(abs(az_amps)) <= 1 always keeps it non-negative."
        )


def _concrete_sum(values):
    """Sum of ``values`` as a float, or ``None`` if any value is traced."""
    values = [concrete(v) for v in values]
    if any(v is None for v in values):
        return None
    return float(sum(onp.sum(v) for v in values))


class SourceModel(zdx.Base):  # type: ignore[reportGeneralTypeIssues]
    """Base class for sky-brightness source models.

    There are two kinds of source model.

    * **Components** ([`Component`][virgil.models.Component] subclasses such as
      [`PointSource`][virgil.models.PointSource]) are single shapes normalized to unit flux. Their
      ``flux`` is a *relative weight*, which only matters once they are mixed
      together in a [`System`][virgil.models.System].
    * **Scenes** ([`System`][virgil.models.System], [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian],
      [`BinaryModelAngular`][virgil.models.BinaryModelAngular], [`HarmonixModel`][virgil.models.HarmonixModel]) describe a whole,
      normalized sky. A scene placed inside a [`System`][virgil.models.System] has weight 1,
      unless it carries its own ``flux`` weight as [`System`][virgil.models.System] does.

    The binary models' ``flux`` is their companion/primary flux ratio, which
    is the same thing as a companion's weight in a System whose primary
    has ``flux=1``.

    Subclasses implement [`model`][virgil.models.SourceModel.model], and ``_image`` if they can be drawn.
    """

    def model(self, u, v, wavel):
        """Evaluate complex visibilities on interferometric baselines.

        Parameters
        ----------
        u, v : array-like
            Baseline coordinates in metres.
        wavel : array-like
            Wavelength(s) in metres.

        Returns
        -------
        array-like
            Complex visibilities, normalized to 1 at zero baseline.
        """
        raise NotImplementedError

    def model_on_grid(self, u, v, wavel, grid):
        """Visibilities at samples ``u, v`` that also lie on a uv ``grid``.

        [`OIData.model`][virgil.oidata.OIData.model] calls this when its
        samples form a regular lattice (a
        [`UVGrid`][virgil.oidata.UVGrid]), so that models able to use
        the lattice, such as a matching
        [`Image`][virgil.models.Image], can. By default it is
        [`model`][virgil.models.SourceModel.model].
        """
        return self.model(u, v, wavel)

    def render(self, npix=256, fov_mas=200.0):
        """Render a unit-sum image of the model.

        The image is ``npix`` x ``npix`` pixels spanning ``fov_mas``
        milliarcseconds, with East to the left (column 0 is the most
        positive ``dra``) and North up (row 0 is the most positive ``ddec``).
        Use [`virgil.plotting.plot_model`][virgil.plotting.plot_model] to display it with the
        correct axes.
        """
        xx, yy = image_coordinates(npix, fov_mas)
        return _normalize_image(
            self._image(xx, yy, float(fov_mas) / float(npix))
        )

    def _weight(self, wavel=None):
        """Relative flux of this model inside a [`System`][virgil.models.System].

        ``wavel`` is the wavelength in metres at which the flux is wanted
        (broadcastable against the baselines), or ``None`` for the model's
        reference flux, as used when rendering. Components and systems whose
        ``flux`` is a spectrum (see [`virgil.spectra`][virgil.spectra])
        evaluate it here.
        """
        return 1.0

    def total_spectrum(self, wavel):
        """The model's total flux at ``wavel`` (metres), e.g. for OI_FLUX.

        The sum of the component spectra, in the same relative units as
        their fluxes: one grey scale (absolute calibration, injection) away
        from a measured spectrum. It is the *intrinsic* total, with no
        fibre coupling. For a [`System`][virgil.models.System] it is the
        sum of its parts' fluxes, where a nested system counts with its own
        ``flux``, as in the visibilities.

        Examples
        --------
        >>> from virgil.spectra import PowerLaw
        >>> scene = System(
        ...     star=PointSource(),
        ...     disk=GaussianDisk(5.0, flux=PowerLaw(0.5, index=1.0, wavel0=2.0e-6)),
        ... )
        >>> [round(float(f), 3) for f in scene.total_spectrum(np.array([2.0e-6, 4.0e-6]))]
        [1.5, 2.0]
        """
        wavel = np.asarray(wavel)
        return np.broadcast_to(self._weight(wavel), wavel.shape)

    @property
    def time_dependent(self):
        """Whether the model changes with time (it contains an
        [`Attached`][virgil.models.Attached] component). Such a model is
        evaluated with [`at`][virgil.models.SourceModel.at], which
        [`OIData.model`][virgil.oidata.OIData.model] does sample by sample."""
        return False

    @property
    def broadcast_in_time(self):
        """Whether ``at`` accepts an array of times and returns a model
        whose ``model`` evaluates all of them at once, one per sample
        (which [`OIData.model`][virgil.oidata.OIData.model] then does
        without a loop over samples). False for an
        [`Attached`][virgil.models.Attached] component with bound angles,
        whose per-sample angles the components cannot take."""
        return True

    def at(self, mjd, t_ref=0.0):
        """This model at time ``t_ref + mjd`` (days, MJD).

        The time is split so that it keeps its precision: ``t_ref`` is a
        float64 number and ``mjd`` may be small offsets from it, even in
        float32 (as [`OIData`][virgil.oidata.OIData] passes them). For a
        single time, ``model.at(60500.3)`` is enough. A model that does not
        change with time returns itself.
        """
        return self

    def is_physical(self):
        """Whether the model is physically valid, as a (traceable) boolean.

        Unlike the checks made when a model is built, this works inside
        ``jax.jit`` and on models changed with ``set``, so likelihoods can
        reject invalid models (see ``reject_unphysical`` in
        [`model_loglike`][virgil.likelihood.model_loglike]). The base
        class has no constraints.
        """
        return np.asarray(True)

    def _image(self, xx, yy, pixel_scale_mas):
        """Un-normalized image on the given coordinate grid."""
        raise NotImplementedError(f"{type(self).__name__} cannot be rendered.")

    def __repr__(self):
        fields = ", ".join(
            f"{field.name}={_format_leaf(getattr(self, field.name))}"
            for field in dataclasses.fields(self)
        )
        return f"{type(self).__name__}({fields})"


class Component(SourceModel):
    """Base class for single shapes with ``flux``, ``dra`` and ``ddec``.

    A component on its own is normalized to unit flux. Inside a
    [`System`][virgil.models.System], ``flux`` is its weight relative to the other components,
    and ``dra``/``ddec`` place its centre (milliarcseconds, positive ``dra``
    to the East, positive ``ddec`` to the North).

    New shapes subclass [`Component`][virgil.models.Component] and implement ``_centred_cvis``
    (the unit-flux visibility of the shape at the origin) and
    ``_centred_image`` (an un-normalized image of the shape at the origin);
    offsets and mixing are handled here.
    """

    flux: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def _centred_cvis(self, uu, vv):
        """Visibility of the shape centred on the origin, normalized to 1."""
        raise NotImplementedError

    def _centred_image(self, xx, yy, pixel_scale_mas):
        """Un-normalized image of the shape centred on the origin."""
        raise NotImplementedError

    def model(self, u, v, wavel):
        uu, vv = u / wavel, v / wavel
        return self._centred_cvis(uu, vv) * offset_phase(
            uu, vv, self.dra, self.ddec
        )

    def _image(self, xx, yy, pixel_scale_mas):
        return self._centred_image(
            xx - self.dra, yy - self.ddec, pixel_scale_mas
        )

    def _weight(self, wavel=None):
        return flux_at(self.flux, wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def __check_init__(self):
        _check_non_negative_flux(self.flux, type(self).__name__)


class PointSource(Component):
    """Unresolved point source.

    Parameters
    ----------
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][virgil.models.System],
        or a spectrum from [`virgil.spectra`][virgil.spectra]
        (default 1). Keep the reference star at ``flux=1`` and a companion's
        ``flux`` is then its companion/star flux ratio.
    dra : float or array-like, optional
        Right-ascension offset in milliarcseconds, positive to the East.
    ddec : float or array-like, optional
        Declination offset in milliarcseconds, positive to the North.

    Examples
    --------
    >>> star = PointSource()
    >>> companion = PointSource(flux=0.01, dra=45.0, ddec=30.0)
    """

    def __init__(self, flux=1.0, dra=0.0, ddec=0.0):
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def _centred_cvis(self, uu, vv):
        return np.ones_like(uu) + 0j

    def _centred_image(self, xx, yy, pixel_scale_mas):
        sigma = max(pixel_scale_mas, 1e-6)
        return np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)


class GaussianDisk(Component):
    """Circular Gaussian brightness distribution.

    Parameters
    ----------
    sigma : float or array-like
        Standard deviation of the Gaussian in milliarcseconds
        (FWHM = 2.3548 ``sigma``, ``FWHM_PER_SIGMA``).
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][virgil.models.System],
        or a spectrum from [`virgil.spectra`][virgil.spectra]
        (default 1).
    dra : float or array-like, optional
        Right-ascension offset of the centre in milliarcseconds, positive to
        the East.
    ddec : float or array-like, optional
        Declination offset of the centre in milliarcseconds, positive to the
        North.

    Examples
    --------
    >>> halo = GaussianDisk(sigma=8.0, flux=0.2)
    """

    sigma: jax.Array

    def __init__(self, sigma, flux=1.0, dra=0.0, ddec=0.0):
        self.sigma = np.asarray(sigma, dtype=float)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def _centred_cvis(self, uu, vv):
        sigma_rad = mas2rad * self.sigma
        return np.exp(-2.0 * np.pi**2 * sigma_rad**2 * (uu**2 + vv**2)) + 0j

    def _centred_image(self, xx, yy, pixel_scale_mas):
        sigma = np.maximum(self.sigma, 1e-9)
        return np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)


class EllipticalGaussian(Component):
    """Elliptical Gaussian brightness distribution.

    Parameters
    ----------
    fwhm : float or array-like
        Full width at half maximum along the major axis, in
        milliarcseconds.
    ratio : float or array-like
        Minor-to-major axis ratio, in (0, 1]; 1 is a circular Gaussian.
    pa : float or array-like
        Position angle of the major axis in degrees, North to East.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System], or a spectrum from
        [`virgil.spectra`][virgil.spectra] (default 1).
    dra, ddec : float or array-like, optional
        Offset of the centre in milliarcseconds, positive to the East and
        North.

    Examples
    --------
    >>> plume = EllipticalGaussian(fwhm=30.0, ratio=0.5, pa=20.0, flux=2.0)
    """

    fwhm: jax.Array
    ratio: jax.Array
    pa: jax.Array

    def __init__(self, fwhm, ratio=1.0, pa=0.0, flux=1.0, dra=0.0, ddec=0.0):
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.ratio = np.asarray(ratio, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def __check_init__(self):
        super().__check_init__()
        _check_shape_params(
            type(self).__name__,
            (
                ("fwhm", self.fwhm, lambda x: x > 0.0, "positive"),
                (
                    "ratio",
                    self.ratio,
                    lambda x: (x > 0.0) & (x <= 1.0),
                    "in (0, 1]",
                ),
            ),
        )

    def is_physical(self):
        return (
            super().is_physical()
            & np.all(self.fwhm > 0.0)
            & np.all((self.ratio > 0.0) & (self.ratio <= 1.0))
        )

    def _centred_cvis(self, uu, vv):
        # In the frame where the ellipse is a circle of the major axis's
        # width, the visibility is that of a circular Gaussian.
        ut, vt = undo_elliptical_transf_spat_freq(uu, vv, self.pa, self.ratio)
        return _cvis_gaussian_envelope(ut, vt, self.fwhm)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        ratio = np.maximum(self.ratio, 1e-9)
        xt, yt = undo_elliptical_transf_coord(xx, yy, self.pa, ratio)
        fwhm = np.maximum(self.fwhm, 1e-9)
        return np.exp(-4.0 * np.log(2.0) * (xt**2 + yt**2) / fwhm**2)


class GaussianArc(Component):
    """A Gaussian ridge bent along a circular arc, e.g. a curved shock front.

    The brightness is a curve convolved with a circular Gaussian of FWHM
    ``width``. The curve is a circle of ``radius`` about the centre of
    curvature ``(dra, ddec)``, weighted along its length by a Gaussian of
    FWHM ``length`` (an arc length, in milliarcseconds) peaking at position
    angle ``pa`` as seen from the centre. For a large radius it tends to an
    elliptical Gaussian with FWHMs ``hypot(length, width)`` along the arc
    and ``width`` across it.

    The weight is a Gaussian in the arc length ``s`` from ``pa``, taken
    once round the circle, ``-π radius <= s <= π radius``: it is cut off
    at the antipode of ``pa``, where the two tails meet, and never wraps
    round the circle more than once. The quadrature covers ``±6σ`` (the
    part beyond, a fraction 2e-9 of the flux, is dropped), or the whole
    circle once ``6σ >= π radius``, i.e. ``length >= 0.39 π radius``.
    The cut at the antipode matters only for longer arcs: at ``length =
    π radius`` it removes 2 % of the Gaussian, and as ``length`` grows
    the brightness tends to a uniform ring.

    The visibility is the Fourier transform of the curve, a line integral
    evaluated by the trapezoidal rule on ``nodes`` equally spaced points
    spanning ``±min(6σ, π radius)``, times the Gaussian's. It is accurate
    while the spacing of the points, ``2 min(6σ, π radius) / (nodes -
    1)``, is below half the shortest fringe spacing, ``λ / 2 B_max``.

    Parameters
    ----------
    radius : float or array-like
        Radius of curvature in milliarcseconds.
    width : float or array-like
        FWHM of the ridge across the arc, in milliarcseconds.
    length : float or array-like
        FWHM of the brightness along the arc, as an arc length in
        milliarcseconds.
    pa : float or array-like
        Position angle of the brightest point of the arc, seen from the
        centre of curvature, in degrees North to East.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System] (default 1).
    dra, ddec : float or array-like, optional
        Offset of the centre of curvature in milliarcseconds, positive to
        the East and North. The arc itself lies ``radius`` away from it.
    nodes : int, optional
        Quadrature points along the arc, at least 2 (default 128).

    Examples
    --------
    >>> shock = GaussianArc(radius=30.0, width=2.0, length=40.0, pa=270.0)
    """

    radius: jax.Array
    width: jax.Array
    length: jax.Array
    pa: jax.Array
    nodes: int = eqx.field(static=True)

    def __init__(
        self,
        radius,
        width,
        length,
        pa=0.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        nodes=128,
    ):
        self.radius = np.asarray(radius, dtype=float)
        self.width = np.asarray(width, dtype=float)
        self.length = np.asarray(length, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        if isinstance(nodes, bool) or int(nodes) != nodes or nodes < 2:
            raise ValueError(f"nodes must be an integer >= 2, got {nodes}.")
        self.nodes = int(nodes)

    def __check_init__(self):
        super().__check_init__()
        positive = lambda x: x > 0.0  # noqa: E731
        _check_shape_params(
            type(self).__name__,
            (
                ("radius", self.radius, positive, "positive"),
                ("width", self.width, positive, "positive"),
                ("length", self.length, positive, "positive"),
            ),
        )

    def is_physical(self):
        return (
            super().is_physical()
            & np.all(self.radius > 0.0)
            & np.all(self.width > 0.0)
            & np.all(self.length > 0.0)
        )

    def curve(self):
        """Points along the arc (mas, East and North of the centre) and
        their normalized trapezoidal-rule weights."""
        sigma = self.length / FWHM_PER_SIGMA
        # Out to 6σ (a flux loss of 2e-9), or once round the circle if the
        # arc is longer: then both ends sit at the antipode of pa, and their
        # half weights add to one full trapezoidal weight.
        span = np.minimum(6.0 * sigma, np.pi * self.radius)
        s = np.linspace(-1.0, 1.0, self.nodes) * span
        angle = self.pa * dtor + s / self.radius
        # Trapezoidal rule: the end points carry half weight.
        ends = np.ones(self.nodes).at[np.array([0, -1])].set(0.5)
        weight = ends * np.exp(-0.5 * (s / sigma) ** 2)
        x = self.radius * np.sin(angle)
        y = self.radius * np.cos(angle)
        return x, y, weight / np.sum(weight)

    def _centred_cvis(self, uu, vv):
        x, y, weight = self.curve()
        shape = np.shape(uu)
        uu, vv = np.ravel(uu)[:, None], np.ravel(vv)[:, None]
        curve = np.matmul(
            offset_phase(uu, vv, x[None, :], y[None, :]),
            weight,
            precision=jax.lax.Precision.HIGHEST,
        )
        envelope = _cvis_gaussian_envelope(uu[:, 0], vv[:, 0], self.width)
        return np.reshape(curve * envelope, shape)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        x, y, weight = self.curve()
        width = np.maximum(self.width, pixel_scale_mas)
        d2 = (xx[..., None] - x) ** 2 + (yy[..., None] - y) ** 2
        return np.sum(weight * np.exp(-4.0 * np.log(2.0) * d2 / width**2), -1)


class TruncatedCone(Component):
    """A thin conical shell, truncated near its apex, e.g. a dust cone.

    The cone's axis points towards position angle ``pa`` and is tilted
    ``tilt`` degrees out of the sky plane; its half-opening angle is
    ``alpha``. Its apex lies ``tip`` milliarcseconds (along the axis, in 3-D)
    behind ``(dra, ddec)``, opposite to ``pa``. The emission starts a slant
    distance ``s0`` from the apex along the walls and falls off as
    ``exp(-(s - s0) / length)``, and the shell has a Gaussian thickness of
    FWHM ``width``. It is optically thin, so the sign of the tilt does not
    change the image of an unmodulated cone.

    The cone is a stack of rings about its axis. A ring of radius ρ in the
    plane perpendicular to an axis tilted β out of the sky projects to an
    ellipse, whose visibility is ``J0(2π ρ q)`` with
    ``q² = q_perp² + (ratio · q_par sin β)²``, where ``q_par`` and
    ``q_perp`` are the spatial frequencies along and across the projected
    axis, times the phase of the ring's centre, which lies
    ``(s cos α - tip) cos β`` along the projected axis. The rings are
    weighted by the area element (∝ ρ) and the emissivity, and integrated
    over ``s`` from ``s0`` to ``s0 + 5 length`` (the last 0.7 % of the
    flux is dropped) by a Gauss-Legendre rule on ``n_rings`` rings.

    The cone is a [`PolarDisk`][virgil.models.PolarDisk]: an
    [`ExpTail`][virgil.models.ExpTail] profile in the ring radius, a
    [`Cone`][virgil.models.Cone] height and
    [`Harmonics`][virgil.models.Harmonics] for the modulation below, with the
    disk's inclination ``90 - |tilt|`` and its major axis at ``pa - 90``.
    The class keeps its name and parameters.

    **Azimuthal modulation.** ``az_amps`` and ``az_pas`` brighten one side
    of the walls, e.g. the leading edge of a colliding-wind shock that the
    orbit sweeps round. Every ring's brightness is multiplied by
    ``1 + Σ_m A_m cos(m (φ - φ_m))`` in its own azimuth φ, with ``A_m`` in
    ``az_amps`` and ``φ_m`` in ``az_pas``, the same modulation along the
    whole length of the cone. Like those of
    [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim], the
    azimuths are measured in the plane of the ring, in the same sense as
    position angle. ``φ = pa + 90`` and ``pa - 90`` are the two walls seen
    across the projected axis, at those position angles on the sky.
    ``φ = pa`` is the side of each ring that projects towards ``pa`` on the
    sky when ``tilt > 0`` (towards ``pa + 180`` when ``tilt < 0``). The
    modulation stays analytic: by the Jacobi-Anger expansion a ring adds
    ``A_m (-i)^m J_m(2π ρ q) cos(m (θ - φ_m))`` to ``J0``, with θ the
    direction of the spatial frequency in the ring's plane. A modulated cone
    no longer looks the same at ``tilt`` and ``-tilt``: the image at
    ``-tilt`` is the one at ``tilt`` with ``az_pas`` mirrored to
    ``2 pa + 180 - az_pas``. That leaves the walls (``pa ± 90``) unchanged.
    Bound to an orbit's frame by [`Attached`][virgil.models.Attached]
    (``bind={"az_pas": ...}``), a sky position angle is deprojected into
    the rings' azimuth, so the bright side points at it on the sky.

    **Choosing ``n_rings``.** The quadrature converges spectrally, not at
    second order as the midpoint rule it replaced did: the error is
    negligible once the nodes resolve the fringes, and then does not change
    with more. Fringes of the rings' centres and radii run at most
    ``q max(sin α max(1, ratio), cos α cos β)`` cycles per milliarcsecond
    of slant distance, so about ``π q 5 length`` times that factor nodes are
    safe (fewer often do). For a cone with ``length`` 13.8 mas and ``alpha``
    62.5°, 24 nodes agree with a converged midpoint stack to 1e-5 for
    baselines out to 0.1 cycles/mas, and 48 to 1e-8 out to 0.3 cycles/mas
    (where 256 midpoint rings are still off by 1e-5 and 64 by 5e-4), so
    the default of 48 suits baselines to about 130 m at 2.2 µm. Too few nodes
    is not a smooth loss of accuracy: the error stays large and then
    drops sharply, so check by doubling ``n_rings``: if |Δχ²| ≳ 1 per
    dataset (|Δ log L| ≳ 0.5), use more, as before. The cost is linear in
    ``n_rings`` and about half the midpoint stack's at the same accuracy.

    Parameters
    ----------
    tip : float or array-like
        Distance from ``(dra, ddec)`` back to the apex along the axis, in
        milliarcseconds.
    alpha : float or array-like
        Half-opening angle of the cone, in degrees (0 < alpha < 90).
    s0 : float or array-like
        Slant distance from the apex where the emission starts, in
        milliarcseconds.
    length : float or array-like
        e-folding length of the emission along the walls, in
        milliarcseconds.
    width : float or array-like
        FWHM thickness of the shell, in milliarcseconds.
    tilt : float or array-like, optional
        Angle of the axis out of the sky plane, in degrees (-90 to 90).
    pa : float or array-like, optional
        Position angle the cone opens towards, in degrees North to East.
    ratio : float or array-like, optional
        Axis ratio of the cross-section (default 1, circular): its axis in
        the plane of the cone's axis and the line of sight is ``ratio``
        times the one across.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System] (default 1).
    dra, ddec : float or array-like, optional
        Offset of the reference point in milliarcseconds, positive to the
        East and North.
    n_rings : int, optional
        Gauss-Legendre nodes (rings) along the walls, at least 2 (default
        48). The error falls spectrally once they resolve the fringes;
        check convergence by doubling it (see above).
    az_amps : float or array-like, optional
        Amplitudes of the cosine azimuthal modulations of the walls, from
        the first order up (see above). A scalar gives a single first-order
        modulation, and the default (empty) an unmodulated cone. The
        brightness must stay non-negative, which ``sum(abs(az_amps)) <= 1``
        guarantees. Concrete values are checked when the model is built.
    az_pas : float or array-like, optional
        Azimuths of the modulations in degrees, one per entry of
        ``az_amps``, in the plane of the rings (see above): ``pa + 90`` and
        ``pa - 90`` are the walls at those position angles on the sky.

    Examples
    --------
    >>> cone = TruncatedCone(tip=5.0, alpha=30.0, s0=4.0, length=10.0,
    ...                      width=1.0, tilt=20.0, pa=90.0)

    A cone brighter on its southern wall (PA 180), half as bright on the
    northern one:

    >>> lopsided = TruncatedCone(tip=5.0, alpha=30.0, s0=4.0, length=10.0,
    ...                          width=1.0, tilt=20.0, pa=90.0,
    ...                          az_amps=1 / 3, az_pas=180.0)
    """

    tip: jax.Array
    alpha: jax.Array
    s0: jax.Array
    length: jax.Array
    width: jax.Array
    tilt: jax.Array
    pa: jax.Array
    ratio: jax.Array
    # None when unmodulated, not an empty array: zero-size leaves crash
    # pmapped samplers (see #282).
    az_amps: jax.Array | None
    az_pas: jax.Array | None
    n_rings: int = eqx.field(static=True)

    def __init__(
        self,
        tip,
        alpha,
        s0,
        length,
        width,
        tilt=0.0,
        pa=0.0,
        ratio=1.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        n_rings=48,
        az_amps=(),
        az_pas=(),
    ):
        self.tip = np.asarray(tip, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.s0 = np.asarray(s0, dtype=float)
        self.length = np.asarray(length, dtype=float)
        self.width = np.asarray(width, dtype=float)
        self.tilt = np.asarray(tilt, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.ratio = np.asarray(ratio, dtype=float)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        if isinstance(n_rings, bool) or int(n_rings) != n_rings or n_rings < 2:
            raise ValueError(
                f"n_rings must be an integer >= 2, got {n_rings}."
            )
        self.n_rings = int(n_rings)
        amps = _modulation_array(az_amps, "az_amps")
        pas = _modulation_array(az_pas, "az_pas")
        if amps.shape != pas.shape:
            raise ValueError(
                f"az_amps has {amps.size} entries but az_pas has "
                f"{pas.size}; give one azimuth per modulation."
            )
        if amps.size:
            _check_non_negative_modulation(amps, pas)
            self.az_amps, self.az_pas = amps, pas
        else:
            self.az_amps = self.az_pas = None

    def __check_init__(self):
        super().__check_init__()
        positive = lambda x: x > 0.0  # noqa: E731
        _check_shape_params(
            type(self).__name__,
            (
                (
                    "alpha",
                    self.alpha,
                    lambda x: (x > 0.0) & (x < 90.0),
                    "in (0, 90)",
                ),
                ("s0", self.s0, lambda x: x >= 0.0, "non-negative"),
                ("length", self.length, positive, "positive"),
                ("width", self.width, positive, "positive"),
                (
                    "tilt",
                    self.tilt,
                    lambda x: onp.abs(x) <= 90.0,
                    "in [-90, 90]",
                ),
                ("ratio", self.ratio, positive, "positive"),
            ),
        )

    def is_physical(self):
        return (
            super().is_physical()
            & np.all((self.alpha > 0.0) & (self.alpha < 90.0))
            & np.all(np.abs(self.tilt) <= 90.0)
            & np.all(self.s0 >= 0.0)
            & np.all(self.length > 0.0)
            & np.all(self.width > 0.0)
            & np.all(self.ratio > 0.0)
            & (
                True
                if self.az_amps is None
                else np.asarray(
                    check_az_prof_nonnegative(self.az_amps, self.az_pas)
                )
            )
        )

    def _orders(self):
        """Orders, amplitudes and in-ring phases (radians) of the
        modulations, with the phase measured like the ring angle of
        ``_centred_image`` (0 on the wall at ``pa + 90``, increasing towards
        the side projected along ``pa``)."""
        orders = range(1, self.az_amps.size + 1)
        phases = (self.pa + 90.0 - self.az_pas) * dtor
        return orders, self.az_amps, phases

    def _modulation(self, angles):
        """Brightness factor of the walls at ring angles ``angles``."""
        if self.az_amps is None:
            return np.ones_like(angles)
        orders, amps, phases = self._orders()
        return 1.0 + sum(
            amps[m - 1] * np.cos(m * (angles - phases[m - 1])) for m in orders
        )

    def _ring_angle(self, sky_pa):
        """The ring azimuth whose projection points at ``sky_pa``.

        A ring point at azimuth ``az`` lies ``cos φ`` across and
        ``ratio sin(tilt) sin φ`` along the projected axis, with
        ``φ = pa + 90 - az``, so a sky angle is deprojected by that squash
        (its sign included). Seen edge-on (``tilt = 0``), every direction
        off the walls is the near or far side of the ring.
        """
        squash = self.ratio * np.sin(self.tilt * dtor)
        squash = np.where(squash < 0.0, -1.0, 1.0) * np.maximum(
            np.abs(squash), 1e-8
        )
        d = (sky_pa - self.pa) * dtor
        phi = np.arctan2(np.cos(d) / squash, np.sin(d))
        return self.pa + 90.0 - phi / dtor

    def _radial_profile(self):
        sin_alpha = np.sin(self.alpha * dtor)
        return ExpTail(
            self.s0 * sin_alpha, self.length * sin_alpha, cutoff=5.0
        ).with_rule(self.n_rings)

    def rings(self):
        """Radius, sky offset of the centre along the projected axis (mas)
        and normalized weight of each ring: the Gauss-Legendre nodes that
        ``model`` integrates and ``render`` draws."""
        r, w = self._radial_profile().quadrature()
        along = Cone(self.alpha, self.tip)(r) * np.cos(self.tilt * dtor)
        return r, along, w / np.sum(w)

    def _centred_cvis(self, uu, vv):
        # The cone is a PolarDisk: an exponential tail in the radius of the
        # rings (the slant distance times sin alpha), a conical height, and
        # the walls' modulation as linear harmonics. The disk's major axis
        # is perpendicular to the projected axis of the cone, and the
        # minor-axis foreshortening ratio * sin(tilt) is passed as is.
        r, w = self._radial_profile().quadrature()
        height = Cone(self.alpha, self.tip)(r)
        c = s = None
        if self.az_amps is not None:
            c, s = self._harmonics(r)
        cvis = polar_disk_cvis(
            uu,
            vv,
            r,
            w,
            inc=90.0 - np.abs(self.tilt),
            pa=self.pa - 90.0,
            c=c,
            s=s,
            height=height,
            stretch=self.ratio * np.sin(self.tilt * dtor),
        )
        return cvis * _cvis_gaussian_envelope(uu, vv, self.width)

    def _harmonics(self, r):
        """The wall modulation as the Cartesian ``(c, s)`` of the engine.

        A mode ``A_m cos(m (az - az_pa_m))`` has the ring angle of
        ``_orders`` (0 on the wall at ``pa + 90``), and the engine's azimuth
        runs from the disk's major axis, at ``pa - 90``, towards the minor
        axis: they differ by half a turn, which is the factor ``(-1)**m``.
        """
        orders = onp.arange(1, self.az_amps.size + 1)
        _, amps, phases = self._orders()
        sign = (-1.0) ** orders
        c = sign * amps * np.cos(orders * phases)
        s = -sign * amps * np.sin(orders * phases)
        ones = np.ones_like(r)
        return c[:, None] * ones, s[:, None] * ones

    def _centred_image(self, xx, yy, pixel_scale_mas):
        rho, along, weight = self.rings()
        sin_pa, cos_pa = np.sin(self.pa * dtor), np.cos(self.pa * dtor)
        squash = self.ratio * np.sin(self.tilt * dtor)
        width = np.maximum(self.width, pixel_scale_mas)
        # Points round each ring no further apart than a third of the blur,
        # so a large thin ring is a continuous band, not a string of spots.
        block = 64
        circumference = concrete(
            2.0 * np.pi * np.max(rho) * np.maximum(1.0, self.ratio)
        )
        blur = concrete(width)
        if circumference is None or blur is None:
            n_blocks = 4  # traced: a fixed sampling
        else:
            needed = float(np.max(circumference)) / (float(np.min(blur)) / 3.0)
            n_blocks = int(min(max(onp.ceil(needed / block), 1), 128))
        phi = np.linspace(0.0, 2.0 * np.pi, n_blocks * block, endpoint=False)
        modulation = self._modulation(phi)

        def ring_points(radius, centre, angles):
            par = centre + radius * np.sin(angles) * squash
            perp = radius * np.cos(angles)
            return (
                par * sin_pa + perp * cos_pa,
                par * cos_pa - perp * sin_pa,
            )

        grid = _orthogonal_grid(xx, yy)
        if grid is not None:
            # On a regular grid the Gaussian spot of each point factorizes
            # into a row and a column profile, so a ring is one matrix
            # product instead of a Gaussian per point and pixel.
            origin, e_col, e_row = grid
            k = 4.0 * np.log(2.0) / width**2
            cols = np.arange(np.shape(xx)[1])[:, None]
            rows = np.arange(np.shape(xx)[0])[:, None]

            def add_ring(image, ring):
                radius, centre, w = ring
                x, y = ring_points(radius, centre, phi)
                dx, dy = x - origin[0], y - origin[1]
                s_col = (dx * e_col[0] + dy * e_col[1]) / (e_col @ e_col)
                s_row = (dx * e_row[0] + dy * e_row[1]) / (e_row @ e_row)
                g_col = np.exp(-k * (e_col @ e_col) * (cols - s_col) ** 2)
                g_row = np.exp(-k * (e_row @ e_row) * (rows - s_row) ** 2)
                ring = np.matmul(
                    g_row * modulation,
                    g_col.T,
                    precision=jax.lax.Precision.HIGHEST,
                )
                return image + w * ring / phi.size, None

        else:
            # Arbitrary coordinates: add the points in blocks of 64 to keep
            # memory small.
            blocks = phi.reshape(n_blocks, block)

            def add_ring(image, ring):
                radius, centre, w = ring

                def add_block(image, angles):
                    x, y = ring_points(radius, centre, angles)
                    d2 = (xx[..., None] - x) ** 2 + (yy[..., None] - y) ** 2
                    spots = np.exp(-4.0 * np.log(2.0) * d2 / width**2)
                    spots = spots * self._modulation(angles)
                    return image + w * np.sum(spots, -1) / phi.size, None

                image, _ = jax.lax.scan(add_block, image, blocks)
                return image, None

        image, _ = jax.lax.scan(
            add_ring, np.zeros(np.shape(xx)), (rho, along, weight)
        )
        return image


def _orthogonal_grid(xx, yy):
    """Origin and column and row steps of a regular grid, or ``None``.

    The grid may be shifted and rotated (as ``Component._image`` and
    ``Image.from_model`` make it) but its axes must be orthogonal. Returns
    ``None`` for traced or irregular coordinates.
    """
    xx, yy = concrete(xx), concrete(yy)
    if xx is None or yy is None or xx.ndim != 2 or min(xx.shape) < 2:
        return None
    xx, yy = xx.astype(float), yy.astype(float)
    origin = onp.array([xx[0, 0], yy[0, 0]])
    e_col = onp.array([xx[0, 1], yy[0, 1]]) - origin
    e_row = onp.array([xx[1, 0], yy[1, 0]]) - origin
    step = min(onp.hypot(*e_col), onp.hypot(*e_row))
    rows, cols = onp.indices(xx.shape)
    tol = 1e-3 * step
    if (
        step <= 0.0
        or abs(e_col @ e_row) > tol * step
        or onp.max(onp.abs(origin[0] + cols * e_col[0] + rows * e_row[0] - xx))
        > tol
        or onp.max(onp.abs(origin[1] + cols * e_col[1] + rows * e_row[1] - yy))
        > tol
    ):
        return None
    return origin, e_col, e_row


class UniformDisk(Component):
    """Uniformly bright (tophat) circular disk, e.g. a resolved stellar photosphere.

    Parameters
    ----------
    diam : float or array-like
        Angular diameter in milliarcseconds.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][virgil.models.System],
        or a spectrum from [`virgil.spectra`][virgil.spectra]
        (default 1).
    dra : float or array-like, optional
        Right-ascension offset of the centre in milliarcseconds, positive to
        the East.
    ddec : float or array-like, optional
        Declination offset of the centre in milliarcseconds, positive to the
        North.

    Examples
    --------
    >>> photosphere = UniformDisk(diam=3.0)
    """

    diam: jax.Array

    def __init__(self, diam, flux=1.0, dra=0.0, ddec=0.0):
        self.diam = np.asarray(diam, dtype=float)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def _centred_cvis(self, uu, vv):
        return cvis_uniform_disk(uu, vv, self.diam)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        radius = np.maximum(self.diam / 2.0, 0.5 * pixel_scale_mas)
        return np.where(xx**2 + yy**2 <= radius**2, 1.0, 0.0)


class _LimbDarkenedDisk(Component):
    r"""Circular disk whose brightness is a sum of powers of $\mu$.

    Subclasses give the profile $I(\mu) = \sum_\nu a_\nu \mu^\nu$, with
    $\mu = \sqrt{1 - (r / R)^2}$, through ``_profile``, which returns the
    coefficients $a_\nu$ (traceable) and the powers $\nu$ (a static tuple);
    the visibility is then
    [`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk].
    """

    diam: jax.Array

    def _profile(self):
        r"""``(coeffs, powers)`` of $I(\mu) = \sum a_\nu \mu^\nu$."""
        raise NotImplementedError

    def _set_position(self, diam, flux, dra, ddec):
        self.diam = np.asarray(diam, dtype=float)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def _centred_cvis(self, uu, vv):
        coeffs, powers = self._profile()
        return cvis_limb_darkened_disk(uu, vv, self.diam, coeffs, powers)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        coeffs, powers = self._profile()
        radius = np.maximum(self.diam / 2.0, 0.5 * pixel_scale_mas)
        r2 = (xx**2 + yy**2) / radius**2
        inside = r2 <= 1.0
        # mu = 0 at the limb; keep it positive so mu**nu has finite gradients
        tiny = np.finfo(r2.dtype).tiny
        mu = np.sqrt(np.where(inside, np.maximum(1.0 - r2, tiny), 1.0))
        brightness = sum(a * mu**nu for a, nu in zip(coeffs, powers))
        return np.where(inside, brightness, 0.0)

    def is_physical(self):
        r"""Positive diameter and a profile that is nowhere negative.

        The profile is checked on a grid of 101 values of $\mu$ from 0 to 1;
        each law has $I(1) = 1$, so this also keeps the flux positive.
        """
        coeffs, powers = self._profile()
        mu = np.linspace(0.0, 1.0, 101)
        brightness = sum(a * mu**nu for a, nu in zip(coeffs, powers))
        return (
            super().is_physical()
            & np.all(self.diam > 0.0)
            & np.all(brightness >= 0.0)
        )


class LimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Circular disk with polynomial limb darkening of order up to 22.

    The brightness is
    $I(\mu) / I(1) = 1 - \sum_{n=1}^{N} u_n (1 - \mu)^n$, with
    $\mu = \sqrt{1 - (r / R)^2}$ the cosine of the angle between the line of
    sight and the surface normal. This is the convention of jaxoplanet,
    *starry* and harmonix, so ``u`` is the same as a jaxoplanet
    ``Surface``'s ``u``: ``u=(u1,)`` is the linear law, ``u=(u1, u2)`` the
    quadratic law, and the default ``u=()`` a uniform disk. The order is at
    most 22, as the Bessel functions needed are of order at most 12. To fit the
    quadratic law with priors that cover exactly the physical profiles, use
    [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk].

    The visibility is analytic (Quirrenbach et al. 1996, eq. 3; see
    [`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk]).
    harmonix ([Dholakia & Pope 2025](https://arxiv.org/abs/2509.25433))
    generalizes the same result to limb-darkened spherical-harmonic maps.

    Parameters
    ----------
    diam : float or array-like
        Limb-darkened angular diameter (of the stellar limb) in
        milliarcseconds.
    u : sequence of float, optional
        Limb-darkening coefficients $u_1, \ldots, u_N$ (default: none, a
        uniform disk). A 1D array; its length, the order of the law (at most
        22), is fixed. [`is_physical`][virgil.models.SourceModel.is_physical]
        is false where the profile goes negative.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System], or a spectrum from
        [`virgil.spectra`][virgil.spectra] (default 1).
    dra : float or array-like, optional
        Right-ascension offset of the centre in milliarcseconds, positive to
        the East.
    ddec : float or array-like, optional
        Declination offset of the centre in milliarcseconds, positive to the
        North.

    Examples
    --------
    >>> linear = LimbDarkenedDisk(3.0, u=[0.6])
    >>> quadratic = LimbDarkenedDisk(3.0, u=[0.4, 0.25])
    """

    u: jax.Array

    def __init__(self, diam, u=(), flux=1.0, dra=0.0, ddec=0.0):
        self._set_position(diam, flux, dra, ddec)
        self.u = np.asarray(u, dtype=float).reshape(-1)

    def _profile(self):
        # (1 - mu)^n = sum_k C(n, k) (-mu)^k
        order = self.u.shape[0]
        expand = onp.zeros((order, order + 1))
        for n in range(1, order + 1):
            for k in range(n + 1):
                expand[n - 1, k] = math.comb(n, k) * (-1.0) ** k
        coeffs = np.zeros(order + 1).at[0].set(1.0) - self.u @ expand
        return coeffs, tuple(float(k) for k in range(order + 1))


class EllipticalLimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Limb-darkened disk with an elliptical outline, e.g. an oblate star.

    A [`LimbDarkenedDisk`][virgil.models.LimbDarkenedDisk] compressed along
    its minor axis: the brightness is
    $I(\mu) / I(1) = 1 - \sum_{n=1}^{N} u_n (1 - \mu)^n$, with
    $\mu = \sqrt{1 - \rho^2}$ and $\rho$ the elliptical radius (1 on the
    limb), so the profile follows the outline. The major axis is
    oriented exactly as an
    [`EllipticalGaussian`][virgil.models.EllipticalGaussian]'s. Like every
    component it has unit flux on its own: compressing the disk does not
    change its total flux, so the visibility at zero baseline is 1 and
    ``flux`` is its weight in a [`System`][virgil.models.System].

    The visibility is that of the circular disk at the spatial frequencies
    stretched by ``ratio`` along the minor axis (the Fourier similarity
    theorem), so it is analytic like
    [`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk].
    This is a geometric ellipse with a radial profile, not a model of a
    rapid rotator's gravity darkening; for that use
    [`GravityDarkenedStar`][virgil.models.GravityDarkenedStar].

    Parameters
    ----------
    diam : float or array-like
        Angular diameter of the limb along the major axis, in
        milliarcseconds.
    ratio : float or array-like, optional
        Minor-to-major axis ratio, in (0, 1] (default 1, a circular
        [`LimbDarkenedDisk`][virgil.models.LimbDarkenedDisk]).
    pa : float or array-like, optional
        Position angle of the major axis in degrees, North to East
        (default 0).
    u : sequence of float, optional
        Limb-darkening coefficients $u_1, \ldots, u_N$ as in
        [`LimbDarkenedDisk`][virgil.models.LimbDarkenedDisk] (default: none,
        a uniform elliptical disk).
        [`is_physical`][virgil.models.SourceModel.is_physical] is false
        where the profile goes negative.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System], or a spectrum from
        [`virgil.spectra`][virgil.spectra] (default 1).
    dra, ddec : float or array-like, optional
        Offset of the centre in milliarcseconds, positive to the East and
        North.

    Examples
    --------
    >>> star = EllipticalLimbDarkenedDisk(5.5, ratio=0.8, pa=30.0, u=[0.5])
    """

    u: jax.Array
    ratio: jax.Array
    pa: jax.Array

    def __init__(
        self, diam, ratio=1.0, pa=0.0, u=(), flux=1.0, dra=0.0, ddec=0.0
    ):
        self._set_position(diam, flux, dra, ddec)
        self.u = np.asarray(u, dtype=float).reshape(-1)
        self.ratio = np.asarray(ratio, dtype=float)
        self.pa = np.asarray(pa, dtype=float)

    # The same law, and the same ``u``, as LimbDarkenedDisk.
    _profile = LimbDarkenedDisk._profile

    def __check_init__(self):
        super().__check_init__()
        _check_shape_params(
            type(self).__name__,
            (
                ("diam", self.diam, lambda x: x > 0.0, "positive"),
                (
                    "ratio",
                    self.ratio,
                    lambda x: (x > 0.0) & (x <= 1.0),
                    "in (0, 1]",
                ),
            ),
        )

    def is_physical(self):
        return super().is_physical() & np.all(
            (self.ratio > 0.0) & (self.ratio <= 1.0)
        )

    def _centred_cvis(self, uu, vv):
        # In the frame where the outline is a circle of the major axis's
        # diameter, the visibility is the circular disk's. The frequencies
        # at zero baseline stay zero, so the visibility there stays 1.
        ut, vt = undo_elliptical_transf_spat_freq(uu, vv, self.pa, self.ratio)
        return super()._centred_cvis(ut, vt)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        # Unnormalized; render divides by the sum, so no 1/ratio is needed.
        ratio = np.maximum(self.ratio, 1e-9)
        xt, yt = undo_elliptical_transf_coord(xx, yy, self.pa, ratio)
        return super()._centred_image(xt, yt, pixel_scale_mas)


class QuadraticLimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Circular disk with quadratic limb darkening in Kipping's (2013) $q_1, q_2$.

    The brightness is $I(\mu) / I(1) = 1 - u_1 (1 - \mu) - u_2 (1 - \mu)^2$,
    with
    $u_1 = 2 \sqrt{q_1}\, q_2$ and $u_2 = \sqrt{q_1}\,(1 - 2 q_2)$
    ([Kipping 2013](https://doi.org/10.1093/mnras/stt1435), eqs. 15-16).
    Every $(q_1, q_2)$ in the unit square gives a profile that is positive
    and decreases from the centre to the limb, and every such profile has one,
    so uniform priors on $[0, 1]$ for both are uninformative over exactly the
    physical laws. The visibility is analytic (Quirrenbach et al. 1996,
    eq. 3; see
    [`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk]).

    Parameters
    ----------
    diam : float or array-like
        Limb-darkened angular diameter in milliarcseconds.
    q1, q2 : float or array-like, optional
        Kipping's coefficients, each in $[0, 1]$; $q_1 = 0$ is a uniform
        disk. Start a fit inside the square, not on its edges: ``fit`` maps
        Uniform priors onto unbounded variables, and the edges map to
        infinity (and $\sqrt{q_1}$ has an infinite derivative at 0).
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System], or a spectrum from
        [`virgil.spectra`][virgil.spectra] (default 1).
    dra : float or array-like, optional
        Right-ascension offset of the centre in milliarcseconds, positive to
        the East.
    ddec : float or array-like, optional
        Declination offset of the centre in milliarcseconds, positive to the
        North.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> star = QuadraticLimbDarkenedDisk(3.0, q1=0.4, q2=0.3)
    >>> priors = {
    ...     "diam": dist.LogUniform(2.0, 4.0),  # a scale: log-uniform
    ...     "q1": dist.Uniform(0.0, 1.0),  # Kipping's uniform prior
    ...     "q2": dist.Uniform(0.0, 1.0),
    ... }

    From tabulated $u_1, u_2$ (e.g. Claret's tables):

    >>> star = QuadraticLimbDarkenedDisk.from_u(3.0, u1=0.4, u2=0.25)
    """

    q1: jax.Array
    q2: jax.Array

    def __init__(self, diam, q1, q2, flux=1.0, dra=0.0, ddec=0.0):
        self._set_position(diam, flux, dra, ddec)
        self.q1 = np.asarray(q1, dtype=float)
        self.q2 = np.asarray(q2, dtype=float)

    @classmethod
    def from_u(cls, diam, u1, u2, **kwargs):
        """Build from the usual $u_1, u_2$ (Kipping 2013, eqs. 17-18)."""
        total = u1 + u2
        q2 = np.where(
            total == 0, 0.0, u1 / (2.0 * np.where(total == 0, 1.0, total))
        )
        return cls(diam, q1=total**2, q2=q2, **kwargs)

    @property
    def u1(self):
        return 2.0 * np.sqrt(self.q1) * self.q2

    @property
    def u2(self):
        return np.sqrt(self.q1) * (1.0 - 2.0 * self.q2)

    def _profile(self):
        u1, u2 = self.u1, self.u2
        # 1 - u1 (1 - mu) - u2 (1 - mu)^2, expanded in powers of mu
        coeffs = np.stack([1.0 - u1 - u2, u1 + 2.0 * u2, -u2])
        return coeffs, (0.0, 1.0, 2.0)

    def is_physical(self):
        return super().is_physical() & _in_unit_square(self.q1, self.q2)


class SquareRootLimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Circular disk with square-root limb darkening in Kipping's (2013) $q_1, q_2$.

    The brightness is
    $I(\mu) / I(1) = 1 - c (1 - \mu) - d (1 - \sqrt{\mu})$
    (Díaz-Cordovés & Giménez 1992), which suits late-type stars in the
    near-infrared better than the quadratic law (van Hamme 1993), with
    $c = \sqrt{q_1}\,(1 - 2 q_2)$ and $d = 2 \sqrt{q_1}\, q_2$, inverting
    [Kipping (2013)](https://doi.org/10.1093/mnras/stt1435), eqs. 23-24. As
    for [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk],
    the unit square in $(q_1, q_2)$ is exactly the set of positive profiles
    that decrease towards the limb, so uniform priors on $[0, 1]$ are
    uninformative over the physical laws. The $\sqrt{\mu}$ term needs a
    Bessel function of order $5/4$ in the analytic visibility (Quirrenbach
    et al. 1996, eq. 3; see
    [`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk]).

    Parameters
    ----------
    diam : float or array-like
        Limb-darkened angular diameter in milliarcseconds.
    q1, q2 : float or array-like, optional
        Kipping's coefficients, each in $[0, 1]$; $q_1 = 0$ is a uniform
        disk. Start a fit inside the square, not on its edges: ``fit`` maps
        Uniform priors onto unbounded variables, and the edges map to
        infinity (and $\sqrt{q_1}$ has an infinite derivative at 0).
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System], or a spectrum from
        [`virgil.spectra`][virgil.spectra] (default 1).
    dra : float or array-like, optional
        Right-ascension offset of the centre in milliarcseconds, positive to
        the East.
    ddec : float or array-like, optional
        Declination offset of the centre in milliarcseconds, positive to the
        North.

    Examples
    --------
    >>> star = SquareRootLimbDarkenedDisk(3.0, q1=0.5, q2=0.4)
    >>> same = SquareRootLimbDarkenedDisk.from_cd(3.0, c=star.c, d=star.d)
    """

    q1: jax.Array
    q2: jax.Array

    def __init__(self, diam, q1, q2, flux=1.0, dra=0.0, ddec=0.0):
        self._set_position(diam, flux, dra, ddec)
        self.q1 = np.asarray(q1, dtype=float)
        self.q2 = np.asarray(q2, dtype=float)

    @classmethod
    def from_cd(cls, diam, c, d, **kwargs):
        """Build from the usual $c, d$ (Kipping 2013, eqs. 23-24)."""
        total = c + d
        q2 = np.where(
            total == 0, 0.0, d / (2.0 * np.where(total == 0, 1.0, total))
        )
        return cls(diam, q1=total**2, q2=q2, **kwargs)

    @property
    def c(self):
        return np.sqrt(self.q1) * (1.0 - 2.0 * self.q2)

    @property
    def d(self):
        return 2.0 * np.sqrt(self.q1) * self.q2

    def _profile(self):
        c, d = self.c, self.d
        # 1 - c (1 - mu) - d (1 - sqrt(mu)), in powers of mu
        return np.stack([1.0 - c - d, c, d]), (0.0, 1.0, 0.5)

    def is_physical(self):
        return super().is_physical() & _in_unit_square(self.q1, self.q2)


def _in_unit_square(q1, q2):
    return np.all((q1 >= 0.0) & (q1 <= 1.0) & (q2 >= 0.0) & (q2 <= 1.0))


class GravityDarkenedStar(Component):
    r"""Rapidly rotating star: Roche shape and gravity darkening (ELR11).

    The equations are those of Espinosa Lara & Rieutord (2011, A&A 533, A43,
    https://arxiv.org/abs/1109.3038): a rigidly
    rotating star has the oblate Roche shape, and its local bolometric flux
    follows the effective gravity without a free gravity-darkening exponent
    $\beta$. The brightness is proportional to that local flux (no limb
    darkening yet), summed over the visible triangles of a surface mesh.
    The implementation is Aaron Dotter's GDit
    (https://github.com/aarondotter/GDit), ported to JAX by Shashank
    Dholakia in his jax-interferometry (`ELR_Model`, commit 70689ed); the physics lives in ``virgil._elr``.

    Parameters
    ----------
    diam_eq : float or array-like
        Equatorial angular diameter in milliarcseconds.
    omega : float or array-like, optional
        Angular velocity as a fraction of the Keplerian (critical) rate at
        the equator, $\Omega/\Omega_K$, in [0, 1) (default 0, a sphere).
    inc : float or array-like, optional
        Inclination in degrees, from 0 (pole-on) to 90 (equator-on, the
        default). The star is symmetric about its equator, so ``inc`` and
        ``180 - inc`` give the same image with the pole flipped; the range
        [0, 90] keeps ``pa`` unambiguous.
    pa : float or array-like, optional
        Position angle in degrees, North through East, of the visible
        rotation pole on the sky (default 0).
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System] (default 1).
    dra, ddec : float or array-like, optional
        Offset of the centre in milliarcseconds, positive to the East and
        North.
    n_lat : int, optional
        Number of latitude rings of the surface mesh (default 32, giving
        2520 triangles). Visibilities cost O(``n_lat``$^2$) per baseline.
    t_pole : float or array-like, optional
        Effective temperature of the pole in kelvin. The default ``None`` is
        the grey model; a value switches on the chromatic model (see Notes).
    wavel0 : float or array-like, optional
        Reference wavelength in metres (default 1.65e-6, H band), at which
        the star's spectrum is normalized to ``flux`` and which
        [`render`][virgil.models.SourceModel.render] shows. Only used
        when ``t_pole`` is set.

    Notes
    -----
    **Grey mode** (``t_pole=None``, Dholakia's model): each triangle is
    weighted by its bolometric flux times its projected area, the same at
    every wavelength, so wavelength enters only through ``u / wavel``.

    **Chromatic mode** (``t_pole`` set): each triangle has temperature
    $T = T_\mathrm{pole}\,T_\mathrm{eff}/T_\mathrm{eff,pole}$ from the ELR11
    gravity darkening, radiates the Planck function $B_\lambda(T)$, and is
    weighted by its projected area times that, at each sample's own
    wavelength. The hot pole and cool equator then have a contrast that
    rises towards short wavelengths. Limb darkening and bandwidth smearing
    are not modelled. The weights have shape ``(n_samples, n_triangles)``,
    so memory is about 200 MB of complex64 for 10^4 samples at the default
    ``n_lat``.

    In chromatic mode the star supplies its own spectrum to a
    [`System`][virgil.models.System]: its weight is ``flux`` times the
    summed Planck flux of its visible surface, relative to that at
    ``wavel0``. A companion then gets a physically consistent flux ratio at
    every wavelength with no separate stellar spectrum, and ``flux`` must be a
    number, not a [`Spectrum`][virgil.spectra.Spectrum], which would
    count the spectrum twice:

    ```python
    star = GravityDarkenedStar(
        1.0, omega=0.9, inc=45.0, t_pole=9000.0
    )
    companion = PointSource(flux=BlackBody(0.01, 3000.0))
    system = System(star=star, companion=companion)
    ```

    Dholakia's ``ELR_Model`` parameters map onto these as follows (his
    angles are in radians):

    | his | here |
    | --- | --- |
    | ``diam`` | ``diam_eq`` (his ``r_eq`` is ``diam_eq / 2``) |
    | ``omega`` | ``omega`` |
    | ``inc`` (0 = equator-on) | ``90 - inc`` degrees |
    | ``obl`` | ``pa`` degrees |

    **Choosing ``n_lat``.** The mesh is a second-order quadrature: the error
    in the visibility falls as ``1 / n_lat**2``, so each doubling of ``n_lat``
    cuts it by about 4. An independent ELR11 reference
    (virgil-validation) measured it. In grey mode the error is 1.4-3.4e-4 in
    visibility at ``n_lat = 128``, falling 4x per doubling. At the default
    ``n_lat = 32`` it is 2-6e-3 for fast rotators. In chromatic mode
    (``t_pole`` 9000 K) it is 1.3-1.4e-4 at 0.7, 1.65 and 2.2 µm,
    also at ``n_lat = 128``. The default suits most data, but well-measured
    data (e.g. GRAVITY) can be sensitive to errors of this size.

    To check, evaluate or refit at the best fit with ``n_lat`` doubled and
    compare χ²: if |Δχ²| ≳ 1 per dataset, use more latitude rings, and double
    again until the change is below that. The cost is O(``n_lat``$^2$) per
    baseline.

    Examples
    --------
    >>> star = GravityDarkenedStar(2.0, omega=0.8, inc=60.0, pa=30.0)
    >>> v0 = star.model(np.zeros(1), np.zeros(1), 1.65e-6)
    >>> round(float(np.abs(v0)[0]), 3)
    1.0

    The chromatic model's pole-to-equator contrast depends on wavelength:

    >>> hot = GravityDarkenedStar(2.0, omega=0.9, inc=45.0, t_pole=9000.0)
    >>> w = hot._weight(np.array([1.0e-6, 2.2e-6]))
    >>> bool(w[0] > 1.0 > w[1])
    True
    """

    diam_eq: jax.Array
    omega: jax.Array
    inc: jax.Array
    pa: jax.Array
    n_lat: int = eqx.field(static=True)
    t_pole: jax.Array | None
    wavel0: jax.Array

    def __init__(
        self,
        diam_eq,
        omega=0.0,
        inc=90.0,
        pa=0.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        n_lat=32,
        t_pole=None,
        wavel0=1.65e-6,
    ):
        self.diam_eq = np.asarray(diam_eq, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        if isinstance(n_lat, bool) or int(n_lat) != n_lat or n_lat < 4:
            raise ValueError(f"n_lat must be an integer >= 4, got {n_lat}.")
        self.n_lat = int(n_lat)
        self.t_pole = None if t_pole is None else np.asarray(t_pole, float)
        self.wavel0 = np.asarray(wavel0, dtype=float)

    def __check_init__(self):
        super().__check_init__()
        name = type(self).__name__
        if self.t_pole is not None and isinstance(self.flux, Spectrum):
            raise ValueError(
                f"{name} with t_pole supplies its own spectrum, so flux "
                "must be a number, not a Spectrum (it would be counted "
                "twice)."
            )
        for key, value in (("t_pole", self.t_pole), ("wavel0", self.wavel0)):
            value = None if value is None else concrete(value)
            if value is not None and onp.any(value <= 0.0):
                raise ValueError(
                    f"{name} has {key} {value.tolist()}; it must be positive."
                )
        diam, omega = concrete(self.diam_eq), concrete(self.omega)
        inc = concrete(self.inc)
        if diam is not None and onp.any(diam <= 0.0):
            raise ValueError(
                f"{name} has diam_eq {diam.tolist()}; it must be positive."
            )
        if omega is not None and onp.any((omega < 0.0) | (omega >= 1.0)):
            raise ValueError(
                f"{name} has omega {omega.tolist()}; it must be in [0, 1)."
            )
        if inc is not None and onp.any((inc < 0.0) | (inc > 90.0)):
            raise ValueError(
                f"{name} has inc {inc.tolist()}; it must be in [0, 90] "
                "degrees (90 = equator-on, 0 = pole-on)."
            )

    def is_physical(self):
        return (
            super().is_physical()
            & np.all(self.diam_eq > 0.0)
            & np.all((self.omega >= 0.0) & (self.omega < 1.0))
            & np.all((self.inc >= 0.0) & (self.inc <= 90.0))
            & (
                True
                if self.t_pole is None
                else np.all(self.t_pole > 0.0) & np.all(self.wavel0 > 0.0)
            )
        )

    def _surface(self, **kwargs):
        # his inc = 0 is equator-on; his obl is the pole position angle
        return _elr.surface(
            self.omega,
            self.diam_eq / 2.0,
            (90.0 - self.inc) * dtor,
            self.pa * dtor,
            self.n_lat,
            **kwargs,
        )

    def _planck_weights(self, wavel):
        """Chromatic mode: ``(x, y, weights)``, weights of shape (*wavel.shape, n_tri).

        Each triangle's temperature is ``t_pole`` times its ELR11 ``Teff``
        over the pole's; its weight is projected area times
        ``B_λ(T) / B_λ(t_pole)`` at ``wavel``, so numbers stay O(1).
        """
        x, y, _, teff, (_, _, cosine, _) = self._surface(return_mesh=True)
        area = np.heaviside(cosine, 0.0) * cosine
        return x, y, area * self._planck_intensity(teff, wavel)

    def _planck_intensity(self, teff, wavel):
        """``B_λ(T) / B_λ(t_pole)`` at ``wavel`` for triangles of ``teff``."""
        # the pole, theta = 0, through the jitted vectorized solver
        teff_pole = _elr.solve_ELR_vec(self.omega, np.zeros(1))[1][0]
        temperature = self.t_pole * teff / teff_pole
        wavel = np.asarray(wavel)[..., None]
        return _planck_ratio(wavel, temperature, self.wavel0, self.t_pole)

    def model(self, u, v, wavel):
        if self.t_pole is None:
            return super().model(u, v, wavel)
        uu, vv = u / wavel, v / wavel
        wavel = np.asarray(wavel)
        # one wavelength for all samples, or one per (flattened) sample
        if wavel.size == 1:
            lam = wavel.reshape(1)
        else:
            lam = np.broadcast_to(wavel, uu.shape).reshape(-1)
        x, y, w = self._planck_weights(lam)
        return _elr.visibilities(x, y, w, uu, vv) * offset_phase(
            uu, vv, self.dra, self.ddec
        )

    def _weight(self, wavel=None):
        if self.t_pole is None:
            return super()._weight(wavel)
        if wavel is None:
            return flux_at(self.flux)
        # SED(λ) = sum of area * B_λ(T) over the surface, relative to wavel0
        sed = self._planck_weights(wavel)[2].sum(axis=-1)
        sed0 = self._planck_weights(self.wavel0)[2].sum(axis=-1)
        return flux_at(self.flux) * sed / sed0

    def _centred_cvis(self, uu, vv):
        x, y, w, _ = self._surface()
        return _elr.visibilities(x, y, w, uu, vv)

    # sub-pixel samples per pixel side when rasterizing the image
    _image_oversample = 4

    def _centred_image(self, xx, yy, pixel_scale_mas):
        """Rasterize the faceted surface, flat-shaded per triangle.

        Each pixel averages ``_image_oversample`` squared sub-pixel samples
        of the surface brightness, so the limb is anti-aliased. A sample
        takes the intensity of the front-facing triangle that contains it in
        projection (those do not overlap, so no depth sorting is needed) and
        0 outside the outline. This is what ``plot_surface`` draws and what
        the visibilities transform, as opposed to splatting the barycentres
        onto pixels, which blurs the limb and aliases against the mesh.
        The sum matches the total weight of the DFT's point sources.
        """
        *_, teff, (pts, tri, cosine, intensity) = self._surface(
            return_mesh=True
        )
        if self.t_pole is not None:  # drawn at wavel0
            intensity = self._planck_intensity(teff, self.wavel0)
        px, py = pts[tri, 0], pts[tri, 1]  # (n_tri, 3) corners on the sky
        ax, ay = px[:, 0], py[:, 0]
        e1x, e1y = px[:, 1] - ax, py[:, 1] - ay
        e2x, e2y = px[:, 2] - ax, py[:, 2] - ay
        det = e1x * e2y - e1y * e2x
        # back-facing and degenerate triangles never contain a sample
        valid = (cosine > 0) & (det != 0)
        inv = 1.0 / np.where(valid, det, 1.0)
        value = np.where(valid, intensity, 0.0)

        def sample_row(pos):
            sx, sy = pos[0] - ax[:, None], pos[1] - ay[:, None]
            wb = (sx * e2y[:, None] - sy * e2x[:, None]) * inv[:, None]
            wc = (e1x[:, None] * sy - e1y[:, None] * sx) * inv[:, None]
            inside = valid[:, None] & (wb >= 0) & (wc >= 0) & (wb + wc <= 1)
            hits = np.sum(inside, axis=0)
            # a sample exactly on a shared edge counts once, not twice
            total = np.sum(np.where(inside, value[:, None], 0.0), axis=0)
            return total / np.maximum(hits, 1)

        # sub-pixel sample positions; the pixel centres are the grid xx, yy
        s = self._image_oversample
        nrow, ncol = xx.shape
        frac = ((np.arange(s) + 0.5) / s - 0.5) * pixel_scale_mas
        sx = (xx[0, :, None] + frac).reshape(-1)  # (ncol * s,)
        sy = (yy[:, 0, None] + frac).reshape(-1)  # (nrow * s,)
        # one sub-row of samples at a time bounds memory by ncol * s * n_tri
        sub_rows = jax.lax.map(
            lambda y: sample_row((sx, y)), sy.astype(intensity.dtype)
        )
        mean = sub_rows.reshape(nrow, s, ncol, s).mean(axis=(1, 3))
        # |normal| = 2 x area, so cosine carries twice the projected area
        # in the DFT weights; the pixel area carries the same factor here
        return 2.0 * pixel_scale_mas**2 * mean

    def plot_surface(self, ax=None, cmap="plasma"):
        """Plot the visible surface, coloured by its local brightness.

        That is the bolometric flux in grey mode, and the Planck intensity at
        ``wavel0`` in chromatic mode. East is to the left and North up, as in
        [`plot_model`][virgil.plotting.plot_model]; the offset
        ``dra``, ``ddec`` is not applied. Returns the matplotlib collection.
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        if ax is None:
            _, ax = plt.subplots()
        *_, teff, (pts, tri, cosine, intensity) = self._surface(
            return_mesh=True
        )
        if self.t_pole is not None:  # chromatic: as seen at wavel0
            intensity = self._planck_intensity(teff, self.wavel0)
        pts, tri = onp.asarray(pts), onp.asarray(tri)
        visible = onp.asarray(cosine) > 0
        triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri[visible])
        coll = ax.tripcolor(
            triang, facecolors=onp.asarray(intensity)[visible], cmap=cmap
        )
        ax.set_aspect("equal")
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.set(xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
        ax.figure.colorbar(coll, ax=ax, label="Relative local flux")
        return coll


def _modulation_array(values, name):
    """Coerce azimuthal-modulation coefficients to a 1D float array."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or 1D sequence, got shape {array.shape}."
        )
    return array


class ModulatedGaussianRim(Component):
    r"""
    Azimuthally modulated, infinitely thin rim convolved with a 2D Gaussian
    that is isotropic in the plane of the rim, optionally inclined and
    rotated.

    Parameters
    ----------
    diam : float or array-like
        Diameter of the rim in milliarcseconds.
    fwhm : float or array-like
        Gaussian FWHM of the rim in milliarcseconds, measured in the plane of
        the rim. On the sky this is the FWHM along the projected major axis;
        along the minor axis it is ``fwhm * cos(inc)``.
    inc : float or array-like
        Apparent inclination of the rim in degrees.
    pa : float or array-like
        Position angle of the rim's projected major axis in degrees, measured North
        to East (i.e. counter-clockwise in conventional astronomical image orientation).
    az_amps : float or array-like, optional
        Amplitudes of the cosine azimuthal modulations. The first element is
        the amplitude for the first-order modulation, the second for the
        second-order modulation, etc. A scalar gives a single first-order
        modulation, and the default (empty) gives an unmodulated, azimuthally
        symmetric rim. The brightness must stay non-negative, which
        ``sum(abs(az_amps)) <= 1`` guarantees; with several orders larger
        amplitudes can also be valid. Concrete values are checked when the
        model is built.
    az_pas : float or array-like, optional
        Azimuths of the cosine azimuthal modulations in degrees, one per entry
        of ``az_amps``, measured in the plane of the rim in the same sense
        as position angle, with ``pa`` on the major axis (see Notes). For an
        inclined rim these are not on-sky position angles.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][virgil.models.System],
        or a spectrum from [`virgil.spectra`][virgil.spectra]
        (default 1).
    dra : float or array-like, optional
        Right-ascension offset of the rim's center in milliarcseconds,
        positive to the East.
    ddec : float or array-like, optional
        Declination offset of the rim's center in milliarcseconds, positive
        to the North.

    Notes
    -----
    The rim is defined in its own plane and then inclined. In polar
    coordinates $(r, \phi)$ in the plane of the rim, the thin ring is
    $\delta(r - \mathrm{diam}/2) \left( 1 + \sum_{m=1}^{n}
    A_m \cos{(m(\phi - \mathrm{pa}_m))} \right)$, convolved with an
    isotropic Gaussian of FWHM ``fwhm`` in that same plane. The in-plane
    azimuth $\phi$ is counted in the same sense as position angle, with
    $\phi = \mathrm{pa}$ along the major axis. This face-on image is then
    compressed by $\cos(\mathrm{inc})$ along the minor axis, so on the sky
    the blur is an elliptical Gaussian with FWHM ``fwhm`` along the major
    axis and ``fwhm * cos(inc)`` along the minor axis. An unmodulated rim
    therefore has the same peak brightness all the way round, without
    bright ansae at the ends of the major axis.

    So ``az_pas`` are in-plane (deprojected) angles, not on-sky position
    angles. A point at in-plane azimuth $\phi$ appears at the on-sky
    position angle $\theta$ with $\tan(\theta - \mathrm{pa}) =
    \cos(\mathrm{inc}) \tan(\phi - \mathrm{pa})$, in the same quadrant.
    The two agree for a face-on rim (``inc = 0``) and along the major and
    minor axes. Otherwise a peak at ``az_pas`` appears on the sky at a
    position angle closer to the major axis.

    This model is achromatic: it does not represent any spectral dependence.
    The rim contains no star; put it in a [`System`][virgil.models.System] with a
    [`PointSource`][virgil.models.PointSource] for that.

    Examples
    --------
    >>> rim = ModulatedGaussianRim(
    ...     diam=40.0, fwhm=4.0, inc=50.0, pa=30.0, az_amps=0.7, az_pas=120.0
    ... )
    """

    diam: jax.Array
    fwhm: jax.Array
    inc: jax.Array
    pa: jax.Array
    az_amps: jax.Array
    az_pas: jax.Array

    def __init__(
        self,
        diam,
        fwhm,
        inc,
        pa,
        az_amps=(),
        az_pas=(),
        flux=1.0,
        dra=0.0,
        ddec=0.0,
    ):
        self.diam = np.asarray(diam, dtype=float)
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.az_amps = _modulation_array(az_amps, "az_amps")
        self.az_pas = _modulation_array(az_pas, "az_pas")
        if self.az_amps.shape != self.az_pas.shape:
            raise ValueError(
                f"az_amps has {self.az_amps.size} entries but az_pas has "
                f"{self.az_pas.size}; give one position angle per modulation."
            )
        _check_non_negative_modulation(self.az_amps, self.az_pas)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def is_physical(self):
        # Non-negative flux, and a brightness that stays non-negative round
        # the rim.
        return super().is_physical() & np.asarray(
            check_az_prof_nonnegative(self.az_amps, self.az_pas)
        )

    def _ring_angle(self, sky_pa):
        """The in-plane azimuth whose projection points at ``sky_pa``."""
        return _rim_angle(sky_pa, self.pa, self.inc)

    def _centred_cvis(self, uu, vv):
        return _cvis_centred_rim(
            uu,
            vv,
            self.diam,
            self.fwhm,
            self.inc,
            self.pa,
            self.az_amps,
            self.az_pas - self.pa,
        )

    def _centred_image(self, xx, yy, pixel_scale_mas):
        stretch = np.maximum(np.cos(self.inc * dtor), 1e-8)
        xx_ell, yy_ell = undo_elliptical_transf_coord(xx, yy, self.pa, stretch)
        r_ell = np.hypot(xx_ell, yy_ell)
        theta_ell = np.arctan2(xx_ell, yy_ell)

        # Anti-aliased thin ring: Gaussian in de-projected radius, at least
        # half a pixel wide on the sky along the compressed minor axis.
        width = 0.5 * pixel_scale_mas / stretch
        ring = np.exp(-0.5 * ((r_ell - self.diam / 2.0) / width) ** 2)

        az_amps = np.concatenate([np.array([1.0]), self.az_amps])
        az_phis_rad = (
            np.concatenate([np.array([0.0]), self.az_pas - self.pa]) * dtor
        )
        az_orders = np.arange(az_amps.size)

        def _az_term(az_amp, az_phi_rad, az_order):
            return az_amp * np.cos(az_order * (theta_ell - az_phi_rad))

        az_factors = jax.vmap(_az_term)(az_amps, az_phis_rad, az_orders)
        ring = ring * np.sum(az_factors, axis=0)

        # Odd-sized kernel centred on a pixel, so mode="same" introduces no shift.
        npix = xx.shape[0]
        nker = npix + 1 - npix % 2
        kx, ky = image_coordinates(nker, nker * pixel_scale_mas)
        sigma_mas = np.maximum(self.fwhm / FWHM_PER_SIGMA, 1e-9)
        # Gaussian kernel that is isotropic in the rim plane: on the sky it
        # has sigma_mas along the major axis and sigma_mas * stretch along
        # the minor axis. The stretch is floored so that the sky-plane
        # minor-axis sigma stays at least half a pixel, or sigma_mas if that
        # is smaller (a near edge-on kernel then cannot fall between pixel
        # centres), and capped at 1 so that a face-on rim keeps its
        # isotropic sky kernel.
        psf_stretch = np.minimum(
            np.maximum(stretch, 0.5 * pixel_scale_mas / sigma_mas), 1.0
        )
        kx_ell, ky_ell = undo_elliptical_transf_coord(
            kx, ky, self.pa, psf_stretch
        )
        psf = np.exp(-0.5 * (kx_ell**2 + ky_ell**2) / sigma_mas**2)
        return fftconvolve(ring, psf, mode="same")


def _check_log_brightness(log_brightness, support):
    """Reject log-brightnesses whose softmax over the support is not finite.

    Skipped under ``jax.jit``.
    """
    if support is None:
        support = np.ones(log_brightness.shape, dtype=bool)
    invalid = np.isnan(log_brightness) | (log_brightness == np.inf)
    bad = concrete(np.any(support & invalid))
    if bad:
        raise ValueError("log_brightness must not be NaN or +inf.")
    finite = concrete(np.any(support & np.isfinite(log_brightness)))
    if finite is not None and not bool(finite):
        raise ValueError(
            "log_brightness needs at least one finite (supported) pixel."
        )


def circular_support(npix, pixel_scale_mas, radius_mas, inner_radius_mas=0.0):
    """Pixels of an ``npix`` x ``npix`` image within ``radius_mas`` of its centre.

    Returns a boolean array for the ``support`` of an
    [`Image`][virgil.models.Image]. With ``inner_radius_mas``, pixels
    closer to the centre than that are left out too: a hole under an
    analytic star, so that the image cannot pile flux onto it.
    """
    offsets = pixel_offsets(int(npix), float(pixel_scale_mas))
    radius = np.hypot(offsets[None, :], offsets[:, None])
    return (radius <= radius_mas) & (radius >= inner_radius_mas)


def _pixel_visibilities(
    fluxes, pixel_scale_mas, rotation_deg, u, v, wavel, grid=None
):
    """Fourier transform of pixel fluxes centred on the origin, unnormalized.

    Uses the exact matrix Fourier transform when the samples lie on a uv
    ``grid`` whose rotation matches the pixels' (and there is a single
    wavelength), and the direct transform otherwise.
    """
    matched = (
        grid is not None
        and abs(grid.rotation_deg - rotation_deg) < 1e-9
        and np.size(wavel) == 1
    )
    if matched:
        wavel = np.reshape(wavel, ())
        vis = grid_visibilities(
            fluxes, grid.u_axis / wavel, grid.v_axis / wavel, pixel_scale_mas
        )
        return vis.ravel()[grid.index]
    uu, vv = rotate(u / wavel, v / wavel, -rotation_deg)
    return image_visibilities(fluxes, uu, vv, pixel_scale_mas)


class Image(Component):
    """Pixelised brightness distribution, for image reconstruction.

    The pixel fluxes are ``brightness = softmax(log_brightness)`` taken over
    the pixels in ``support``: positive, summing to one, and exactly zero
    outside the support. Like any other component, the image has a ``flux``
    weight and an offset inside a [`System`][virgil.models.System], so
    unresolved sources can stay analytic (e.g. a
    [`PointSource`][virgil.models.PointSource] star) while resolved
    emission goes in the pixels.

    Visibilities are the exact Fourier transform of the pixels, each treated
    as a point at its centre.

    Parameters
    ----------
    log_brightness : array-like, shape (nrow, ncol), or GaussianField
        Log pixel fluxes, up to an additive constant, in the orientation of
        [`render`][virgil.models.SourceModel.render]: row 0 is the top
        (North) and column 0 the left (East) of the image. Alternatively an
        object with an ``evaluate(pixel_scale_mas)`` method returning them,
        such as a [`GaussianField`][virgil.fields.GaussianField] (a
        Gaussian-process prior on the log-brightness).
    pixel_scale_mas : float
        Pixel size in milliarcseconds. The image centre, at index
        ``((nrow - 1) / 2, (ncol - 1) / 2)``, is at ``(dra, ddec)``.
    support : array-like of bool, optional
        Pixels allowed to carry flux (default: all); see
        [`circular_support`][virgil.models.circular_support].
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System] (default 1).
    dra, ddec : float, optional
        Offset of the image centre in milliarcseconds, positive to the East
        and North.
    rotation_deg : float, optional
        Position angle (North to East) of the pixel grid's "up" axis, so
        that pixels can follow a detector rather than the sky; default 0
        (North up). An image whose rotation matches the uv lattice of its
        data (``data.uv_grid.rotation_deg``, e.g. AMI data at their
        parallactic angle) is transformed with an exact two-sided matrix
        Fourier transform, which is much faster.

    Examples
    --------
    >>> image = Image(np.zeros((32, 32)), pixel_scale_mas=2.0)
    >>> envelope = Image.from_model(GaussianDisk(sigma=8.0), 32, 2.0, flux=0.3)
    >>> scene = System(star=PointSource(), env=envelope)
    """

    log_brightness: Any
    support: jax.Array | None
    pixel_scale_mas: float = eqx.field(static=True)
    rotation_deg: float = eqx.field(static=True)

    def __init__(
        self,
        log_brightness,
        pixel_scale_mas,
        support=None,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        rotation_deg=0.0,
    ):
        if not hasattr(log_brightness, "evaluate"):
            log_brightness = np.asarray(log_brightness, dtype=float)
        self.log_brightness = log_brightness
        if not (onp.isfinite(pixel_scale_mas) and pixel_scale_mas > 0.0):
            raise ValueError(
                f"pixel_scale_mas must be finite and positive, not {pixel_scale_mas}."
            )
        self.pixel_scale_mas = float(pixel_scale_mas)
        eta = self.eta
        if eta.ndim != 2:
            raise ValueError("log_brightness must be a 2D array.")
        if support is not None:
            support = np.asarray(support, dtype=bool)
            if support.shape != eta.shape:
                raise ValueError(
                    f"support has shape {support.shape}, but log_brightness "
                    f"has shape {eta.shape}."
                )
            any_pixel = concrete(np.any(support))
            if any_pixel is not None and not bool(any_pixel):
                raise ValueError("support must contain at least one pixel.")
        self.support = support
        _check_log_brightness(eta, support)
        self.rotation_deg = float(rotation_deg)
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    @classmethod
    def from_brightness(
        cls, brightness, pixel_scale_mas, *, floor=1e-6, **kwargs
    ):
        """Build an image from non-negative pixel fluxes.

        Pixels fainter than ``floor`` times the brightest are raised to that
        level, so that their logarithm is finite. Other keyword arguments
        (``support``, ``flux``, ``dra``, ``ddec``, ``rotation_deg``) go to
        [`Image`][virgil.models.Image].
        """
        brightness = np.asarray(brightness, dtype=float)
        support = kwargs.get("support")
        inside = (
            brightness
            if support is None
            else np.where(support, brightness, 0.0)
        )
        peak = np.max(inside)
        if concrete(peak) is not None and not float(peak) > 0.0:
            raise ValueError("brightness needs a positive (supported) pixel.")
        brightness = np.maximum(brightness, floor * peak)
        return cls(np.log(brightness), pixel_scale_mas, **kwargs)

    @classmethod
    def from_model(cls, model, npix, pixel_scale_mas, **kwargs):
        """Pixelise a model, e.g. a parametric fit, as a starting image.

        The pixels sample ``model`` on an ``npix`` x ``npix`` grid (rotated
        by ``rotation_deg``, if given); see
        [`from_brightness`][virgil.models.Image.from_brightness] for
        the keyword arguments.
        """
        xx, yy = image_coordinates(npix, npix * pixel_scale_mas)
        xx, yy = rotate(xx, yy, kwargs.get("rotation_deg", 0.0))
        image = _normalize_image(model._image(xx, yy, pixel_scale_mas))
        return cls.from_brightness(image, pixel_scale_mas, **kwargs)

    @property
    def eta(self):
        """The log-brightness array, evaluated if it is a field."""
        if hasattr(self.log_brightness, "evaluate"):
            return self.log_brightness.evaluate(self.pixel_scale_mas)
        return self.log_brightness

    @property
    def brightness(self):
        """Pixel fluxes: positive, unit sum, zero outside ``support``."""
        eta = self.eta
        # As a mask: some zodiax versions return leaves as floats from get().
        where = None if self.support is None else self.support.ravel() != 0
        flat = jax.nn.softmax(eta.ravel(), where=where)
        return flat.reshape(eta.shape)

    def _centred_cvis(self, uu, vv):
        uu, vv = rotate(uu, vv, -self.rotation_deg)
        return image_visibilities(
            self.brightness, uu, vv, self.pixel_scale_mas
        )

    def model_on_grid(self, u, v, wavel, grid):
        vis = _pixel_visibilities(
            self.brightness,
            self.pixel_scale_mas,
            self.rotation_deg,
            u,
            v,
            wavel,
            grid,
        )
        return vis * offset_phase(u / wavel, v / wavel, self.dra, self.ddec)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        # Bilinear resampling of the pixels, exact at their centres.
        xx, yy = rotate(xx, yy, -self.rotation_deg)
        brightness = self.brightness
        nrow, ncol = brightness.shape
        rows = 0.5 * (nrow - 1) - yy / self.pixel_scale_mas
        cols = 0.5 * (ncol - 1) - xx / self.pixel_scale_mas
        return map_coordinates(brightness, [rows, cols], order=1)


class FlaredDisk(Component):
    r"""Flared, inclined scattered-light disk (Blakely et al. 2024, §III).

    The geometrical disk model that Blakely et al. (2024, arXiv:2404.13032)
    fitted to JWST AMI data of PDS 70: a skewed Gaussian ring on a flared
    surface, with a forward-scattering peak on its near side. It is an
    abstract base: use [`FlaredDiskHG`][virgil.models.FlaredDiskHG],
    [`FlaredDiskGaussian`][virgil.models.FlaredDiskGaussian] or
    [`FlaredDiskPowerLaw`][virgil.models.FlaredDiskPowerLaw], which
    differ only in the azimuthal phase function. The disk contains no star
    or planets; compose them in a [`System`][virgil.models.System], and
    any over-resolved flux with a [`Resolved`][virgil.models.Resolved]
    component (the paper's $I_o$).

    The brightness has no analytic Fourier transform, so the visibilities
    are the exact Fourier transform of the brightness sampled on a fixed,
    centred grid of ``npix`` x ``npix`` pixels of ``pixel_scale_mas``. The
    grid must cover the whole disk, and its pixels must be small enough to
    resolve the ring and its sharp inner edge.

    Parameters
    ----------
    radius : float or array-like
        Radius of peak brightness $r_0$ in milliarcseconds (before the
        skew, which moves the peak outwards).
    fwhm : float or array-like
        Radial FWHM of the Gaussian ring, $2\sqrt{2\ln 2}\,\sigma_r$, in
        milliarcseconds.
    inc : float or array-like
        Inclination in degrees, from 0 (face-on) up to but not including
        90 (edge-on, where the surface cannot be deprojected).
    pa : float or array-like
        Position angle of the projected major axis in degrees, North to
        East. The near side, where forward scattering peaks, is at
        ``pa + 90``: ``pa`` and ``pa + 180`` are mirror images.
    npix : int
        Pixels on a side of the grid the visibilities are computed from;
        must be even, so that no pixel centre falls on the star, where the
        surface is singular.
    pixel_scale_mas : float
        Pixel size of that grid in milliarcseconds.
    skew : float or array-like, optional
        Truncation $\alpha$ of the ring's inner edge (default 0, a
        symmetric Gaussian ring).
    aspect : float or array-like, optional
        Aspect ratio $z/\rho$ of the scattering surface at ``radius``
        (default 0, a flat disk).
    flaring : float or array-like, optional
        Flaring index $\beta$ of the surface (default 1.25).
    symmetric : float or array-like, optional
        Brightness of the axisymmetric part relative to the phase function,
        $A_s/A_a$ in the paper (default 0).
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System] (default 1): the disk/star
        flux ratio when the star has ``flux=1``.
    dra, ddec : float or array-like, optional
        Offset of the disk centre in milliarcseconds, positive to the East
        and North.

    Notes
    -----
    The brightness follows Eqs. 2–9 of the paper. Sky offsets are rotated
    so that $x$ runs along the major axis and $y$ along the minor axis,
    positive towards the near side, and $y$ is divided by $\cos i$ to give
    mid-plane coordinates. The surface height
    $z = h\,r_0\,(\rho/r_0)^\beta$, with $\rho = \sqrt{x^2 + y^2}$ and
    $h$ = ``aspect``, raises the apparent radius to
    $r = \sqrt{x^2 + (y + z\sin i)^2 + z^2}$, which shifts the ring
    towards the far side. The brightness is

    $$I(r, \theta) = \left(f(\theta) + A_s/A_a\right)
    \exp\left(-\frac{(r - r_0)^2}{2\sigma_r^2}\right)
    \frac{1}{2}\left(1 + \mathrm{erf}\left(
    \frac{\alpha (r - r_0)}{\sqrt{2}\sigma_r}\right)\right),$$

    where $\theta = \arctan(x / y)$ is the mid-plane azimuth from the
    near-side minor axis and $f$ the phase function. The paper gives the
    height as $H_{100}$ (au) at 100 au, which is
    ``aspect`` $= (H_{100} / 100\,\mathrm{au})(r_0 / 100\,\mathrm{au})^{\beta - 1}$
    with $r_0$ in au; its fitted fluxes $A_a, A_s$ are absolute, and here
    only their ratio and the disk's total ``flux`` enter.
    """

    radius: jax.Array
    fwhm: jax.Array
    inc: jax.Array
    pa: jax.Array
    skew: jax.Array
    aspect: jax.Array
    flaring: jax.Array
    symmetric: jax.Array
    npix: int = eqx.field(static=True)
    pixel_scale_mas: float = eqx.field(static=True)

    def __init__(
        self,
        radius,
        fwhm,
        inc,
        pa,
        npix,
        pixel_scale_mas,
        skew=0.0,
        aspect=0.0,
        flaring=1.25,
        symmetric=0.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
    ):
        self.radius = np.asarray(radius, dtype=float)
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.skew = np.asarray(skew, dtype=float)
        self.aspect = np.asarray(aspect, dtype=float)
        self.flaring = np.asarray(flaring, dtype=float)
        self.symmetric = np.asarray(symmetric, dtype=float)
        self.npix = int(npix)
        if self.npix <= 0 or self.npix % 2:
            raise ValueError(f"npix must be positive and even, got {npix}.")
        self.pixel_scale_mas = float(pixel_scale_mas)
        if not (
            onp.isfinite(self.pixel_scale_mas) and self.pixel_scale_mas > 0
        ):
            raise ValueError(
                f"pixel_scale_mas must be positive, got {pixel_scale_mas}."
            )
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def _phase_function(self, theta):
        """Azimuthal brightness ``f(θ)``, with θ in radians from the near side."""
        raise NotImplementedError

    def is_physical(self):
        # The ring is divided by its radius and width.
        return (
            super().is_physical()
            & (self.radius > 0.0)
            & (self.fwhm > 0.0)
            & (self.symmetric >= 0.0)
            # Edge-on (and beyond) cannot be deprojected.
            & (np.abs(self.inc) < 90.0)
        )

    def _centred_cvis(self, uu, vv):
        xx, yy = image_coordinates(self.npix, self.npix * self.pixel_scale_mas)
        pixels = self._centred_image(xx, yy, self.pixel_scale_mas)
        return image_visibilities(
            pixels / np.sum(pixels), uu, vv, self.pixel_scale_mas
        )

    def _centred_image(self, xx, yy, pixel_scale_mas):
        # Mid-plane coordinates: x along the major axis, y along the minor
        # axis (towards the near side, at pa + 90), deprojected.
        cos_inc = np.maximum(np.cos(self.inc * dtor), 1e-8)
        y, x = undo_elliptical_transf_coord(xx, yy, self.pa, cos_inc)

        # Flared surface (Eq. 2) and apparent radius (Eq. 3). The floors
        # keep gradients finite at the star, where both radii vanish.
        tiny = np.finfo(np.result_type(x, float)).tiny
        rho = np.sqrt(np.maximum(x**2 + y**2, tiny))
        z = self.aspect * self.radius * (rho / self.radius) ** self.flaring
        r2 = x**2 + (y + z * np.sin(self.inc * dtor)) ** 2 + z**2
        r = np.sqrt(np.maximum(r2, tiny))

        # Skewed Gaussian ring (Eqs. 4-5) times the azimuthal term (Eq. 9).
        sigma = self.fwhm / FWHM_PER_SIGMA
        offset = (r - self.radius) / sigma
        ring = (
            np.exp(-0.5 * offset**2)
            * 0.5
            * (1.0 + jax.scipy.special.erf(self.skew * offset / np.sqrt(2.0)))
        )
        theta = np.arctan2(x, y)
        return (self._phase_function(theta) + self.symmetric) * ring


class FlaredDiskHG(FlaredDisk):
    r"""[`FlaredDisk`][virgil.models.FlaredDisk] with a Henyey–Greenstein phase function.

    $f(\theta) = \dfrac{1 - g^2}{4\pi\,(1 + g^2 - 2g\cos\theta)^{3/2}}$
    (Blakely et al. 2024, Eq. 6).

    Parameters
    ----------
    g : float or array-like
        Asymmetry parameter, with ``-1 < g < 1``: 0 is isotropic, positive
        values scatter forwards (peaking on the near side) and negative
        values backwards (peaking on the far side).
    **geometry
        The parameters of [`FlaredDisk`][virgil.models.FlaredDisk].

    Examples
    --------
    >>> disk = FlaredDiskHG(
    ...     g=0.3, radius=440.0, fwhm=290.0, inc=52.0, pa=160.0,
    ...     npix=96, pixel_scale_mas=20.0, flux=0.05,
    ... )
    """

    g: jax.Array

    def __init__(self, g, **geometry):
        self.g = np.asarray(g, dtype=float)
        super().__init__(**geometry)

    def is_physical(self):
        return super().is_physical() & (np.abs(self.g) < 1.0)

    def _phase_function(self, theta):
        g = self.g
        return (1.0 - g**2) / (
            4.0 * np.pi * (1.0 + g**2 - 2.0 * g * np.cos(theta)) ** 1.5
        )


class FlaredDiskGaussian(FlaredDisk):
    r"""[`FlaredDisk`][virgil.models.FlaredDisk] with a Gaussian phase function.

    $f(\theta) = \exp\left(-\theta^2 / 2\sigma_\theta^2\right)$, with
    $\theta \in (-180°, 180°]$ (Blakely et al. 2024, Eq. 7).

    Parameters
    ----------
    sigma_theta : float or array-like
        Azimuthal width in degrees.
    **geometry
        The parameters of [`FlaredDisk`][virgil.models.FlaredDisk].
    """

    sigma_theta: jax.Array

    def __init__(self, sigma_theta, **geometry):
        self.sigma_theta = np.asarray(sigma_theta, dtype=float)
        super().__init__(**geometry)

    def is_physical(self):
        return super().is_physical() & (self.sigma_theta > 0.0)

    def _phase_function(self, theta):
        return np.exp(-0.5 * (theta / (self.sigma_theta * dtor)) ** 2)


class FlaredDiskPowerLaw(FlaredDisk):
    r"""[`FlaredDisk`][virgil.models.FlaredDisk] with a power-law phase function.

    $f(\theta) = \cos^N(\theta / 2)$, with $\theta \in (-180°, 180°]$
    (Blakely et al. 2024, Eq. 8), the best-fitting form for PDS 70.

    Parameters
    ----------
    n : float or array-like
        Power $N$; larger is more concentrated towards the near side.
    **geometry
        The parameters of [`FlaredDisk`][virgil.models.FlaredDisk].
    """

    n: jax.Array

    def __init__(self, n, **geometry):
        self.n = np.asarray(n, dtype=float)
        super().__init__(**geometry)

    def is_physical(self):
        return super().is_physical() & (self.n >= 0.0)

    def _phase_function(self, theta):
        # Clipped: cos(θ/2) rounds to slightly below zero at θ = ±π.
        return np.maximum(np.cos(0.5 * theta), 0.0) ** self.n


# ---------------------------------------------------------------------------
# PolarDisk: radial profiles and a quadrature (Hankel-transform) engine
# ---------------------------------------------------------------------------

# Largest number of (sample, node) Bessel evaluations held at once; bigger
# problems are chunked over the samples with ``jax.lax.map``.
_POLAR_CHUNK_ELEMENTS = 2**24
# Half-width, in widths, of the quadrature range of a Gaussian or soft edge:
# the mass beyond it is about 2e-9.
_POLAR_TAIL_WIDTHS = 6.0
# numpy's Gauss-Laguerre weights underflow to 0 and then NaN beyond this.
_LAGUERRE_MAX_NODES = 150


def _legendre_rule(n_r):
    """Gauss-Legendre nodes and weights on [-1, 1] (NumPy, built once)."""
    x, w = onp.polynomial.legendre.leggauss(int(n_r))
    return np.asarray(x), np.asarray(w)


def _laguerre_rule(n_r):
    """Gauss-Laguerre nodes and weights for the weight ``exp(-t)`` on [0, inf)."""
    if n_r > _LAGUERRE_MAX_NODES:
        raise ValueError(
            f"A Gauss-Laguerre rule has at most {_LAGUERRE_MAX_NODES} nodes "
            f"(its weights underflow beyond that), not {n_r}; use a "
            "RadialPowerLaw with a large r_out and a soft edge for a tail "
            "resolved that finely."
        )
    t, w = onp.polynomial.laguerre.laggauss(int(n_r))
    return np.asarray(t), np.asarray(w)


def _legendre_nodes(rule, lo, hi):
    """Nodes on [lo, hi] and their ``dr`` weights from a [-1, 1] rule.

    The rule arrays are inputs of the computation (never baked into the
    compiled program) and carry no gradient.
    """
    x, w = (jax.lax.stop_gradient(a) for a in rule)
    half = 0.5 * (hi - lo)
    return 0.5 * (hi + lo) + half * x, half * w


def _soft_step(x, width):
    """Smooth 0 -> 1 step, an error function of standard deviation ``width``."""
    return 0.5 * (1.0 + jax.scipy.special.erf(x / (math.sqrt(2.0) * width)))


class _RadialProfile(zdx.Base):
    """Base class of the radial profiles of a [`PolarDisk`][virgil.models.PolarDisk].

    A profile is a non-negative brightness ``I(r)`` of the radius ``r`` in
    the plane of the disk (milliarcseconds). It owns its quadrature rule:
    ``quadrature`` returns nodes ``r_k`` and weights ``w_k`` such that
    ``sum_k w_k f(r_k)`` approximates ``int I(r) f(r) r dr`` for smooth
    ``f``. The rule's reference arrays are fields of the profile, filled by
    ``with_rule(n_r)`` (which [`PolarDisk`][virgil.models.PolarDisk] calls), so
    they are traced inputs and changing a parameter never recompiles.

    ``weight`` is the profile's share of the light when it is one part of a
    [`RadialSum`][virgil.models.RadialSum]; alone it has no effect.
    """

    weight: jax.Array
    rule_x: Any
    rule_w: Any

    @property
    def rule(self):
        """The quadrature rule's reference nodes and weights, or ``None``."""
        if self.rule_x is None:
            return None
        return self.rule_x, self.rule_w

    def with_rule(self, n_r):
        """This profile with an ``n_r``-node quadrature rule."""
        return eqx.tree_at(
            lambda p: (p.rule_x, p.rule_w),
            self,
            self._reference_rule(int(n_r)),
            is_leaf=lambda x: x is None,
        )

    def _reference_rule(self, n_r):
        return _legendre_rule(n_r)

    def _check_rule(self):
        if self.rule is None:
            raise ValueError(
                f"{type(self).__name__} has no quadrature rule; build it "
                "into a PolarDisk, or call with_rule(n_r)."
            )

    def quadrature(self):
        """Nodes ``r`` (mas) and weights ``w`` including ``I(r) r dr``."""
        raise NotImplementedError

    def intensity(self, r, pixel_scale_mas):
        """``I(r)`` on a radius array, with the weight scale of ``quadrature``."""
        raise NotImplementedError

    def extent(self):
        """Concrete (inner, outer) radius of the support, for ``suggest_n_r``."""
        raise NotImplementedError

    def _n_needed(self, q_max):
        """Nodes for oscillations up to the frequency ``q_max`` (cycles/rad)."""
        lo, hi = self.extent()
        phase = onp.pi * float(q_max) * (hi - lo) * mas2rad
        return int(onp.ceil(phase)) + 12

    def is_physical(self):
        return np.asarray(self.weight >= 0.0)

    def _normalized_quadrature(self):
        """``quadrature`` scaled to unit total, for mixing."""
        r, w = self.quadrature()
        return r, w / np.sum(w)


class ThinRing(_RadialProfile):
    r"""Infinitely thin ring of radius ``r0``, $I \propto \delta(r - r_0)$.

    Its Hankel transform is $J_0(2\pi q r_0)$, so it needs no quadrature
    (``n_r`` is irrelevant). Rendered, it is a Gaussian a pixel wide.

    Parameters
    ----------
    r0 : float or array-like
        Radius in milliarcseconds, in the plane of the disk.
    weight : float or array-like, optional
        Relative weight inside a [`RadialSum`][virgil.models.RadialSum].
    """

    r0: jax.Array

    def __init__(self, r0, weight=1.0):
        self.r0 = np.asarray(r0, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def with_rule(self, n_r):
        return self

    def quadrature(self):
        return np.reshape(self.r0, (1,)), np.ones((1,), dtype=self.r0.dtype)

    def intensity(self, r, pixel_scale_mas):
        s = np.maximum(0.5 * pixel_scale_mas, 1e-9)
        norm = math.sqrt(2.0 * math.pi) * s * np.maximum(self.r0, s)
        return np.exp(-0.5 * ((r - self.r0) / s) ** 2) / norm

    def extent(self):
        r0 = float(onp.max(onp.asarray(self.r0)))
        return r0, r0

    def _n_needed(self, q_max):
        return 1

    def is_physical(self):
        return super().is_physical() & np.asarray(self.r0 >= 0.0)


class GaussianRing(_RadialProfile):
    r"""Gaussian ring, $I \propto \exp[-(r - r_0)^2 / 2\sigma^2]$, with $r \ge 0$.

    The quadrature runs over $r_0 \pm 6\sigma$ (clipped at the centre) with
    a Gauss-Legendre rule, which converges spectrally.

    Parameters
    ----------
    r0 : float or array-like
        Radius of the peak in milliarcseconds.
    sigma : float or array-like
        Standard deviation of the radial profile in milliarcseconds.
    weight : float or array-like, optional
        Relative weight inside a [`RadialSum`][virgil.models.RadialSum].

    References
    ----------
    Gaussian rings are the basic building block of the parametric radial
    models of vis-r (Kennedy 2025, RASTI 4, rzaf004;
    https://github.com/drgmk/vis-r).
    """

    r0: jax.Array
    sigma: jax.Array

    def __init__(self, r0, sigma, weight=1.0):
        self.r0 = np.asarray(r0, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def _range(self):
        half = _POLAR_TAIL_WIDTHS * self.sigma
        return np.maximum(self.r0 - half, 0.0), self.r0 + half

    def quadrature(self):
        self._check_rule()
        r, dr = _legendre_nodes(self.rule, *self._range())
        return r, dr * r * self.intensity(r, 0.0)

    def intensity(self, r, pixel_scale_mas):
        return np.exp(-0.5 * ((r - self.r0) / self.sigma) ** 2)

    def extent(self):
        r0, s = (float(onp.max(onp.asarray(a))) for a in (self.r0, self.sigma))
        return max(
            r0 - _POLAR_TAIL_WIDTHS * s, 0.0
        ), r0 + _POLAR_TAIL_WIDTHS * s

    def is_physical(self):
        return (
            super().is_physical()
            & np.asarray(self.r0 >= 0.0)
            & np.asarray(self.sigma > 0.0)
        )


class RadialPowerLaw(_RadialProfile):
    r"""Power law $I \propto r^{\rm index}$ between ``r_in`` and ``r_out``.

    With ``edge`` (a length in milliarcseconds) both edges are soft, error
    functions of that standard deviation, as in vis-r; the quadrature then
    runs over ``edge`` * 6 beyond them, and ``n_r`` must resolve ``edge``.
    With ``edge=None`` the edges are sharp and the Gauss-Legendre rule spans
    exactly ``[r_in, r_out]``, so the transform is spectrally accurate for
    a smooth power law (``index=0``, ``r_in=0`` is a uniform disk).

    Parameters
    ----------
    r_in, r_out : float or array-like
        Inner and outer radius in milliarcseconds.
    index : float or array-like, optional
        Power-law index (default 0).
    edge : float, array-like or None, optional
        Standard deviation of the soft edges in milliarcseconds, or ``None``
        for sharp edges (default).
    weight : float or array-like, optional
        Relative weight inside a [`RadialSum`][virgil.models.RadialSum].

    References
    ----------
    Power-law profiles with error-function edges follow vis-r
    (Kennedy 2025, RASTI 4, rzaf004; https://github.com/drgmk/vis-r).
    """

    r_in: jax.Array
    r_out: jax.Array
    index: jax.Array
    edge: Any

    def __init__(self, r_in, r_out, index=0.0, edge=None, weight=1.0):
        self.r_in = np.asarray(r_in, dtype=float)
        self.r_out = np.asarray(r_out, dtype=float)
        self.index = np.asarray(index, dtype=float)
        self.edge = None if edge is None else np.asarray(edge, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def _range(self):
        if self.edge is None:
            return self.r_in, self.r_out
        half = _POLAR_TAIL_WIDTHS * self.edge
        return np.maximum(self.r_in - half, 0.0), self.r_out + half

    def quadrature(self):
        self._check_rule()
        r, dr = _legendre_nodes(self.rule, *self._range())
        return r, dr * r * self.intensity(r, 0.0)

    def intensity(self, r, pixel_scale_mas):
        base = np.maximum(r, 1e-9) ** self.index
        if self.edge is None:
            inside = (r >= self.r_in) & (r <= self.r_out)
            return np.where(inside, base, 0.0)
        return (
            base
            * _soft_step(r - self.r_in, self.edge)
            * _soft_step(self.r_out - r, self.edge)
        )

    def extent(self):
        lo, hi = self._range()
        return float(onp.min(onp.asarray(lo))), float(onp.max(onp.asarray(hi)))

    def is_physical(self):
        valid = (
            super().is_physical()
            & np.asarray(self.r_in >= 0.0)
            & np.asarray(self.r_out > self.r_in)
        )
        if self.edge is not None:
            valid = valid & np.asarray(self.edge > 0.0)
        return valid


class ExpTail(_RadialProfile):
    r"""Exponential tail, $I \propto \exp[-(r - r_0)/L]$ for $r \ge r_0$.

    By default a Gauss-Laguerre rule integrates the whole tail to infinity,
    so no outer radius is needed and none of the tail is dropped. ``r0 = 0``
    is an exponential disk. Gauss-Laguerre nodes cover the oscillating tail
    less efficiently than Gauss-Legendre does a finite range: about
    $16 \cdot 2\pi q L$ nodes give 1e-5 at the longest baseline $q$ (24
    nodes at $2\pi q L = 1$, 100 at 5). For a tail that is resolved much
    more finely, give a ``cutoff``: the tail is then cut at
    $r_0 + \mathrm{cutoff}\, L$ (the light beyond is dropped) and
    integrated by a Gauss-Legendre rule, which needs about
    $\pi q\, \mathrm{cutoff}\, L + 12$ nodes (a cutoff of 5 drops 0.7 %
    of the flux of a thin disk). See ``PolarDisk.suggest_n_r``.

    Parameters
    ----------
    r0 : float or array-like
        Inner radius in milliarcseconds, where the tail starts.
    scale : float or array-like
        e-folding length $L$ in milliarcseconds.
    weight : float or array-like, optional
        Relative weight inside a [`RadialSum`][virgil.models.RadialSum].
    cutoff : float or None, optional
        Cut the tail at ``r0 + cutoff * scale`` (a static number; ``None``,
        the default, is the whole tail).
    """

    r0: jax.Array
    scale: jax.Array
    cutoff: float | None = eqx.field(static=True)

    def __init__(self, r0, scale, weight=1.0, cutoff=None):
        if cutoff is not None and not float(cutoff) > 0.0:
            raise ValueError(f"cutoff must be positive or None: {cutoff!r}")
        self.r0 = np.asarray(r0, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.rule_x = None
        self.rule_w = None

    def _reference_rule(self, n_r):
        if self.cutoff is not None:
            return _legendre_rule(n_r)
        return _laguerre_rule(n_r)

    def quadrature(self):
        self._check_rule()
        if self.cutoff is not None:
            r, dr = _legendre_nodes(
                self.rule, self.r0, self.r0 + self.cutoff * self.scale
            )
            return r, dr * r * self.intensity(r, 0.0)
        t, w = (jax.lax.stop_gradient(a) for a in self.rule)
        r = self.r0 + self.scale * t
        # int_{r0}^inf e^{-(r-r0)/L} f(r) r dr = L int e^{-t} f r dt.
        return r, self.scale * w * r

    def intensity(self, r, pixel_scale_mas):
        s = np.maximum(r - self.r0, 0.0) / self.scale
        inside = r >= self.r0
        if self.cutoff is not None:
            inside = inside & (s <= self.cutoff)
        return np.where(inside, np.exp(-s), 0.0)

    def extent(self):
        r0, s = (float(onp.max(onp.asarray(a))) for a in (self.r0, self.scale))
        return r0, r0 + (10.0 if self.cutoff is None else self.cutoff) * s

    def _n_needed(self, q_max):
        lo, hi = self.extent()
        if self.cutoff is not None:
            return super()._n_needed(q_max)
        a = 2.0 * onp.pi * float(q_max) * (hi - lo) / 10.0 * mas2rad
        return min(int(onp.ceil(16.0 * a)) + 16, _LAGUERRE_MAX_NODES)

    def is_physical(self):
        return (
            super().is_physical()
            & np.asarray(self.r0 >= 0.0)
            & np.asarray(self.scale > 0.0)
        )


class Gap(_RadialProfile):
    r"""A Gaussian gap cut in another profile.

    The brightness is that of ``base`` times
    $1 - d \exp[-(r - r_0)^2 / 2 w^2]$, so ``depth=1`` empties the gap
    and ``depth=0`` leaves ``base``. The quadrature is the base profile's:
    its ``n_r`` must resolve ``width``. The base profile's own ``weight`` is
    ignored; the gap's ``weight`` is used inside a
    [`RadialSum`][virgil.models.RadialSum].

    Parameters
    ----------
    base : profile
        The profile the gap is cut in. It must have a brightness, so a
        [`ThinRing`][virgil.models.ThinRing] cannot be gapped.
    r0, width : float or array-like
        Radius of the gap's centre and its standard deviation in
        milliarcseconds.
    depth : float or array-like, optional
        Fractional depth between 0 and 1 (default 1).
    weight : float or array-like, optional
        Relative weight inside a [`RadialSum`][virgil.models.RadialSum].

    References
    ----------
    Gaussian gaps in parametric radial profiles follow vis-r
    (Kennedy 2025, RASTI 4, rzaf004; https://github.com/drgmk/vis-r).
    """

    base: _RadialProfile
    r0: jax.Array
    width: jax.Array
    depth: jax.Array

    def __init__(self, base, r0, width, depth=1.0, weight=1.0):
        if not isinstance(base, _RadialProfile) or isinstance(base, ThinRing):
            raise TypeError(
                "Gap needs a base profile with a brightness (not a "
                f"ThinRing): {base!r}"
            )
        self.base = base
        self.r0 = np.asarray(r0, dtype=float)
        self.width = np.asarray(width, dtype=float)
        self.depth = np.asarray(depth, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def with_rule(self, n_r):
        return eqx.tree_at(lambda p: p.base, self, self.base.with_rule(n_r))

    def _factor(self, r):
        return 1.0 - self.depth * np.exp(
            -0.5 * ((r - self.r0) / self.width) ** 2
        )

    def quadrature(self):
        r, w = self.base.quadrature()
        return r, w * self._factor(r)

    def intensity(self, r, pixel_scale_mas):
        return self.base.intensity(r, pixel_scale_mas) * self._factor(r)

    def extent(self):
        return self.base.extent()

    def _n_needed(self, q_max):
        lo, hi = self.extent()
        width = float(onp.min(onp.asarray(self.width)))
        resolve = int(onp.ceil(4.0 * (hi - lo) / max(width, 1e-12)))
        return max(self.base._n_needed(q_max), resolve)

    def is_physical(self):
        return (
            super().is_physical()
            & self.base.is_physical()
            & np.asarray(self.width > 0.0)
            & np.asarray((self.depth >= 0.0) & (self.depth <= 1.0))
        )


class RadialSum(_RadialProfile):
    """Sum of named radial profiles, mixed by their ``weight``.

    Each part is normalized to unit total light and then weighted by its
    ``weight`` (relative to the others), so the weights are the parts' shares
    of the disk's flux, as the ``flux`` of a [`System`][virgil.models.System]'s
    components are. Keep the parts' weights as parameters and fit
    their ratios. Parts are reached by name: ``disk.profile.inner.r0``.
    Each part keeps its own quadrature rule, so the sum has
    ``n_r`` nodes per part.

    Examples
    --------
    >>> profile = RadialSum(
    ...     inner=GaussianRing(r0=3.0, sigma=0.5),
    ...     outer=GaussianRing(r0=8.0, sigma=1.0, weight=0.5),
    ... )
    >>> disk = PolarDisk(profile, inc=40.0, pa=20.0, n_r=32)
    """

    names: tuple = eqx.field(static=True)
    parts: tuple

    def __init__(self, parts=None, /, *, weight=1.0, **named):
        parts = {**(parts or {}), **named}
        if not parts:
            raise ValueError("RadialSum needs at least one profile.")
        for name, part in parts.items():
            check_part_name(
                name, RadialSum, "part", "inner.r0", "a RadialSum attribute"
            )
            if not isinstance(part, _RadialProfile):
                raise TypeError(f"Part '{name}' is not a profile: {part!r}")
        self.names = tuple(parts)
        self.parts = tuple(parts.values())
        self.weight = np.asarray(weight, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def __getattr__(self, name):
        try:
            names = object.__getattribute__(self, "names")
            parts = object.__getattribute__(self, "parts")
        except AttributeError:
            raise AttributeError(name) from None
        if name in names:
            return parts[names.index(name)]
        raise AttributeError(
            f"RadialSum has no part or attribute '{name}'; its parts are "
            f"{list(names)}."
        )

    def with_rule(self, n_r):
        parts = tuple(part.with_rule(n_r) for part in self.parts)
        return eqx.tree_at(lambda p: p.parts, self, parts)

    def _shares(self):
        weights = np.stack([np.asarray(p.weight) for p in self.parts])
        return weights / np.sum(weights)

    def quadrature(self):
        shares = self._shares()
        rs, ws = [], []
        for share, part in zip(shares, self.parts):
            r, w = part._normalized_quadrature()
            rs.append(r)
            ws.append(share * w)
        return np.concatenate(rs), np.concatenate(ws)

    def intensity(self, r, pixel_scale_mas):
        shares = self._shares()
        total = 0.0
        for share, part in zip(shares, self.parts):
            _, w = part.quadrature()
            total = total + share * part.intensity(
                r, pixel_scale_mas
            ) / np.sum(w)
        return total

    def extent(self):
        extents = [part.extent() for part in self.parts]
        return min(e[0] for e in extents), max(e[1] for e in extents)

    def _n_needed(self, q_max):
        return max(part._n_needed(q_max) for part in self.parts)

    def is_physical(self):
        valid = super().is_physical()
        for part in self.parts:
            valid = valid & part.is_physical()
        total = sum(np.asarray(p.weight) for p in self.parts)
        return valid & (total > 0.0)


class _RadialFunction(zdx.Base):
    """A smooth function of the radius ``r`` (milliarcseconds), e.g. for a
    harmonic coefficient of [`Harmonics`][virgil.models.Harmonics]."""

    def __call__(self, r):
        raise NotImplementedError


class RadialLinear(_RadialFunction):
    """``value + slope (r - r0)``: a coefficient that varies linearly.

    Parameters
    ----------
    value : float or array-like
        The function at ``r = r0``.
    slope : float or array-like, optional
        Gradient per milliarcsecond (default 0).
    r0 : float or array-like, optional
        Reference radius in milliarcseconds (default 0).
    """

    value: jax.Array
    slope: jax.Array
    r0: jax.Array

    def __init__(self, value, slope=0.0, r0=0.0):
        self.value = np.asarray(value, dtype=float)
        self.slope = np.asarray(slope, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)

    def __call__(self, r):
        return self.value + self.slope * (r - self.r0)


class RadialPower(_RadialFunction):
    """``value (r / r0) ** index``: a coefficient that follows a power law.

    Parameters
    ----------
    value : float or array-like
        The function at ``r = r0``.
    index : float or array-like
        Power-law index.
    r0 : float or array-like
        Reference radius in milliarcseconds (positive).
    """

    value: jax.Array
    index: jax.Array
    r0: jax.Array

    def __init__(self, value, index, r0):
        self.value = np.asarray(value, dtype=float)
        self.index = np.asarray(index, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)

    def __call__(self, r):
        return self.value * np.maximum(r / self.r0, 1e-12) ** self.index


def _harmonic_entries(values, name):
    """A tuple of coefficients: constants (0-d arrays) or radial functions."""
    if isinstance(values, _RadialFunction):
        return (values,)
    if isinstance(values, (tuple, list)):
        items = values
    else:
        items = np.atleast_1d(np.asarray(values, dtype=float))
    out = []
    for item in items:
        if isinstance(item, _RadialFunction):
            out.append(item)
        else:
            item = np.asarray(item, dtype=float)
            if item.ndim:
                raise ValueError(
                    f"each entry of {name} must be a scalar or a radial "
                    f"function of r, got an array of shape {item.shape}."
                )
            out.append(item)
    return tuple(out)


class Harmonics(zdx.Base):
    r"""Azimuthal harmonics of a [`PolarDisk`][virgil.models.PolarDisk],
    in the linear form.

    The brightness at radius $r$ and azimuth $\theta$ in the plane of the
    disk is the profile $I_0(r)$ times
    $1 + \sum_{m=1}^{M} c_m(r) \cos m\theta + s_m(r) \sin m\theta$.
    The azimuth $\theta$ is measured from the projected major axis (at
    position angle ``pa`` of the disk) towards ``pa + 90`` on the sky, the
    azimuth of [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim].
    The model is exactly linear in the Cartesian coefficients $c_m$, $s_m$,
    which are better behaved than an amplitude and phase (no funnel as the
    amplitude goes to zero and no wrap of the phase), and a Gaussian prior on
    them is a uniform prior on the phase. Each is a constant or a
    function of the radius, such as
    [`RadialLinear`][virgil.models.RadialLinear] or
    [`RadialPower`][virgil.models.RadialPower]. The visibility needs only
    the orders ``0..M``, from one call to ``bessel_jn``.

    The brightness stays non-negative only if
    $1 + \sum_m (\dots) \ge 0$ everywhere; ``sum(c_m**2 + s_m**2) ** 0.5
    <= 1`` guarantees it, and
    ``PolarDisk.min_intensity``
    reports violations.

    Parameters
    ----------
    c, s : float, radial function or sequence of them, optional
        Coefficients of $\cos m\theta$ and $\sin m\theta$ from the first
        order up. A scalar is the first order only. Missing orders are zero.
    degree : int, optional
        The highest order ``M`` (static: it sets shapes, so changing it
        recompiles). By default the length of the longer of ``c`` and ``s``.
    form : str, optional
        Only ``"linear"`` is available.

    Examples
    --------
    A lopsided ring, brighter towards the major axis's far end, with a
    second order that grows outwards:

    >>> harmonics = Harmonics(c=(0.3, RadialLinear(0.0, 0.02, 5.0)), s=(0.1,))
    >>> disk = PolarDisk(GaussianRing(6.0, 1.0), harmonics=harmonics, inc=30.0)
    """

    c: tuple
    s: tuple
    degree: int = eqx.field(static=True)

    def __init__(self, c=(), s=(), degree=None, form="linear"):
        if form != "linear":
            raise ValueError(
                f"Harmonics form must be 'linear' (the only one available), "
                f"not {form!r}."
            )
        c, s = _harmonic_entries(c, "c"), _harmonic_entries(s, "s")
        top = max(len(c), len(s)) if degree is None else degree
        if isinstance(top, bool) or int(top) != top or top < 0:
            raise ValueError(f"degree must be a non-negative integer: {top}.")
        top = int(top)
        if max(len(c), len(s)) > top:
            raise ValueError(
                f"degree is {top} but c has {len(c)} and s has {len(s)} "
                "entries."
            )
        zero = np.asarray(0.0)
        self.c = c + (zero,) * (top - len(c))
        self.s = s + (zero,) * (top - len(s))
        self.degree = top

    @staticmethod
    def _at(entries, r):
        r = np.asarray(r)
        if not entries:
            return np.zeros((0,) + r.shape)
        return np.stack(
            [
                e(r) if isinstance(e, _RadialFunction) else e + 0.0 * r
                for e in entries
            ]
        ).reshape((len(entries),) + r.shape)

    def coefficients(self, r):
        """``(c, s)``, each of shape ``(M, len(r))``, at the radii ``r``."""
        return self._at(self.c, r), self._at(self.s, r)

    def factor(self, r, theta):
        """``1 + sum_m c_m cos m theta + s_m sin m theta`` on ``r, theta``
        (broadcast against each other)."""
        shape = np.broadcast_shapes(np.shape(r), np.shape(theta))
        rr = np.broadcast_to(r, shape).ravel()
        tt = np.broadcast_to(theta, shape).ravel()
        c, s = self.coefficients(rr)
        out = np.ones_like(rr)
        for m in range(1, self.degree + 1):
            out = out + c[m - 1] * np.cos(m * tt) + s[m - 1] * np.sin(m * tt)
        return out.reshape(shape)

    def min_intensity(self, r, n_theta=720):
        """The minimum of the brightness relative to the azimuthal mean
        over the radii ``r`` and ``n_theta`` azimuths (negative if the
        image goes negative)."""
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        return np.min(self.factor(np.ravel(r)[:, None], theta[None, :]))


class _Height(zdx.Base):
    """Height of a ring of radius ``r`` above the disk plane (mas)."""

    def __call__(self, r):
        raise NotImplementedError

    def is_physical(self):
        return np.asarray(True)


class Cone(_Height):
    r"""A conical surface: the ring at radius $r$ lies at the height
    $h = r \cot\alpha - \mathrm{tip}$ along the disk's normal.

    This is the geometry of [`TruncatedCone`][virgil.models.TruncatedCone]:
    the normal is the cone's axis, so the disk's inclination is
    $90^\circ - $ the axis's tilt out of the sky, and ``pa`` is the
    position angle of the *major* axis, 90 degrees from the projection of the
    axis. Positive heights shift towards position angle ``pa + 90``. In
    slant distance $s$ from the apex, $r = s \sin\alpha$.

    Parameters
    ----------
    alpha : float or array-like
        Half-opening angle in degrees (0 < alpha < 90).
    tip : float or array-like, optional
        Distance in milliarcseconds from the disk's centre back to the apex
        along the axis (the ring at $r = 0$ lies at $-\mathrm{tip}$).
    """

    alpha: jax.Array
    tip: jax.Array

    def __init__(self, alpha, tip=0.0):
        self.alpha = np.asarray(alpha, dtype=float)
        self.tip = np.asarray(tip, dtype=float)

    def __call__(self, r):
        return r / np.tan(self.alpha * dtor) - self.tip

    def is_physical(self):
        return np.all((self.alpha > 0.0) & (self.alpha < 90.0))


class PowerLawHeight(_Height):
    r"""A flared surface: $h = h_0 (r / r_0)^\psi$.

    Positive heights shift towards position angle ``pa + 90`` on the sky
    (see [`Cone`][virgil.models.Cone]); a negative ``h0`` puts the surface
    on the other side.

    Parameters
    ----------
    h0 : float or array-like
        Height at ``r0`` in milliarcseconds.
    r0 : float or array-like
        Reference radius in milliarcseconds (positive).
    psi : float or array-like, optional
        Flaring index (default 1, a cone through the centre).
    """

    h0: jax.Array
    r0: jax.Array
    psi: jax.Array

    def __init__(self, h0, r0, psi=1.0):
        self.h0 = np.asarray(h0, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)
        self.psi = np.asarray(psi, dtype=float)

    def __call__(self, r):
        return self.h0 * np.maximum(r / self.r0, 1e-12) ** self.psi

    def is_physical(self):
        return np.all(self.r0 > 0.0)


def polar_disk_cvis(
    u,
    v,
    r,
    w,
    inc=0.0,
    pa=0.0,
    blur_fwhm=None,
    c=None,
    s=None,
    height=None,
    stretch=None,
):
    r"""Unit-flux visibility of a disk from radial quadrature.

    The disk is a sum of thin rings, the nodes ``r`` weighted by ``w``,
    after undoing the inclination ``inc`` and the position angle ``pa`` of
    the projected major axis (North to East, as
    [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim]). In the
    plane of the disk, the brightness of ring ``k`` is
    $1 + \sum_m c_{mk} \cos m\theta + s_{mk} \sin m\theta$ times its
    weight, with the azimuth $\theta$ measured from the projected major
    axis (the North of the deprojected frame) towards East, so that

    $$
    V = 2\pi \sum_{|m| \le M} (-i)^m e^{im\phi} \sum_k w_k I_m(r_k)
        J_m(2\pi q r_k),
    $$

    with $I_{\pm m} = (c_m \mp i s_m)/2$ (and $I_0 = 1$), $q$ and $\phi$
    the length and direction of the spatial frequency in the disk plane.
    The sum is normalized, so $V(0) = 1$, and an optional Gaussian
    ``blur_fwhm`` (in the disk plane) multiplies it as a Fourier envelope.
    If ``height`` is given, ring ``k`` is lifted by ``height[k]`` along the
    disk's normal, a shift on the sky of ``height[k] sin(inc)`` along the
    projected minor axis (towards position angle ``pa + 90``), applied as a
    phase factor after the stretch; the surface is thin and nothing is
    occulted.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units (cycles/rad); any shape.
    r : array-like
        Quadrature radii in milliarcseconds, shape ``(n_r,)``.
    w : array-like
        Quadrature weights ($I_0(r_k) r_k\,dr$ folded in), shape ``(n_r,)``.
    inc, pa : float or array-like, optional
        Inclination and position angle in degrees.
    blur_fwhm : float, array-like or None, optional
        FWHM in milliarcseconds, in the disk plane, of a Gaussian that
        smooths the profile.
    c, s : array-like or None, optional
        Cartesian coefficients of the linear azimuthal harmonics at the
        nodes, shape ``(M, n_r)``: row ``m - 1`` is order ``m``. The
        brightness must stay non-negative (the caller's task).
    height : array-like or None, optional
        Height of each ring above the disk plane in milliarcseconds, shape
        ``(n_r,)`` (``None`` is a flat disk).
    stretch : float or array-like or None, optional
        The factor by which the minor axis is foreshortened on the sky,
        ``cos(inc)`` by default. Pass it to change the axis ratio of the
        rings independently of ``inc`` (which still sets the lift of
        ``height``); a negative value mirrors the azimuth.

    Returns
    -------
    array-like
        Complex visibilities with the shape of ``u``.

    Notes
    -----
    Large problems are evaluated in chunks over the samples. The Bessel
    functions of every order come from one call to ``bessel_jn``.

    References
    ----------
    Jennings et al. 2020, MNRAS 495, 3209 (frank,
    https://github.com/discsim/frank): visibilities of a radial profile as
    a Hankel transform on nodes. Kennedy 2025, RASTI 4, rzaf004 (vis-r,
    https://github.com/drgmk/vis-r): parametric profiles, the uv
    deprojection and a Gaussian envelope. No code was copied.
    """
    if stretch is None:
        stretch = np.maximum(np.cos(inc * dtor), 1e-8)
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa, stretch)
    q2 = ut**2 + vt**2
    positive = q2 > 0.0
    # sqrt has an infinite derivative at 0: keep gradients finite at u = v = 0,
    # where the visibility is exactly 1.
    q_safe = np.sqrt(np.where(positive, q2, 1.0))
    q = np.where(positive, q_safe, 0.0)
    # Direction of the frequency in the disk plane, e^{i phi}, with phi from
    # the deprojected North (v) towards East (u); J_m(0) = 0 for m >= 1, so
    # its value at q = 0 is irrelevant.
    phase = np.where(positive, (vt + 1j * ut) / q_safe, 1.0 + 0j)
    # Distance along the projected minor axis (towards PA pa + 90): the
    # frequency conjugate to the lift of a ring.
    q_minor = u * np.cos(pa * dtor) - v * np.sin(pa * dtor)

    w = w / np.sum(w)
    r_rad = r * mas2rad
    n_orders = 0 if c is None else int(np.shape(c)[0])
    lift = None
    if height is not None:
        lift = height * np.sin(inc * dtor)

    def ring_sum(args):
        q_flat, z_flat, qm_flat = args
        arg = 2.0 * np.pi * q_flat[..., None] * r_rad
        bessel = bessel_jn(n_orders, arg)
        # Even orders feed the real part and odd orders the imaginary part
        # through (-i)^m, so only one complex product is needed at the end.
        re = bessel[0]
        im = np.zeros_like(re)
        zm = np.ones_like(z_flat)
        for m in range(1, n_orders + 1):
            zm = zm * z_flat
            amp = (
                c[m - 1] * np.real(zm)[..., None]
                + s[m - 1] * np.imag(zm)[..., None]
            ) * bessel[m]
            sign = (-1.0) ** (m // 2)
            if m % 2 == 0:
                re = re + sign * amp
            else:
                im = im - sign * amp
        ring = jax.lax.complex(re, im)
        if lift is not None:
            ring = ring * offset_phase(qm_flat[..., None], 0.0, lift, 0.0)
        return np.sum(ring * w, axis=-1)

    n_r = r.shape[-1]
    flat = (q.ravel(), phase.ravel(), q_minor.ravel())
    if q.size * n_r * (1 + n_orders) > _POLAR_CHUNK_ELEMENTS:
        batch = max(1, _POLAR_CHUNK_ELEMENTS // (n_r * (1 + n_orders)))
        cvis = jax.lax.map(ring_sum, flat, batch_size=batch)
    else:
        cvis = ring_sum(flat)
    cvis = cvis.reshape(q.shape)

    if blur_fwhm is not None:
        fwhm_rad = blur_fwhm * mas2rad
        cvis = cvis * np.exp(
            -(np.pi**2) * fwhm_rad**2 * q2 / (4.0 * np.log(2.0))
        )
    return cvis + 0j


class PolarDisk(Component):
    r"""Axisymmetric disk with a radial profile, evaluated by quadrature.

    The brightness is a radial profile $I(r)$ in the plane of the disk,
    inclined by ``inc`` and rotated by ``pa``. Its visibility is the
    Hankel transform $2\pi \int I(r) J_0(2\pi q r) r\,dr$, evaluated by
    summing over ``n_r`` nodes of a rule that the profile owns (a
    Gauss-Legendre rule, or Gauss-Laguerre for an
    [`ExpTail`][virgil.models.ExpTail]); both converge spectrally, so a narrow
    ring needs only tens of nodes. Many profiles are available:
    [`ThinRing`][virgil.models.ThinRing], [`GaussianRing`][virgil.models.GaussianRing],
    [`RadialPowerLaw`][virgil.models.RadialPowerLaw] (with sharp or soft edges),
    [`ExpTail`][virgil.models.ExpTail], [`Gap`][virgil.models.Gap] (a gap in
    another profile) and [`RadialSum`][virgil.models.RadialSum].

    Parameters
    ----------
    profile : profile
        The radial profile (see above); its quadrature rule is set to
        ``n_r`` nodes.
    harmonics : Harmonics or None, optional
        Azimuthal harmonics in the linear form: the brightness is the
        profile times ``1 + sum_m c_m(r) cos(m theta) + s_m(r) sin(m theta)``,
        with the azimuth ``theta`` measured in the disk plane from the
        projected major axis towards ``pa + 90``. They make closure phases
        non-zero for a centred disk (odd orders) and are linear in the
        coefficients. ``min_intensity``
        reports where the image would go negative.
    height : Cone, PowerLawHeight or None, optional
        The height of each ring above the disk plane, as a function of its
        radius: a cone ``r cot(alpha) - tip`` or a flared power law
        ``h0 (r / r0) ** psi``. The ring is shifted on the sky by
        ``h sin(inc)`` towards position angle ``pa + 90`` (a phase factor
        after the stretch). The surface is thin and nothing is occulted.
    inc : float or array-like, optional
        Inclination in degrees (0 is face-on).
    pa : float or array-like, optional
        Position angle of the projected major axis in degrees, North to
        East (see the image coordinate convention).
    blur_fwhm : float, array-like or None, optional
        FWHM of a Gaussian that smooths the profile, in milliarcseconds, in
        the plane of the disk. It is a Fourier envelope: exact and free.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][virgil.models.System],
        or a spectrum from [`virgil.spectra`][virgil.spectra]
        (default 1).
    dra, ddec : float or array-like, optional
        Offset of the disk's centre in milliarcseconds, positive to the East
        and North.
    n_r : int, optional
        Number of quadrature nodes per profile (default 64). It sets the
        shapes of arrays, so it is static: changing it recompiles, and
        changing any parameter does not. [`suggest_n_r`][virgil.models.PolarDisk.suggest_n_r]
        gives a starting value; it is never applied silently.

    Notes
    -----
    Without ``harmonics`` or ``height`` the disk is azimuthally symmetric
    and point-symmetric (whatever ``inc``), so when centred its
    visibilities are real and its closure phases are 0 or 180 degrees (the
    visibility changes sign past its nulls); other values appear with an
    offset ``dra``/``ddec``, an odd harmonic, a height, or in
    a [`System`][virgil.models.System] with other components. A height
    needs ``n_r`` to resolve the extra phase ``2 pi q h sin(inc)`` as well.
    Rendering a disk with a ``height`` splats rings on a fine radial grid and
    needs concrete parameters. The disk has
    unit flux, and in a system its weight is ``flux``. Light on scales much larger than
    the shortest baseline's resolution appears only as a loss of visibility,
    which [`Resolved`][virgil.models.Resolved] also models: the data
    constrain their sum.

    The transform needs ``n_r`` nodes enough to sample the Bessel oscillations
    across the profile's range: about $\pi q_{\max} (r_{\rm out} - r_{\rm in})$,
    plus a dozen.

    Examples
    --------
    A ring with a gap beyond it, seen at 50 degrees:

    >>> profile = Gap(GaussianRing(r0=10.0, sigma=3.0), r0=10.0, width=0.8)
    >>> disk = PolarDisk(profile, inc=50.0, pa=30.0, blur_fwhm=1.0, n_r=48)
    >>> scene = System(star=PointSource(), disk=disk.set("flux", 0.3))
    >>> vis = scene.model(np.array([30.0]), np.array([10.0]), 2.2e-6)

    References
    ----------
    The Hankel-transform representation of a radial profile on a fixed set of
    nodes follows frank (Jennings et al. 2020, MNRAS 495, 3209;
    https://github.com/discsim/frank), and the parametric profiles, uv
    deprojection and Gaussian envelope follow vis-r (Kennedy 2025, RASTI 4,
    rzaf004; https://github.com/drgmk/vis-r). The code is a new
    implementation; none was copied.
    """

    profile: _RadialProfile
    harmonics: Harmonics | None
    height: _Height | None
    inc: jax.Array
    pa: jax.Array
    blur_fwhm: Any
    n_r: int = eqx.field(static=True)

    def __init__(
        self,
        profile,
        harmonics=None,
        height=None,
        inc=0.0,
        pa=0.0,
        blur_fwhm=None,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        n_r=64,
    ):
        if not isinstance(profile, _RadialProfile):
            raise TypeError(f"profile must be a radial profile: {profile!r}")
        if int(n_r) != n_r or n_r < 1:
            raise ValueError(f"n_r must be a positive integer, not {n_r!r}.")
        self.n_r = int(n_r)
        self.profile = profile.with_rule(self.n_r)
        if harmonics is not None and not isinstance(harmonics, Harmonics):
            raise TypeError(f"harmonics must be a Harmonics: {harmonics!r}")
        if height is not None and not isinstance(height, _Height):
            raise TypeError(
                f"height must be a Cone or PowerLawHeight: {height!r}"
            )
        self.harmonics = harmonics
        self.height = height
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.blur_fwhm = (
            None if blur_fwhm is None else np.asarray(blur_fwhm, dtype=float)
        )
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def is_physical(self):
        valid = super().is_physical() & self.profile.is_physical()
        if self.blur_fwhm is not None:
            valid = valid & np.all(self.blur_fwhm >= 0.0)
        if self.height is not None:
            valid = valid & self.height.is_physical()
        if self.harmonics is not None and self.harmonics.degree:
            valid = valid & (self.min_intensity() >= -1e-6)
        return valid

    def min_intensity(self, n_theta=720):
        """The least brightness relative to the azimuthal mean.

        The minimum over the quadrature nodes and ``n_theta`` azimuths of
        ``I(r, theta) / I0(r) = 1 + sum_m c_m cos m theta + s_m sin m
        theta``: 1 without harmonics, and negative if the image goes
        negative somewhere (the harmonics are linear, so nothing prevents
        it). ``is_physical`` is false
        then.
        """
        if self.harmonics is None or not self.harmonics.degree:
            return np.asarray(1.0)
        r, w = self.profile.quadrature()
        return self.harmonics.min_intensity(r, n_theta)

    def suggest_n_r(self, data):
        """A starting number of nodes for ``data``'s longest baselines.

        Enough to sample the Bessel oscillations across the profile and the
        phase of a ``height``'s lift (at the current parameter values, which
        must be concrete), rounded up to a
        multiple of 8 (but at most the 144 nodes buildable for a Gauss-Laguerre
        tail). A starting point only: check convergence by
        doubling ``n_r``.

        Parameters
        ----------
        data : OIData
            Anything with ``u``, ``v`` (metres) and ``wavel`` (metres).
        """
        u, v, wavel = (onp.asarray(a) for a in (data.u, data.v, data.wavel))
        q_max = float(onp.max(onp.hypot(u, v) / wavel))
        n = self.profile._n_needed(q_max)
        if self.height is not None:
            # The lift h(r) sin(inc) adds a phase 2 pi q (lift) across the
            # profile, which dominates when |dh/dr| sin(inc) > 1.
            lo, hi = self.profile.extent()
            h = onp.asarray(self.height(np.linspace(lo, hi, 65)), float)
            lift = float(onp.sum(onp.abs(onp.diff(h))))
            lift *= float(
                onp.max(onp.abs(onp.sin(onp.asarray(self.inc) * dtor)))
            )
            n = max(n, int(onp.ceil(onp.pi * q_max * lift * mas2rad)) + 12)
        n = int(8 * onp.ceil(max(n, 16) / 8))
        # A Gauss-Laguerre tail (ExpTail, alone or inside a RadialSum/Gap)
        # has at most 150 nodes; step down in 8s until the rule builds.
        while n > 16:
            try:
                self.profile.with_rule(n)
                break
            except ValueError:
                n -= 8
        return n

    def _centred_cvis(self, uu, vv):
        r, w = self.profile.quadrature()
        c = s = None
        if self.harmonics is not None and self.harmonics.degree:
            c, s = self.harmonics.coefficients(r)
        height = None if self.height is None else self.height(r)
        return polar_disk_cvis(
            uu,
            vv,
            r,
            w,
            self.inc,
            self.pa,
            self.blur_fwhm,
            c=c,
            s=s,
            height=height,
        )

    def _lifted_image(self, xx, yy, pixel_scale_mas, stretch):
        """The image of a disk with a height, by splatting thin rings.

        The ring at each radius of a fine grid is a set of points round it,
        lifted and projected, each a Gaussian spot a pixel wide, so
        overlapping parts of the surface add (it is optically thin). Needs
        concrete parameters, for the number of points.
        """
        lo, hi = self.profile.extent()
        pad = 3.0 * pixel_scale_mas
        lo, hi = max(lo - pad, 0.0), hi + pad
        n_rad = int(
            min(max(onp.ceil(3.0 * (hi - lo) / pixel_scale_mas), 64), 768)
        )
        n_theta = int(
            min(max(onp.ceil(6.0 * onp.pi * hi / pixel_scale_mas), 128), 1536)
        )
        dr = (hi - lo) / n_rad
        radii = lo + (np.arange(n_rad) + 0.5) * dr
        theta = (np.arange(n_theta) + 0.5) * (2.0 * np.pi / n_theta)
        brightness = (
            self.profile.intensity(radii, pixel_scale_mas) * radii * dr
        )
        lift = self.height(radii) * np.sin(self.inc * dtor)
        if self.harmonics is not None and self.harmonics.degree:
            factor = self.harmonics.factor(radii[:, None], theta[None, :])
        else:
            factor = np.ones((n_rad, n_theta))
        weights = brightness[:, None] * factor / n_theta
        # Unit vectors of the projected minor axis (towards PA pa + 90).
        minor_x, minor_y = np.cos(self.pa * dtor), -np.sin(self.pa * dtor)
        width = pixel_scale_mas
        k = 4.0 * np.log(2.0) / width**2
        grid = _orthogonal_grid(xx, yy)

        def ring_points(radius, height):
            x, y = apply_elliptical_transf_coord(
                radius * np.sin(theta),
                radius * np.cos(theta),
                self.pa,
                stretch,
            )
            return x + height * minor_x, y + height * minor_y

        if grid is not None:
            origin, e_col, e_row = grid
            cols = np.arange(np.shape(xx)[1])[:, None]
            rows = np.arange(np.shape(xx)[0])[:, None]

            def add_ring(image, ring):
                radius, height, wt = ring
                x, y = ring_points(radius, height)
                dx, dy = x - origin[0], y - origin[1]
                s_col = (dx * e_col[0] + dy * e_col[1]) / (e_col @ e_col)
                s_row = (dx * e_row[0] + dy * e_row[1]) / (e_row @ e_row)
                g_col = np.exp(-k * (e_col @ e_col) * (cols - s_col) ** 2)
                g_row = np.exp(-k * (e_row @ e_row) * (rows - s_row) ** 2)
                spots = np.matmul(
                    g_row * wt, g_col.T, precision=jax.lax.Precision.HIGHEST
                )
                return image + spots, None

        else:
            blocks = n_theta // 64 + (n_theta % 64 > 0)
            pad_to = blocks * 64

            def add_ring(image, ring):
                radius, height, wt = ring
                x, y = ring_points(radius, height)
                x, y = (np.pad(a, (0, pad_to - n_theta)) for a in (x, y))
                wt = np.pad(wt, (0, pad_to - n_theta))

                def add_block(image, block):
                    bx, by, bw = block
                    d2 = (xx[..., None] - bx) ** 2 + (yy[..., None] - by) ** 2
                    spots = np.exp(-k * d2) * bw
                    return image + np.sum(spots, -1), None

                image, _ = jax.lax.scan(
                    add_block,
                    image,
                    tuple(a.reshape(blocks, 64) for a in (x, y, wt)),
                )
                return image, None

        image, _ = jax.lax.scan(
            add_ring, np.zeros(np.shape(xx)), (radii, lift, weights)
        )
        return image

    def _centred_image(self, xx, yy, pixel_scale_mas):
        stretch = np.maximum(np.cos(self.inc * dtor), 1e-8)
        if self.height is not None:
            image = self._lifted_image(xx, yy, pixel_scale_mas, stretch)
        else:
            image = self._flat_image(xx, yy, pixel_scale_mas, stretch)
        return self._blurred(image, pixel_scale_mas, stretch)

    def _flat_image(self, xx, yy, pixel_scale_mas, stretch):
        xx_ell, yy_ell = undo_elliptical_transf_coord(xx, yy, self.pa, stretch)
        r_ell = np.hypot(xx_ell, yy_ell)
        image = self.profile.intensity(r_ell, pixel_scale_mas / stretch)
        if self.harmonics is not None and self.harmonics.degree:
            # The azimuth runs from the deprojected North towards East.
            origin = (xx_ell == 0.0) & (yy_ell == 0.0)
            theta = np.arctan2(xx_ell, np.where(origin, 1.0, yy_ell))
            image = image * self.harmonics.factor(r_ell, theta)
        return image

    def _blurred(self, image, pixel_scale_mas, stretch):
        if self.blur_fwhm is None:
            return image

        # As for ModulatedGaussianRim: an odd kernel centred on a pixel,
        # isotropic in the disk plane.
        npix = image.shape[0]
        nker = npix + 1 - npix % 2
        kx, ky = image_coordinates(nker, nker * pixel_scale_mas)
        sigma_mas = np.maximum(self.blur_fwhm / FWHM_PER_SIGMA, 1e-9)
        psf_stretch = np.minimum(
            np.maximum(stretch, 0.5 * pixel_scale_mas / sigma_mas), 1.0
        )
        kx_ell, ky_ell = undo_elliptical_transf_coord(
            kx, ky, self.pa, psf_stretch
        )
        psf = np.exp(-0.5 * (kx_ell**2 + ky_ell**2) / sigma_mas**2)
        return fftconvolve(image, psf, mode="same")


class Resolved(SourceModel):
    """Fully resolved (over-resolved) flux, e.g. a large, diffuse envelope.

    Its visibility is 0 on every non-zero baseline, so inside a
    [`System`][virgil.models.System] it only adds to the normalization,
    lowering every other component's visibility by the same factor.

    Parameters
    ----------
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System] (default 1), or a spectrum from
        [`virgil.spectra`][virgil.spectra].

    Notes
    -----
    A resolved component spreads its light far beyond any image, so it
    cannot be rendered on its own, and a rendered
    [`System`][virgil.models.System] shows only its other components,
    renormalized to unit sum. The Fourier transform of such an image is
    therefore the visibility of the unresolved components alone, i.e.
    ``model()`` divided by the unresolved fraction of the flux, not
    ``model()`` itself.

    Examples
    --------
    >>> scene = System(star=PointSource(), background=Resolved(flux=0.1))
    """

    flux: jax.Array

    def __init__(self, flux=1.0):
        self.flux = _as_flux(flux)

    def model(self, u, v, wavel):
        at_origin = (np.asarray(u) == 0) & (np.asarray(v) == 0)
        return np.where(at_origin, 1.0, 0.0) + 0j

    def render(self, npix=256, fov_mas=200.0):
        raise ValueError(
            "A Resolved component has no image: its light is spread far "
            "beyond any field of view. Render the System it belongs to, which "
            "shows the other components."
        )

    def _image(self, xx, yy, pixel_scale_mas):
        # Contributes nothing inside a rendered System (see Notes).
        return np.zeros_like(xx)

    def _weight(self, wavel=None):
        return flux_at(self.flux, wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def __check_init__(self):
        _check_non_negative_flux(self.flux, "Resolved")


class System(SourceModel):
    r"""Flux-weighted mixture of named source models.

    A [`System`][virgil.models.System] is how you describe a scene with more than one part:
    a star with a disk, a binary inside a ring, a companion with its own
    circumstellar material. Each component is given a name, and the
    visibility is the flux-weighted mean
    $V = \sum_i f_i V_i \,/\, \sum_i f_i$, where ``f_i`` is each component's ``flux``. Because interferometric
    visibilities are normalized to 1 at zero baseline, only the *ratios* of
    the fluxes can be measured: keep one reference component (usually the
    star) at ``flux=1`` and fit the others relative to it.

    Components are reached by name, both as attributes (``system.comp.flux``)
    and as zodiax paths (``system.get("comp.flux")``,
    ``system.set("comp.flux", 0.02)``). These paths are how the fitting tools
    in [`virgil.grid_fit`][virgil.grid_fit] and [`numpyro_model`][virgil.likelihood.numpyro_model] address
    parameters. Components keep the order in which they were given.

    A [`System`][virgil.models.System] can itself be a component. Its own ``flux`` is then the
    total flux of the group relative to its siblings, and ``dra``/``ddec``
    move the whole group together.

    Parameters
    ----------
    components : dict[str, SourceModel], optional
        Components as a mapping from name to model. Usually it is clearer to
        pass them as keyword arguments instead.
    flux : float or array-like, optional
        Weight of the whole system when nested inside another
        [`System`][virgil.models.System] (default 1). It has no effect at the top level.
    dra, ddec : float or array-like, optional
        Offset of the whole system in milliarcseconds (positive to the East
        and North).
    **named : SourceModel
        Components as keyword arguments, e.g. ``star=PointSource()``. Names
        must be valid Python identifiers that do not start with ``_`` and do
        not clash with a [`System`][virgil.models.System] attribute (``model``, ``render``,
        ``set``, ``flux``, ...).

    Notes
    -----
    Fluxes are physical brightnesses, so they must be non-negative, and they
    must not all be zero. Both are checked when a model is built from
    concrete values; changing values afterwards with ``set`` is not
    checked. Inside a traced computation (a fit or grid search) the
    values cannot be checked, so positivity is the job of the priors and grid
    axes: [`numpyro_model`][virgil.likelihood.numpyro_model] rejects flux priors that allow negative
    values, and the grid tools reject negative flux axes.

    Examples
    --------
    A star with a faint companion:

    >>> binary = System(
    ...     star=PointSource(),
    ...     comp=PointSource(dra=45.0, ddec=30.0, flux=0.01),
    ... )
    >>> binary.comp
    PointSource(flux=0.01, dra=45, ddec=30)

    A star with a rim, and a companion that has its own disk:

    >>> scene = System(
    ...     star=PointSource(),
    ...     rim=ModulatedGaussianRim(diam=40.0, fwhm=4.0, inc=50.0, pa=30.0, flux=0.5),
    ...     comp=System(
    ...         core=PointSource(),
    ...         disk=GaussianDisk(sigma=4.0, flux=0.5),
    ...         dra=-30.0,
    ...         ddec=25.0,
    ...         flux=0.3,
    ...     ),
    ... )
    >>> list(scene.components)
    ['star', 'rim', 'comp']
    >>> moved = scene.set(["comp.dra", "comp.ddec"], [30.0, -25.0])
    """

    names: tuple = eqx.field(static=True)
    parts: tuple
    flux: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def __init__(
        self, components=None, /, *, flux=1.0, dra=0.0, ddec=0.0, **named
    ):
        components = {**(components or {}), **named}
        if not components:
            raise ValueError("System needs at least one component.")
        for name, component in components.items():
            check_part_name(
                name,
                System,
                "Component",
                "comp.flux",
                "a System attribute or method",
            )
            if not isinstance(component, SourceModel):
                raise TypeError(
                    f"Component '{name}' is not a SourceModel: {component!r}"
                )
        total = _concrete_sum(c._weight() for c in components.values())
        if total == 0.0:
            raise ValueError(
                "The component fluxes sum to zero, so the system has no "
                "light. Keep a reference component (usually the star) at "
                "flux=1."
            )
        self.names = tuple(components)
        self.parts = tuple(components.values())
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    @property
    def components(self):
        """The components as a ``{name: model}`` dictionary, in order."""
        return dict(zip(self.names, self.parts))

    def __getattr__(self, name):
        try:
            names = object.__getattribute__(self, "names")
            parts = object.__getattribute__(self, "parts")
        except AttributeError:
            raise AttributeError(name) from None
        if name in names:
            return parts[names.index(name)]
        raise AttributeError(
            f"System has no component or attribute '{name}'; its components "
            f"are {list(names)}."
        )

    def __repr__(self):
        lines = [
            f"{name}={component!r},"
            for name, component in self.components.items()
        ]
        lines += [
            f"{field}={_format_leaf(getattr(self, field))},"
            for field in ("flux", "dra", "ddec")
        ]
        body = textwrap.indent("\n".join(lines), "    ")
        return f"System(\n{body}\n)"

    def model(self, u, v, wavel):
        return self._mix(u, v, wavel, lambda c: c.model(u, v, wavel))

    @property
    def time_dependent(self):
        return any(part.time_dependent for part in self.parts)

    @property
    def broadcast_in_time(self):
        return all(part.broadcast_in_time for part in self.parts)

    def at(self, mjd, t_ref=0.0):
        if not self.time_dependent:
            return self
        parts = tuple(part.at(mjd, t_ref) for part in self.parts)
        return eqx.tree_at(lambda s: s.parts, self, parts)

    def model_on_grid(self, u, v, wavel, grid):
        return self._mix(
            u, v, wavel, lambda c: c.model_on_grid(u, v, wavel, grid)
        )

    def _mix(self, u, v, wavel, part_model):
        """Flux-weighted mean of ``part_model(part)``, then the offset."""
        weights = [c._weight(wavel) for c in self.parts]
        total = sum(
            w * part_model(c) for w, c in zip(weights, self.parts)
        ) / sum(weights)
        uu, vv = u / wavel, v / wavel
        return total * offset_phase(uu, vv, self.dra, self.ddec)

    def _image(self, xx, yy, pixel_scale_mas):
        xx, yy = xx - self.dra, yy - self.ddec
        weights = [c._weight() for c in self.parts]
        return sum(
            w * _unit_flux(c._image(xx, yy, pixel_scale_mas))
            for w, c in zip(weights, self.parts)
        ) / sum(weights)

    def _weight(self, wavel=None):
        return flux_at(self.flux, wavel)

    def total_spectrum(self, wavel):
        wavel = np.asarray(wavel)
        return sum(
            np.broadcast_to(c._weight(wavel), wavel.shape) for c in self.parts
        )

    def is_physical(self):
        valid = _flux_is_non_negative(self.flux)
        for part in self.parts:
            valid = valid & part.is_physical()
        # All parts are non-negative when valid, so a zero sum means no light.
        total = sum(np.sum(c._weight()) for c in self.parts)
        return valid & (total > 0.0)

    def __check_init__(self):
        _check_non_negative_flux(self.flux, "System")


class Rotated(SourceModel):
    """A model rotated on the sky about the phase centre.

    The rotation is by ``rotation_deg`` from North towards East, and the
    angle is an ordinary (fittable) parameter, unlike
    [`Image`][virgil.models.Image]'s ``rotation_deg``, which only orients
    its pixel grid. Use it for a scene seen at several epochs, e.g. a
    spiral rotating between them (see [`fit`][virgil.fitting.fit] with
    a model per dataset).

    Parameters
    ----------
    source : SourceModel
        The model to rotate.
    rotation_deg : float
        Position angle of the rotation, North towards East, in degrees.

    Examples
    --------
    A companion to the North, rotated by 90°, lands to the East:

    >>> import jax.numpy as np
    >>> from virgil.models import GaussianDisk, Rotated, System
    >>> north = System(a=GaussianDisk(1.0), b=GaussianDisk(1.0, ddec=10.0))
    >>> east = System(a=GaussianDisk(1.0), b=GaussianDisk(1.0, dra=10.0))
    >>> u, v = np.array([3.0, 5.0]), np.array([1.0, -2.0])
    >>> rotated = Rotated(north, 90.0).model(u, v, 1e-6)
    >>> bool(np.allclose(rotated, east.model(u, v, 1e-6), atol=1e-6))
    True
    """

    source: SourceModel
    rotation_deg: jax.Array

    def __init__(self, source, rotation_deg):
        self.source = source
        self.rotation_deg = np.asarray(rotation_deg, dtype=float)

    def model(self, u, v, wavel):
        u, v = rotate(u, v, -self.rotation_deg)
        return self.source.model(u, v, wavel)

    def _image(self, xx, yy, pixel_scale_mas):
        xx, yy = rotate(xx, yy, -self.rotation_deg)
        return self.source._image(xx, yy, pixel_scale_mas)

    def _weight(self, wavel=None):
        return self.source._weight(wavel)

    def total_spectrum(self, wavel):
        return self.source.total_spectrum(wavel)

    def is_physical(self):
        return self.source.is_physical()

    @property
    def time_dependent(self):
        return self.source.time_dependent

    @property
    def broadcast_in_time(self):
        return self.source.broadcast_in_time

    def at(self, mjd, t_ref=0.0):
        if not self.time_dependent:
            return self
        return eqx.tree_at(
            lambda r: r.source, self, self.source.at(mjd, t_ref)
        )


_FRAME_ANGLES = (
    "line_pa",
    "towards_primary",
    "line_tilt",
    "node_pa",
    "inc",
    "apparent_inc",
)


class Attached(SourceModel):
    """A component placed and oriented in the frame of a binary's orbit.

    At each time the component is moved to its anchor on the orbit, and
    its angle attributes are set from angles of the binary frame (see
    [`KeplerOrbit.frame`][virgil.orbits.KeplerOrbit.frame]). A companion
    on its orbit is ``Attached(PointSource(flux), orbit)``; a disc around
    it in the orbital plane, brighter on the side facing the primary, is

    ``Attached(ModulatedGaussianRim(...), orbit, bind={"pa": "node_pa",
    "inc": "apparent_inc", "az_pas": "towards_primary"})``.

    The model changes with time, so it is evaluated through
    [`at`][virgil.models.SourceModel.at]; inside a
    [`System`][virgil.models.System],
    [`OIData.model`][virgil.oidata.OIData.model] evaluates every sample at
    its own time. Components that should share one orbit (a companion and
    its disc) are best built from shared parameters in a model function
    (see [`fit`][virgil.fitting.fit]).

    Parameters
    ----------
    component : Component
        The component (its ``dra``, ``ddec`` and bound angles are set).
    orbit : KeplerOrbit
        The orbit of the secondary about the primary (the scene's origin).
    anchor : {"secondary", "primary"} or float, optional
        Where the component sits: on the secondary (default), on the
        primary, or at this fraction of the way from the primary to the
        secondary (e.g. ``q / (1 + q)`` for the barycentre).
    bind : dict, optional
        ``{attribute: frame angle}``, e.g. ``{"pa": "line_pa"}``. Frame
        angles: ``line_pa``, ``towards_primary``, ``line_tilt``,
        ``node_pa``, ``inc`` and ``apparent_inc``. An ``az_pas`` binding is a
        sky position angle, converted to the component's deprojected rim
        angle after its ``pa`` and ``inc`` are set (as
        [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim]
        measures it).
    offsets : dict, optional
        ``{attribute: degrees}`` added to the bound angles (fittable, e.g. a
        skew); zero by default.
    """

    component: SourceModel
    orbit: Any
    offsets: dict
    bind: tuple = eqx.field(static=True)
    anchor: Any = eqx.field(static=True)

    def __init__(
        self, component, orbit, anchor="secondary", bind=None, offsets=None
    ):
        bind = dict(bind or {})
        for attr, angle in bind.items():
            if not hasattr(component, attr):
                raise ValueError(
                    f"{type(component).__name__} has no attribute {attr!r} "
                    "to bind."
                )
            if angle not in _FRAME_ANGLES:
                raise ValueError(
                    f"Unknown frame angle {angle!r}; use one of "
                    f"{', '.join(_FRAME_ANGLES)}."
                )
        if "az_pas" in bind and (
            not hasattr(component, "_ring_angle")
            or component.az_pas is None
            or onp.size(component.az_pas) == 0
        ):
            raise ValueError(
                f"{type(component).__name__} has no azimuthal modulation to "
                "bind az_pas to; give it az_amps and az_pas when building it."
            )
        offsets = dict(offsets or {})
        unknown = set(offsets) - set(bind)
        if unknown:
            raise ValueError(
                f"Offsets for unbound attributes: {sorted(unknown)}."
            )
        if anchor not in ("primary", "secondary"):
            anchor = float(anchor)
        self.component = component
        self.orbit = orbit
        self.anchor = anchor
        # az_pas last: it is deprojected with the bound orientation.
        self.bind = tuple(sorted(bind.items(), key=lambda b: b[0] == "az_pas"))
        self.offsets = {
            attr: np.asarray(offsets.get(attr, 0.0), dtype=float)
            for attr in bind
        }

    @property
    def time_dependent(self):
        return True

    @property
    def broadcast_in_time(self):
        return not self.bind

    def at(self, mjd, t_ref=0.0):
        dt = _days_since(mjd, self.orbit.t_ref - t_ref)
        _warn_if_mjd_without_t_ref(self.orbit.t_ref, dt)
        dra, ddec, _ = self.orbit._relative(dt)
        fraction = {"primary": 0.0, "secondary": 1.0}.get(
            self.anchor, self.anchor
        )
        out = eqx.tree_at(
            lambda c: (c.dra, c.ddec),
            self.component,
            (fraction * dra, fraction * ddec),
        )
        frame = self.orbit._frame(dt)
        for attr, angle in self.bind:
            value = frame[angle] + self.offsets[attr]
            if attr == "az_pas":
                value = out._ring_angle(value)
            old = getattr(out, attr)
            out = eqx.tree_at(
                lambda c: getattr(c, attr),
                out,
                np.broadcast_to(value, np.shape(old)).astype(
                    np.result_type(old)
                ),
            )
        return out

    def model(self, u, v, wavel):
        raise ValueError(
            "Attached changes with time: evaluate model.at(mjd), or let "
            "OIData.model evaluate each sample at its own time."
        )

    def _image(self, xx, yy, pixel_scale_mas):
        raise ValueError(
            "Attached changes with time: render model.at(mjd) instead."
        )

    def _weight(self, wavel=None):
        return self.component._weight(wavel)

    def total_spectrum(self, wavel):
        return self.component.total_spectrum(wavel)

    def is_physical(self):
        return self.component.is_physical()


class OrbitalBinary(SourceModel):
    """A primary and a point-source companion moving on a Keplerian orbit.

    The time-dependent form of
    [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]: at each
    time, [`at`][virgil.models.SourceModel.at] returns the
    ``BinaryModelCartesian`` with the companion at the orbit's sky
    position, so each snapshot keeps the fast binary path. It is the same
    scene as ``System(primary=PointSource(), companion=Attached(
    PointSource(flux), orbit))``, for the common case of two unresolved
    stars. The primary is the scene's origin and reference (``flux`` is
    the companion's flux relative to it).

    Evaluate it with one snapshot per dataset or epoch through
    [`Epochs`][virgil.epochs.Epochs], or per sample by
    [`OIData.model`][virgil.oidata.OIData.model].

    Parameters
    ----------
    orbit : KeplerOrbit
        The companion's orbit about the primary (any orbit class with
        ``t_ref`` and ``_relative``).
    flux : float
        Companion/primary flux ratio.

    Examples
    --------
    >>> from virgil.models import OrbitalBinary
    >>> from virgil.orbits import KeplerOrbit
    >>> orbit = KeplerOrbit(700.0, 0.0, 0.3, 50.0, 60.0, 120.0, 20.0,
    ...                     t_ref=60500.0)
    >>> snap = OrbitalBinary(orbit, 0.1).at(60500.0)
    >>> type(snap).__name__, round(float(snap.flux), 2)
    ('BinaryModelCartesian', 0.1)
    """

    orbit: Any
    flux: jax.Array

    def __init__(self, orbit, flux):
        self.orbit = orbit
        self.flux = np.asarray(flux, dtype=float)

    @property
    def time_dependent(self):
        return True

    def at(self, mjd, t_ref=0.0):
        dt = _days_since(mjd, self.orbit.t_ref - t_ref)
        _warn_if_mjd_without_t_ref(self.orbit.t_ref, dt)
        dra, ddec, _ = self.orbit._relative(dt)
        return BinaryModelCartesian(dra, ddec, self.flux)

    def model(self, u, v, wavel):
        raise ValueError(
            "OrbitalBinary changes with time: evaluate model.at(mjd), or "
            "let OIData.model evaluate each sample at its own time."
        )

    def _image(self, xx, yy, pixel_scale_mas):
        raise ValueError(
            "OrbitalBinary changes with time: render model.at(mjd) instead."
        )

    def total_spectrum(self, wavel):
        return BinaryModelCartesian(0.0, 0.0, self.flux).total_spectrum(wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)


def _rim_angle(sky_pa, pa, inc):
    """The in-plane rim angle whose projection points at ``sky_pa``.

    [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim] measures
    ``az_pas - pa`` in the deprojected disc and then compresses its minor
    axis by ``cos(inc)``, so a sky angle is deprojected first.
    """
    d = np.deg2rad(sky_pa - pa)
    return pa + np.rad2deg(
        np.arctan2(np.sin(d) / np.cos(np.deg2rad(inc)), np.cos(d))
    )


class BinaryModelAngular(SourceModel):
    """
    A primary star and a point-source companion, in polar coordinates.

    Parameters
    ----------
    sep : float or array-like
        On-sky separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees, measured East of North.
    flux : float or array-like
        Companion/primary flux ratio (e.g. 0.01 for a companion 100 times
        fainter, i.e. contrast 100 or 5 mag; see
        [`flux_to_contrast`][virgil.limits.flux_to_contrast]).

    Notes
    -----
    Equivalent to [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]
    at ``dra = sep sin(pa)``, ``ddec = sep cos(pa)``; convenient for
    reporting astrometry directly in separation and position angle.
    """

    sep: jax.Array
    pa: jax.Array
    flux: jax.Array

    def __init__(self, sep, pa, flux):
        self.sep = np.asarray(sep, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

    def to_cartesian(self):
        """Return the equivalent [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]."""
        th = self.pa * dtor
        return BinaryModelCartesian(
            self.sep * np.sin(th), self.sep * np.cos(th), self.flux
        )

    def total_spectrum(self, wavel):
        return self.to_cartesian().to_system().total_spectrum(wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def model(self, u, v, wavel):
        """Complex visibilities on baselines ``u``, ``v`` (m) at ``wavel`` (m)."""
        uu, vv = u / wavel, v / wavel
        return _cvis_binary_angular(uu, vv, self.sep, self.pa, self.flux)

    def _image(self, xx, yy, pixel_scale_mas):
        return self.to_cartesian()._image(xx, yy, pixel_scale_mas)


class BinaryModelCartesian(SourceModel):
    """
    A primary star and a point-source companion, in Cartesian sky offsets.

    Parameters
    ----------
    dra : float or array-like
        Right-ascension offset of the companion in milliarcseconds, positive
        to the East.
    ddec : float or array-like
        Declination offset of the companion in milliarcseconds, positive to
        the North.
    flux : float or array-like
        Companion/primary flux ratio (e.g. 0.01 for a companion 100 times
        fainter, i.e. contrast 100 or 5 mag).

    Notes
    -----
    This is the fast form of ``System(primary=PointSource(),
    companion=PointSource(flux, dra, ddec))`` (see :meth:`to_system`), and
    the most convenient parameterization for grid searches and fits.
    """

    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array

    def __init__(self, dra, ddec, flux):
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

    def to_angular(self):
        """Return the equivalent [`BinaryModelAngular`][virgil.models.BinaryModelAngular]."""
        sep, pa = separation_pa(self.dra, self.ddec)
        return BinaryModelAngular(sep, pa, self.flux)

    def total_spectrum(self, wavel):
        return self.to_system().total_spectrum(wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def model(self, u, v, wavel):
        """Complex visibilities on baselines ``u``, ``v`` (m) at ``wavel`` (m)."""
        uu, vv = u / wavel, v / wavel
        return cvis_binary(uu, vv, self.dra, self.ddec, self.flux)

    def to_system(self):
        """Return the equivalent ``System(primary=..., companion=...)``.

        The [`System`][virgil.models.System] form is slower to evaluate
        but can be extended, e.g. by adding a disk around the primary.
        """
        return System(
            primary=PointSource(),
            companion=PointSource(
                flux=self.flux, dra=self.dra, ddec=self.ddec
            ),
        )

    def _image(self, xx, yy, pixel_scale_mas):
        return self.to_system()._image(xx, yy, pixel_scale_mas)


class HarmonixModel(SourceModel):
    """
    Wrapper for external source models with harmonix-like visibility methods.

    Parameters
    ----------
    source : object
        External model. ``getattr(source, visibility_method)`` is called with
        the baselines (and ``observation_time`` if given) and must return
        complex visibilities normalized to 1 at zero baseline.
    visibility_method : str, optional
        Name of the visibility method (default ``"model"``).
    render_method : str, optional
        Name of a ``render(npix, fov_mas)`` method used by [`render`][virgil.models.HarmonixModel.render]
        (default ``"render"``). Sources without one but with a ``surface``
        and a ``radius`` in milliarcseconds (harmonix stars) are drawn on the
        sky from the surface's intensity, East left and North up, like every
        other model.
    expects_wavelength_units : bool, optional
        If True (default), pass spatial frequencies ``u / wavel``,
        ``v / wavel``; otherwise pass ``u``, ``v`` in metres.
    observation_time : optional
        Extra argument passed after the baselines, e.g. a time for rotating
        stars.

    Notes
    -----
    ``source`` is an ordinary field, so a source that is a pytree (such as a
    harmonix ``Harmonix``, an equinox module) has parameters that zodiax
    paths reach and fits can vary, e.g. ``"source.radius"`` (mas) or
    ``"source.data"`` (harmonix's map coefficients for l >= 1, with
    Y00 = 1). Sources that are not pytrees need ``eqx.filter_jit`` rather
    than ``jax.jit``. ``observation_time`` is static (hashable):
    array-valued times are not supported under ``jax.jit``.

    A wrapped source has weight 1 inside a
    [System][virgil.models.System]; only harmonix stars can be drawn
    there.
    """

    source: Any
    visibility_method: str = eqx.field(static=True)
    render_method: str = eqx.field(static=True)
    expects_wavelength_units: bool = eqx.field(static=True)
    observation_time: Any = eqx.field(static=True)

    def __init__(
        self,
        source,
        visibility_method="model",
        render_method="render",
        expects_wavelength_units=True,
        observation_time=None,
    ):
        self.source = source
        self.visibility_method = str(visibility_method)
        self.render_method = str(render_method)
        self.expects_wavelength_units = bool(expects_wavelength_units)
        self.observation_time = observation_time

    def model(self, u, v, wavel):
        method = getattr(self.source, self.visibility_method)
        args = (
            [u / wavel, v / wavel] if self.expects_wavelength_units else [u, v]
        )
        if self.observation_time is not None:
            args.append(self.observation_time)
        return np.asarray(method(*args))

    def render(self, npix=256, fov_mas=200.0):
        if hasattr(self.source, self.render_method):
            return _normalize_image(
                getattr(self.source, self.render_method)(npix, fov_mas)
            )
        return super().render(npix, fov_mas)

    def _image(self, xx, yy, pixel_scale_mas):
        source = self.source
        if not (hasattr(source, "surface") and hasattr(source, "radius")):
            raise NotImplementedError(
                "Wrapped source does not expose a compatible render method."
            )
        theta = 0.0
        if hasattr(source, "rotational_phase"):
            theta = source.rotational_phase(
                0.0 if self.observation_time is None else self.observation_time
            )
        # harmonix's visibilities use the sign convention of offset_phase
        # with the surface's x (East) and y (North) in stellar radii, so the
        # intensity at sky offset (dra, ddec) is the surface's at
        # (dra, ddec) / radius. (jaxoplanet's Surface.render puts East on
        # the right instead.)
        surface = source.surface
        if hasattr(source, "data"):
            # harmonix reads the map from ``data`` (the l >= 1 coefficients,
            # with Y00 = 1) rather than ``surface.y``, so draw that map.
            y00 = np.ones((1,), dtype=source.data.dtype)
            ylm = type(surface.y).from_dense(
                np.concatenate([y00, source.data]), normalize=False
            )
            surface = eqx.tree_at(lambda s: s.y, surface, ylm)
        x, y = xx / source.radius, yy / source.radius
        on_disk = x**2 + y**2 < 1.0
        x, y = np.where(on_disk, x, 0.0), np.where(on_disk, y, 0.0)
        z = np.sqrt(1.0 - x**2 - y**2)
        intensity = surface._intensity(
            x.ravel(), y.ravel(), z.ravel(), theta=theta
        )
        return np.where(on_disk, np.reshape(intensity, xx.shape), 0.0)


def _cvis_binary_angular(u, v, sep, pa, flux):
    """Complex visibilities of a binary in polar coordinates.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units.
    sep : float or array-like
        Separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees, measured East of North.
    flux : float or array-like
        Companion/primary flux ratio.

    Returns
    -------
    array-like
        Complex visibility samples, normalized to 1 at zero baseline.
    """
    th = pa * dtor
    return cvis_binary(u, v, sep * np.sin(th), sep * np.cos(th), flux)


def cvis_binary(u, v, dra, ddec, flux):
    """Compute complex visibilities for a Cartesian-parameterized binary model.

    Adapted from Frantz Martinache's ``xara`` (formerly ``pysco``,
    https://github.com/fmartinache/xara; Martinache 2010,
    https://arxiv.org/abs/1009.3933). For closure-phase contrast limits in
    sparse aperture masking see Ireland (2013, MNRAS 433, 1718,
    https://arxiv.org/abs/1301.6205).

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    dra : float or array-like
        Right-ascension offset of the companion in milliarcseconds, positive
        to the East.
    ddec : float or array-like
        Declination offset of the companion in milliarcseconds, positive to
        the North.
    flux : float or array-like
        Companion/primary flux ratio.

    Returns
    -------
    array-like
        Complex visibility samples, normalized to 1 at zero baseline.
    """

    # normalize visibilities so total power is 1
    primary = 1.0 / (1.0 + flux)
    companion = flux / (1.0 + flux)

    return primary + companion * offset_phase(u, v, dra, ddec)


@renamed()
def cvis_uniform_disk(u, v, diam, dra=0.0, ddec=0.0):
    """Compute complex visibilities for a uniform (tophat) disk.

    The visibility amplitude follows the classic uniform-disk form
    ``2 * J1(x) / x``, with ``x = pi * diam_rad * base_norm`` the product of
    the disk diameter (in radians) and the baseline length in wavelength
    units (``base_norm = hypot(u, v)``, i.e. baseline length divided by
    wavelength).

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    diam : float or array-like
        Diameter of the uniform disk in milliarcseconds.
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    ddec : float or array-like
        Declination offset in milliarcseconds.

    Returns
    -------
    array-like
        Complex visibility samples.
    """
    diam_rad = mas2rad * diam
    base_norm = np.hypot(u, v)
    kernel = np.pi * base_norm * diam_rad

    # Keep 0 out of the division so the unused branch has finite gradients.
    at_zero = kernel == 0
    safe_kernel = np.where(at_zero, 1.0, kernel)
    envelope = np.where(
        at_zero,
        1.0 + 0j,
        (2.0 * bessel_jn(1, safe_kernel)[1]) / safe_kernel + 0j,
    )

    return envelope * offset_phase(u, v, dra, ddec)


def cvis_limb_darkened_disk(u, v, diam, coeffs, powers, dra=0.0, ddec=0.0):
    r"""Complex visibilities of a disk whose brightness is a sum of powers of $\mu$.

    For $I(\mu) = \sum_\nu a_\nu \mu^\nu$, with $\mu = \sqrt{1 - (r/R)^2}$,
    Quirrenbach et al. (1996, A&A 312, 160,
    https://ui.adsabs.harvard.edu/abs/1996A&A...312..160Q, eqs. 1-4) show that

    $$
    V(x) = \frac{1}{C} \sum_\nu a_\nu\, 2^{\nu/2}\,
    \Gamma\!\left(\frac{\nu}{2} + 1\right)
    \frac{J_{\nu/2+1}(x)}{x^{\nu/2+1}}, \qquad
    C = \sum_\nu \frac{a_\nu}{\nu + 2},
    $$

    with $x = \pi\,\theta\,|b| / \lambda$ for diameter $\theta$, normalized to
    1 at zero baseline. A uniform disk ($a_0 = 1$) gives $2 J_1(x)/x$. The
    powers need not be integers: the square-root law has $\nu = 1/2$, which
    needs order $5/4$, from jaxbessel's ``bessel_jv_over_xv``, which takes
    orders up to 12, so $-2 < \nu \le 22$ (the lower limit keeps the flux
    finite). harmonix
    ([Dholakia & Pope 2025](https://arxiv.org/abs/2509.25433)) generalizes
    the result to polynomial limb darkening of spherical-harmonic maps.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    diam : float or array-like
        Limb-darkened diameter in milliarcseconds.
    coeffs : array-like
        Coefficients $a_\nu$, one per power (traceable).
    powers : sequence of float
        Powers $\nu$ of $\mu$, each with $-2 < \nu \le 22$ (static).
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    ddec : float or array-like
        Declination offset in milliarcseconds.

    Returns
    -------
    array-like
        Complex visibility samples.
    """
    bad = [nu for nu in powers if not -2.0 < float(nu) <= 22.0]
    if bad:
        raise ValueError(f"Powers of mu must be in (-2, 22], got {bad}.")
    x = np.pi * np.hypot(u, v) * mas2rad * diam
    total = 0.0
    norm = 0.0
    for i, nu in enumerate(powers):
        nu = float(nu)
        scale = 2.0 ** (nu / 2) * math.gamma(nu / 2 + 1)
        total = total + coeffs[i] * scale * bessel_jv_over_xv(nu / 2 + 1, x)
        norm = norm + coeffs[i] / (nu + 2)
    return (total / norm + 0j) * offset_phase(u, v, dra, ddec)


def cvis_radial_dirac_delta_modulated(u, v, r0, az_amps, az_phis):
    r"""Compute the complex visibility for an azimuthally modulated radial dirac
    delta ring. The image intensity can be described in polar image coordinates as
    $I(r, \theta) \propto \delta(r-r_0) \left( 1 + \sum_{m=1}^{n} A_m
    \cos{(m(\theta - \phi_m))} \right)$, where $r_0$ is the ring's radial position,
    $A_m$ the amplitude and $\phi_m$ the position phase angle (defined counter-clockwise
    , North to East) for the m-th order modulation.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    r0 : float or array-like
        Scalar with the radial position of the ring in milliarcseconds.
    az_amps : array-like
        1D array containing amplitude coefficients for cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_phis : array-like
        1D array containing offset angles of the cosine azimuthal modulations,
        relative to the position angle of the rim's projected major axis, in
        degrees. The first element is seen as the offset for the first-order
        modulation, the second for the second-order modulation, etc.

    Returns
    -------
    array-like
        Complex visibility samples.

    Notes
    -----
    This function does not account for rotation and geometric stretching (e.g. due to
    inclination). A separate transformation of $uv$ coordinates should account for this.
    The phase angles of the cosine modulations are defined relative to the
    spatial y-axis (North), turning counterclockwise to the x-axis (East). This means
    that a single 1st order modulation with a phase angle of $0 \, \mathrm{deg}$
    results in a bright peak towards the North, and a faint peak towards the South.
    A phase angle of $90 \, \mathrm{deg}$ would result in a bright peak towards the
    East, and a faint one towards the West.
    """
    # Add radially symmetric component (order m=0) to beginning of the order arrays.
    az_amps = np.concatenate([np.array([1.0]), az_amps])
    az_phis = np.concatenate([np.array([0.0]), az_phis])
    az_orders = np.arange(az_amps.size)

    r0_rad = r0 * mas2rad
    az_phis_rad = az_phis * dtor

    # Get length of baseline and baseline projection angle (i.e. counterclockwise
    # angle in uv-plane, turning from top, i.e. positive v, to left, i.e. positive u).
    # At u = v = 0 (e.g. flagged samples) hypot and arctan2 have undefined
    # derivatives; the double where keeps gradients finite there, where the
    # visibility is exactly 1.
    origin = (u == 0) & (v == 0)
    u_safe, v_safe = np.where(origin, 1.0, u), np.where(origin, 1.0, v)
    base_norm = np.where(origin, 0.0, np.hypot(u_safe, v_safe))
    base_proj_ang_rad = np.where(origin, 0.0, np.arctan2(u_safe, v_safe))

    az_order_max = np.size(az_orders) - 1
    xbes = 2.0 * np.pi * base_norm * r0_rad
    bessel_vals = bessel_jn(az_order_max, xbes)

    def _azmod_cvis_term(az_amp, az_phi_rad, az_order):
        return (
            az_amp
            * np.exp(-0.5j * np.pi * az_order)
            * np.cos(az_order * (base_proj_ang_rad - az_phi_rad))
            * bessel_vals[az_order]
        )

    azmod_cvis_terms = jax.vmap(
        _azmod_cvis_term, in_axes=(0, 0, 0), out_axes=0
    )(az_amps, az_phis_rad, az_orders)

    return np.sum(azmod_cvis_terms, axis=0)


def _cvis_gaussian_envelope(u, v, fwhm):
    """Complex visibility envelope of a centered isotropic 2D Gaussian PSF, used
    as the convolution kernel of [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim]. Not offered as a public
    function: this is a plain Gaussian envelope
    with no point-source/companion mixture. The rim evaluates it on deprojected
    spatial frequencies, so the kernel is isotropic in the rim plane.
    """
    fwhm_rad = fwhm * mas2rad
    base_norm = np.hypot(u, v)
    return (
        np.exp(-(np.pi**2) * fwhm_rad**2 * base_norm**2 / (4.0 * np.log(2.0)))
        + 0j
    )


def _cvis_centred_rim(u, v, diam, fwhm, inc, pa, az_amps, az_phis):
    """Unit-flux visibility of an inclined, modulated thin ring at the origin, blurred by a Gaussian that is isotropic in the rim plane."""
    # Transform spatial frequency coordinates to the frame of reference where the
    # model rim is uninclined and the major axis is pointed North.
    stretch_factor = np.maximum(np.cos(inc * dtor), 1e-8)
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa, stretch_factor)

    cvis = cvis_radial_dirac_delta_modulated(
        ut, vt, diam / 2.0, az_amps, az_phis
    )

    # Gaussian blur that is isotropic in the rim plane, so it is evaluated in
    # the same (deprojected) frame as the ring; on the sky it is an elliptical
    # Gaussian that is narrower along the minor axis.
    return cvis * _cvis_gaussian_envelope(ut, vt, fwhm)
