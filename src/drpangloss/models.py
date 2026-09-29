"""Source models: sky-brightness distributions and their visibilities.

* Components ([`PointSource`][drpangloss.models.PointSource],
  [`GaussianDisk`][drpangloss.models.GaussianDisk],
  [`UniformDisk`][drpangloss.models.UniformDisk],
  [`ModulatedGaussianRim`][drpangloss.models.ModulatedGaussianRim]) are
  single shapes, combined with flux weights in a
  [`System`][drpangloss.models.System].
* [`BinaryModelCartesian`][drpangloss.models.BinaryModelCartesian] and
  [`BinaryModelAngular`][drpangloss.models.BinaryModelAngular] are fast
  forms of a primary plus a point-source companion.
* The ``cvis_*`` functions are the analytic visibilities behind them.

Every flux is relative: for a companion it is the companion/primary flux
ratio. Likelihoods of these models are in
[`drpangloss.likelihood`][drpangloss.likelihood].
"""

import dataclasses
import textwrap
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zx
from jax.scipy.signal import fftconvolve

from ._geometry import (
    check_az_prof_nonnegative,
    image_coordinates,
    offset_phase,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)
from ._utils import concrete, dtor, mas2rad
from .bessel import bessel_jn
from .spectra import Spectrum, flux_at, reference_flux


def _normalize_image(image):
    """Return a finite unit-sum image for render outputs."""
    image = np.nan_to_num(np.asarray(image), nan=0.0, posinf=0.0, neginf=0.0)
    total = np.sum(image)
    if bool(np.isfinite(total)) and bool(total > 0.0):
        return image / total
    raise ValueError("Rendered image must contain positive finite flux.")


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
    """Traced check that a number or spectrum has non-negative flux."""
    return np.all(np.asarray(reference_flux(flux)) >= 0.0)


def _check_non_negative_flux(flux, owner):
    """Raise if a concrete ``flux`` is negative; traced values are not checked."""
    value = concrete(reference_flux(flux))
    if value is not None and onp.any(value < 0.0):
        raise ValueError(
            f"{owner} has flux {value.tolist()}; fluxes must be non-negative."
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


class SourceModel(zx.Base):  # type: ignore[reportGeneralTypeIssues]
    """Base class for sky-brightness source models.

    There are two kinds of source model.

    * **Components** ([`Component`][drpangloss.models.Component] subclasses such as
      [`PointSource`][drpangloss.models.PointSource]) are single shapes normalized to unit flux. Their
      ``flux`` is a *relative weight*, which only matters once they are mixed
      together in a [`System`][drpangloss.models.System].
    * **Scenes** ([`System`][drpangloss.models.System], [`BinaryModelCartesian`][drpangloss.models.BinaryModelCartesian],
      [`BinaryModelAngular`][drpangloss.models.BinaryModelAngular], [`HarmonixModel`][drpangloss.models.HarmonixModel]) describe a whole,
      normalized sky. A scene placed inside a [`System`][drpangloss.models.System] has weight 1,
      unless it carries its own ``flux`` weight as [`System`][drpangloss.models.System] does.

    The binary models' ``flux`` is their companion/primary flux ratio, which
    is the same thing as a companion's weight in a System whose primary
    has ``flux=1``.

    Subclasses implement [`model`][drpangloss.models.SourceModel.model], and ``_image`` if they can be drawn.
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

    def render(self, npix=256, fov_mas=200.0):
        """Render a unit-sum image of the model.

        The image is ``npix`` x ``npix`` pixels spanning ``fov_mas``
        milliarcseconds, with East to the left (column 0 is the most
        positive ``dra``) and North up (row 0 is the most positive ``ddec``).
        Use [`drpangloss.plotting.plot_model`][drpangloss.plotting.plot_model] to display it with the
        correct axes.
        """
        xx, yy = image_coordinates(npix, fov_mas)
        return _normalize_image(
            self._image(xx, yy, float(fov_mas) / float(npix))
        )

    def _weight(self, wavel=None):
        """Relative flux of this model inside a [`System`][drpangloss.models.System].

        ``wavel`` is the wavelength in metres at which the flux is wanted
        (broadcastable against the baselines), or ``None`` for the model's
        reference flux, as used when rendering. Components and systems whose
        ``flux`` is a spectrum (see [`drpangloss.spectra`][drpangloss.spectra])
        evaluate it here.
        """
        return 1.0

    def is_physical(self):
        """Whether the model is physically valid, as a (traceable) boolean.

        Unlike the checks made when a model is built, this works inside
        ``jax.jit`` and on models changed with ``set``, so likelihoods can
        reject invalid models (see ``reject_unphysical`` in
        [`model_loglike`][drpangloss.likelihood.model_loglike]). The base
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
    [`System`][drpangloss.models.System], ``flux`` is its weight relative to the other components,
    and ``dra``/``ddec`` place its centre (milliarcseconds, positive ``dra``
    to the East, positive ``ddec`` to the North).

    New shapes subclass [`Component`][drpangloss.models.Component] and implement ``_centred_cvis``
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
        Weight relative to the other components of a [`System`][drpangloss.models.System],
        or a spectrum from [`drpangloss.spectra`][drpangloss.spectra]
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
        (FWHM = 2.3548 ``sigma``).
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][drpangloss.models.System],
        or a spectrum from [`drpangloss.spectra`][drpangloss.spectra]
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


class UniformDisk(Component):
    """Uniformly bright (tophat) circular disk, e.g. a resolved stellar photosphere.

    Parameters
    ----------
    diam : float or array-like
        Angular diameter in milliarcseconds.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][drpangloss.models.System],
        or a spectrum from [`drpangloss.spectra`][drpangloss.spectra]
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
    Azimuthally modulated, infinitely thin rim convolved with an isotropic 2D
    Gaussian, optionally inclined and rotated.

    Parameters
    ----------
    diam : float or array-like
        Diameter of the rim in milliarcseconds.
    fwhm : float or array-like
        Gaussian FWHM of the rim in milliarcseconds.
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
        Position angles of the cosine azimuthal modulations in degrees, North
        to East, one per entry of ``az_amps``.
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a [`System`][drpangloss.models.System],
        or a spectrum from [`drpangloss.spectra`][drpangloss.spectra]
        (default 1).
    dra : float or array-like, optional
        Right-ascension offset of the rim's center in milliarcseconds,
        positive to the East.
    ddec : float or array-like, optional
        Declination offset of the rim's center in milliarcseconds, positive
        to the North.

    Notes
    -----
    The intensity profile is separable into a symmetric radial profile and cosine
    azimuthal modulations, meaning the image intensity can be described in polar image
    coordinates as $I(r, \theta) = f(r) \left( 1 + \sum_{m=1}^{n}
    A_m \cos{(m(\theta - \mathrm{pa}_m))} \right)$, where $f(r)$ is a thin ring radial
    profile convolved with an isotropic Gaussian.

    This model is achromatic: it does not represent any spectral dependence.
    The rim contains no star; put it in a [`System`][drpangloss.models.System] with a
    [`PointSource`][drpangloss.models.PointSource] for that.

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
        sigma_mas = np.maximum(
            self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-9
        )
        psf = np.exp(-0.5 * (kx**2 + ky**2) / sigma_mas**2)
        return fftconvolve(ring, psf, mode="same")


class Resolved(SourceModel):
    """Fully resolved (over-resolved) flux, e.g. a large, diffuse envelope.

    Its visibility is 0 on every non-zero baseline, so inside a
    [`System`][drpangloss.models.System] it only adds to the normalization,
    lowering every other component's visibility by the same factor.

    Parameters
    ----------
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][drpangloss.models.System] (default 1), or a spectrum from
        [`drpangloss.spectra`][drpangloss.spectra].

    Notes
    -----
    A resolved component spreads its light far beyond any image, so
    [`render`][drpangloss.models.SourceModel.render] draws nothing for it,
    and a rendered [`System`][drpangloss.models.System] shows only its other
    components (renormalized to unit sum).

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

    def _image(self, xx, yy, pixel_scale_mas):
        return np.zeros_like(xx)

    def _weight(self, wavel=None):
        return flux_at(self.flux, wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def __check_init__(self):
        _check_non_negative_flux(self.flux, "Resolved")


class System(SourceModel):
    r"""Flux-weighted mixture of named source models.

    A [`System`][drpangloss.models.System] is how you describe a scene with more than one part:
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
    in [`drpangloss.grid_fit`][drpangloss.grid_fit] and [`numpyro_model`][drpangloss.likelihood.numpyro_model] address
    parameters. Components keep the order in which they were given.

    A [`System`][drpangloss.models.System] can itself be a component. Its own ``flux`` is then the
    total flux of the group relative to its siblings, and ``dra``/``ddec``
    move the whole group together.

    Parameters
    ----------
    components : dict[str, SourceModel], optional
        Components as a mapping from name to model. Usually it is clearer to
        pass them as keyword arguments instead.
    flux : float or array-like, optional
        Weight of the whole system when nested inside another
        [`System`][drpangloss.models.System] (default 1). It has no effect at the top level.
    dra, ddec : float or array-like, optional
        Offset of the whole system in milliarcseconds (positive to the East
        and North).
    **named : SourceModel
        Components as keyword arguments, e.g. ``star=PointSource()``. Names
        must be valid Python identifiers that do not start with ``_`` and do
        not clash with a [`System`][drpangloss.models.System] attribute (``model``, ``render``,
        ``set``, ``flux``, ...).

    Notes
    -----
    Fluxes are physical brightnesses, so they must be non-negative, and they
    must not all be zero. Both are checked when a model is built from
    concrete values; changing values afterwards with ``set`` is not
    checked. Inside a traced computation (a fit or grid search) the
    values cannot be checked, so positivity is the job of the priors and grid
    axes: [`numpyro_model`][drpangloss.likelihood.numpyro_model] rejects flux priors that allow negative
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
            _check_component_name(name)
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
        weights = [c._weight(wavel) for c in self.parts]
        total = sum(
            w * c.model(u, v, wavel) for w, c in zip(weights, self.parts)
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

    def is_physical(self):
        valid = _flux_is_non_negative(self.flux)
        for part in self.parts:
            valid = valid & part.is_physical()
        return valid

    def __check_init__(self):
        _check_non_negative_flux(self.flux, "System")


_RESERVED_COMPONENT_NAMES = frozenset({"components", "names", "parts"})


def _check_component_name(name):
    """Reject component names that cannot be used as parameter paths."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"Component name {name!r} must be a valid Python identifier, "
            "so that it can be used in parameter paths such as 'comp.flux'."
        )
    if (
        name.startswith("_")
        or name in _RESERVED_COMPONENT_NAMES
        or name in {"flux", "dra", "ddec"}
        or hasattr(System, name)
    ):
        raise ValueError(
            f"'{name}' cannot be a component name because it clashes with a "
            "System attribute or method; choose another name."
        )


def GaussianDiskModel(sigma, flux, dra=0.0, ddec=0.0):
    """Point source at the origin plus a Gaussian disk with disk/star ratio ``flux``.

    Convenience constructor equivalent to
    ``System(star=PointSource(), disk=GaussianDisk(sigma, flux, dra, ddec))``,
    whose parameters are addressed as ``"disk.sigma"``, ``"disk.flux"`` etc.
    It can still be passed as a model class with plain parameter names
    (``sigma``, ``flux``, ``dra``, ``ddec``) to the fitting tools.

    ``GaussianDiskModel`` used to be a class. It now returns a
    [`System`][drpangloss.models.System], so ``isinstance(model, GaussianDiskModel)`` no longer
    works.
    """
    return System(
        star=PointSource(),
        disk=GaussianDisk(sigma, flux=flux, dra=dra, ddec=ddec),
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
        [`flux_to_contrast`][drpangloss.limits.flux_to_contrast]).

    Notes
    -----
    Equivalent to [`BinaryModelCartesian`][drpangloss.models.BinaryModelCartesian]
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
        """Return the equivalent [`BinaryModelCartesian`][drpangloss.models.BinaryModelCartesian]."""
        th = self.pa * dtor
        return BinaryModelCartesian(
            self.sep * np.sin(th), self.sep * np.cos(th), self.flux
        )

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def model(self, u, v, wavel):
        """Complex visibilities on baselines ``u``, ``v`` (m) at ``wavel`` (m)."""
        uu, vv = u / wavel, v / wavel
        return cvis_binary_angular(uu, vv, self.sep, self.pa, self.flux)

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
        """Return the equivalent [`BinaryModelAngular`][drpangloss.models.BinaryModelAngular]."""
        sep = np.sqrt(self.dra**2 + self.ddec**2)
        pa = np.mod(np.rad2deg(np.arctan2(self.dra, self.ddec)), 360.0)
        return BinaryModelAngular(sep, pa, self.flux)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def model(self, u, v, wavel):
        """Complex visibilities on baselines ``u``, ``v`` (m) at ``wavel`` (m)."""
        uu, vv = u / wavel, v / wavel
        return cvis_binary(uu, vv, self.dra, self.ddec, self.flux)

    def to_system(self):
        """Return the equivalent ``System(primary=..., companion=...)``.

        The [`System`][drpangloss.models.System] form is slower to evaluate
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
        Name of a ``render(npix, fov_mas)`` method used by [`render`][drpangloss.models.HarmonixModel.render]
        (default ``"render"``). Sources without one but with a ``surface``
        (harmonix stars) are rendered from ``surface.render``.
    expects_wavelength_units : bool, optional
        If True (default), pass spatial frequencies ``u / wavel``,
        ``v / wavel``; otherwise pass ``u``, ``v`` in metres.
    observation_time : optional
        Extra argument passed after the baselines, e.g. a time for rotating
        stars.

    Notes
    -----
    ``source`` and ``observation_time`` are static (hashable) fields: the
    wrapped model's parameters cannot be fitted through zodiax paths, and
    array-valued ``observation_time`` values are not supported under
    ``jax.jit``. A wrapped source has weight 1 inside a
    [System][drpangloss.models.System], and cannot be drawn there (only on
    its own, with [`render`][drpangloss.models.HarmonixModel.render]).
    """

    source: Any = eqx.field(static=True)
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
        if not hasattr(self.source, self.render_method):
            if hasattr(self.source, "surface"):
                theta = 0.0
                if hasattr(self.source, "rotational_phase"):
                    theta = self.source.rotational_phase(
                        0.0
                        if self.observation_time is None
                        else self.observation_time
                    )
                image = np.asarray(
                    self.source.surface.render(res=npix, theta=theta)
                )
                return _normalize_image(image)
            raise NotImplementedError(
                "Wrapped source does not expose a compatible render method."
            )
        return _normalize_image(
            getattr(self.source, self.render_method)(npix, fov_mas)
        )


def cvis_binary_angular(u, v, sep, pa, flux):
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
    # adapted from pymask
    """Compute complex visibilities for a Cartesian-parameterized binary model.

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


def cvis_gaussian_disk(
    u,
    v,
    sigma,
    flux,
    dra: jax.Array | float = 0.0,
    ddec: jax.Array | float = 0.0,
):
    """Compute complex visibilities for a Gaussian-disk companion mixed with
    an unresolved point source, using the ``flux`` companion/primary flux ratio
    convention shared with [`cvis_binary`][drpangloss.models.cvis_binary].
    """
    sigma_rad = mas2rad * sigma
    rho2 = u**2 + v**2
    envelope = np.exp(-2.0 * (np.pi**2) * (sigma_rad**2) * rho2)

    phase = offset_phase(u, v, dra, ddec)

    l2 = flux / (flux + 1.0)
    l1 = 1.0 - l2
    return l1 + l2 * envelope * phase


def cvis_uniform_disk(u, v, ud, dra=0.0, ddec=0.0):
    """Compute complex visibilities for a uniform (tophat) disk.

    The visibility amplitude follows the classic uniform-disk form
    ``2 * J1(x) / x``, with ``x = pi * ud_rad * base_norm`` the product of
    the disk diameter (in radians) and the baseline length in wavelength
    units (``base_norm = hypot(u, v)``, i.e. baseline length divided by
    wavelength).

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units (cycles / rad).
    v : array-like
        Baseline ``v`` coordinates in wavelength units (cycles / rad).
    ud : float or array-like
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
    ud_rad = mas2rad * ud
    base_norm = np.hypot(u, v)
    kernel = np.pi * base_norm * ud_rad

    # Keep 0 out of the division so the unused branch has finite gradients.
    at_zero = kernel == 0
    safe_kernel = np.where(at_zero, 1.0, kernel)
    envelope = np.where(
        at_zero,
        1.0 + 0j,
        (2.0 * bessel_jn(1, safe_kernel)[1]) / safe_kernel + 0j,
    )

    return envelope * offset_phase(u, v, dra, ddec)


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
    base_norm = np.hypot(u, v)
    base_proj_ang_rad = np.arctan2(u, v)

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
    as the convolution kernel of [`ModulatedGaussianRim`][drpangloss.models.ModulatedGaussianRim]. Not offered as a public
    function: unlike [`cvis_gaussian_disk`][drpangloss.models.cvis_gaussian_disk], this is a plain Gaussian envelope
    with no point-source/companion mixture.
    """
    fwhm_rad = fwhm * mas2rad
    base_norm = np.hypot(u, v)
    return (
        np.exp(-(np.pi**2) * fwhm_rad**2 * base_norm**2 / (4.0 * np.log(2.0)))
        + 0j
    )


def _cvis_centred_rim(u, v, diam, fwhm, inc, pa, az_amps, az_phis):
    """Unit-flux visibility of a Gaussian-blurred, inclined, modulated thin ring at the origin."""
    # Transform spatial frequency coordinates to the frame of reference where the
    # model rim is uninclined and the major axis is pointed North.
    stretch_factor = np.maximum(np.cos(inc * dtor), 1e-8)
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa, stretch_factor)

    cvis = cvis_radial_dirac_delta_modulated(
        ut, vt, diam / 2.0, az_amps, az_phis
    )

    # Image-plane Gaussian blur, evaluated in the original (untransformed) frame.
    return cvis * _cvis_gaussian_envelope(u, v, fwhm)
