import dataclasses
import textwrap
from typing import Any

import jax.numpy as np
import jax
from jax.scipy.signal import fftconvolve

import numpy as onp

import equinox as eqx
import zodiax as zx

from ._utils import (
    bessel_jn as bessel_jn,
    check_az_prof_nonnegative,
    dtor as dtor,
    i2pi as i2pi,
    mas2rad as mas2rad,
    rad2mas as rad2mas,
    renamed_argument as _renamed_argument,
    undo_elliptical_transf_coord as undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq as undo_elliptical_transf_spat_freq,
)
from .inference import (
    fisher_matrix as _fisher_matrix,
    laplace_covariance as _laplace_covariance,
)

from .oidata import OIData as OIData
from .oidata import closure_phases as closure_phases
from .oidata import cp_indices as cp_indices


def _image_coordinates(npix, fov_mas):
    """Return Cartesian image-plane coordinates in milliarcseconds."""
    npix = int(npix)
    pixel_scale_mas = float(fov_mas) / npix
    pixel_indices = np.arange(npix)
    center = 0.5 * (npix - 1)
    x = -(pixel_indices - center) * pixel_scale_mas
    y = (center - pixel_indices) * pixel_scale_mas
    return np.meshgrid(x, y, indexing="xy")


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


def _offset_phase(uu, vv, dra, ddec):
    """Fourier shift factor for an offset of ``(dra, ddec)`` milliarcseconds."""
    arg = 2.0 * np.pi * mas2rad * (uu * dra + vv * ddec)
    return jax.lax.complex(np.cos(arg), -np.sin(arg))


def _format_leaf(x):
    """Short human-readable form of a parameter value for ``repr``."""
    try:
        value = onp.asarray(x)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return repr(x)
    if value.ndim == 0 and value.dtype.kind in "fiu":
        return f"{float(value):g}"
    if value.dtype.kind in "fiu":
        return "[" + ", ".join(f"{float(v):g}" for v in value.ravel()) + "]"
    return repr(x)


def _check_non_negative_flux(flux, owner):
    """Raise if a concrete ``flux`` is negative; traced values are not checked."""
    try:
        value = onp.asarray(flux)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return
    if onp.any(value < 0.0):
        raise ValueError(
            f"{owner} has flux {value.tolist()}; fluxes must be non-negative."
        )


def _check_non_negative_modulation(az_amps, az_pas):
    """Raise if concrete azimuthal modulations make the brightness negative."""
    try:
        amps, pas = onp.asarray(az_amps), onp.asarray(az_pas)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return
    if not bool(check_az_prof_nonnegative(np.asarray(amps), np.asarray(pas))):
        raise ValueError(
            f"Azimuthal modulations az_amps={amps.tolist()}, "
            f"az_pas={pas.tolist()} make the rim brightness negative "
            "somewhere; sum(abs(az_amps)) <= 1 always keeps it non-negative."
        )


def _concrete_sum(values):
    """Sum of ``values`` as a float, or ``None`` if any value is traced."""
    try:
        return float(sum(onp.sum(onp.asarray(v)) for v in values))
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


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

    For historical reasons the binary models use ``flux`` (and ``contrast``)
    for a companion/primary ratio rather than a weight. New models should use
    ``flux`` only to mean a relative weight.

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
        xx, yy = _image_coordinates(npix, fov_mas)
        return _normalize_image(
            self._image(xx, yy, float(fov_mas) / float(npix))
        )

    def _weight(self, wavel=None):
        """Relative flux of this model inside a [`System`][drpangloss.models.System].

        ``wavel`` is the wavelength in metres at which the flux is wanted
        (broadcastable against the baselines), or ``None`` for the model's
        reference flux, as used when rendering. Every current model is
        achromatic and ignores it; chromatic fluxes (e.g. spectral indices)
        will override this.
        """
        return 1.0

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
        return self._centred_cvis(uu, vv) * _offset_phase(
            uu, vv, self.dra, self.ddec
        )

    def _image(self, xx, yy, pixel_scale_mas):
        return self._centred_image(
            xx - self.dra, yy - self.ddec, pixel_scale_mas
        )

    def _weight(self, wavel=None):
        return self.flux

    def __check_init__(self):
        _check_non_negative_flux(self.flux, type(self).__name__)


class PointSource(Component):
    """Unresolved point source.

    Parameters
    ----------
    flux : float or array-like, optional
        Weight relative to the other components of a [`System`][drpangloss.models.System]
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
        self.flux = np.asarray(flux, dtype=float)
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
    flux : float or array-like, optional
        Weight relative to the other components of a [`System`][drpangloss.models.System]
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
        self.flux = np.asarray(flux, dtype=float)
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
    flux : float or array-like, optional
        Weight relative to the other components of a [`System`][drpangloss.models.System]
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
        self.flux = np.asarray(flux, dtype=float)
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
    flux : float or array-like, optional
        Weight relative to the other components of a [`System`][drpangloss.models.System]
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
        self.flux = np.asarray(flux, dtype=float)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

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
        kx, ky = _image_coordinates(nker, nker * pixel_scale_mas)
        sigma_mas = np.maximum(
            self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-9
        )
        psf = np.exp(-0.5 * (kx**2 + ky**2) / sigma_mas**2)
        return fftconvolve(ring, psf, mode="same")


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
    in [`drpangloss.grid_fit`][drpangloss.grid_fit] and [`numpyro_model`][drpangloss.models.numpyro_model] address
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
    axes: [`numpyro_model`][drpangloss.models.numpyro_model] rejects flux priors that allow negative
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
        self.flux = np.asarray(flux, dtype=float)
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
        return total * _offset_phase(uu, vv, self.dra, self.ddec)

    def _image(self, xx, yy, pixel_scale_mas):
        xx, yy = xx - self.dra, yy - self.ddec
        weights = [c._weight() for c in self.parts]
        return sum(
            w * _unit_flux(c._image(xx, yy, pixel_scale_mas))
            for w, c in zip(weights, self.parts)
        ) / sum(weights)

    def _weight(self, wavel=None):
        return self.flux

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
    Represent a binary companion using angular separation and position angle.

    Parameters
    ----------
    sep : float or array-like
        On-sky separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees, measured East of North.
    contrast : float or array-like
        Brightness contrast ratio ``star/companion``.

    Notes
    -----
    This parameterization is often convenient for reporting astrophysical
    constraints directly in polar-like coordinates. The model evaluates complex
    visibilities on the provided interferometric baseline geometry.
    """

    sep: jax.Array
    pa: jax.Array
    contrast: jax.Array

    def __init__(self, sep, pa, contrast):
        """
        Initialize a binary model in angular coordinates.

        Parameters
        ----------
        sep : float or array-like
            Separation in milliarcseconds.
        pa : float or array-like
            Position angle in degrees.
        contrast : float or array-like
            Contrast ratio between primary and companion (``star/companion``).

        """

        self.sep = np.asarray(sep, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.contrast = np.asarray(contrast, dtype=float)

    def unpack_all(self):
        """
        Return all model parameters in angular form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(sep, pa, contrast)``.
        """
        return self.sep, self.pa, self.contrast

    def to_cartesian(self):
        """
        Convert this angular parameterization into Cartesian sky offsets.

        Returns
        -------
        BinaryModelCartesian
            Equivalent binary model expressed as ``(dra, ddec, flux)``.
        """
        th = self.pa * dtor
        dra = self.sep * np.sin(th)
        ddec = self.sep * np.cos(th)
        flux = 1.0 / self.contrast
        return BinaryModelCartesian(dra, ddec, flux)

    def model(self, u, v, wavel):
        """
        Evaluate complex visibilities for this angular binary model.

        Parameters
        ----------
        u : array-like
            Baseline ``u`` coordinates in meters.
        v : array-like
            Baseline ``v`` coordinates in meters.
        wavel : array-like
            Effective wavelength(s) in meters.

        Returns
        -------
        array-like
            Complex visibility samples on the provided baselines.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary_angular(uu, vv, self.sep, self.pa, self.contrast)

    def _image(self, xx, yy, pixel_scale_mas):
        return self.to_cartesian()._image(xx, yy, pixel_scale_mas)


class BinaryModelCartesian(SourceModel):
    """
    Represent a binary companion using Cartesian sky offsets.

    Parameters
    ----------
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    ddec : float or array-like
        Declination offset in milliarcseconds.
    flux : float or array-like
        Companion-to-primary flux ratio.

    Notes
    -----
    This parameterization is useful for optimization and inference workflows
    that operate directly in Cartesian offsets.
    """

    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array

    def __init__(self, dra, ddec, flux):
        """
        Initialize a binary model in Cartesian offsets.

        Parameters
        ----------
        dra : float or array-like
            Right-ascension offset in milliarcseconds.
        ddec : float or array-like
            Declination offset in milliarcseconds.
        flux : float or array-like
            Flux ratio for the companion component.

        """

        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

    def unpack_all(self):
        """
        Return all model parameters in Cartesian form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(dra, ddec, flux)``.
        """
        return self.dra, self.ddec, self.flux

    def to_angular(self):
        """
        Convert this Cartesian parameterization into angular coordinates.

        Returns
        -------
        BinaryModelAngular
            Equivalent binary model expressed as ``(sep, pa, contrast)``.
        """
        sep = np.sqrt(self.dra**2 + self.ddec**2)
        pa = np.mod(np.rad2deg(np.arctan2(self.dra, self.ddec)), 360.0)
        contrast = 1.0 / self.flux
        return BinaryModelAngular(sep, pa, contrast)

    def model(self, u, v, wavel):
        """
        Evaluate complex visibilities for this Cartesian binary model.

        Parameters
        ----------
        u : array-like
            Baseline ``u`` coordinates in meters.
        v : array-like
            Baseline ``v`` coordinates in meters.
        wavel : array-like
            Effective wavelength(s) in meters.

        Returns
        -------
        array-like
            Complex visibility samples on the provided baselines.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary(uu, vv, self.dra, self.ddec, self.flux)

    def to_system(self):
        """Return the equivalent ``System(primary=..., companion=...)``.

        The [`System`][drpangloss.models.System] form is slower to evaluate but can be extended,
        e.g. by adding a disk around the primary.
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


HarmonixAdapter = HarmonixModel


def cvis_binary_angular(u, v, sep, pa, contrast):
    # adapted from pymask
    """Compute complex visibilities for an angular-parameterized binary model.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    sep : float or array-like
        Separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees, measured East of North.
    contrast : float or array-like
        Contrast ratio ``star/companion``.

    Returns
    -------
    array-like
        Complex visibility samples.
    """

    # normalize visibilities so total power is 1

    th = pa * dtor

    ddec = mas2rad * (sep * np.cos(th))
    dra = mas2rad * (sep * np.sin(th))

    # decompose into two "luminosity"
    l2 = 1.0 / (contrast + 1)
    l1 = 1 - l2

    # phase-factor
    phi = np.exp(-i2pi * (u * dra + v * ddec))
    cvis = l1 + l2 * phi

    return cvis


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

    return primary + companion * _offset_phase(u, v, dra, ddec)


def cvis_gaussian_disk(
    u,
    v,
    sigma,
    flux,
    dra: jax.Array | float = 0.0,
    ddec: jax.Array | float = 0.0,
):
    """Compute complex visibilities for a Gaussian-disk companion mixed with
    an unresolved point source, using the ``flux`` companion/star contrast
    convention shared with [`cvis_binary`][drpangloss.models.cvis_binary].
    """
    sigma_rad = mas2rad * sigma
    rho2 = u**2 + v**2
    envelope = np.exp(-2.0 * (np.pi**2) * (sigma_rad**2) * rho2)

    dra_rad = mas2rad * dra
    ddec_rad = mas2rad * ddec
    phase = np.exp(-i2pi * (u * dra_rad + v * ddec_rad))

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

    dra_rad = mas2rad * dra
    ddec_rad = mas2rad * ddec
    phase = np.exp(-i2pi * (u * dra_rad + v * ddec_rad))
    return envelope * phase


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
    with no point-source/flux-contrast mixture.
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


def _gaussian_loglike(residuals, errors):
    """Sum of independent Gaussian log densities of ``residuals``."""
    return jax.scipy.stats.norm.logpdf(residuals, loc=0.0, scale=errors).sum()


def model_loglike(model_object, data_obj):
    """Evaluate a Gaussian log likelihood for an instantiated model object.

    Phase residuals are wrapped into ``[-π, π)`` (see
    [`residuals`][drpangloss.oidata.OIData.residuals]).
    """
    _, errors = data_obj.flatten_data()
    residuals = data_obj.residuals(data_obj.model(model_object))
    return _gaussian_loglike(residuals, errors)


def joint_prediction(params, observations, model_fn):
    """Concatenate predictions for a parameter pytree and multiple observations.

    ``model_fn(params, index)`` defines which parameters are shared and which
    are specific to each observation.
    """
    return np.concatenate(
        [
            observation.model(model_fn(params, index))
            for index, observation in enumerate(observations)
        ]
    )


def joint_data(observations):
    """Concatenate observed vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.standardize_data() for observation in observations]
    )


def joint_errors(observations):
    """Concatenate uncertainty vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.standardize_errors() for observation in observations]
    )


def joint_loglike(params, observations, model_fn):
    """Sum independent Gaussian log likelihoods over multiple observations."""
    return sum(
        model_loglike(model_fn(params, index), observation)
        for index, observation in enumerate(observations)
    )


def build_model(model, params, values):
    """Build a model from parameter names and values.

    ``model`` is either a class/callable, called as ``model(**dict(zip(params,
    values)))``, or a [`SourceModel`][drpangloss.models.SourceModel] instance used as a template whose
    leaves at the (dot-separated) paths ``params`` are replaced by ``values``.
    """
    if isinstance(model, SourceModel):
        return model.set(list(params), list(values))
    return model(**dict(zip(params, values)))


@_renamed_argument("model_class", "model")
def loglike(values, params, data_obj, model):
    """
    Gaussian log-likelihood of a model with the given parameter values, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][drpangloss.models.build_model]).

    Returns
    -------
    float
        Log-likelihood value.
    """

    return model_loglike(build_model(model, params, values), data_obj)


def _is_flux_name(name):
    """Whether a prior key names a (non-negative) brightness parameter."""
    leaf = name.rsplit(".", 1)[-1]
    return leaf in {"flux", "contrast"} or leaf.endswith("_flux")


def _check_positive_flux_prior(name, distribution):
    """Reject flux priors whose support includes negative values."""
    from numpyro.distributions import constraints

    support = distribution.support
    lower = getattr(support, "lower_bound", None)
    if lower is None:
        unbounded = support in (constraints.real, constraints.real_vector)
    else:
        try:
            unbounded = bool(onp.any(onp.asarray(lower) < 0.0))
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            unbounded = False
    if unbounded:
        raise ValueError(
            f"The prior on {name!r} allows negative values, but fluxes must "
            "be non-negative. Use a prior with non-negative support, e.g. "
            "dist.LogUniform or dist.Uniform(0, ...)."
        )


def numpyro_model(model, priors, data_obj):
    """Return a numpyro model sampling the parameters in ``priors``.

    Parameters
    ----------
    model : SourceModel or callable
        Either a template model whose leaves at the paths in ``priors`` are
        sampled, or a function called with the sampled values as keyword
        arguments that returns a [`SourceModel`][drpangloss.models.SourceModel]. A function lets you
        sample parameters that are not leaves of the model, such as a
        separation and position angle, or one inclination shared by two
        components (see [`build_model`][drpangloss.models.build_model]).
    priors : dict[str, numpyro.distributions.Distribution]
        Mapping from parameter path (e.g. ``"comp.flux"``) or function
        argument name to prior; each key is also used as the numpyro
        sample-site name. Priors on brightnesses (keys named ``flux`` or
        ``contrast``, or ending in ``.flux``, ``.contrast`` or ``_flux``)
        must have non-negative support.
    data_obj : OIData or sequence of OIData
        Data whose Gaussian log likelihood is added with ``numpyro.factor``.

    Returns
    -------
    callable
        Zero-argument numpyro model, e.g. for ``numpyro.infer.NUTS``.
    """
    import numpyro

    paths = list(priors)
    for path in paths:
        if _is_flux_name(path):
            _check_positive_flux_prior(path, priors[path])
    observations = (
        tuple(data_obj) if isinstance(data_obj, (list, tuple)) else (data_obj,)
    )

    def numpyro_fn():
        values = [numpyro.sample(path, priors[path]) for path in paths]
        source = build_model(model, paths, values)
        numpyro.factor(
            "loglike",
            sum(model_loglike(source, obs) for obs in observations),
        )

    return numpyro_fn


@_renamed_argument("model_class", "model")
def loglike_nosignal(values, params, data_obj, model):
    """
    Gaussian log-likelihood of the no-signal (unresolved point source) data under a model, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][drpangloss.models.build_model]).

    Returns
    -------
    float
        Log-likelihood value.
    """

    model_data = data_obj.model(build_model(model, params, values))
    _, errors = data_obj.flatten_data()
    unity_cvis = np.ones_like(data_obj.u, dtype=complex)
    null_data = data_obj.standardize_model(unity_cvis)

    return _gaussian_loglike(data_obj.residuals(model_data, null_data), errors)


@_renamed_argument("model_class", "model")
def laplace_cov(values, params, data_obj, model):
    """
    Compute the full Laplace covariance matrix for all model parameters jointly.

    Computes the inverse of the Hessian of the negative log-likelihood with
    respect to all parameters in ``params`` simultaneously, returning an
    ``N x N`` covariance matrix (where ``N = len(params)``).

    This returns the *full* covariance matrix over all ``N`` parameters. For
    only the flux uncertainty at a fixed position, use
    [`laplace_contrast_uncertainty`][drpangloss.models.laplace_contrast_uncertainty]
    instead.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][drpangloss.models.build_model]).

    Returns
    -------
    array-like
        ``N x N`` covariance matrix, where ``N = len(params)``.
    """

    objective = lambda vals: -loglike(vals, params, data_obj, model)
    return _laplace_covariance(objective, np.asarray(values, dtype=float))


@_renamed_argument("model_class", "model")
def laplace_contrast_uncertainty(
    flux, dra, ddec, data_obj, model, params=None
):
    """
    Compute the Laplace uncertainty in flux at a fixed sky position.

    Unlike [`laplace_cov`][drpangloss.models.laplace_cov], which inverts the *full* N-parameter Hessian,
    this function **fixes** ``dra`` and ``ddec`` and computes only the scalar
    curvature of the negative log-likelihood along the **flux axis alone**:

    $$
    \\sigma_f = \\left(\\frac{\\partial^2 (-\\log L)}{\\partial f^2}\\right)^{-1/2}
    $$

    This is a 1-D (scalar) second derivative, not a matrix inversion.  It is
    appropriate when the position is held fixed (e.g. on a detection grid) and
    only the contrast uncertainty at that grid point is needed.  For the joint
    uncertainty over all parameters, use [`laplace_cov`][drpangloss.models.laplace_cov] instead.

    Parameters
    ----------
    flux : float
        Flux ratio value at which the local Laplace uncertainty is evaluated.
    dra : float
        Right ascension offset in mas (held fixed).
    ddec : float
        Declination offset in mas (held fixed).
    data_obj : OIData
        Object containing the data to be fitted.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][drpangloss.models.build_model]).
    params : list[str] or tuple[str, str, str], optional
        Parameter names corresponding to ``(dra, ddec, flux)``. Defaults to
        ``["dra", "ddec", "flux"]``.

    Returns
    -------
    float
        Scalar uncertainty in the contrast (standard deviation along flux axis).
    """

    if params is None:
        params = ["dra", "ddec", "flux"]

    values = np.asarray([dra, ddec, flux], dtype=float)
    return laplace_parameter_uncertainty(
        values,
        params,
        data_obj,
        model,
        target_param=params[-1],
    )


@_renamed_argument("model_class", "model")
def laplace_parameter_uncertainty(
    values, params, data_obj, model, target_param
):
    """Compute scalar Laplace uncertainty for one parameter with all others fixed.

    Parameters
    ----------
    values : array-like
        Parameter values at which to evaluate the curvature.
    params : list[str]
        Parameter names corresponding to ``values``.
    data_obj : OIData
        Data to fit.
    model : SourceModel or callable
        Template model or class, as for [`loglike`][drpangloss.models.loglike].
    target_param : str
        The parameter whose uncertainty is returned.

    Returns
    -------
    float
        ``(d² -log L / d target²)^(-1/2)``. It is NaN where the curvature is
        not positive, i.e. away from a likelihood maximum along
        ``target_param``.
    """
    params = list(params)
    if target_param not in params:
        raise ValueError(
            f"target_param '{target_param}' is not present in params={params}."
        )
    idx = params.index(target_param)
    values = np.asarray(values, dtype=float)

    objective = lambda x: -loglike(
        values.at[idx].set(x), params, data_obj, model
    )
    d2_axis = jax.grad(jax.grad(objective))(values[idx])
    return np.sqrt(1.0 / np.asarray(d2_axis, dtype=float))


@_renamed_argument("model_class", "model")
def fisher(values, params, data_obj, model, ridge=0.0):
    """Observed information (Hessian of ``-log L``) at a parameter point.

    At the maximum-likelihood point this approximates the Fisher matrix.

    Parameters
    ----------
    values : array-like
        Parameter vector at which to evaluate the local curvature.
    params : list[str]
        Parameter names corresponding to ``values``.
    data_obj : OIData
        Observational data object.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][drpangloss.models.build_model]).
    ridge : float, optional
        Diagonal regularization term.

    Returns
    -------
    array-like
        Observed information matrix, ``N x N`` for ``N = len(params)``.
    """
    objective = lambda vals: -loglike(vals, params, data_obj, model)
    return _fisher_matrix(
        objective, np.asarray(values, dtype=float), ridge=ridge
    )


def chi2ppf(p, df):
    """
    Percentile function for chi-square.

    For ``df=1`` (the path used in ``nsigma``), use the closed-form identity
    based on the standard normal quantile, i.e. square ``norm.ppf((p+1)/2)``.
    This remains JAX-native, differentiable, and fast.

    For ``df != 1``, this falls back to numpyro's gammaincinv backend.

    Parameters
    ----------
    p : array-like
        Percentile value.
    df : array-like
        Degrees of freedom.

    Returns
    -------
    array-like
        Corresponding chi2 value to the percentile.

    Notes
    -----
    ``p`` is clipped to ``[eps, 1 - eps]`` of its own floating-point type, so
    the result stays finite. Near ``p = 1`` this loses precision; to convert
    small tail probabilities, use [`nsigma`][drpangloss.models.nsigma], which works with the upper
    tail directly.
    """
    p = np.asarray(p, dtype=float)
    eps = np.finfo(p.dtype).eps
    p = np.clip(p, eps, 1.0 - eps)

    try:
        if float(onp.asarray(df)) == 1.0:
            z = jax.scipy.stats.norm.ppf((p + 1.0) / 2.0)
            return z**2
    except (
        TypeError,
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        pass

    from numpyro.distributions.util import gammaincinv

    return np.asarray(gammaincinv(df / 2.0, p), dtype=float) * 2.0


def nsigma(chi2r_test, chi2r_true, ndof):
    """
    Convert a reduced-chi-squared ratio to a Gaussian-equivalent significance.

    The statistic ``x = ndof * chi2r_test / chi2r_true`` is compared with a
    chi-squared distribution of ``ndof`` degrees of freedom, and its
    upper-tail probability is expressed as the equivalent two-sided Gaussian
    significance (as in Absil et al. 2011).

    Parameters
    ----------
    chi2r_test: float
        Reduced chi-squared of test model.
    chi2r_true: float
        Reduced chi-squared of true model.
    ndof: int
        Number of degrees of freedom.

    Returns
    -------
    nsigma: float
        Detection significance in Gaussian sigma.

    Notes
    -----
    The upper tail is computed directly (with the regularized incomplete
    gamma function) rather than as ``1 - cdf``, so significances stay finite
    and accurate far beyond 8σ, including in float32 (up to about 13σ).
    """
    x = ndof * chi2r_test / chi2r_true
    half_tail = 0.5 * jax.scipy.special.gammaincc(ndof / 2.0, x / 2.0)
    # Floor at the smallest normal number, so the result saturates (about
    # 13σ in float32, 37σ in float64) instead of becoming infinite.
    half_tail = np.maximum(half_tail, np.finfo(half_tail.dtype).tiny)
    return -jax.scipy.special.ndtri(half_tail)
