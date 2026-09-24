from typing import Any

import jax.numpy as np
import jax
from jax.scipy.signal import fftconvolve

import numpy as onp

import equinox as eqx
import zodiax as zx

from ._utils import (
    bessel_jn as bessel_jn,
    dtor as dtor,
    i2pi as i2pi,
    mas2rad as mas2rad,
    rad2mas as rad2mas,
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


class SourceModel(zx.Base):  # type: ignore[reportGeneralTypeIssues]
    """Base class for sky-brightness source models."""

    def model(self, u, v, wavel):
        """Evaluate complex visibilities on interferometric baselines."""
        raise NotImplementedError

    def render(self, npix=256, fov_mas=200.0):
        """Render an image-plane model in milliarcseconds."""
        raise NotImplementedError


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
        dra = -self.sep * np.sin(th)
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

    def render(self, npix=256, fov_mas=200.0):
        """
        Render a two-point-source approximation on a Cartesian image grid.
        """
        xx, yy = _image_coordinates(npix, fov_mas)
        th = self.pa * dtor
        ddec = self.sep * np.cos(th)
        dra = -1.0 * self.sep * np.sin(th)

        l2 = 1.0 / (self.contrast + 1.0)
        l1 = 1.0 - l2
        sigma = max(float(fov_mas) / float(npix), 1e-6)
        star = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
        comp = np.exp(
            -0.5 * (((xx - dra) / sigma) ** 2 + ((yy - ddec) / sigma) ** 2)
        )
        image = l1 * star + l2 * comp
        return image / np.sum(image)


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
        pa = np.mod(np.rad2deg(np.arctan2(-self.dra, self.ddec)), 360.0)
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
        return cvis_binary(uu, vv, self.ddec, self.dra, self.flux)

    def render(self, npix=256, fov_mas=200.0):
        """
        Render a two-point-source approximation on a Cartesian image grid.
        """
        xx, yy = _image_coordinates(npix, fov_mas)
        l2 = self.flux / (1.0 + self.flux)
        l1 = 1.0 - l2
        sigma = max(float(fov_mas) / float(npix), 1e-6)
        star = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
        comp = np.exp(
            -0.5
            * (
                ((xx - self.dra) / sigma) ** 2
                + ((yy - self.ddec) / sigma) ** 2
            )
        )
        image = l1 * star + l2 * comp
        return image / np.sum(image)


class GaussianDiskModel(SourceModel):
    """
    Resolved circular Gaussian disk companion added to an unresolved point
    source, using the same ``flux`` (companion/star) contrast convention as
    :class:`BinaryModelCartesian`.
    """

    sigma: jax.Array
    flux: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def __init__(self, sigma, flux, dra=0.0, ddec=0.0):
        self.sigma = np.asarray(sigma, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def __repr__(self):
        return (
            "GaussianDiskModel("
            f"sigma={self.sigma}, flux={self.flux}, "
            f"dra={self.dra}, ddec={self.ddec})"
        )

    def unpack_all(self):
        return self.sigma, self.flux, self.dra, self.ddec

    def model(self, u, v, wavel):
        uu, vv = u / wavel, v / wavel
        return cvis_gaussian_disk(
            uu, vv, self.sigma, self.flux, self.dra, self.ddec
        )

    def render(self, npix=256, fov_mas=200.0):
        xx, yy = _image_coordinates(npix, fov_mas)
        sigma_mas = np.maximum(self.sigma, 1e-9)
        point_sigma_mas = max(float(fov_mas) / float(npix), 1e-6)
        l2 = self.flux / (self.flux + 1.0)
        l1 = 1.0 - l2
        star = np.exp(
            -0.5 * ((xx / point_sigma_mas) ** 2 + (yy / point_sigma_mas) ** 2)
        )
        disk = np.exp(
            -0.5
            * (
                ((xx - self.dra) / sigma_mas) ** 2
                + ((yy - self.ddec) / sigma_mas) ** 2
            )
        )
        image = l1 * star + l2 * disk
        return _normalize_image(image)


class UniformDiskModel(SourceModel):
    """Centered or offset uniform (tophat) circular disk model."""

    ud: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def __init__(self, ud, dra=0.0, ddec=0.0):
        self.ud = np.asarray(ud, dtype=float)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def __repr__(self):
        return (
            f"UniformDiskModel(ud={self.ud}, dra={self.dra}, ddec={self.ddec})"
        )

    def unpack_all(self):
        return self.ud, self.dra, self.ddec

    def model(self, u, v, wavel):
        uu, vv = u / wavel, v / wavel
        return cvis_uniform_disk(uu, vv, self.ud, self.dra, self.ddec)

    def render(self, npix=256, fov_mas=200.0):
        xx, yy = _image_coordinates(npix, fov_mas)
        pixel_scale_mas = float(fov_mas) / float(npix)
        radius_mas = np.maximum(self.ud / 2.0, 0.5 * pixel_scale_mas)
        rr2 = (xx - self.dra) ** 2 + (yy - self.ddec) ** 2
        image = np.where(rr2 <= radius_mas**2, 1.0, 0.0)
        return _normalize_image(image)


class ModulatedGaussianRimModel(SourceModel):
    r"""
    Represents an azimuthally modulated, infinitely thin rim convolved with an
    isotropic 2D Gaussian, optionally inclined and rotated.

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
    az_amps : array-like
        1D array containing amplitude coefficients for cosine azimuthal modulations.
        The first element is the amplitude for the first-order modulation, the second
        for the second-order modulation, etc. An empty array gives an unmodulated,
        azimuthally symmetric rim.
    az_pas : array-like
        1D array containing position angles of the cosine azimuthal modulations, in
        degrees. The first element is the angle for the first-order modulation, the
        second for the second-order modulation, etc.
    dra : float or array-like
        Right-ascension offset of the rim's center in milliarcseconds.
    ddec : float or array-like
        Declination offset of the rim's center in milliarcseconds.

    Notes
    -----
    The intensity profile is separable into a symmetric radial profile and cosine
    azimuthal modulations, meaning the image intensity can be described in polar image
    coordinates as $I(r, \theta) = f(r) \left( 1 + \sum_{m=1}^{n}
    A_m \cos{(m(\theta - \mathrm{pa}_m))} \right)$, where $f(r)$ is a thin ring radial
    profile convolved with an isotropic Gaussian.

    This model is achromatic: it does not represent any spectral dependence.
    """

    diam: jax.Array
    fwhm: jax.Array
    inc: jax.Array
    pa: jax.Array
    az_amps: jax.Array
    az_pas: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def __init__(
        self,
        diam,
        fwhm,
        inc,
        pa,
        az_amps=(),
        az_pas=(),
        dra=0.0,
        ddec=0.0,
    ):
        self.diam = np.asarray(diam, dtype=float)
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.az_amps = np.asarray(az_amps, dtype=float)
        self.az_pas = np.asarray(az_pas, dtype=float)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def __repr__(self):
        return (
            "ModulatedGaussianRimModel("
            f"diam={self.diam}, fwhm={self.fwhm}, inc={self.inc}, "
            f"pa={self.pa}, az_amps={self.az_amps}, az_pas={self.az_pas}, "
            f"dra={self.dra}, ddec={self.ddec})"
        )

    def unpack_all(self):
        return (
            self.diam,
            self.fwhm,
            self.inc,
            self.pa,
            self.az_amps,
            self.az_pas,
            self.dra,
            self.ddec,
        )

    def model(self, u, v, wavel):
        uu, vv = u / wavel, v / wavel
        az_phis = self.az_pas - self.pa
        return cvis_gaussian_rim(
            uu,
            vv,
            self.dra,
            self.ddec,
            self.diam,
            self.fwhm,
            self.inc,
            self.pa,
            self.az_amps,
            az_phis,
        )

    def render(self, npix=256, fov_mas=200.0):
        xx, yy = _image_coordinates(npix, fov_mas)
        pixel_scale_mas = float(fov_mas) / float(npix)

        # Ring pattern is centered on (dra, ddec).
        xx_centered = xx - self.dra
        yy_centered = yy - self.ddec

        stretch = np.maximum(np.cos(self.inc * dtor), 1e-8)
        xx_ell, yy_ell = undo_elliptical_transf_coord(
            xx_centered, yy_centered, self.pa, stretch
        )
        r_ell = np.hypot(xx_ell, yy_ell)
        theta_ell = np.arctan2(xx_ell, yy_ell)

        ring = np.where(
            np.abs(r_ell - self.diam / 2.0) <= pixel_scale_mas, 1.0, 0.0
        )

        az_amps = np.concatenate([np.array([1.0]), self.az_amps])
        az_phis_rad = (
            np.concatenate([np.array([0.0]), self.az_pas - self.pa]) * dtor
        )
        az_orders = np.arange(az_amps.size)

        def _az_term(az_amp, az_phi_rad, az_order):
            return az_amp * np.cos(az_order * (theta_ell - az_phi_rad))

        az_factors = jax.vmap(_az_term)(az_amps, az_phis_rad, az_orders)
        ring = ring * np.sum(az_factors, axis=0)

        # The Gaussian PSF is centered on the pixel grid's own origin (not
        # on (dra, ddec)), so convolving with it blurs in place instead of
        # also shifting the image.
        sigma_mas = np.maximum(
            self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-9
        )
        r_psf = np.hypot(xx, yy)
        psf = np.exp(-0.5 * (r_psf / sigma_mas) ** 2)

        image = fftconvolve(ring, psf, mode="same")
        return _normalize_image(image)


class HarmonixModel(SourceModel):
    """
    Wrapper for external source models with harmonix-like visibility methods.
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
        Position angle in degrees.
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
    dra = -1 * mas2rad * (sep * np.sin(th))

    # decompose into two "luminosity"
    l2 = 1.0 / (contrast + 1)
    l1 = 1 - l2

    # phase-factor
    phi = np.exp(-i2pi * (u * dra + v * ddec))
    cvis = l1 + l2 * phi

    return cvis


def cvis_binary(u, v, ddec, dra, planet):
    # adapted from pymask
    """Compute complex visibilities for a Cartesian-parameterized binary model.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    ddec : float or array-like
        Declination offset in milliarcseconds.
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    planet : float or array-like
        Flux ratio of the companion.

    Returns
    -------
    array-like
        Complex visibility samples.
    """

    star = 1

    # normalize visibilities so total power is 1
    p3 = star / (star + planet)
    p2 = planet / (star + planet)

    # relative locations
    ddec = ddec * np.pi / (180.0 * 3600.0 * 1000.0)
    dra = dra * np.pi / (180.0 * 3600.0 * 1000.0)
    phi_r = np.cos(-2 * np.pi * (u * dra + v * ddec))
    phi_i = np.sin(-2 * np.pi * (u * dra + v * ddec))

    cvis = p3 + p2 * phi_r + p2 * phi_i * 1.0j

    return cvis


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
    convention shared with :func:`cvis_binary`.
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

    envelope = np.where(
        kernel == 0, 1.0 + 0j, (2.0 * bessel_jn(1, kernel)[1]) / kernel + 0j
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
            * bessel_vals[az_order, :]
        )

    azmod_cvis_terms = jax.vmap(
        _azmod_cvis_term, in_axes=(0, 0, 0), out_axes=0
    )(az_amps, az_phis_rad, az_orders)

    return np.sum(azmod_cvis_terms, axis=0)


def _cvis_gaussian_envelope(u, v, fwhm):
    """Complex visibility envelope of a centered isotropic 2D Gaussian PSF, used
    as the convolution kernel in :func:`cvis_gaussian_rim`. Not offered as a public
    function: unlike :func:`cvis_gaussian_disk`, this is a plain Gaussian envelope
    with no point-source/flux-contrast mixture.
    """
    fwhm_rad = fwhm * mas2rad
    base_norm = np.hypot(u, v)
    return (
        np.exp(-(np.pi**2) * fwhm_rad**2 * base_norm**2 / (4.0 * np.log(2.0)))
        + 0j
    )


def cvis_gaussian_rim(u, v, dra, ddec, diam, fwhm, inc, pa, az_amps, az_phis):
    """Compute complex visibilities for a (modulated) rim, consisting of a radial
    Dirac delta ring (infinitely thin) subsequently convolved with an isotropic 2D
    Gaussian.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    dra : float or array-like
        Right-ascension offset of the rim in milliarcseconds.
    ddec : float or array-like
        Declination offset of the rim in milliarcseconds.
    diam : float or array-like
        Diameter of the rim in milliarcseconds.
    fwhm : float or array-like
        Gaussian FWHM of the rim in milliarcseconds.
    inc : float or array-like
        Apparent inclination of the rim in degrees.
    pa : float or array-like
        Position angle of the rim's projected major axis in degrees, measured North to
        East (i.e. counter-clockwise in conventional astronomical image orientation).
    az_amps : array-like
        1D array containing amplitude coefficients for cosine azimuthal modulations.
        The first element is seen as the amplitude for the first-order modulation,
        the second as the amplitude for the second-order modulation, etc.
    az_phis : array-like
        1D array containing offset angles of the rim's cosine azimuthal modulations,
        relative to the position angle of the rim's projected major axis, in
        degrees. The first element is seen as the offset for the first-order
        modulation, the second for the second-order modulation, etc.

    Returns
    -------
    array-like
        Complex visibility samples.
    """
    inc_rad, dra_rad, ddec_rad = inc * dtor, dra * mas2rad, ddec * mas2rad

    # Transform spatial frequency coordinates to the frame of reference where the
    # model rim is uninclined and the major axis is pointed North.
    stretch_factor = np.maximum(np.cos(inc_rad), 1e-8)
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa, stretch_factor)

    # Complex visibility of the (inclined) Dirac delta modulated ring.
    cvis = cvis_radial_dirac_delta_modulated(
        ut, vt, diam / 2.0, az_amps, az_phis
    )

    # Effect of convolution in image-plane with an isotropic Gaussian of the given
    # FWHM, evaluated in the original (un-transformed) image frame of reference.
    cvis = cvis * _cvis_gaussian_envelope(u, v, fwhm)

    # Apply offset phase-factor.
    phi = np.exp(-i2pi * (u * dra_rad + v * ddec_rad))
    return cvis * phi


def model_loglike(model_object, data_obj):
    """Evaluate a Gaussian log likelihood for an instantiated model object."""
    model_data = data_obj.model(model_object)
    data, errors = data_obj.flatten_data()
    return jax.scipy.stats.norm.logpdf(
        model_data, loc=data, scale=errors
    ).sum()


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


def loglike(values, params, data_obj, model_class):
    """
    Abstract log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    """

    param_dict = dict(zip(params, values))

    return model_loglike(model_class(**param_dict), data_obj)


def loglike_nosignal(values, params, data_obj, model_class):
    """
    Abstract null log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    """

    param_dict = dict(zip(params, values))

    model_data = data_obj.model(model_class(**param_dict))
    _, errors = data_obj.flatten_data()
    unity_cvis = np.ones_like(data_obj.u, dtype=complex)
    data = data_obj.standardize_model(unity_cvis)

    return jax.scipy.stats.norm.logpdf(
        model_data, loc=data, scale=errors
    ).sum()


def laplace_cov(values, params, data_obj, model_class):
    """
    Compute the full Laplace covariance matrix for all model parameters jointly.

    Computes the inverse of the Hessian of the negative log-likelihood with
    respect to all parameters in ``params`` simultaneously, returning an
    ``N x N`` covariance matrix (where ``N = len(params)``).

    .. note::
        This function returns the *full* covariance matrix over all ``N``
        parameters.  To obtain only the marginal flux uncertainty at a fixed
        position, use :func:`laplace_contrast_uncertainty` instead.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    array-like
        ``N x N`` covariance matrix, where ``N = len(params)``.
    """

    objective = lambda vals: -loglike(vals, params, data_obj, model_class)
    return _laplace_covariance(objective, np.asarray(values, dtype=float))


def laplace_contrast_uncertainty(
    flux, dra, ddec, data_obj, model_class, params=None
):
    """
    Compute the Laplace uncertainty in flux at a fixed sky position.

    Unlike :func:`laplace_cov`, which inverts the *full* N-parameter Hessian,
    this function **fixes** ``dra`` and ``ddec`` and computes only the scalar
    curvature of the negative log-likelihood along the **flux axis alone**:

    .. math::

        \\sigma_f = \\left(\\frac{\\partial^2 (-\\log L)}{\\partial f^2}\\right)^{-1/2}

    This is a 1-D (scalar) second derivative, not a matrix inversion.  It is
    appropriate when the position is held fixed (e.g. on a detection grid) and
    only the contrast uncertainty at that grid point is needed.  For the joint
    uncertainty over all parameters, use :func:`laplace_cov` instead.

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
    model_class : class
        Model class to be fitted to the data.
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
        model_class,
        target_param=params[-1],
    )


def laplace_parameter_uncertainty(
    values, params, data_obj, model_class, target_param
):
    """Compute scalar Laplace uncertainty for one parameter with all others fixed."""
    params = list(params)
    if target_param not in params:
        raise ValueError(
            f"target_param '{target_param}' is not present in params={params}."
        )
    idx = params.index(target_param)
    values = np.asarray(values, dtype=float)

    objective = lambda x: -loglike(
        values.at[idx].set(x), params, data_obj, model_class
    )
    d2_axis = jax.grad(jax.grad(objective))(values[idx])
    return np.sqrt(1.0 / np.asarray(d2_axis, dtype=float))


def fisher(values, params, data_obj, model_class, ridge=0.0):
    """Approximate the local Fisher matrix at a parameter point.

    Parameters
    ----------
    values : array-like
        Parameter vector at which to evaluate the local curvature.
    params : list[str]
        Parameter names corresponding to ``values``.
    data_obj : OIData
        Observational data object.
    model_class : class
        Model class used to evaluate the likelihood.
    ridge : float, optional
        Diagonal regularization term.

    Returns
    -------
    array-like
        Fisher information matrix.
    """
    objective = lambda vals: -loglike(vals, params, data_obj, model_class)
    return _fisher_matrix(
        objective, np.asarray(values, dtype=float), ridge=ridge
    )


def chi2ppf(p, df):
    """
    Percentile function for chi-square.

    For ``df=1`` (the path used in ``nsigma``), use the closed-form identity
    based on the standard normal quantile, i.e. square ``norm.ppf((p+1)/2)``.
    This remains JAX-native,
    differentiable, and fast.

    For ``df != 1``, this falls back to numpyro's gammaincinv backend when
    available.

    Parameters
    ----------
    p : array-like
        Percentile value
    df : array-like
        Degrees of freedom

    Returns
    -------
    array-like
        Corresponding chi2 value to the percentile
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    try:
        if float(onp.asarray(df)) == 1.0:
            z = jax.scipy.stats.norm.ppf((p + 1.0) / 2.0)
            return z**2
    except Exception:
        pass

    from numpyro.distributions.util import gammaincinv

    return np.asarray(gammaincinv(df / 2.0, p), dtype=float) * 2.0


def nsigma(chi2r_test, chi2r_true, ndof):
    """
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
        Detection significance.
    """

    percentile = jax.scipy.stats.chi2.cdf(ndof * chi2r_test / chi2r_true, ndof)
    nsigma = np.sqrt(chi2ppf(percentile, 1.0))

    return nsigma
