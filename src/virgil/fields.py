"""Gaussian-process log-brightness fields for pixel images.

A [`GaussianField`][virgil.fields.GaussianField] can stand in for the
``log_brightness`` array of an [`Image`][virgil.models.Image]. The image's
log-brightness is then a stationary Gaussian process with a Matérn-like
spectrum, written in its whitened form: independent standard-normal
``latent`` coefficients on the image's cosine (DCT-II) basis. This is the
basis that diagonalizes the Laplacian with reflecting boundaries. Fitting the
latents under a standard-normal prior (from
[`image_priors`][virgil.imaging.image_priors]) is MAP estimation with a
GP prior, and the same parameterization suits sampling.

The construction, a Gaussian-process field with a stationary spectrum on a
whitened basis, is abstracted from the NIFTy library (Edenhofer et al.
2024, https://arxiv.org/abs/2402.16683) and the correlated field model used
for M87* by Arras et al. (2022, Nature Astron. 6, 259,
https://arxiv.org/abs/2002.05218), within the framework of information
field theory (Enßlin et al. 2009, https://arxiv.org/abs/0806.3474; review:
Enßlin 2019, https://arxiv.org/abs/1804.03350). The first NIFTy is Selig et
al. (2013, https://arxiv.org/abs/1301.4499). That quadratic regularizers
are Gaussian priors is the view of Thiébaut & Young (2017,
https://arxiv.org/abs/1708.08390). No code is taken from NIFTy.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zdx
from jax.scipy.fft import idctn

from ._geometry import pixel_offsets, undo_elliptical_transf_coord
from ._utils import concrete
from .models import _check_nu, _gp_factor, _log_azimuthal_spectrum

__all__ = [
    "GaussianField",
    "MultipoleField",
    "field_spectrum",
    "multipole_field_priors",
    "polar_interpolate",
]

__provisional__ = (
    "MultipoleField",
    "multipole_field_priors",
)


def _check_positive(name, value):
    """Reject a concrete value that is not finite and positive."""
    value = concrete(value)
    if value is not None and not (
        onp.all(onp.isfinite(value)) and onp.all(value > 0.0)
    ):
        raise ValueError(f"{name} must be finite and positive, not {value}.")


def field_spectrum(shape, pixel_scale_mas, sigma, length_mas, order=2):
    """Prior variances of a field's cosine coefficients.

    ``S_jk ∝ (κ² + λ_jk)^(-order)``, with ``κ = 1 / length_mas`` and
    ``λ_jk`` the eigenvalues of the reflecting-boundary (Neumann) Laplacian
    on the pixel grid, ``(2/h)² [sin²(πj/2n) + sin²(πk/2m)]``. With two
    lengths ``(ℓ_row, ℓ_col)`` the field is anisotropic,
    ``S_jk ∝ (1 + ℓ_row² λ_j + ℓ_col² λ_k)^(-order)``: correlated over
    ``ℓ_row`` along the image's columns (down a column, from row to row) and
    over ``ℓ_col`` along its rows. One length gives the isotropic case. The constant
    mode is set to zero, since a softmax image does not depend on it, and
    the spectrum is scaled so that the field's variance, averaged over
    pixels, is ``sigma²``.

    Parameters
    ----------
    shape : tuple of int
        ``(nrow, ncol)``.
    pixel_scale_mas : float
        Pixel size ``h`` in milliarcseconds.
    sigma : float or array-like
        Standard deviation of the field, in units of log-brightness.
    length_mas : float or array-like, shape () or (2,)
        Correlation length ``1/κ`` in milliarcseconds, or the pair
        ``(ℓ_row, ℓ_col)`` for an anisotropic field.
    order : int, optional
        Exponent of the spectrum (default 2). ``order=1`` makes the prior
        exactly a total-squared-variation plus L2 penalty on the field.

    Returns
    -------
    jax.Array
        Variances, of shape ``shape``.
    """
    _check_positive("pixel_scale_mas", pixel_scale_mas)
    _check_positive("length_mas", length_mas)
    eigen = [
        (2.0 / pixel_scale_mas * np.sin(np.pi * np.arange(n) / (2 * n))) ** 2
        for n in shape
    ]
    length_mas = np.asarray(length_mas)
    if length_mas.ndim == 0:
        laplacian = eigen[0][:, None] + eigen[1][None, :]
        spectrum = (length_mas**-2.0 + laplacian) ** (-float(order))
    elif length_mas.shape == (2,):
        stretched = (
            length_mas[0] ** 2 * eigen[0][:, None]
            + length_mas[1] ** 2 * eigen[1][None, :]
        )
        spectrum = (1.0 + stretched) ** (-float(order))
    else:
        raise ValueError(
            "length_mas must be one length or a pair (row, column), not "
            f"shape {length_mas.shape}."
        )
    spectrum = spectrum.at[0, 0].set(0.0)
    return sigma**2 * spectrum * spectrum.size / np.sum(spectrum)


class GaussianField(zdx.Base):  # type: ignore[reportGeneralTypeIssues]
    r"""A Gaussian-process log-brightness for an [`Image`][virgil.models.Image].

    The log-brightness is

    $$\eta = \log\left(\frac{\mu}{\max\mu} + \epsilon\right)
    + \mathrm{IDCT}\left[\sqrt{S} \odot z\right],$$

    with the orthonormal inverse DCT-II, the spectrum ``S`` of
    [`field_spectrum`][virgil.fields.field_spectrum], latent
    coefficients ``z`` (``latent``) and an optional positive template image
    ``μ`` (``mean``, floored at ``ε`` = ``mean_floor`` of its peak). With a
    standard-normal prior on ``latent``, ``η`` is a Gaussian process with
    standard deviation ``sigma`` and correlation length ``length_mas``
    about the template. Two lengths make it anisotropic: with the Image's
    ``rotation_deg`` set to a structure's position angle, ``(ℓ_along,
    ℓ_across)`` correlates the field along the structure (the grid's "up"
    axis) and across it, which suits filaments. ``latent = 0`` gives the template raised by the
    floor, ``μ/max μ + ε``, as an image.

    Use it in place of an Image's log-brightness, with priors from
    [`image_priors`][virgil.imaging.image_priors]:

    ```python
    field = GaussianField(np.zeros((32, 32)), sigma=2.0, length_mas=3.0,
                          mean=start.env.brightness)
    env = Image(field, pixel_scale_mas=0.6, flux=0.3)
    ```

    Fix ``sigma`` and ``length_mas``, or sample them: their MAP values are
    biased (``fit`` warns), and Stage 5's evidence helpers choose them from
    the data.

    Parameters
    ----------
    latent : array-like, shape (nrow, ncol)
        Whitened cosine coefficients.
    sigma : float or array-like, optional
        Standard deviation of the field (default 1).
    length_mas : float or array-like, optional
        Correlation length in milliarcseconds (default 1), or a pair
        ``(ℓ_row, ℓ_col)``: correlated over ``ℓ_row`` along the image's
        vertical axis and ``ℓ_col`` along its horizontal axis.
    order : int, optional
        Exponent of the spectrum (default 2).
    mean : array-like, optional
        Positive template image, of the same shape (default: none, a flat
        template).
    mean_floor : float, optional
        Floor on the template, as a fraction of its peak (default 1e-3).
    """

    latent: jax.Array
    sigma: jax.Array
    length_mas: jax.Array
    mean: jax.Array | None
    order: int = eqx.field(static=True)
    mean_floor: float = eqx.field(static=True)

    def __init__(
        self,
        latent,
        sigma=1.0,
        length_mas=1.0,
        order=2,
        mean=None,
        mean_floor=1e-3,
    ):
        self.latent = np.asarray(latent, dtype=float)
        if self.latent.ndim != 2:
            raise ValueError("latent must be a 2D array.")
        self.sigma = np.asarray(sigma, dtype=float)
        self.length_mas = np.asarray(length_mas, dtype=float)
        value = concrete(self.sigma)
        if value is not None and not (
            onp.all(onp.isfinite(value)) and onp.all(value >= 0.0)
        ):
            raise ValueError(
                f"sigma must be finite and non-negative, not {value}."
            )
        _check_positive("length_mas", self.length_mas)
        if self.length_mas.shape not in ((), (2,)):
            raise ValueError(
                "length_mas must be one length or a pair (row, column), "
                f"not shape {self.length_mas.shape}."
            )
        self.order = int(order)
        if self.order < 1:
            raise ValueError(f"order must be a positive integer, not {order}.")
        if mean is not None:
            mean = np.asarray(mean, dtype=float)
            if mean.shape != self.latent.shape:
                raise ValueError(
                    f"mean has shape {mean.shape}, but latent has shape "
                    f"{self.latent.shape}."
                )
            value = concrete(mean)
            if value is not None and not (
                onp.all(onp.isfinite(value)) and value.max() > 0.0
            ):
                raise ValueError(
                    "mean must be a finite template with a positive peak."
                )
        self.mean = mean
        _check_positive("mean_floor", mean_floor)
        self.mean_floor = float(mean_floor)

    @property
    def shape(self):
        """The image's shape, ``(nrow, ncol)``."""
        return self.latent.shape

    def evaluate(self, pixel_scale_mas):
        """The log-brightness ``η`` on pixels of ``pixel_scale_mas``."""
        spectrum = field_spectrum(
            self.shape,
            pixel_scale_mas,
            self.sigma,
            self.length_mas,
            self.order,
        )
        # The constant mode has zero variance; a double where keeps the
        # gradient of its square root finite.
        positive = spectrum > 0.0
        amplitude = np.where(
            positive, np.sqrt(np.where(positive, spectrum, 1.0)), 0.0
        )
        eta = idctn(amplitude * self.latent, type=2, norm="ortho")
        if self.mean is not None:
            template = np.maximum(self.mean, 0.0) / np.max(self.mean)
            eta = eta + np.log(template + self.mean_floor)
        return eta


# ---------------------------------------------------------------------------
# MultipoleField: a multipole-kernel residual field aligned with a disk
# ---------------------------------------------------------------------------

_DEG = np.pi / 180.0


def _radial_factor(nodes, length, nu):
    """Cholesky factor of the radial correlation on ``nodes``.

    This is [`RadialGP`][virgil.models.RadialGP]'s factor (one definition,
    ``models._gp_factor``), with a jitter of at least ``10 n`` times the
    machine epsilon of the working precision (1.5e-5 for 128 nodes in
    float32, below the default 1e-6 in float64), so that the factorization
    of a Matérn matrix whose length is many node spacings stays finite in
    float32 (a jitter of 1e-6 gives NaN for 128 nodes, ``nu`` = 5/2 and a
    length of 5 spacings or more).
    """
    n = nodes.shape[0]
    return _gp_factor(
        nodes,
        length,
        nu,
        max(1e-6, 10.0 * n * float(np.finfo(nodes.dtype).eps)),
    )


def _azimuthal_filter(latent, angle, nu):
    """Apply the azimuthal square root of the periodic kernel, along axis 1.

    The spectral density at the integer order ``m`` is the one that
    [`MultipoleKernel`][virgil.models.MultipoleKernel] gives the harmonics,
    ``S(m) = [1 + (m angle)^2 / (2 nu)]^(-nu - 1/2)`` (``angle`` is its
    ``ell_theta / r``), the same function. The kernel is circulant on the
    ``n_theta`` azimuths, so the filter is two FFTs, scaled so that the
    mean of ``S`` over the orders is one and the variance of the filtered
    white noise is exactly one.
    """
    n_theta = latent.shape[1]
    m = np.minimum(np.arange(n_theta), n_theta - np.arange(n_theta))
    # In log space, so that no order underflows (a gradient of NaN).
    log_power = _log_azimuthal_spectrum(m, angle, nu)
    amplitude = np.exp(
        0.5
        * (
            log_power
            - jax.scipy.special.logsumexp(log_power)
            + np.log(n_theta)
        )
    )
    return np.real(np.fft.ifft(amplitude * np.fft.fft(latent, axis=1), axis=1))


def polar_interpolate(values, r_min, spacing, r, theta, fill=0.0):
    """Bilinear interpolation of a polar grid at ``(r, theta)``.

    ``values[k, j]`` sits at radius ``r_min + k * spacing`` and azimuth
    ``2π j / n_theta`` (periodic). Outside the radial range the field falls
    linearly to ``fill`` over one ``spacing`` and is ``fill`` beyond, so
    there is no edge discontinuity. The error is second order in the grid
    spacings.

    Parameters
    ----------
    values : array-like, shape (n_r, n_theta)
    r_min, spacing : float
        Radius of the first node and the node spacing, in milliarcseconds.
    r, theta : array-like
        Radii (mas) and azimuths (radians); any common shape.
    fill : float, optional
        The value outside the grid (default 0).

    Returns
    -------
    jax.Array
        The interpolated values, of the shape of ``r``.
    """
    values = np.asarray(values)
    n_r, n_theta = values.shape
    edge = np.full((1, n_theta), fill, dtype=values.dtype)
    padded = np.concatenate([edge, values, edge], axis=0)
    f = np.clip((r - r_min) / spacing + 1.0, 0.0, n_r + 1.0 - 1e-9)
    i0 = np.floor(f)
    wr = f - i0
    i0 = i0.astype(int)
    t = np.mod(theta, 2.0 * np.pi) * (n_theta / (2.0 * np.pi))
    j0 = np.floor(t)
    wt = t - j0
    j0 = j0.astype(int) % n_theta
    j1 = (j0 + 1) % n_theta
    low = padded[i0, j0] * (1.0 - wt) + padded[i0, j1] * wt
    high = padded[i0 + 1, j0] * (1.0 - wt) + padded[i0 + 1, j1] * wt
    return low * (1.0 - wr) + high * wr


class MultipoleField(zdx.Base):  # type: ignore[reportGeneralTypeIssues]
    r"""A multipole-kernel residual field for an [`Image`][virgil.models.Image].

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The log-brightness of an Image is a Gaussian process on a polar grid
    in the frame of a disk: ``n_r`` radii ``r_k`` from ``r_min_mas`` to
    ``r_max_mas`` and ``n_theta`` azimuths, in the disk plane (deprojected
    by ``inc``, rotated by ``pa`` and centred on ``(dra, ddec)``), then
    interpolated onto the pixels. The prior is separable,

    $$\mathrm{Cov}[\eta(r, \theta), \eta(r', \theta')] =
    \sigma^2\, k_r(|r - r'|;\, \ell_r)\, k_\theta(\Delta\theta;\, \ell_\theta),$$

    with a Matérn kernel in radius (that of
    [`RadialGP`][virgil.models.RadialGP]) and a periodic Matérn kernel in
    azimuth, given by the spectral density that
    [`MultipoleKernel`][virgil.models.MultipoleKernel] gives the harmonics of a
    [`MultipoleDisk`][virgil.models.MultipoleDisk],
    $S(m) \propto [1 + (m\,\ell_\theta)^2 / 2\nu_\theta]^{-\nu_\theta - 1/2}$
    at the integer order $m$ ($\ell_\theta$ in radians, the angle
    ``ell_theta / r`` of a ``MultipoleKernel``). A long ``length_theta`` favours rings
    and arcs without imposing them, and a short ``length_r`` allows radial
    structure such as gaps. Use it, in the frame of a
    [`MultipoleDisk`][virgil.models.MultipoleDisk], for emission that the
    parametric disk misses, as a residual field aligned with the disk,
    instead of a Cartesian [`GaussianField`][virgil.fields.GaussianField]
    that knows nothing about the geometry.

    The parameters are whitened: independent standard-normal ``latent``
    values, ``η = σ L_r Z F_θ`` with ``L_r`` the Cholesky factor of the radial
    correlation and ``F_θ`` the (circulant) square root of the azimuthal one,
    scaled so that ``σ`` is the standard deviation of the field on the grid.
    Fitting the latents under a standard-normal prior
    ([`image_priors`][virgil.imaging.image_priors]) is MAP estimation, and
    [`log_evidence`][virgil.imaging.log_evidence] and the numpyro model work
    as for a ``GaussianField``. Positivity is that of the Image: its
    brightness is the softmax of ``η``. Beyond the polar grid (outside
    ``r_max_mas``, or inside ``r_min_mas``) ``η`` falls to ``outside``
    (default -20), so those pixels are dark and the Image's flux goes to the
    grid's ellipse; a value of 0 would leave a flat background there that
    the latents cannot remove. Give ``σ`` and the two lengths
    bounded log-uniform priors with
    [`multipole_field_priors`][virgil.fields.multipole_field_priors], so that the
    evidence is proper; their MAP values are biased (``fit`` warns).

    ```python
    field = MultipoleField(np.zeros((24, 32)), sigma=1.0, length_r=4.0,
                       length_theta=1.0, inc=50.0, pa=30.0,
                       image_shape=(64, 64), r_max_mas=40.0)
    env = Image(field, pixel_scale_mas=1.5, flux=0.3)
    ```

    Parameters
    ----------
    latent : array-like, shape (n_r, n_theta)
        Whitened coefficients. ``n_r`` and ``n_theta`` set the resolution of
        the polar grid (``n_theta`` should be about ``2π r_max_mas`` over
        the pixel scale at most).
    image_shape : tuple of int
        ``(nrow, ncol)`` of the Image's pixel grid.
    r_max_mas : float
        Outermost radius of the polar grid in milliarcseconds, in the disk
        plane. The field fades to zero over one node spacing beyond it.
    r_min_mas : float, optional
        Innermost radius (default 0).
    sigma : float or array-like, optional
        Standard deviation of the field on the polar grid (default 1).
    length_r : float or array-like, optional
        Radial correlation length in milliarcseconds (default: five node
        spacings).
    length_theta : float or array-like, optional
        Azimuthal correlation length in radians (default 1).
    inc, pa : float or array-like, optional
        Inclination and position angle of the disk's projected major axis,
        in degrees (North to East), in the Image's grid frame.
    dra, ddec : float or array-like, optional
        Centre of the disk relative to the Image's centre, in
        milliarcseconds (East and North positive).
    nu : {0.5, 1.5, 2.5, inf}, optional
        Smoothness of the radial Matérn kernel (default 1.5), and of the
        azimuthal one unless ``nu_theta`` is given.
    nu_theta : {0.5, 1.5, 2.5, inf} or None, optional
        Smoothness of the azimuthal kernel (default: ``nu``).
    geometry_from : str or None, optional
        Name of a sibling component of the [`System`][virgil.models.System]
        (a [`MultipoleDisk`][virgil.models.MultipoleDisk], or anything with ``inc``,
        ``pa``, ``dra`` and ``ddec``) whose geometry the field follows when
        the System is evaluated (default ``None``: the field's own). See
        Notes.
    mean : array-like, optional
        Positive template image of shape ``image_shape``, as for
        [`GaussianField`][virgil.fields.GaussianField] (default: flat).
    mean_floor : float, optional
        Floor on the template, as a fraction of its peak (default 1e-3).
    outside : float, optional
        Log-brightness beyond the polar grid (default -20, effectively
        dark).

    Notes
    -----
    By default the geometry parameters belong to the field: they are copies
    of a ``MultipoleDisk``'s (see [`from_disk`][virgil.fields.MultipoleField.from_disk]),
    and if both are fitted the caller must sample the same values for both.
    With ``geometry_from="disk"`` they are tied instead. The field then has
    no ``inc``, ``pa``, ``dra`` or ``ddec`` parameters (they are ``None``:
    nothing to fit, no path to set), and when the enclosing ``System`` is
    evaluated (``model``, ``render``, the likelihood, ``fit``,
    ``log_evidence``) it takes the geometry of its sibling ``disk``, in the
    Image's grid frame: ``inc`` and ``pa - rotation_deg`` of the disk, and the
    disk's centre relative to the Image's, along the Image's rotated axes. The
    gradient therefore flows into the disk's parameters. The ties are
    resolved by [`System`][virgil.models.System], so they hold for siblings
    of one System (not for a disk in another nested System), and the Image
    is tied only inside a System: evaluated alone (``Image.brightness``,
    or a regularizer given the bare Image) the field uses the fallback
    geometry it was built with, which is also what ``from_disk`` stores.
    `System.resolved` returns the
    System with the ties baked in, for code that reads the pixels directly.
    The pixel scale comes from the Image. The azimuth is measured from the
    projected major axis towards ``pa + 90``, as for ``MultipoleDisk``'s harmonics.

    References
    ----------
    The whitened Gaussian-process latents of a radial profile follow FRAP
    (Yoshida et al. 2026, arXiv:2607.12618; https://github.com/tomyoshida/frap),
    and the polar-grid representation of a disk's brightness follows frank
    (Jennings et al. 2020, MNRAS 495, 3209;
    https://github.com/discsim/frank). The code is a new implementation;
    none was copied.
    """

    latent: jax.Array
    sigma: jax.Array
    length_r: jax.Array
    length_theta: jax.Array
    inc: jax.Array | None
    pa: jax.Array | None
    dra: jax.Array | None
    ddec: jax.Array | None
    mean: jax.Array | None
    image_shape: tuple = eqx.field(static=True)
    r_min_mas: float = eqx.field(static=True)
    r_max_mas: float = eqx.field(static=True)
    nu: float = eqx.field(static=True)
    nu_theta: float = eqx.field(static=True)
    geometry_from: str | None = eqx.field(static=True)
    fallback: tuple = eqx.field(static=True)
    mean_floor: float = eqx.field(static=True)
    outside: float = eqx.field(static=True)

    def __init__(
        self,
        latent,
        image_shape,
        r_max_mas,
        r_min_mas=0.0,
        sigma=1.0,
        length_r=None,
        length_theta=1.0,
        inc=0.0,
        pa=0.0,
        dra=0.0,
        ddec=0.0,
        nu=1.5,
        mean=None,
        mean_floor=1e-3,
        outside=-20.0,
        nu_theta=None,
        geometry_from=None,
    ):
        self.latent = np.asarray(latent, dtype=float)
        if self.latent.ndim != 2 or min(self.latent.shape) < 2:
            raise ValueError("latent must be a 2D array of at least 2 x 2.")
        self.image_shape = tuple(int(n) for n in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError(
                f"image_shape must be (nrow, ncol): {image_shape}"
            )
        self.r_min_mas = float(r_min_mas)
        self.r_max_mas = float(r_max_mas)
        if not (
            onp.isfinite(self.r_max_mas)
            and 0.0 <= self.r_min_mas < self.r_max_mas
        ):
            raise ValueError(
                "need finite 0 <= r_min_mas < r_max_mas, not "
                f"{r_min_mas} and {r_max_mas}."
            )
        self.nu = _check_nu(nu, "MultipoleField")
        self.nu_theta = _check_nu(
            nu if nu_theta is None else nu_theta, "MultipoleField"
        )
        if length_r is None:
            length_r = 5.0 * self.spacing
        self.sigma = np.asarray(sigma, dtype=float)
        self.length_r = np.asarray(length_r, dtype=float)
        self.length_theta = np.asarray(length_theta, dtype=float)
        _check_positive("sigma", self.sigma)
        _check_positive("length_r", self.length_r)
        _check_positive("length_theta", self.length_theta)
        for name in ("sigma", "length_r", "length_theta"):
            if np.shape(getattr(self, name)) != ():
                raise ValueError(f"{name} must be a scalar.")
        geometry = tuple(
            np.asarray(g, dtype=float) for g in (inc, pa, dra, ddec)
        )
        if geometry_from is None:
            self.geometry_from = None
            self.fallback = ()
            self.inc, self.pa, self.dra, self.ddec = geometry
        else:
            # Tied: the geometry is the disk's, so it has no parameters; the
            # values given are only the fallback for an Image evaluated
            # outside a System, and must be concrete.
            if not isinstance(geometry_from, str) or not geometry_from:
                raise ValueError(
                    f"geometry_from must be a component name: {geometry_from!r}"
                )
            values = tuple(concrete(g) for g in geometry)
            if any(v is None or v.shape != () for v in values):
                raise ValueError(
                    "with geometry_from, inc, pa, dra and ddec are only the "
                    "fallback geometry and must be concrete scalars."
                )
            self.geometry_from = geometry_from
            self.fallback = tuple(float(v) for v in values)
            self.inc = self.pa = self.dra = self.ddec = None
        if mean is not None:
            mean = np.asarray(mean, dtype=float)
            if mean.shape != self.image_shape:
                raise ValueError(
                    f"mean has shape {mean.shape}, but image_shape is "
                    f"{self.image_shape}."
                )
            value = concrete(mean)
            if value is not None and not (
                onp.all(onp.isfinite(value)) and value.max() > 0.0
            ):
                raise ValueError(
                    "mean must be a finite template with a positive peak."
                )
        self.mean = mean
        _check_positive("mean_floor", mean_floor)
        self.mean_floor = float(mean_floor)
        self.outside = float(outside)

    @classmethod
    def from_disk(
        cls,
        disk,
        latent,
        image_shape,
        r_max_mas=None,
        geometry_from=None,
        **kwargs,
    ):
        """A field in the frame of ``disk`` (its ``inc``, ``pa``, ``dra``, ``ddec``).

        ``r_max_mas`` defaults to the outer edge of the disk's profile
        (``disk.profile.extent()``) plus 20 per cent. The values are copied
        unless ``geometry_from`` is the disk's name in the System the Image
        will join, which ties them: see the notes of
        [`MultipoleField`][virgil.fields.MultipoleField].
        """
        if r_max_mas is None:
            r_max_mas = 1.2 * float(disk.profile.extent()[1])
        return cls(
            latent,
            image_shape,
            r_max_mas,
            inc=disk.inc,
            pa=disk.pa,
            dra=disk.dra,
            ddec=disk.ddec,
            geometry_from=geometry_from,
            **kwargs,
        )

    def with_geometry(self, inc, pa, dra, ddec):
        """This field untied, with the given geometry as its own parameters.

        What [`System`][virgil.models.System] calls to resolve a tie; the
        arguments are the grid-frame ``inc``, ``pa`` (degrees) and centre
        (milliarcseconds).
        """
        return type(self)(
            self.latent,
            self.image_shape,
            self.r_max_mas,
            r_min_mas=self.r_min_mas,
            sigma=self.sigma,
            length_r=self.length_r,
            length_theta=self.length_theta,
            inc=inc,
            pa=pa,
            dra=dra,
            ddec=ddec,
            nu=self.nu,
            mean=self.mean,
            mean_floor=self.mean_floor,
            outside=self.outside,
            nu_theta=self.nu_theta,
        )

    def _geometry(self):
        if self.inc is not None:
            return self.inc, self.pa, self.dra, self.ddec
        return tuple(np.asarray(g, dtype=float) for g in self.fallback)

    @property
    def shape(self):
        """The shape of the latents, ``(n_r, n_theta)``."""
        return self.latent.shape

    @property
    def spacing(self):
        """Radial node spacing in milliarcseconds."""
        return (self.r_max_mas - self.r_min_mas) / (self.latent.shape[0] - 1)

    def polar_values(self):
        """The field ``η`` on its polar grid, shape ``(n_r, n_theta)``."""
        n_r = self.latent.shape[0]
        nodes = self.r_min_mas + self.spacing * np.arange(n_r, dtype=float)
        factor = _radial_factor(nodes, self.length_r, self.nu)
        azimuthal = _azimuthal_filter(
            self.latent, self.length_theta, self.nu_theta
        )
        # HIGHEST: the default matmul precision on A100/H100 is TF32.
        return self.sigma * np.matmul(
            factor, azimuthal, precision=jax.lax.Precision.HIGHEST
        )

    def evaluate(self, pixel_scale_mas):
        """The log-brightness ``η`` on pixels of ``pixel_scale_mas``."""
        nrow, ncol = self.image_shape
        inc, pa, dra, ddec = self._geometry()
        x = pixel_offsets(ncol, pixel_scale_mas)[None, :] - dra
        y = pixel_offsets(nrow, pixel_scale_mas)[:, None] - ddec
        x, y = np.broadcast_arrays(x, y)
        stretch = np.maximum(np.cos(inc * _DEG), 1e-8)
        xe, ye = undo_elliptical_transf_coord(x, y, pa, stretch)
        r = np.hypot(xe, ye)
        # The azimuth runs from the deprojected North towards East, as for
        # MultipoleDisk's harmonics; the centre has none, so avoid arctan2(0, 0).
        origin = (xe == 0.0) & (ye == 0.0)
        theta = np.arctan2(xe, np.where(origin, 1.0, ye))
        eta = polar_interpolate(
            self.polar_values(),
            self.r_min_mas,
            self.spacing,
            r,
            theta,
            fill=self.outside,
        )
        if self.mean is not None:
            template = np.maximum(self.mean, 0.0) / np.max(self.mean)
            eta = eta + np.log(template + self.mean_floor)
        return eta


def multipole_field_priors(
    path="env.log_brightness",
    sigma=(0.05, 5.0),
    length_r=(0.5, 50.0),
    length_theta=(0.1, 6.0),
):
    """Bounded log-uniform priors on a [`MultipoleField`][virgil.fields.MultipoleField]'s scales.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Returns a priors dict for [`fit`][virgil.fitting.fit] or
    [`numpyro_model`][virgil.likelihood.numpyro_model]: ``LogUniform`` priors,
    which are invariant under rescaling and proper because they are
    bounded. Add the latents' priors with
    [`image_priors`][virgil.imaging.image_priors]. Each range is
    ``(low, high)``: ``sigma`` in log-brightness, ``length_r`` in
    milliarcseconds and ``length_theta`` in radians. The variance of the
    azimuthal order ``m`` falls as the spectrum of
    [`MultipoleKernel`][virgil.models.MultipoleKernel] in ``m * length_theta``, so a
    length above about 2 leaves almost only the order 0, an axisymmetric
    field (for ``nu`` = 3/2 the first order, cos θ and sin θ together, has about
    1 per cent of the variance at a length of 6), and the default upper limit is effectively
    axisymmetric.

    Parameters
    ----------
    path : str, optional
        Path of the field in the model (default ``"env.log_brightness"``).
    """
    import numpyro.distributions as dist

    priors = {}
    for name, (low, high) in (
        ("sigma", sigma),
        ("length_r", length_r),
        ("length_theta", length_theta),
    ):
        if not 0.0 < low < high:
            raise ValueError(f"{name} range must satisfy 0 < low < high.")
        priors[f"{path}.{name}"] = dist.LogUniform(low, high)
    return priors
