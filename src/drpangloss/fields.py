"""Gaussian-process log-brightness fields for pixel images.

A [`GaussianField`][drpangloss.fields.GaussianField] can stand in for the
``log_brightness`` array of an [`Image`][drpangloss.models.Image]. The image's
log-brightness is then a stationary Gaussian process with a Matérn-like
spectrum, written in its whitened form: independent standard-normal
``latent`` coefficients on the image's cosine (DCT-II) basis. This is the
basis that diagonalises the Laplacian with reflecting boundaries. Fitting the
latents under a standard-normal prior (from
[`image_priors`][drpangloss.imaging.image_priors]) is MAP estimation with a
GP prior, and the same parameterisation suits sampling.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zx
from jax.scipy.fft import idctn

from ._utils import concrete

__all__ = ["GaussianField", "field_spectrum"]


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
    on the pixel grid, ``(2/h)² [sin²(πj/2n) + sin²(πk/2m)]``. The constant
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
    length_mas : float or array-like
        Correlation length ``1/κ`` in milliarcseconds.
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
    laplacian = eigen[0][:, None] + eigen[1][None, :]
    spectrum = (length_mas**-2.0 + laplacian) ** (-float(order))
    spectrum = spectrum.at[0, 0].set(0.0)
    return sigma**2 * spectrum * spectrum.size / np.sum(spectrum)


class GaussianField(zx.Base):  # type: ignore[reportGeneralTypeIssues]
    r"""A Gaussian-process log-brightness for an [`Image`][drpangloss.models.Image].

    The log-brightness is

    $$\eta = \log\left(\frac{\mu}{\max\mu} + \epsilon\right)
    + \mathrm{IDCT}\left[\sqrt{S} \odot z\right],$$

    with the orthonormal inverse DCT-II, the spectrum ``S`` of
    [`field_spectrum`][drpangloss.fields.field_spectrum], latent
    coefficients ``z`` (``latent``) and an optional positive template image
    ``μ`` (``mean``, floored at ``ε`` = ``mean_floor`` of its peak). With a
    standard-normal prior on ``latent``, ``η`` is a Gaussian process with
    standard deviation ``sigma`` and correlation length ``length_mas``
    about the template. ``latent = 0`` gives the template raised by the
    floor, ``μ/max μ + ε``, as an image.

    Use it in place of an Image's log-brightness, with priors from
    [`image_priors`][drpangloss.imaging.image_priors]:

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
        Correlation length in milliarcseconds (default 1).
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
