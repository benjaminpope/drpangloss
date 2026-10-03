"""Wavelength-dependent fluxes for source components.

A component's ``flux`` is either a number (achromatic) or a spectrum from this
module, which gives its weight at each wavelength. Inside a
[`System`][virgil.models.System] the visibility is then
``V(λ) = Σ f_i(λ) V_i / Σ f_i(λ)``, as in SPARCO (Kluska et al. 2014):

```python
star = UniformDisk(0.5, flux=PowerLaw(1.0, index=-4.0, wavel0=1.65e-6))
disk = GaussianDisk(5.0, flux=PowerLaw(0.3, index=1.0, wavel0=1.65e-6))
ring = GaussianDisk(5.0, flux=BlackBody(0.3, temperature=1200.0))
line = PointSource(flux=Tabulated(ratios, channel_wavelengths))
```

Spectrum parameters are reached by path like any other, e.g.
``"disk.flux.ratio"``, ``"disk.flux.index"`` or ``"ring.flux.temperature"``
(for `Tabulated`, ``"line.flux.ratio"`` is one value per channel).
Flux ratios are relative: component ``i``'s fraction of the total at the
reference wavelength is ``ratio_i / Σ ratio``, as SPARCO's ``f_i``.
"""

import jax
import jax.numpy as np
import numpy as onp
import zodiax as zx

from ._utils import concrete


__all__ = [
    "BlackBody",
    "PowerLaw",
    "Spectrum",
    "Tabulated",
    "flux_at",
    "reference_flux",
]


class Spectrum(zx.Base):  # type: ignore[reportGeneralTypeIssues]
    """Base class for component spectra.

    Subclasses implement ``__call__(wavel)``, returning the flux at
    ``wavel`` (metres, any shape), or the reference flux when ``wavel`` is
    ``None`` (as used when rendering images), and ``ratio``, the reference
    flux, which must be non-negative.
    """

    ratio: jax.Array

    def __call__(self, wavel=None):
        raise NotImplementedError

    def is_physical(self):
        """Whether the spectrum is valid, as a (traceable) boolean."""
        return np.all(self.ratio >= 0.0)


class PowerLaw(Spectrum):
    """Power-law spectrum ``ratio * (λ / wavel0) ** index``.

    Parameters
    ----------
    ratio : float or array-like
        Flux at the reference wavelength, relative to the other components.
    index : float or array-like, optional
        Spectral index (default 0, i.e. achromatic). A star in the
        Rayleigh-Jeans regime of F_λ has index -4.
    wavel0 : float or array-like, optional
        Reference wavelength in metres (default 1.65e-6, H band).

    Examples
    --------
    >>> round(float(PowerLaw(0.2, index=-4.0, wavel0=1.6e-6)(3.2e-6)), 6)
    0.0125
    """

    ratio: jax.Array
    index: jax.Array
    # An ordinary (traceable) field rather than a static one, so that fixed
    # parameters passed through jitted functions can set it.
    wavel0: jax.Array

    def __init__(self, ratio, index=0.0, wavel0=1.65e-6):
        self.ratio = np.asarray(ratio, dtype=float)
        self.index = np.asarray(index, dtype=float)
        self.wavel0 = np.asarray(wavel0, dtype=float)

    def __call__(self, wavel=None):
        if wavel is None:
            return self.ratio
        return self.ratio * (np.asarray(wavel) / self.wavel0) ** self.index

    def is_physical(self):
        return np.all(self.ratio >= 0.0) & np.all(self.wavel0 > 0.0)

    def __check_init__(self):
        value = concrete(self.ratio)
        if value is not None and (value < 0.0).any():
            raise ValueError(
                f"PowerLaw ratio {value.tolist()} is negative; fluxes must be "
                "non-negative."
            )
        wavel0 = concrete(self.wavel0)
        if wavel0 is not None and (wavel0 <= 0.0).any():
            raise ValueError(
                f"PowerLaw wavel0 {wavel0.tolist()} must be a positive "
                "wavelength (in metres)."
            )


class BlackBody(Spectrum):
    """Planck spectrum ``ratio * B_λ(T, λ) / B_λ(T, wavel0)``.

    The shape of a blackbody at ``temperature`` in F_λ, normalised to
    ``ratio`` at ``wavel0``, as SPARCO uses for dust and companions (e.g.
    Hillen et al. 2016). At long wavelengths (``hc/λkT`` small) it tends to
    the Rayleigh-Jeans ``PowerLaw`` with index -4.

    Parameters
    ----------
    ratio : float or array-like
        Flux at the reference wavelength, relative to the other components.
    temperature : float or array-like
        Temperature in kelvin.
    wavel0 : float or array-like, optional
        Reference wavelength in metres (default 1.65e-6, H band).

    Examples
    --------
    >>> round(float(BlackBody(0.2, 1500.0)(1.65e-6)), 6)
    0.2
    """

    ratio: jax.Array
    temperature: jax.Array
    wavel0: jax.Array

    def __init__(self, ratio, temperature, wavel0=1.65e-6):
        self.ratio = np.asarray(ratio, dtype=float)
        self.temperature = np.asarray(temperature, dtype=float)
        self.wavel0 = np.asarray(wavel0, dtype=float)

    def __call__(self, wavel=None):
        if wavel is None:
            return self.ratio
        wavel = np.asarray(wavel)
        return self.ratio * _planck_ratio(
            wavel, self.temperature, self.wavel0, self.temperature
        )

    def is_physical(self):
        return (
            np.all(self.ratio >= 0.0)
            & np.all(self.temperature > 0.0)
            & np.all(self.wavel0 > 0.0)
        )

    def __check_init__(self):
        for name, value, ok in (
            ("ratio", self.ratio, lambda x: x >= 0.0),
            ("temperature", self.temperature, lambda x: x > 0.0),
            ("wavel0", self.wavel0, lambda x: x > 0.0),
        ):
            value = concrete(value)
            if value is not None and not ok(value).all():
                raise ValueError(
                    f"BlackBody {name} {value.tolist()} must be "
                    + ("non-negative." if name == "ratio" else "positive.")
                )


class Tabulated(Spectrum):
    """A free flux in every spectral channel, interpolated linearly between.

    **Provisional**: not exported from the top-level ``virgil`` namespace, and
    to be replaced by the node spectra of Stage 6a.

    For fitting a spectrum channel by channel, e.g. a companion's flux ratio
    across emission lines: give ``wavel`` the data's channel wavelengths and
    fit ``ratio`` (one value per channel) with a prior of that shape.

    Parameters
    ----------
    ratio : array-like, shape (n,)
        Flux at each node, relative to the other components: finite and
        non-negative, with n >= 1.
    wavel : array-like, shape (n,)
        Node wavelengths in metres: finite, positive and strictly
        increasing. Beyond the end nodes the flux is constant.

    Notes
    -----
    The reference flux (``wavel=None``, used when rendering) is the mean
    over the nodes.

    Examples
    --------
    >>> spectrum = Tabulated([0.2, 0.4], [2.0e-6, 2.2e-6])
    >>> round(float(spectrum(2.1e-6)), 6)
    0.3
    """

    ratio: jax.Array
    # Traceable, like PowerLaw.wavel0.
    wavel: jax.Array

    def __init__(self, ratio, wavel):
        self.ratio = np.asarray(ratio, dtype=float)
        self.wavel = np.asarray(wavel, dtype=float)

    def __call__(self, wavel=None):
        if wavel is None:
            return np.mean(self.ratio)
        return np.interp(np.asarray(wavel), self.wavel, self.ratio)

    def is_physical(self):
        return (
            np.all(self.ratio >= 0.0)
            & np.all(self.wavel > 0.0)
            & np.all(np.diff(self.wavel) > 0.0)
        )

    def __check_init__(self):
        if (
            self.ratio.shape != self.wavel.shape
            or self.ratio.ndim != 1
            or self.ratio.size == 0
        ):
            raise ValueError(
                f"Tabulated needs non-empty 1D ratio and wavel of the same "
                f"length, not shapes {self.ratio.shape} and {self.wavel.shape}."
            )
        value = concrete(self.ratio)
        if value is not None and not (
            onp.isfinite(value).all() and (value >= 0.0).all()
        ):
            raise ValueError(
                "Tabulated ratios must be finite and non-negative; fluxes "
                "cannot be negative."
            )
        wavel = concrete(self.wavel)
        if wavel is not None and not (
            onp.isfinite(wavel).all()
            and (wavel > 0.0).all()
            and (onp.diff(wavel) > 0.0).all()
        ):
            raise ValueError(
                "Tabulated wavel must be finite, positive and strictly "
                "increasing."
            )


# Planck's second radiation constant h c / k, in metre kelvin.
_HC_OVER_K = 1.438776877e-2


def _planck_ratio(wavel, temperature, wavel0, temperature0):
    """``B_λ(wavel, temperature) / B_λ(wavel0, temperature0)``, no overflow.

    Wavelengths in metres, temperatures in kelvin; all broadcast together.
    """
    x = _HC_OVER_K / (wavel * temperature)
    x0 = _HC_OVER_K / (wavel0 * temperature0)
    # (wavel0 / wavel)**5 * expm1(x0) / expm1(x), formed entirely in log
    # space: expm1(x) = exp(x) * -expm1(-x), so its log is
    # x + log(-expm1(-x)) without overflow. Exponentiating only the total
    # keeps the result finite whenever it is representable (in float32 the
    # factors alone can overflow when the ratio does not).
    log_ratio = (
        5.0 * np.log(wavel0 / wavel)
        + (x0 - x)
        + np.log(-np.expm1(-x0))
        - np.log(-np.expm1(-x))
    )
    return np.exp(log_ratio)


def flux_at(flux, wavel=None):
    """Evaluate a number or a spectrum at ``wavel`` (``None`` = reference)."""
    if isinstance(flux, Spectrum):
        return flux(wavel)
    return flux


def reference_flux(flux):
    """The reference flux of a number or a spectrum."""
    if isinstance(flux, Spectrum):
        return flux.ratio
    return flux
