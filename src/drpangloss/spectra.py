"""Wavelength-dependent fluxes for source components.

A component's ``flux`` is either a number (achromatic) or a spectrum from this
module, which gives its weight at each wavelength. Inside a
[`System`][drpangloss.models.System] the visibility is then
``V(λ) = Σ f_i(λ) V_i / Σ f_i(λ)``, as in SPARCO (Kluska et al. 2014):

```python
star = UniformDisk(0.5, flux=PowerLaw(1.0, index=-4.0, wavel0=1.65e-6))
disk = GaussianDisk(5.0, flux=PowerLaw(0.3, index=1.0, wavel0=1.65e-6))
ring = GaussianDisk(5.0, flux=BlackBody(0.3, temperature=1200.0))
```

Spectrum parameters are reached by path like any other, e.g.
``"disk.flux.ratio"``, ``"disk.flux.index"`` or ``"ring.flux.temperature"``.
Flux ratios are relative: component ``i``'s fraction of the total at the
reference wavelength is ``ratio_i / Σ ratio``, as SPARCO's ``f_i``.
"""

import jax
import jax.numpy as np
import zodiax as zx

from ._utils import concrete


__all__ = ["BlackBody", "PowerLaw", "Spectrum", "flux_at", "reference_flux"]


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
        x, x0 = (
            _HC_OVER_K / (wavel * self.temperature),
            _HC_OVER_K / (self.wavel0 * self.temperature),
        )
        return (
            self.ratio
            * (self.wavel0 / wavel) ** 5
            * np.expm1(x0)
            / np.expm1(x)
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


# Planck's second radiation constant h c / k, in metre kelvin.
_HC_OVER_K = 1.438776877e-2


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
