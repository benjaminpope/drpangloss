"""Wavelength-dependent fluxes for source components.

A component's ``flux`` is either a number (achromatic) or a spectrum from this
module, which gives its weight at each wavelength. Inside a
[`System`][drpangloss.models.System] the visibility is then
``V(λ) = Σ f_i(λ) V_i / Σ f_i(λ)``, as in SPARCO (Kluska et al. 2014):

```python
star = UniformDisk(0.5, flux=PowerLaw(1.0, index=-4.0, wavel0=1.65e-6))
disk = GaussianDisk(5.0, flux=PowerLaw(0.3, index=1.0, wavel0=1.65e-6))
```

Spectrum parameters are reached by path like any other, e.g.
``"disk.flux.ratio"`` and ``"disk.flux.index"``.
"""

import jax
import jax.numpy as np
import zodiax as zx

from ._utils import concrete


__all__ = ["PowerLaw", "Spectrum", "flux_at", "reference_flux"]


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
    >>> float(PowerLaw(0.2, index=-4.0, wavel0=1.6e-6)(3.2e-6))
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

    def __check_init__(self):
        value = concrete(self.ratio)
        if value is not None and (value < 0.0).any():
            raise ValueError(
                f"PowerLaw ratio {value.tolist()} is negative; fluxes must be "
                "non-negative."
            )


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
