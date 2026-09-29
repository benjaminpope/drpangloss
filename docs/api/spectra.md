# `drpangloss.spectra`

Wavelength-dependent fluxes. A component's `flux` can be a number or a
spectrum; inside a `System` each component is weighted by its spectrum at each
sample's wavelength, as in SPARCO.

::: drpangloss.spectra
    options:
      members:
        - PowerLaw
        - Spectrum
