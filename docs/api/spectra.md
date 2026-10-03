# `virgil.spectra`

Wavelength-dependent fluxes. A component's `flux` can be a number or a
spectrum; inside a `System` each component is weighted by its spectrum at each
sample's wavelength, as in SPARCO.

::: virgil.spectra
    options:
      members:
        - BlackBody
        - PowerLaw
        - Spectrum
        - Tabulated
        - flux_at
        - reference_flux

`Tabulated` is **provisional**: it is not exported from the top-level
`virgil` namespace (import it from `virgil.spectra`), and it will be replaced
by the node spectra of Stage 6a.
