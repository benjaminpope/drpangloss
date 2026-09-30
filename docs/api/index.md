# API Reference

The everyday names are importable from the top level, e.g.
`from drpangloss import OIData, System, PointSource, likelihood_grid`.

- [OIData](oidata.md): observables and their conventions
- [OIFITS](oifits.md): reading and writing OIFITS files
- [AMIGO](amigo.md): AMIGO mixed-DISCO products
- [Models](models/index.md): source models and visibilities
- [Likelihood](likelihood.md): likelihoods and numpyro models
- [Inference](inference.md): Laplace and Fisher curvature
- [Grid Fit](grid_fit.md): grid searches
- [Limits](limits.md): contrast limits and flux/contrast/Δmag conversions
- [Spectra](spectra.md): wavelength-dependent fluxes
- [Fitting](fitting.md): `fit`, maximum a posteriori fits
- [Imaging](imaging.md): regularisers and helpers for image reconstruction
- [Scenes](scenes.md): synthetic truth images for testing reconstructions
- [Plotting](plotting.md): figures
- [Bessel](bessel.md): Bessel functions in JAX
- [Legacy](legacy.md): ImPlaneIA-derived OIFITS tools
