# API Reference

The everyday names are importable from the top level, e.g.
`from virgil import OIData, System, PointSource, likelihood_grid`.

Everything here is stable except the entries marked **Provisional**, which
may change in any 0.4.x release without a deprecation. See
[Stability](../stability.md) for the list.

- [OIData](oidata.md): observables and their conventions
- [OIFITS](oifits.md): reading and writing OIFITS files
- [AMIGO](amigo.md): AMIGO mixed-DISCO products
- [Models](models/index.md): source models and visibilities
- [Harmonics](harmonics.md): azimuthal harmonics of a `MultipoleDisk` and conversions between their forms
- [Likelihood](likelihood.md): likelihoods and numpyro models
- [Gains](gains.md): calibration gains correlated across channels, marginalized analytically
- [Linear marginalization](linear.md): the shared algebra for parameters marginalized analytically
- [Observables](observables.md): OI_FLUX spectra, |V|, T3AMP and differential phases after the visibilities and phases
- [Inference](inference.md): Laplace and Fisher curvature
- [Grid Fit](grid_fit.md): grid searches
- [Limits](limits.md): contrast limits and flux/contrast/Δmag conversions
- [Priors](priors.md): isotropic inclination and latitude priors, and hierarchical (population) scales
- [Detection](detection.md): detection statistics for false-alarm rates and ROC curves
- [Spectra](spectra.md): wavelength-dependent fluxes
- [Orbits](orbits.md): Keplerian orbits in virgil's conventions, solved with jaxoplanet
- [Orbit aliases](aliases.md): `fit_orbit_aliases`, evidences of the period alias bands of an undersampled orbit (provisional)
- [Tempering](tempering.md): the effective-channel factor that tempers an orbit evidence for correlated spectral channels (provisional)
- [Epochs](epochs.md): datasets grouped into epochs, and starting multi-epoch orbit fits (provisional)
- [Radio](radio.md): `ComplexVisData`, `CalibrationNuisance`, complex visibilities from CASA measurement sets with per-observation calibration nuisances (provisional)
- [Orbit Search](orbit_search.md): `score_orbits`, candidate orbits scored on all epochs with shared nuisances (provisional)
- [Radio Toeplitz](radio_toeplitz.md): `ToeplitzVisData`, the radio χ² of a one-Image fit through precomputed dirty beams (provisional)
- [Radio CLEAN](radio_clean.md): `clean`, CLEAN deconvolution of complex visibilities (provisional)
- [Cross-validation](crossval.md): K-fold cross-validation of a hyperparameter (provisional)
- [Short arcs](arcs.md): each night's velocity, the periods its positions and velocities admit, and each alias band's seed orbits (provisional)
- [Fields](fields.md): Gaussian-process log-brightness priors for pixel images
- [Fitting](fitting.md): `fit`, maximum a posteriori fits; `gauss_newton_mass`, a NUTS mass matrix from a fit
- [SVI](svi.md): `variational`, a guide fitted to the posterior of `numpyro_model` with numpyro's SVI
- [Angles](angles.md): `AngleVector`, angles sampled as 2-D vectors with no wrap boundary, and von Mises priors in least-squares form
- [Imaging](imaging.md): regularizers and helpers for image reconstruction
- [Scenes](scenes.md): synthetic truth images for testing reconstructions
- [Metrics](metrics.md): image-recovery scores for reconstructions
- [Ensemble](ensemble.md): randomized reconstruction ensembles, selected and averaged into a mean image with a per-pixel spread
- [Coverage](coverage.md): synthetic uv coverage and noise for simulations
- [Pipeline](pipeline.md): `BinaryPipeline`, `StarPipeline`, `DiskPipeline`, `OrbitPipeline` and `ImagingPipeline`, scriptable pipelines that write a run folder
- [Simulate](simulate.md): a scene observed with a template's sampling, and bias tests
- [Plotting](plotting.md): figures
- [Legacy](legacy.md): ImPlaneIA-derived OIFITS tools
