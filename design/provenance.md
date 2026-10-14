# Provenance of virgil's methods

Where each major method came from: a paper, a package, or a person. The
machine-readable list is `references/provenance.toml` (bibkeys below refer
to it); `scripts/fetch_references.py` resolves it against NASA ADS and writes
`references/virgil.bib`. "Ported" means code was adapted; "after" means the
idea and its equations were reimplemented from the paper or the package's
documentation, with no code copied. IDs in the TOML marked
`verified = true` were checked against the arXiv abstract page or Crossref;
the rest carry no arXiv ID or DOI yet, and wait for an ADS run.

## Origin of the package

- Functions for the PDS 70 JWST/AMI analysis by Dori Blakely (`blakely2024`),
  rewritten by Benjamin Pope and Louis Desdoigts as `drpangloss`. Louis
  Desdoigts wrote the JAX grid-search syntax and the first kernel-phase
  support; everything is built on his `zodiax` (dot-path fitting interface).
  Name from Voltaire's *Candide*, after CANDID (`gallenne2015`).
- Jonah Goldfine: the acronym, PDS 70 geometry, and the parametric scenes in
  `virgil.scenes` (from his `frito`).
- Toon De Prins: `ModulatedGaussianRim` (azimuthally modulated thin ring times
  a Gaussian envelope), the elliptical (u, v) transform helpers, from his
  post-AGB disk work. The literature for such rim models is
  `lazareff2017` and `hillen2016`; neither is cited in the code he wrote.
  He did not write `virgil.pipeline`: Claude wrote it on Ben's instructions,
  and he made about 13 review-fix commits to it.
- Grid searches (`virgil.grid_fit`): Dori Blakely and Louis Desdoigts, in the
  `drpangloss` era. Only `linear_flux_grid` notes its equivalence to
  fouriever's `lincmap`, as a comparison.
- Shashank Dholakia: the gravity-darkened star (`espinosalara2011`, ported
  from his jax-interferometry, who took the equations from Aaron Dotter's
  GDit), the spherical-harmonic star (`harmonix`), the Bessel functions
  (CEPHES, via harmonix, now `jaxbessel`), and spectrally dispersed data.
- Max Charles: the image reconstruction in `virgil.imaging` is a port and
  extension of the visibility part of `dorito` (`dorito`), with ideas from
  Ian Czekala's MPoL.
- Louis Desdoigts: `virgil.amigo` reads AMIGO's mixed-DISCO format
  (`amigo`); `virgil.coverage` follows its latent visibility basis; the
  matrix Fourier transform was checked against dLux (`dlux`).

## Models and data

| Feature | Where | Source |
|---|---|---|
| Limb-darkened disk, sum of powers of μ | `models.cvis_limb_darkened_disk` | `quirrenbach1996` (eqs 1–4), generalized by `harmonix` |
| Quadratic and square-root LD in q1, q2 | `models` | `kipping2013` |
| Rapid rotator, Roche shape, gravity darkening | `_elr`, `GravityDarkenedStar` | `espinosalara2011`; ported via Dholakia from GDit |
| Spotted star | `HarmonixModel` | `harmonix`; conventions of jaxoplanet and `starry` |
| Flared scattered-light disk, phase functions | `FlaredDisk*` | `blakely2024`, §III |
| Matrix Fourier transform | `_geometry.grid_visibilities` | `soummer2007` |
| Chromatic scenes with spectra | `spectra`, `System` | `kluska2014` (SPARCO); ratio convention after `hillen2016` |
| Node spectra, error floors | `Nodes`, `OIData.with_error_floor` | PMOIRED (`merand2022`) |
| Correlated closure phases | `_closure.ClosureNoise` | `kammerer2020` |
| Correlated calibration errors as gain modes | `gains` | `lachaume2021` |
| Analytic marginalization of linear parameters | `_linear`, `grid_fit` | `luger2017linear` |
| Error-scale marginalization (Jeffreys) | `epochs.marginal_loglike` | `jeffreys1946`, re-estimation of `mackay1992` |
| Angle vectors, North angle, plate scale | `angles`, `oidata` | Octofitter's `UniformCircular` and calibration terms (`thompson2023`) |
| OIFITS reading and writing | `oifits` | `pauls2005`, `duvert2017` |
| Legacy OIFITS writer | `legacy.oifits_implaneia` | ImPlaneIA (`greenbaum2015`) |
| Closure-phase binary visibilities | `models.cvis_binary` | adapted from `pymask` (Pope), which took it from pysco/`xara` (Martinache, `xara`) |
| AMI mask geometry, NRM simulation | `coverage` | `sivaramakrishnan2023` |

## Detection and limits

| Feature | Where | Source |
|---|---|---|
| Bayesian upper limits | `limits.ruffio_upperlimit` | `ruffio2018` |
| Chi-squared ratio limits | `limits.absil_limits`, `nsigma` | `absil2011` |
| Injection limits, bootstrap | `limits.injection_limits`, `detection.bootstrap_null` | CANDID (`gallenne2015`), `efron1979` |
| Linearized flux map | `grid_fit.linear_flux_grid` | equivalent to fouriever's `lincmap` (a comparison; no paper) |
| Local significance of Δχ² | `detection.local_nsigma` | `wilks1938`, `chernoff1954` |
| Look-elsewhere effect, calibration by simulation | `detection` | `gross2010` |
| Injection–recovery, ROC | `detection.DetectionMC` | direct-imaging practice (`mawet2014`), `fawcett2006` |

## Orbits

| Feature | Where | Source |
|---|---|---|
| Thiele–Innes linear fit at fixed (P, e, T) | `orbits._thiele_innes_fit`, `starting_orbits` | Thiele 1883; `hartkopf1989`; `wright2009`; `lucy2014` |
| Frequency-spaced period grid | `orbits.period_grid` | `lucy2014`, The Joker (`pricewhelan2017`) |
| Position angle at a reference epoch; orbit fits to closure phases | `KeplerOrbit.from_position_angle` | Octofitter (`thompson2023`) |
| Alias bands, spectral window | `aliases` | `vanderplas2018`; Laplace evidence |
| Compared, not ported | design notes | orbitize! (`blunt2020`), OFTI (`blunt2017`), orvara (`brandt2021`); The Joker (`pricewhelan2017`) for period resolution only |

No code from orbitize!, Octofitter or orvara is in virgil.

## Imaging and inference

| Feature | Where | Source |
|---|---|---|
| TSV, TV regularizers | `imaging.TSV`, `TV` | `kuramochi2018`, `akiyama2017`, `rudin1992` |
| MaxEnt regularizer; classic MaxEnt weight | `MaxEntropy`, `LCurve.classic_maxent` | Skilling & Bryan 1984; `gull1989`, `skilling1989`; BSMEM (`baron2008`) |
| L1 sparsity on starlets; log-sum | `StarletL1`, `LogSum` | `honma2014`, `akiyama2017`, `starck2010`; `candes2008`, SQUEEZE (`baron2010squeeze`) |
| Gauge pitfalls, quadratic regularizers as priors | `Laplacian`, `diagnose` | `thiebaut2017` |
| CLEAN, major cycle, multi-scale | `imaging.clean` | Högbom 1974, Clark 1980, Cotton–Schwab, `cornwell2008`, `rau2011` |
| L-curve, discrepancy principle | `l_curve`, `LCurve` | `hansen1993`, Morozov |
| Laplace evidence, noise re-estimation | `log_evidence`, `error_scale` | `mackay1992` |
| Gaussian-process image fields | `fields` | NIFTy (`selig2013`, `edenhofer2024`), `arras2022`, `ensslin2009`, `ensslin2019` |
| Randomized ensembles of reconstructions | `ensemble` | `drevon2026` (PYRA, MYTHRA); written from the paper |
| Variational inference | `svi` | `kucukelbir2017`, `decao2019`, `kingma2016`, `vehtari2024` |
| NUTS, Adam, L-BFGS | `fitting`, `inference` | `hoffman2014`, `kingma2015`, `liu1989` |
| Group-invariant priors | `priors`, `_flat` | `jeffreys1946`, `hogg2010ecc` |
| Beauty-contest metrics | `metrics` | `malbet2010`, `baron2012` |

## Open points

- `docs/contributors.md` credits Skilling 1989 with DOI `…_4`; Crossref gives
  that DOI to Gull 1989 ("Developments in Maximum Entropy Data Analysis") and
  `…_3` to Skilling's "Classic Maximum Entropy".
- `virgil.ensemble` cited Drevon et al. as 2025; arXiv:2609.15365 is dated
  2026.
- fouriever has no paper; the author attribution (Kammerer) is from the
  GitHub repository owner.
- jaxoplanet, exoplanet and AMICAL are in `virgil.bib` for the paper
  (`role = "bibliography"`) but are imports or separate repositories, not
  provenance. eht-imaging is a comparison only; nothing was taken from it.
  Entries with `role = "compared"` were studied for comparison, not ported.
- Entries without arXiv ID or DOI (Quirrenbach 1996, Högbom 1974, Skilling &
  Bryan 1984, Thiele 1883, Starck et al. 2010) and the package-only heritage
  entries are unverified until `scripts/fetch_references.py` runs with an ADS
  token.
