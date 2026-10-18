# Changelog

All notable changes to this project are recorded here, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project follows
[semantic versioning](https://semver.org/), with the usual caveat that
anything before 1.0 may change between minor versions.

## Unreleased

### Stability

virgil 0.4.0 has two tiers (see the [Stability page](docs/stability.md)).
**Stable**: a change needs a deprecation cycle. **Provisional**: may change
in any 0.4.x release without a deprecation, so that new features can settle.
Everything public that is not listed here is stable. Nothing warns at
runtime; provisional pages and API entries carry a "Provisional" notice, and
each module lists its provisional names in `__provisional__`, which
`tests/test_stability.py` checks. New public features start provisional.

Provisional in 0.4:

- the modules `virgil.epochs`, `virgil.aliases`, `virgil.orbit_search` and
  `virgil.tempering` (the orbit fitting, aliases and search);
- the `MultipoleDisk` family in `virgil.models`: `MultipoleDisk`, `ThinRing`,
  `GaussianRing`, `RadialPowerLaw`, `ExpTail`, `Gap`, `RadialSum`,
  `RadialLinear`, `RadialPower`, `RadialGP`, `RadialGPFunction`, `MultipoleKernel`, `Harmonics`,
  `VonMises`, `Cone`, `PowerLawHeight`, `multipole_disk_cvis` and `gp_priors`;
  and in `virgil.fields`, `MultipoleField` and `multipole_field_priors`;
- the Gaussian-process members of `virgil.ensemble`: `evidence_grid`,
  `gp_members`, `GPFit` and `EvidenceGrid`;
- `virgil.svi` (`variational`);
- all of `virgil.pipeline`: `BinaryPipeline` and `StarPipeline` were
  documented as stable and are now provisional, to be promoted after a
  release of use (`global_nsigma` is to be settled first). The run schema
  `virgil-pipeline-run-v1` is unchanged, and a provisional pipeline no
  longer emits a `FutureWarning`;
- `virgil.imaging.clean` and `CleanResult`;
- `virgil.metrics`.

### Added

- **`DiskPipeline` defaults from imaging nine PIONIER post-AGB discs**
  (provisional, so no deprecation; the lessons are from
  virgil-validation#133). For `model="gp"`, `error_scale="auto"` (the new
  default, `"rescale"` for the Gaussian process and `"quoted"` for the
  rings) estimates one MacKay error scale from a baseline fit and
  evaluates every fit and Laplace evidence on the rescaled data
  (`chi2.error_scale_factor`; χ²/N stays on the quoted errors);
  `imaging.error_scale` now accepts a `MultipoleDisk` with Gaussian-process
  parts as `log_evidence` does. The ℓ grid cannot choose its own edge: the
  best ℓ on an edge extends the grid once that way (`ell_floor_halffringes`,
  0.5, is the lowest ℓ/√3), `on_grid_edge` is recorded and `length_scale`
  warns if it persists; the evidences are now computed in `profile`. The
  star can be a uniform disk (`star_diameter_mas`, which also floors the
  inner radius at its radius) with a power-law spectrum (`star_index`), and
  the summary's `model.star` records it, since the disk's spectral index
  is conditional on it. New checks: `structure_unconstrained` (flat
  evidence and unstable radial profiles across ℓ), `render_consistency`
  (the rendered image against the model's χ²) and `at_bound` (parameters
  within 0.1% of a prior bound). `r_out` still comes from `image_field`.
  `design/pipeline.md` records the planned changes (an optional `Resolved`
  term) and the guidance on `"gp"` against the Cartesian field.

- `OIData(..., insname=)` selects the instrument table(s) when reading an
  OIFITS file, as `read_oifits` does; the GRAVITY fringe-tracker/science
  error message already told users to pass it.

- `starting_orbits(batch_size=)` bounds the memory of the grid solve (the
  grid is solved in fixed-size batches).

- **Tempered orbit evidence for correlated spectral channels.**
  `fit_orbit_aliases(temper=True)` (default) divides every block's log
  likelihood by the effective-channel factor `f = N / N_eff`, estimated as
  the empirical within-chain covariance factor
  `sum (sum_chain z)^2 / sum z^2` of the residuals of each night's best
  static fit, over the chains of consecutive channels of a baseline or
  triangle. It is exact for AR(1) and for common-mode (shared calibration)
  correlation alike, where a lag-1 AR(1) estimate undercounts the latter
  (8 channels, c = 0.6: 3.2 for a truth of 5.2); it is 1 for white noise,
  at most the channel count, used as `max(f, 1)` (tempering never sharpens;
  no significance threshold, a slight upward bias that is the safe
  direction). The lag-1 and lag-2 correlations are kept as diagnostics
  (`AliasResult.temper_rho`); a flat signature (rho2 well above rho1^2) warns
  and sets `tempering`. Tempering by 1/f is exact for signals flat across
  channels (a position) and conservative for chromatic templates. The
  untempered evidence gave the best alias band p = 1.000 at chi2/N 12 and
  posteriors far too narrow; on 20 seeds of such noise (AR(1) 0.85 and
  common mode 0.6) the median sky-plane offset/spread is 1.1 and 1.3
  tempered, 2.6 and 3.0 untempered. `temper=False` is the old behaviour and
  a finite float >= 1 a fixed `f`, which also tempers the
  `primary_diam="auto"` resolution test. `AliasResult.temper_f`,
  `.temper_blocks`, `.temper_rho` and `.tempering` (largest block `f` above
  2, raw `f` below 0.5, or a flat correlation) record it; `save` and `load`
  keep them (older files load with `temper_f=None`), and
  `AliasResult.block_dof()` / `virgil.tempering.block_dof` give the
  block-keyed `dof` for later stages. New module `virgil.tempering`
  (`channel_chains`, `estimate_factors`, `ChannelFactors`); the block score
  takes a per-block `dof`: `marginal_loglike(block_dof=)` and
  `estimate_primary_diameter(block_dof=)` (keyed by block, unlike the
  dataset-keyed `epoch_positions(dof=)`). `LABELS_FREE_NATS` (3) and
  `RESOLVE_NATS` (5) keep their values and are now in tempered nats.

- **Night flags and UD-aware seeds for orbit starts.** `epoch_positions`
  flags nights that should not seed an orbit (`alias-peaks`, `low-gap`,
  `chi2-high`, `sep-below-resolution`, `flux-at-bound`), each with a
  reason (`NightFlag`, `EpochPositions.flags`, `.flagged`,
  `.flag_codes()`), with the keywords `flag_nights`, `flag_alias_nats`,
  `flag_min_gap`, `flag_chi2_factor`, `flag_sep_fraction` and
  `flag_flux_tol`. `primary_diam=` fits the per-night companion beside a
  uniform-disc primary (also `start_from_positions(primary_diam=,
  flag_options=)`). `AliasResult.unseeded` records the nights that seeded
  nothing, and is kept by `save` and `load`.


- **`image_field`, `second_moment_fwhm` and `OIData.unflagged`.** The image
  field is now set by the data. `image_field(data)` is `field_fwhms` (6)
  times the source's FWHM, at least `inner_elements` (4) resolution
  elements λ/(2 B_max) across, and at most `field_of_view(data)`; the
  FWHM is `second_moment_fwhm(data)`, a straight-line fit of −ln V² against
  (B/λ)² on the unflagged samples with V² ≥ 0.3 (or the FWHM you give).
  `OIData.unflagged` is the mask of the samples an observable uses: a
  flagged sample keeps its u, v in the data but measures nothing.
  `field_of_view` also takes `aperture_mas` (a telescope's field), and
  stops at the bandwidth-smearing field R λ/B_max when the data carry
  channel widths. `starting_image` takes `field_fwhms`, `inner_elements`
  and `aperture_mas`; `EnsembleSpec` takes `field_mas` (the field that the
  factors multiply) and `aperture_mas`.
- **`gauss_newton_mass(..., noise=...)`** takes the fitted error terms of
  `fit` and `numpyro_model` and covers them in the matrix, under their
  numpyro site names (`"noise.vis_scale"`). For Gaussian data the matrix
  is the Fisher information, with the extra rows √2 log σ for each
  visibility and phase error. It now also accepts data with gains, any
  elementwise prior (HalfNormal, Beta, Uniform and so on, each with the
  curvature of the sampler's transform) and likelihood terms with a
  `log_norm`. With more than 4096 coordinates it returns the diagonal of
  the covariance, with a `UserWarning`, instead of an n² matrix.
  `SingularCurvature` now means a direction that neither the data nor a
  proper prior constrains (an `ImproperUniform` or flat-in-coordinate
  parameter the data do not fix), or a matrix that is not positive
  definite: a parameter under a bounded prior no longer raises. The
  result's `condition` attribute is the condition number of its
  correlation matrix. The matrix checks positive definiteness on the
  covariance itself, which a float32 Woodbury result could fail while
  the old check passed. A prior that is neither elementwise nor an
  angle vector (`MultivariateNormal`, `Dirichlet`) raises
  `NoLeastSquaresForm`, as its curvature is not diagonal.

### Changed

- `Epochs.snapshots` evaluates a single scene at all the datasets' times with one `vmap`, so the gradient graph no longer grows with the number of datasets (about half the jaxpr equations for an `OrbitalBinary`), and the `DiskPipeline` ring stages reuse one compiled model instead of recompiling every fit. Results are unchanged.
- `score_orbits`, `rank_orbits` and `chain_starts` compile their kernel once
  per layout of the data instead of on every call. Results are unchanged.
  `epoch_positions(batch_size=)` now defaults from the data's size, as the
  other grid functions do (it was 4096), and `batch_size=0` raises.

- **Imaging and metrics performance.** The evidence helpers (`log_evidence`,
  `error_scale`, `laplace_samples`, `chi2_by_observable`, `classic_maxent`)
  take the Jacobian's SVD on the device in float64, QR-reduced when it is
  far from square, instead of NumPy's SVD on the host (#328). `dirty_image`
  sums the visibilities in chunks of samples, so its memory no longer grows
  as samples x pixels. `metrics.align` sweeps its shift table in one
  compiled `lax.map` instead of a call per shift. Results are unchanged up to
  rounding, except that `laplace_samples` with a given `key` returns
  different draws than before (the singular vectors' signs differ between
  the host and device SVDs); they have the same distribution.
- **One naming scheme for the `MultipoleDisk` family** (new since 0.3.0, so
  there are no shims). The classes are `MultipoleDisk`, `MultipoleKernel`
  and `MultipoleField` (were `PolarDisk`, `PolarKernel`, `PolarField`),
  with `multipole_disk_cvis` and `multipole_field_priors` for
  `polar_disk_cvis` and `polar_field_priors`: the disk is a sum of
  azimuthal multipoles, and "polar" now means only polar coordinates.
  Names follow the stable conventions: `radius`,
  `sigma` (a standard deviation, as `GaussianDisk`), `inc`/`pa` in degrees,
  a unit suffix nowhere else. Renames:
  `ThinRing(r0)` is `ThinRing(radius)`; `GaussianRing(r0, sigma, weight)`
  is `GaussianRing(radius, sigma)`; `RadialPowerLaw(edge=)` is
  `edge_sigma=`; `ExpTail(r0, scale, weight, cutoff)` is `ExpTail(r_in,
  scale_length, *, cutoff)` (`cutoff` is keyword-only, so an old
  `ExpTail(r0, L, w)` raises instead of cutting the tail); `Gap(base, r0, width, depth, weight)` is
  `Gap(base, radius, sigma, depth)`; `RadialLinear`, `RadialPower`,
  `PowerLawHeight` and `MultipoleKernel` take the reference radius `r_ref`
  (was `r0`), and `PowerLawHeight` its height `h_ref` (was `h0`). The
  profile `weight` is gone: a weight belongs to a term of the sum, as
  `RadialSum(a, (b, 0.3))` or `RadialSum(core=ring, halo=(tail, 0.2))`
  (paths `profile.halo.radius`, `profile.halo.weight`), and
  `RadialSum` takes no `weight=` of its own. `RadialGP` splits in two:
  `RadialGP(r_in, r_out, ...)` is the profile only, and the new
  `RadialGPFunction` is the process as a callable function (spectral
  index, harmonic coefficients, `MultipoleKernel` radial part); both share the
  fit, prior and `draws` machinery. `Harmonics(c, s, *, degree, form, re,
  im, w, prior)` takes everything after `s` by keyword. `_multipole_disks`
  (used by `freeze_geometry`, `gp_priors` and `log_evidence`) now finds
  the `MultipoleDisk` inside an `Attached` component. The profile
  parametrization is unchanged.

- The dense-matrix limit `fitting._DENSE_MASS_MAX` is 4096 coordinates
  (was 8192; a dense 8192 × 8192 matrix is 537 MB in float64, and its
  eigendecomposition for `GaussNewtonMass.condition` is slower still). For
  4096 < n ≤ 8192 (images of 65² to 90²) `gauss_newton_mass` now returns
  the diagonal, with `condition=None` and a `UserWarning`, where it gave a
  dense mass: NUTS started from it has no dense mass. `variational` with
  `"bnaf"`, `"iaf"`, `"mvn"` (an n × n Cholesky factor) or `"laplace"` (an
  n × n Hessian) raises `ValueError` above the limit, before allocating:
  use `laplace_samples` or a diagonal guide such as `autoguide.AutoNormal`.
- `fit(method="lm")` compiles its step kernel once for every chunk and
  every `max_steps`: the step bound is traced, so the short last chunk no
  longer recompiles. Results are unchanged.
- `fit_orbit_aliases` now tempers by default, so its evidences, band
  probabilities and posteriors are not comparable with those of earlier
  untempered runs on data with correlated channels (pass `temper=False` to
  reproduce them). It warns when data with several channels have no
  `EpochPositions` to estimate `f` from, when a block takes the pooled `f`
  for want of chains of its own, and when `f` is outside the range or flat
  across channels (`AliasResult.tempering`). Checkpoints of earlier
  versions are refitted (format 4).

- **`fit_orbit_aliases` needs no per-system arguments.** The defaults
  changed, so results change: `p_range` is optional and defaults to a
  window computed from the data by `period_window` (a Nyquist-style
  sampling limit at the bottom: the companion moves less than half the
  resolution during the longest night, with the median unflagged separation
  as the field, or λ/B_min without positions; 4 times the span at the top;
  no published period; capped at `n_bands_max=200` bands with a warning);
  `sign_free=True` (sign-free seeds are added to the signed ones, so the
  label swap is always explored, and a twin sets `labels_free`); `primary_diam="auto"` (a first pass fits a shared
  uniform-disc diameter to the per-night positions, resolved at a gain of
  5 nats over a point primary, then the diameter is fitted and the seeds
  are refined with it; a point primary otherwise, and also without
  `EpochPositions`); `vis_amp=True` (a per-night V² amplitude in 0.5 to
  1.2 for squared-visibility data, none otherwise); `flip_flux` stays on.
  Pass `p_range=`, `sign_free=False`, `primary_diam=None` or
  `vis_amp=None` for the old behaviour. `AliasResult` gains `labels_free`,
  `period_bound` (the best period is at the top of the window, or its
  σ(log P) is wide: a short arc), `period_at_lower_bound`, `window_capped`,
  `p_range` and `primary_fit`, all kept by `save` and `load`. New:
  `period_window`, `estimate_primary_diameter` and `PrimaryDiameter`.

- **Seeding skips flagged nights by default.** `fit_orbit_aliases`
  accepts the `EpochPositions` of `epoch_positions` and, with
  `skip_flagged=True` (default), a flagged night seeds nothing, though
  its data still count in the likelihood; fewer than two seeding nights
  fall back to random starts with a warning, and seeds from three or
  fewer nights keep random candidates beside them.
  `EpochPositions.positions` and `start_from_positions` also skip flagged
  nights by default (`skip_flagged=False` restores the old behaviour).

- **The default image field comes from the data, not from λ/B_min.**
  `starting_image` took `min(max(500 mas, 6 FWHM), λ/B_min)`; the 500 mas
  was a holdover from one system, and since λ/B_min is 14 to 80 mas for
  every VLTI or CHARA dataset, the field was always λ/B_min and the size of
  the source made no difference (2.9 mas sources on 58 mas fields). It is
  now six FWHMs of the envelope (or of the short baselines' second moment,
  if larger), between four resolution elements across and the
  interferometric field of view; for a 2.9 mas Gaussian seen by 130 m
  baselines at 2.2 µm that is 17 mas. Pass `field_fwhms` to change the
  six.
- **Resolution and field use unflagged samples only.** `beam`,
  `fringe_scales`, `nyquist_pixel_scale`, `field_of_view`, `Epochs.resolution_mas`
  and the pipelines' geometry report no longer count flagged samples,
  whose u, v stay in the data (for example, the provider flags 59 to 95%
  of V² of the 2024 contest's Obj1 GRAVITY at SNR < 1, which gave a beam of
  4.3 × 3.3 mas against 11.6 × 8.1 mas from the unflagged data). Data
  without flags, and projected (kernel, DISCO) data, are unchanged.
- **The ensemble sweeps the data's field.** `EnsembleSpec.field_factors`
  defaults to `(0.75, 1.0, 1.5)` times `image_field(data)` (or `field_mas`),
  never beyond `field_of_view(data)`, instead of `(1, 2, 4)` times
  λ/B_min, which gave fields beyond what the data constrain. The field is
  sized from the envelope fit of `starting_image` (`envelope_fwhm`), so a
  bright star does not shrink it. Every field is now capped at
  `field_of_view(data)` (λ/B_min at most), `field_mas` included, so the
  old 2× and 4× λ/B_min fields are no longer possible; `field_factors`
  above 1 only reach the ceiling. `reference_starts` takes `aperture_mas`.
- **`variational` starts error terms and gains data at their
  Gauss–Newton width.** The start passes `noise=` to `gauss_newton_mass`,
  so fitted error terms get widths from the curvature instead of
  `init_scale`; data with gains, and priors without a least-squares form
  (HalfNormal, Beta), now get a Gauss–Newton start rather than
  `init_scale`. The posterior is unchanged; only the starting width, so
  the number of steps to converge.
- A bounded elementwise prior (`Uniform`, `LogUniform` and so on) now adds
  the curvature of the sampler's transform to `gauss_newton_mass`'s
  matrix, where `Uniform` added none before. The matrix is a
  preconditioner, so this changes the efficiency of NUTS with it, not its
  samples.
- The builders of `gain_modes`, `ClosureNoise`, `DifferentialPhase` and the
  OIFITS reader's frame grouping sort once instead of passing over all
  samples per group (quadratic before); their output is identical. The
  numeric prior fields of `FluxSpectrum` (`widths`, `scale`, `poly_width`)
  and `DifferentialPhase.prior_width` are now NumPy arrays that jit traces,
  so scanning a prior width does not recompile. Read them with `tuple(...)`
  where you compared them with a tuple.
- **Models simplification (no numerical change).** The long derivations and
  accuracy notes in the docstrings of `virgil.models` and
  `virgil.fields.MultipoleField` moved to the new
  [model notes page](docs/api/models/notes.md); the docstrings keep the
  summary, parameters and an example. The class shape checks
  (`__check_init__` and `is_physical`) share one definition, and the ring
  splatting of `TruncatedCone` and `MultipoleDisk` and the in-plane blur of
  `ModulatedGaussianRim` and `MultipoleDisk` each share one helper. Results
  are unchanged to rounding.

### Removed

- **Unused helpers, with no caller in virgil or its analyses.**
  `virgil.tempering.lag_correlation`, `lag1_correlation` and
  `factor_from_correlation` (`estimate_factors` computes the lag statistics
  itself) and `chain_factor` (the exact AR(1) factor, now a helper of the
  tests); `virgil.plotting.plot_hmc_fisher_chainconsumer` (call
  `plot_chainconsumer_diagnostics` for each coordinate system); and
  `virgil.harmonics.image_harmonics_from_latents`
  (`image_harmonics_from_h(step_up(reflection_coefficients(re, im)))`). No
  result changes.

- **`virgil.spectra.Tabulated`.** The 0.3.0 entry promised its removal in
  0.4.0. Replace `Tabulated(ratio, wavel)` with `Nodes(ratio, wavel)`:
  the flux at every wavelength is the same (identical between the end nodes,
  within an ulp beyond them). Two things change. The free values are
  `flux.values`, not `flux.ratio` (fit and prior paths such as
  `"comp.flux.ratio"` become `"comp.flux.values"`), and the reference flux
  `spectrum()` (used by `render` and the zero-baseline normalization) is the
  value at `wavel0`, the first node, not the mean over the nodes; pass
  `wavel0=` to set it. Negative values now pass the constructor and are
  checked on the total flux, as for every other spectrum. Priors on
  `comp.flux.values` are therefore not checked for non-negative support (as
  `comp.flux.ratio` was): keep a non-negative prior (e.g. a log-uniform
  scale) if every channel must stay positive, and pass `flux_param=` where a
  grid or plot auto-detected the old key.

### Migration from 0.3.0

- **`GaussianArc` has a different shape.** Fits of long arcs made with 0.3.0
  change when refitted: the weight is now a von Mises arc, not a Gaussian
  in arc length (about 1e-2 in V at a fifth of the semicircle, more for
  longer arcs; no warning is given). See Changed.
- **`likelihood_grid` takes `data` and `grid` by keyword.** It is
  `likelihood_grid(model, data=..., grid=...)`: only the model is
  positional, so no positional call can swap the model, the data and the
  grid axes. A 0.3-order call, `likelihood_grid(data, model, grid)`, still
  works with a `FutureWarning` until 0.5; any other positional call (such
  as `likelihood_grid(model, data, grid)`) raises a `TypeError` naming the
  keyword call. The 0.4 order change was never silent: the shim recognizes
  data by type and rebinds the arguments, and any other permutation failed
  with an `AttributeError`; the keyword form makes the failure a clear one
  (`virgil-validation` finding F21).
- **Model before data in the grid, limit and detection tools.**
  `likelihood_grid`, `optimized_likelihood_grid`, `optimized_flux_grid`,
  `linear_flux_grid`, `laplace_flux_uncertainty_grid`, `absil_limits`,
  `injection_limits` and `detection_statistics` are now `f(model, data,
  grid, ...)`, as `fit` already was; `gaussian_null`, `rescale_errors`
  and `bootstrap_null` take the null scene first; `injection_recovery`
  is `(model, null_scene, template, grid, key, ...)`. Calls in the 0.3
  order still work, with a `FutureWarning` naming the new call; from 0.5
  they raise a `TypeError` naming it.
- **Model before data in the likelihoods.** `loglike`, `laplace_cov`,
  `laplace_parameter_uncertainty` and `fisher` are now `f(values, params,
  model, data, ...)`, and `joint_prediction` and `joint_loglike` are
  `f(params, model, data)`, with the same `FutureWarning` for the 0.3
  order until 0.5.
- **One name for each argument.** The model is `model` and the data
  `data` everywhere: `whitened_residuals`, `model_loglike`,
  `flux_scale_posterior`, `numpyro_model` and
  `posterior_predictive_summary` (whose order was already right) and the
  reordered functions accept `model_object=`, `model_fn=`, `data_obj=`
  and `observations=` with a `FutureWarning` until 0.5.
- **`samples_dict` is now `grid`** in these functions and in
  `best_grid_point`, `plot_grid_map` and `plot_contrast_curve`, and their
  data argument is `data`. `samples_dict=` and `data_obj=` still work
  with a `FutureWarning` until 0.5.
- **Curvature tools are named for what they compute (`virgil.inference`).**
  `fisher` was the observed information (the Hessian of -log L), while
  `gaussian_fisher` was the expected Fisher matrix, and `laplace_cov` (model
  and data) sat beside `laplace_covariance` (a flat objective). They are now
  `observed_information` (also `virgil.observed_information`),
  `expected_fisher` and `objective_laplace_covariance`; `laplace_cov` keeps
  its name. The old names `fisher`, `gaussian_fisher` and
  `laplace_covariance` still work with a `FutureWarning` until 0.5.
  `laplace_parameter_uncertainty` takes `dtype=` like `laplace_cov` and
  `observed_information` and now computes in float64 by default (it used
  the ambient float32, whose second derivatives are noisy); the grid tools
  keep the ambient precision. `virgil.inference` has an `__all__`.
- **`starting_image` takes everything after `data` by keyword.**
  `starting_image(data, star, oversample, largest_mas, start, hole_mas)`
  in the 0.3 positional form still works with a `FutureWarning` until 0.5;
  `field_fwhms`, `inner_elements` and `aperture_mas` (new in 0.4) are keyword-only outright.
  `clean` already took its options by keyword.
- **`OIData.observable_kind` is now `OIData.vis_encoding`** (`"split"` or
  `"mixed_log_complex"`), so that it is not mistaken for the record key
  `vis_mode`, which sets the visibility channel (`"auto"`, `"v2"`, `"amp"`,
  `"logamp"`; the docstring wrongly called `observable_kind` an alias of it).
  The old attribute still reads, with a `FutureWarning` outside virgil, until 0.5.
- **`DetectionMC.load` checks the file format.** `meta["format"]` (1) was
  written but never read; `load` now raises a `ValueError` for any other
  version and reads a file without it as version 1.
- **`Tabulated` is removed in 0.4.0**, as the 0.3.0 entry promised (see
  Removed above); the other deprecations wait for 0.5.
- To find the remaining 0.3-order calls in your own code before 0.5, add
  `"error:.*stops working in virgil 0.5:FutureWarning"` to your pytest
  `filterwarnings` (as virgil's own test suite now does), or run
  `pytest -W error::FutureWarning`. A message regex given to `-W` on the
  command line is matched literally, so it would not catch them.

### Added

- **Opt-in bandwidth smearing.** `OIData(path, bandwidth=True)` (or `read_oifits(..., bandwidth=True)`) keeps `EFF_BAND`
  as `OIData.bandwidth` (`with_bandwidth(dlambda)` sets it for simulated data);
  models are then smeared over boxcar channels by a `sinc` envelope on each
  component offset (`models.smeared_model`). Off by default: results are
  unchanged.

- **Tutorial: a planet in a gap between inclined rings (stage P5).**
  `notebooks/multipole_disk_rings.ipynb` (docs page `multipole_disk_rings.md`)
  simulates chromatic VLTI-like V² and closure phases of a star, three
  coplanar blackbody rings (the outer one lopsided) and a planet in the
  gap; fits the parametric family with invariant priors (MAP, Laplace
  covariance and evidence); then recovers the rings, their colours, the
  lopsidedness and the planet with one `MultipoleDisk` whose profile, spectral
  index and harmonics are Gaussian processes, and compares the Laplace
  evidences and the prior and posterior of the profile and index.
- **`RadialGP`, `RadialGPFunction`, `MultipoleKernel`, `spectral_index` and
  `freeze_geometry` (stage P4).** `RadialGP` is a whitened Matérn
  Gaussian-process profile (positive, $\exp g$) and `RadialGPFunction` the
  same process as a radial function $g(r)$ (a spectral index or a harmonic
  coefficient), each with latents `z`; `MultipoleKernel` is a prior over the
  harmonic coefficients (`Harmonics(degree=M, prior=...)`); `MultipoleDisk` takes
  `spectral_index=` (number, function or `RadialGPFunction`) with `wavel0=`;
  `freeze_geometry(data)` caches the Bessel matrices; `gp_priors` builds the
  priors, and `fit`, `numpyro_model` (NUTS, flat coordinates) and
  `imaging.log_evidence` (Laplace over the latents) support them. The
  Gaussian-process idea follows FRAP (Yoshida et al. 2026), now credited in
  the `RadialGP` docstring, the contributors page and the bibliography.
- **`MultipoleField` (stage P6).** A multipole-kernel residual field for an `Image`:
  a Gaussian process on a polar grid in a disk's frame (`inc`, `pa`, centre;
  `MultipoleField.from_disk` copies them, or with `geometry_from="disk"` the field
  takes the geometry of its sibling disk whenever the `System` is evaluated,
  with no geometry parameters of its own; `System.resolved()` bakes the ties
  in), with the Matérn kernel of `RadialGP`/`RadialGPFunction` in radius and the azimuthal
  spectrum of `MultipoleKernel` (one shared definition, `nu` up to infinity,
  `nu_theta`), interpolated onto the pixels. Whitened latents,
  so `image_priors`, `fit`, `log_evidence` and `numpyro_model` work as for
  `GaussianField`; `multipole_field_priors` gives bounded log-uniform priors on
  `sigma`, `length_r` and `length_theta`. After the whitened latents of FRAP
  (Yoshida et al. 2026) and frank's polar representation (Jennings et al.
  2020). `FlaredDisk*` is unchanged: its comparison with `MultipoleDisk` is
  tested in `tests/test_flared_vs_multipole_disk.py`.

- **Positive azimuthal harmonics, `VonMises` and `virgil.harmonics` (stage P3).**
  `Harmonics(degree=M)` (or `Harmonics(w=...)` / `re=`, `im=`) is now the
  *positive* form by default: brightness `|h(e^{i theta})|^2` normalized to the
  azimuthal mean, with `h(z) = 1 + h_1 z + ... + h_M z^M` the minimum-phase
  polynomial of complex reflection coefficients `k_j = w_j tanh|w_j| / |w_j|`
  (unconstrained Cartesian latents `w_j`, a Levinson step-up recursion).
  Every non-negative image of degree `M` is reachable (Fejér-Riesz), the image
  is positive for every parameter value, its degree is exactly `M`, and the
  root-flip multimodality of `|h|^2` is removed. Value and gradient are finite
  at `w = 0`. The linear form (`c=`, `s=`) is unchanged and is still chosen
  whenever `c` or `s` is given. `VonMises(kappa, theta0, degree=M)` is a
  `MultipoleDisk` arc with analytic harmonics (a small FFT), and
  `truncation_power()` reports the dropped power. The new module
  `virgil.harmonics` converts every form to and from the image harmonics
  (`to_image_harmonics`, `from_image_harmonics`, `Harmonics.convert`), including
  image-to-positive by spectral factorization, and the legacy `az_amps`/`az_pas`
  conventions of `ModulatedGaussianRim` and `TruncatedCone` (`from_legacy`,
  `to_legacy`). The positive form is credited to Dumitrescu 2007 and
  Barndorff-Nielsen & Schou 1973 in the docstrings and `references/`.

- **`spectra.Scaled(template, ratio)`**: a fixed spectral shape with one
  free reference flux (`ratio` at the template's `wavel0`), so any spectrum (a library SED, a `Sum` with lines) can be a
  companion in the grids, limits and detection statistics, with
  `comp.flux.ratio` as the flux axis; the template's parameters get no
  gradient. Regression tests (`tests/test_chromatic_search.py`) run
  `likelihood_grid`, `linear_flux_grid`, `injection_limits`, `absil_limits`,
  `detection_statistics` and `injection_recovery` on a WR + OB system with V²,
  closure phases, differential phases and a flux spectrum, and show a
  differential-phase-only detection of a 0.1 mas companion that a grey V²/T3 fit
  misses. Tutorial: *Chromatic companion search*.
- **`MultipoleDisk` height and harmonics (stage P2).** `height=` takes `Cone(alpha,
  tip)` or `PowerLawHeight(h_ref, r_ref, psi)` (each ring is lifted along the disk's
  normal: a shift of `h sin(inc)` on the sky), and `harmonics=` takes the
  linear-form `Harmonics(c=, s=)` of Cartesian coefficients, constants or
  functions of the radius (`RadialLinear`, `RadialPower`). `min_intensity()`
  reports where the image would go negative. `ExpTail` gains `cutoff=` (a
  Gauss-Legendre rule on a finite tail). `multipole_disk_cvis` takes `c`, `s`,
  `height` and `stretch`.
- **`TruncatedCone` runs on the `MultipoleDisk` engine** with the same name and
  parameters. Its quadrature along the walls is Gauss-Legendre, which
  converges spectrally instead of as `1 / n_rings**2`: for the Apep-like cone
  (edge-on, baselines to 130 m at 2.2 µm) 40 nodes agree with a converged
  midpoint stack to 1e-6, where 64 midpoint rings are off by 5e-4 and 256 by
  3e-5, at less than half the cost. The model is unchanged (the tail is still
  cut at five lengths). The released behaviour changes: see Changed.
  Credits carry over from the earlier cone code, `frank` and `vis-r`.
- **`MultipoleDisk`**: an axisymmetric disk with a radial profile (`ThinRing`,
  `GaussianRing`, `RadialPowerLaw` with sharp or soft edges, `ExpTail`, `Gap`
  and `RadialSum`), inclined and rotated, with an optional Gaussian blur,
  built on `multipole_disk_cvis`, a Hankel transform by quadrature that each
  profile owns the nodes of (Gauss-Legendre, or Gauss-Laguerre for an
  exponential tail). It follows the ideas of frank (Jennings et al. 2020) and
  vis-r (Kennedy 2025), which are credited in the docstrings and on the
  contributors page; no code was copied. This first stage is the
  azimuthally symmetric disk (degree 0).
- **`fit_orbit_aliases` handles a resolved primary, contrast loss and a
  bright companion.** `primary_diam` gives the primary a uniform disk (fixed
  or `"fit"`, one shared log-uniform diameter); `vis_amp=(lo, hi)` gives every
  epoch's V² its own profiled amplitude (a contrast loss independent of
  baseline and wavelength); `flip_flux` (default) refits a mode whose flux
  ratio is pinned at the top of `flux_range` with the stars' labels swapped
  (`AliasBandResult.flipped`); `init=` is a warm start that is always refined
  in the band holding its period.
- **Alias fit for near-equal binaries.** `fit_orbit_aliases(...,
  sign_free=True)` and `starting_orbits(..., sign_free=True)` seed from
  positions known only up to a sign: with flux ratios of 0.9-1 the closure
  phases cannot tell `(dra, ddec)` from `(-dra, -ddec)` epoch by epoch, and
  seeding with the wrong sign mix landed on the wrong alias (AL Dor, HD 41255,
  HD 188088). The fit also seeds fluxes near the top of `flux_range`.
  L-BFGS is now followed by Levenberg-Marquardt polishing, aimed at the
  `not-converged` and `hessian-not-positive-definite` flags.

- **Variational inference** (virgil#12). `virgil.svi.variational(model,
  priors, data, ..., start=fit_result)` fits a guide to `numpyro_model`'s
  posterior with numpyro's SVI and returns a `VariationalResult`: draws in
  the model's parameters under NUTS's site names, the loss history, the
  guide and its parameters, a convergence flag and the PSIS k̂. Every guide
  starts at the Laplace approximation (the fit and its Gauss–Newton
  covariance, at the nominal errors and without regularizers; with
  regularizers it starts at `init_scale` instead) and learns in units of
  its widths. The default is a block
  neural autoregressive flow (`"bnaf"`), which follows curved, skewed and
  bounded posteriors; `"iaf"`, `"mvn"` (a Gaussian, ≈ Laplace) and
  `"laplace"` are the alternatives. float64 by default, as for `fit`;
  `flat_coordinates` is passed through. New page: Variational inference.
- **`virgil.aliases.fit_orbit_aliases`.** Orbits whose epochs undersample the
  period: alias bands `N = round(T/P)`, each fitted to the closure phases of
  all epochs with per-epoch error scales integrated out, with Laplace and
  importance-sampling evidences, band probabilities and posterior samples.
  `AliasResult.table()` gives one row per band (`scales` flat over epochs
  and blocks, `epoch_scales` per epoch). `checkpoint_dir=` writes each band
  (`band_<N>.pkl`, `band_<N>.json`) as soon as it is fitted, and a rerun
  with the same data, positions and settings loads it instead of
  refitting; `on_band=` is called with each `AliasBandResult`. Each band
  has its own random stream, so a resumed run reproduces an uninterrupted
  one. A band that raises becomes a row with the flag `band-failed` and
  `AliasBandResult.error`, and a singular or ill-conditioned Hessian gets
  the flag `hessian-singular`. The checkpoint key hashes every field of
  the data and positions, static ones included. The likelihood and its
  gradient are evaluated once before any band is fitted, so a
  misconfiguration raises there; afterwards only a `TypeError` propagates.
  Other failed optimizer starts are counted (`starts-failed`,
  `AliasBandResult.n_starts_failed`), a band with no finite mode is a
  `no-valid-mode` row, and a `RuntimeError` is raised when every band
  fails. The likelihood compiles once per data shape, not once per mode.
- **Orbit API tidy-up before release.** `score_orbits(model, epochs,
  orbits, ...)` takes the model before the data like the rest of the
  package (the order `score_orbits(epochs, model, ...)` works with a
  `FutureWarning` for one release); `score_orbits` takes `model(orbit,
  flux)` and `rank_orbits` takes `model(orbit)`. `rank_orbits` and
  `start_from_positions` default to `scales="marginal"`, without a
  warning, and `EpochPositions.gap` is gone (`gap_marginal` is the gap).
  `AliasResult.save` writes a `format` field (1); `load` reads files
  without it and refuses a newer format; `UNRECORDED` is no longer in
  `virgil.aliases.__all__`. `aliases.spectral_window` is removed (unused;
  `|mean exp(2πi f t)|²` is one line of numpy).
- **Orbit-search scorer.** `virgil.orbit_search.score_orbits(model,
  epochs, orbits, shared=SharedFlux(...), terms=(...), scales="marginal")`
  scores a stack of candidate orbits on all epochs in one compiled,
  `lax.map`-batched kernel. The companion flux is shared: one flux per
  band (optionally f₀(λ/λ₀)^β), at most 1 in a reference band, integrated
  out under a log-uniform prior and also reported profiled, with an
  uncertainty from the curvature. The coarse grid only seeds Newton steps
  to the profiled peak; Gauss–Legendre nodes on each side, out to a 12-nat
  drop or the prior's bound, then integrate it, so the marginal is within
  0.05 nat of brute-force quadrature wherever the peak falls (a coarse-grid
  fallback is flagged in `OrbitScores.fallback`). Each dataset's error scales are
  marginalized as in `marginal_loglike`, bounded by default to
  [1/5, 5] (`s_max=5`; `s_max=None` for the unbounded closed form). The
  scale each dataset needs at the profiled flux is reported in
  `OrbitScores.scale`, and `OrbitScores.scale_at_bound` flags candidates
  whose scale posterior presses against `s_max`. Gains (`OIData.with_gains`) and
  closure-phase offsets (`OIData.with_closure_offsets`), which
  `marginal_loglike` refuses, are profiled analytically, at one degree of
  freedom each. `RVData`/`PositionData` terms or callables are added.
  Positions are evaluated at every sample's own time, and the visibility
  of a unit companion is computed once per candidate, with every flux
  point cheap arithmetic on it. `max_evaluations` refuses a run whose
  predicted cost (candidates × datasets × score evaluations) is over budget.
  `rank_scores` / `OrbitScores.order` rank with ties broken on quantized
  scores, then by index.
- **Pipelines (`virgil.pipeline`, the `pipeline` extra).**
  `BinaryPipeline` (companion search, contrast limits, fit and NUTS
  posterior) and `StarPipeline` (angular diameter and limb darkening,
  with a registry of models by short name) run fixed, named stages on
  public virgil functions and write a run folder: `run.json`,
  `summary.json`, HDF5 grids and samples, plots, the processed OIFITS and
  an executed quicklook notebook, with deterministic quality checks
  (`Check`). Runs resume; a run folder whose settings, data or template
  differ raises `ConfigMismatchError`. `load` reloads a run as a `Result`,
  and the `virgil-pipeline` command runs a pipeline and prints, validates
  or re-renders the quicklook of a run folder.
  `import virgil` does not import the pipelines. See the Pipelines page.
  `StarPipeline` compares diameter lobes and models on −2 log L
  (`model_loglike`, so with `error_scale="fit"` each candidate is scored
  at its own maximum-likelihood error scales, not on a χ² that those
  scales make equal to N), walks the registry's parent chain with a BIC
  penalty (`summary["comparison"]` decides, `summary["comparisons"]` lists
  all), and keeps the uniform disk's diameter scan with its scales
  profiled, refining the best diameter between grid points. Re-running a
  stage deletes every later stage's outputs, and `Result` serves only
  files and grid groups of complete stages; `run.json` records provenance
  per stage, and `virgil-pipeline validate` fails a run left `running`. The look-elsewhere significance is computed in log
  space, so it stays finite for large searches.
- **`virgil.ensemble`**: PYRA/MYTHRA-style reconstruction ensembles.
  `EnsembleSpec`, `draw_groups`, `run_group` (one L-curve per geometry
  group), `combine` (selection and the iterative mean) and `ensemble`
  return an `Ensemble` (mean `Image`, per-pixel σ, raw χ²/N per dataset).
  The L-curve window keeps the weights from `window_dex` below the corner
  up to it; the default weight ranges (`tsv` 0.1–10⁴, the others 10⁻⁴–10
  per data point) were set from the corners of the 60 datasets of
  virgil-validation's contest bench, and each sweep draws one log-uniform
  weight per equal bin of `log w`. The default `field_factors` are
  (1, 2, 4) times `field_of_view(data)`, capped at 1 on a uv lattice, and
  a field too large for `max_npix` keeps its size with coarser pixels.
  The iterative mean admits a member if every dataset's χ² stays within
  `mean_rtol` (by default each dataset's χ²/N noise √(2/N)) of the best
  member's; members whose fit diverged are dropped with
  `reason="diverged"`, and `combine` warns when fewer than `min_kept` (3)
  are kept.
- **`virgil.metrics`**: scores for a reconstructed image against a known
  truth, from the imaging contests' published definitions:
  `lawson_sigma_over_peak` (both images normalized over the whole array),
  `rms_convolved` (after a Gaussian beam; `pixel_scale_mas` is needed only
  with a beam), `l1_score` and `ncc`.
- **Azimuthally modulated cones.** `TruncatedCone(..., az_amps=, az_pas=)`
  multiplies every ring's brightness by `1 + Σ A_m cos(m (φ - φ_m))` in
  its own azimuth, e.g. for a colliding-wind shock brighter on its
  leading edge. The azimuths follow `ModulatedGaussianRim`'s convention
  (in the ring's plane, in the sense of position angle, `pa ± 90` on the
  walls). The visibilities stay analytic (one `J_m` per ring and order)
  and the render matches them. Unmodulated cones are unchanged and carry
  no empty leaves.
- **Fit budgets and guards.** `fit(..., time_limit=, progress=)`: LM and
  L-BFGS run in chunks of steps, check the wall clock between them and
  stop, unconverged, with `info["stop"] == "time"` (L-BFGS carries its
  whole state across chunks, so its path is unchanged; LM restarts each
  chunk with its damping reset); `progress` reports steps, loss, gradient
  norm and elapsed time between chunks. `info["stop"]` gives every
  unconverged fit's reason (`"limit"`, `"time"`, `"stalled"`,
  `"non-finite"`, `"failed"`), a non-finite loss now stops L-BFGS, a fit
  by any method (Adam included) that ends on a non-finite loss is now
  marked unconverged, and `info["at_bound"]` lists parameters that ended
  at a genuine edge of a Uniform, LogUniform or other interval prior
  (not at the poles of isotropic inclinations or latitudes).
  `start_from_positions` forwards `time_limit`, keeps going when one
  refinement fit raises (recorded in `OrbitStart.failed`; a `TypeError`
  or `ValueError`, such as a misspelt option, still raises at once, and
  if every fit fails the `RuntimeError` is chained to the last error),
  sorts non-finite losses last and leaves them out of `modes()`.
- `orbits.period_grid(times, p_min, p_max, k=9)`: trial periods uniform
  in frequency with δP ≤ P²/(kT) over the baseline T, finer than
  ARMADA's and The Joker's rules; `start_from_positions` warns when its
  `periods` are coarser than that over the seeding datasets.
- `OIData.select` takes `ranges=` (a union of wavelength windows),
  `exclude=` (windows to drop, e.g. a line) and `observables=` (`"vis"`,
  `"phi"` and the extra observables' kinds), so closure-phase-only or
  line-free subsets need no hand-flagged records. Dropping the closure
  phases rebuilds differential phases with their closure part, and gains
  go with the visibilities.
- Starting and sampling orbit fits to multi-epoch visibilities
  (`design/visibility_orbits.md`, stage 2), all model first:
  `rank_orbits(model, data, orbits)` ranks trial orbits (e.g. from
  `starting_orbits`) by the log likelihood of all the epochs, one snapshot
  per dataset; `chain_starts` picks the best distinct orbits (modes judged
  by the companion's positions at the epochs), one per chain;
  `epoch_positions` fits the companion's position and covariance in each
  dataset; `start_from_positions(model, priors, data, start_values, ...)`
  chains these into a start (positions, starting orbits, ranking, `fit`
  from distinct orbits) returned as an `OrbitStart`, whose `chain_values`
  give one start per distinct mode; and `likelihood.chain_init_params`
  turns one start per chain into numpyro's `init_params`.
  `Epochs.resolution_mas` gives the data's finest λ/B.
  - `Epochs` and `OrbitalBinary` (a binary on its orbit, observed as one
    snapshot per dataset; `design/visibility_orbits.md`, stage 1).
  - `epoch_positions` scores its grid on the scale-marginalized surface
    m = -Σ_b (ν_b/2) ln χ²_b, in which each dataset's V² and
    closure-phase error scales are integrated out under their Jeffreys
    priors (`epochs.marginal_loglike`; `s_max=` integrates each block's
    scale adaptively in ln s over a bounded range, matching
    `scipy.integrate.quad` to about 1e-11, and `dof=` sets an
    effective-dof fraction). Positions, covariances and
    `EpochPositions.gap_marginal` therefore do not change when one
    dataset's errors are rescaled. It refines the top `n_peaks=5` distinct
    local maxima of the grid, takes the best refined one, and measures
    `gap_marginal` against its best refined rival; `EpochPositions.peaks`
    holds each dataset's catalogue (`EpochPeaks`: positions, fluxes,
    scores, Laplace-mass weights, `edge` and `laplace` flags). `decisive`, `positions(min_gap=...)` and
    `start_from_positions(min_gap=...)` compare `gap_marginal`. Raw χ²/N
    and the profiled scales are recorded (`chi2_raw`, `scale`), with a
    `UserWarning` naming datasets with raw χ²/N > 4. Data with gains,
    closure-phase offsets or a model-dependent covariance raise
    `NotImplementedError` on the marginalized surface.
  - `rank_orbits` and `start_from_positions` take `scales="quoted" |
    "marginal"`; `"marginal"` integrates each dataset's error scales out,
    so the worst-calibrated nights do not dominate the ranking, and is the
    default (as in `score_orbits`). Rankings use
    `orbit_search.rank_scores` (ties within 1e-3 nat broken by index,
    non-finite scores last). `OrbitStart.seeded` and `.ambiguous` name the
    datasets that seeded the orbits and those left out (with a
    `UserWarning`).
- The orbit tutorial now fits one snapshot per night with `Epochs` and
  `OrbitalBinary` (the default for orbits whose motion within a night is
  negligible), starts from `start_from_positions`, and starts one NUTS
  chain per distinct mode.
- A private `virgil._deprecate` module implements the 0.4 change to a
  single model-before-data argument order and the renamed keywords (see
  "Migration from 0.3.0"): `old_order` accepts calls in the 0.3
  positional order, recognized by which argument is the `OIData`, with a
  `FutureWarning` that names the new call (with `removed=True`, for 0.5,
  a `TypeError`); `old_order` and `renamed` accept the old keyword names
  with a `FutureWarning`, and passing an old name with its new one is a
  `TypeError`.

### Changed

- **`GaussianArc` is a von Mises arc (a change of shape).** The weight along
  the arc is now `exp(kappa cos(angle - pa))` with `kappa = (2 sqrt(2 ln 2)
  radius / length)^2`, where it was a Gaussian in arc length taken once round
  the circle and cut off at the antipode of `pa`. `length` is still the FWHM
  along the arc and the parameter names and paths are unchanged, and the two
  shapes agree to about 1e-3 in V for arcs with `length` below a tenth of the
  semicircle (1e-2 at a fifth) but differ for long arcs, so refit if you rely
  on those. The visibility is analytic (a `ThinRing` with `VonMises` harmonics
  and a blur on the `MultipoleDisk` engine); `degree` (default chosen from the
  starting `radius` and `length`) replaces `nodes` for the visibility, and
  `nodes` only samples the rendered image. Arcs narrower than 256 harmonics
  allow (`length / radius` below about 0.05) are integrated along the arc
  with the von Mises weight, at `nodes` points, as before (`quadrature=True`),
  so their cost does not grow; `is_physical` is false when the series drops
  more than 1e-4 of its power. The shape differs from the one released
  in 0.3.0, and building a `GaussianArc` gives no warning (see Migration
  from 0.3.0).
- **`ModulatedGaussianRim` runs on the `MultipoleDisk` engine** (a `ThinRing`,
  linear harmonics converted from `az_amps`/`az_pas`, and the blur), with the
  same name and parameters; it agrees with the old code to rounding.
- **`fit_orbit_aliases` and `epoch_positions` optimize with damped Newton
  steps.** Each start of the alias search, and each peak refinement of
  `epoch_positions`, now runs Levenberg–Marquardt on the
  iteratively-reweighted curvature of the scale-marginalized likelihood,
  instead of L-BFGS-B with restarts and a Newton polish; L-BFGS-B remains
  the fallback for a start that does not converge. It reaches the same
  optima (positions agree to 1e-4 mas) and is faster. Which local
  optimum a start far from the ridge reaches can differ, so a band's modes
  and evidence can change for such starts. `fit_orbit_aliases(max_steps=)`
  now caps the kernel's iterations (rejected steps count) and the
  L-BFGS-B fallback's, instead of each L-BFGS-B run's; a refinement of
  `epoch_positions` that does not converge keeps its grid point.
  `_Surface.weights` is new
  (private).
- **`TruncatedCone` (released in 0.3.0) changes numerically.** Its default
  `n_rings` goes from 32 to 48, since too few Gauss-Legendre nodes fail
  sharply at long baselines, and its visibilities at fixed parameters shift
  by the old midpoint rule's error (about 8e-4 at the old default, worth
  Delta log L of about 4 per GRAVITY epoch), so a 0.3 fit rerun gives a
  different chi-squared and best fit with no API change. The per-call cost
  at the default rises (48 Gauss-Legendre nodes against 32 midpoint rings);
  it is lower only at equal accuracy. `rings()` and `render()` now use the
  same Gauss-Legendre nodes as `model()`.
- **`fit_orbit_aliases` flips a pinned flux ratio by default** (`flip_flux=True`): a mode at the top of
  `flux_range` is refit with the stars' labels swapped, which can add a mode to a band's evidence. It also adds
  settings to the checkpoint key, so existing `checkpoint_dir`s are refit. `flip_flux=False` restores the old fits.
- **The evidence helpers take several field Images.** `log_evidence`,
  `error_scale` and `laplace_samples` accept `path=None` (every Image whose
  log-brightness is a `GaussianField`) or a tuple of paths, and take the
  Jacobian over all those latents together, so a compact and an extended
  GP Image in one `System` get a joint Laplace evidence. `laplace_samples`
  returns `"latents"` as a dict keyed by Image path when it draws more than
  one. Single-Image results and the default `path="env"` are unchanged.

- **NUTS can sample each prior in its flat coordinate (opt-in,
  `flat_coordinates=True`, #301).** `numpyro_model(...,
  flat_coordinates=True)` maps a `LogUniform` scale to NUTS's
  unconstrained coordinate through log x, an `IsotropicInclination`
  through cos i and an `IsotropicLatitude` through sin(lat), for
  parameters and `noise=` terms alike, as `fit` optimizes them. Sites keep
  their names, samples are the model's own parameters, and the posterior
  is unchanged. It is off by default: simulation-based calibration (1000
  replicates) showed no gain in effective sample size per second or in
  divergences, and rank uniformity was the same. The 62² imaging notebook,
  with `gauss_newton_mass`, ran in 67 s instead of 550 s with it, so the
  tutorials that sample large images pass `flat_coordinates=True`. Pass
  the same value to `gauss_newton_mass`, `variational` and
  `chain_init_params`' model. See `design/sampler_flat_coordinates.md`.
- **Warnings and precision (#306).** `Tabulated` now warns with `FutureWarning` (as every other deprecation does) instead of the default-hidden `DeprecationWarning`; the four warnings that set no category (`Ensemble` too few members kept, every closure phase flagged, unmatched T3 legs, `t_ref = 0` with MJD times) now say `UserWarning`. `cvis_uniform_disk` takes `diam` like the other disks (`ud=` still works with a `FutureWarning`). The two pixel-sum matmuls in `GaussianArc`/`TruncatedCone` use `Precision.HIGHEST`, as the rest of the forward model does (TF32 on A100/H100 otherwise). The AMIGO covariance check takes its tiny-number floor from the array's dtype. `tests/_test_data.py` finds `data/` from its own location, not the working directory.
- **Oxford -ize spelling in names (#306).** `fit`, `numpyro_model` and `diagnose` take `regularizers` (not `regularisers`), `imaging.l_curve` takes `regularizer` (not `regulariser`), `RVData.term` takes `marginalize_offsets` (not `marginalise_offsets`), and the numpyro factor sites for regularizers are named `regularizer_<i>`. The old keywords still work with a `FutureWarning` until 0.5 (`_deprecate.renamed`). Private names (`_linear.standardize`, `_diagonalized`, `_summarize`, `_normalize`) and the spelling in docstrings, comments and the hand-written docs follow. `barycenter` and the already -ize public names (`regularized_inverse`, `optimized_*`, `standardize_model`, `Image(normalize=)`) are unchanged.
- **Deprecated parameter names are gone from public signatures (#306).** `OIData.model`, `OIData.with_model` and every `observables.*.predict` take `model` (not `model_object`), `inflated_errors` takes `data` (not `data_obj`), and `joint_data` / `joint_errors` take `data` (not `observations`). The old keywords still work with a `FutureWarning` until 0.5, through the same `_deprecate.renamed` path as the other functions. Private helpers in `likelihood`, `inference` and `oidata` were renamed to match.
- **One import convention (#306).** `import jax.numpy as np` and `import numpy as onp` everywhere in `src`, `tests`, `examples`, `scripts` and `design/sketches` (six `src` modules and three tests had the two swapped, so code moved between files silently changed backend). Ruff's `ICN001` now enforces it; notebooks keep their old aliases until they are next re-executed.
- **Duplicated and dead code removed (part of #310; no behaviour change).**
  `legacy.oifits_implaneia.GetWavelength`, `Format_STAINDEX_V2`,
  `Format_STAINDEX_T3`, `rad2mas` and the `cp_indices` re-export are
  deprecated (`FutureWarning`; removal in 0.6), since nothing uses them.
  Internally: `FWHM_PER_SIGMA` moved to `_utils` (still importable from
  `detection`) and replaces the hand-written `2.3548` and
  `2*sqrt(2 ln 2)` (the two literal `2.3548`s in `imaging` now use the
  exact value, a 1e-5 relative change in an envelope-fit starting width);
  `coverage` uses `_geometry.rotate` (which now stays in NumPy for NumPy inputs); `meshgrid_vectors` is
  `coordinate_points`; one `_unit_sum`; the `batch_size` and noise-term
  "Priors" docstrings are stated once and referred to. Unused
  `_utils.i2pi` removed.
- **Warnings name the caller's line.** Deprecation `FutureWarning`s and
  virgil's other warnings are attributed to the first frame outside virgil
  (`_deprecate.external_stacklevel`), not to a fixed depth, which pointed
  into equinox's method wrappers (`OIData.model`, `with_model`,
  `observables.*.predict`, `RVData.term`, `Tabulated`) or into the
  `renamed` wrapper (`fit`'s own warnings); frames of JAX, numpyro and
  the optimizers are looked through too. Shim messages name methods as
  `Class.method`, and a new name passed positionally beside its old
  keyword raises the shim's "got both" `TypeError`.
- `_geometry.position_angle` never returns exactly 360 (it did for tiny
  negative offsets; a scalar input still gives a NumPy scalar),
  `KeplerOrbit.frame()["towards_primary"]` is wrapped to `[0, 360)`, and
  the geometry helpers compute lists and array-likes (e.g. a pandas
  Series) in NumPy float64 rather than JAX float32.
- Differential-phase steps and anchors use one wrap, `[-π, π)`, in data
  and model alike (a step of exactly ±π used to wrap differently).
- `Centroid` sums over pixels at `Precision.HIGHEST`; the `oifits` module
  no longer imports JAX itself (`import virgil` still does).
- `fit` falls back to L-BFGS only when the objective has no least-squares
  form (`fitting.NoLeastSquaresForm`), and `variational` starts at
  `init_scale` only for that, a singular curvature
  (`fitting.SingularCurvature`) or a parameter with no starting value
  (`fitting.NoStartingValue`, a `ValueError`); other errors propagate
  instead of being taken for one of these. `gauss_newton_mass` raises
  `NoLeastSquaresForm` for data with gains, as `fit`'s Gauss–Newton path
  does, instead of ignoring them. SVI warnings state their category. `fit(progress=)`
  also reports when the fit finishes, and `fitting._bound_fraction` gives
  a LogUniform's fraction in x when `flat=False`.
- `numpyro_model`'s function and `gauss_newton_mass`'s result record
  `flat_coordinates`, and `chain_init_params(..., mass=)` raises if they
  differ (a mass matrix in the other coordinates is never adapted). The
  docs now describe flat-coordinate sampling as opt-in.

### Fixed

- **The pipelines no longer fail with numpyro's "Cannot find valid initial parameters" on a non-detection.** With no companion the fitted flux runs to the lower end of its prior (and an unresolved star's diameter to the edge of `diam_range_mas`); NUTS then started exactly on the bound, where the unconstrained value is infinite. `BinaryPipeline`, `StarPipeline` and `DiskPipeline` now start the chains at least 1e-3 of the prior's width inside any bound (so a start within 0.1% of a bound, which `fit` reports in `at_bound`, moves slightly and gives different draws for the same seed), a start that is still not finite raises a `ValueError` naming the start values instead of numpyro's `RuntimeError`, and `BinaryPipeline` and `StarPipeline` add a `start_at_bound` warning check when the best fit sits at a bound.

- **Projected correlated closure phases keep only their real directions in float32.** When an operator (a kernel-phase or DISCO phase projection) is applied to closure phases with four or more telescopes, their correlation matrix is singular, and `OIData` rotates the operator onto the eigenvectors of the projected covariance with a cut at 1e-10 of the largest eigenvalue. That is below float32 eps, so the numerically zero directions were kept (33 to 36 outputs where 30 are real, for ten four-telescope frames, in float32), with errors of 1e-5 against 1e-2 and so a weight 10⁶ times that of a real one. The covariance and its `eigh` are now computed on the host in float64, as this runs once at construction and is never traced; the arrays returned are in the default dtype. Operators that already whiten the covariance (as DISCO operators do) take the unchanged diagonal branch.

- **`MultipoleDisk.is_physical` works with a `VonMises` azimuthal model.** It read `harmonics.prior`, which `VonMises` does not have, so `model_loglike(..., reject_unphysical=True)` and `System.is_physical` raised `AttributeError` on any scene containing one.

- **`SharedFlux` grids are no longer misread as `(lo, hi, n)`.** A three-value tuple was a range whenever its last element was at least 1, so `SharedFlux(grid=(0.01, 0.1, 1.0))` became the single flux 0.01. The `(lo, hi, n)` form now needs a tuple with an integer `n` (Python or NumPy); a list, an array or a tuple whose last element is a float is a list of flux values (the same for `slope`).

- **`Precision.HIGHEST` in more contractions.** The limb-darkening coefficient matmul (binomials up to C(22, 11) with alternating signs), the `RadialGP` and `MultipoleKernel` contractions and the DISCO operator application now request full precision, as the other forward-model matmuls do; on A100/H100 the default (TF32) loses the cancellation.

- **Bandwidth smearing for binaries, rotated scenes and resolved backgrounds.** With `bandwidth` set, `BinaryModelCartesian`, `BinaryModelAngular`, `Rotated` scenes and `System(star, bg=Resolved())` raised a `TypeError` saying only that Systems of Components are supported. They are now smeared (binaries as their `to_system()`, `Resolved` has no offset to smear, `Rotated` rotates the baselines), and an unsupported model, such as an `OrbitalBinary` that is not a snapshot, raises a `TypeError` that names it and lists the supported ones.

- **`fit(method="lbfgs")` restarts a round that crawls.** L-BFGS now runs in rounds of 2000 steps and restarts from the last point with fresh curvature memory (keeping the first round's gradient tolerance) while a round that used all its steps still gains in the loss, and runs the rest of `max_steps` as one last round once it does not, instead of spending all 20 000 steps with a memory spoilt by a parameter running into a bound. On macOS the limb-darkened `StarPipeline(error_scale="fit")` fit crawled to the step limit with χ² at the fitted scales 94 against N = 81; Linux converged in under 700 steps. Fits that converge, stall or time out within 2000 steps are unchanged, and `info["steps"]` is the total over the rounds.

- **Alias fits reach the likelihood ridge.** `fit_orbit_aliases`'
  L-BFGS-B refinement stopped on its relative-reduction test 50-800 nats
  short of the optimum on a many-cycle orbit's narrow ridge, so about one
  noise draw in four settled in a neighbouring alias. Each start now
  restarts from its own optimum until it stops improving, and a winning
  band with χ²/ν above 10 on the quoted errors (a missed ridge gives
  65-800) is flagged `poor-fit` with a `UserWarning` instead of being
  reported silently; errors underestimated by more than 3x also trip it.
- **Differential-phase steps near ±π wrap the same in data and model.**
  A step of exactly π wrapped to -π in the data but, one ulp lower from
  `arctan2` on macOS, to +π in the model, a 2π disagreement. The model's
  steps are now wrapped to within π of the data's, as its anchor already
  was, so no cut can separate them.
- **Repeated refinements in `start_from_positions` compile once.**
- **Faster, steadier CI.** The slowest tests share their expensive
  fixtures (kept on one xdist worker with `--dist loadgroup`) and use
  smaller problems; one pull-request job runs the slow tests, so they no
  longer fail first on main; the `dev` extra includes `orbits`, so the
  orbit tests run locally rather than skipping.
- **Isotropic priors have finite gradients at the poles in float32.**
  `IsotropicInclination.icdf` and `IsotropicLatitude.icdf` gave a NaN
  derivative within about 1e-8 of a pole (the branch not taken of a
  `where` had an infinite slope), and their final clip a slope of 1/2 at
  the bounds; both now keep the true derivative.
- **Drifted copies unified (part of #310).**
  - **Zero errors.** `OIData` now drops V² and phase samples with a zero
    (or negative) error, as it already did for extra observables
    (`OI_FLUX`, `|V|`, `T3AMP`); a zero error gives an infinite whitened
    residual, which made every likelihood and fit non-finite. Flagged
    and non-finite samples were already dropped. Raw errors that
    `vis_mat`/`phi_mat` project may still be zero; a zero *projected*
    error raises a `ValueError`, also when a correlating operator would
    otherwise drop that direction silently.
  - **Field of view.** `imaging.field_of_view` and `nyquist_pixel_scale`
    (and the new `Epochs.resolution_mas` and pipelines) share
    `_geometry.fringe_scales`: per-sample `B/λ` over non-zero baselines,
    rather than a longest wavelength and a shortest baseline that are not
    observed together when the uv sampling differs between channels.
  - **Position angles.** One `_geometry.position_angle` /
    `separation_pa` (North through East, `[0, 360)`) serves the models,
    orbits and plotting. `KeplerOrbit.frame(...)["line_pa"]`
    was in (-180, 180]; it is now in `[0, 360)` (equal modulo 360, and
    `towards_primary` is still `line_pa + 180`).
  - **Phase units** (`oifits` columns and `OIData` records) go through
    one parser, with one error message; a blank `TUNIT` or `phi_unit` now
    means the default unit in both. One check for the names of
    `System` components and `Sum` parts (derived from the class's own
    fields) and one `wrap_phase`.
- **`read_oifits` reads closure triangles whose leg has no visibility
  row** ([#299](https://github.com/benjaminpope/virgil/issues/299)). A
  triangle leg stored in no `OI_VIS2`/`OI_VIS` row in either orientation
  raised a `ValueError`, so MIRC-X files and some ESO phase-3 PIONIER
  products (which omit a baseline's V²) could not be read. `OI_T3` is
  self-contained in the OIFITS standard, so such a leg now gets a flagged
  sample at the T3 row's own coordinates (`(U1COORD, V1COORD)`,
  `(U2COORD, V2COORD)` or their sum), shared by later triangles of the
  same frame, as reversed legs already were. One `UserWarning` per file
  gives the number of legs placed this way. The `ValueError` remains when
  those coordinates are NaN or zero, or disagree with the triangle's other
  legs (a table written with the opposite baseline direction).
- Parallel numpyro chains (`chain_method="parallel"`) on data with
  `with_gains` no longer segfault. `GainModes` held zero-size arrays when
  no mode spans frames, and XLA's Shardy pass crashes compiling a
  `jax.pmap` that captures one (JAX 0.11.2). `GainModes.spanning` and
  `spanning_group` are now `None` then. `numpyro_model` warns, suggesting
  `chain_method="vectorized"`, when the data still hold zero-size arrays
  (e.g. no closure phases) and more than one device is visible.
- `limits.chi2ppf` works for any number of degrees of freedom on a
  standard install. For `df != 1` it called numpyro's `gammaincinv`,
  which needs `tensorflow_probability`, not a virgil dependency; it now
  inverts `jax.scipy.special.gammainc` by Halley's method for every `df`,
  matching `scipy.stats.chi2.ppf` to about 1e-14 relative in float64
  for p in [1e-10, 1 - 1e-10], and stays differentiable and `jit`-able.
  `df = 1` also uses it, gaining precision at small p. It returns NaN
  for `df <= 0`, and no longer needs `tensorflow_probability`. This
  supersedes the removal of `chi2ppf` in #293, which is reverted. Found
  by virgil-validation (F17).

## 0.3.0 (2026-10-06)

Everything listed as new was added after 0.2.0.

### Migration from 0.2.0

Most of 0.3.0 is additive. These are the changes that can alter an existing
analysis or warn:

- **`fit` with a `LogUniform` prior now returns the MAP in `log x`.** This
  covers priors on parameters and on `noise=` terms (#226). Each prior is
  optimized in its flat coordinate (`LogUniform(a, b)` in `log x`; isotropic
  priors in `cos i` or `sin(lat)`), where it is constant, so the fit is the
  maximum of the likelihood inside the prior's range. In 0.2.0 it was the mode
  of likelihood × `1/x` in `x`, which pulled scales towards small values, so
  fitted values of scales (fluxes, diameters, noise terms) can shift.
  Levenberg–Marquardt is now chosen automatically for these priors, where
  0.2.0 used L-BFGS and `method="lm"` raised. Uniform, Normal and other
  priors are unchanged. See "Priors and the MAP" in the conventions.
- **Frames are grouped by `MJD` and `TIME` (#203), which changes χ² for
  OIFITS v1 files** that keep the snapshot in `TIME` rather than `MJD`
  (OYSTER output, the 2004 Interferometry Beauty Contest files). Snapshots
  that share an `MJD` used to be one frame, and closure phases from
  different snapshots were whitened as one correlated group. They are now
  separate frames: for the 2004 contest files the independent closure phases
  go from 10/130 to 130/130, and every χ² and uncertainty from them
  changes. Files with a constant `TIME`, and the other files we tested
  (AMBER, CHARA, GRAVITY, MIRC), are unchanged.
- **`starting_image` uses `LogUniform` priors** on the envelope width and
  flux (flux bounds 1e-4 to 100), so the starting point, and a fit that
  begins from it, can differ slightly from 0.2.0.
- **Keyword-only grid and limit arguments.** `flux_param` and `batch_size`
  must be passed by keyword to `likelihood_grid`, `optimized_likelihood_grid`,
  `optimized_flux_grid` and `linear_flux_grid`, and `flux_param`,
  `flux_bounds` and `batch_size` to `absil_limits` and `injection_limits`;
  positional calls from 0.2.0 raise `TypeError`. A bare `(mean, sd)` tuple
  as `linear_flux_grid(prior=)` is a `TypeError` (use `dist.Normal`), and
  `RVData.term(marginalise_offsets=True)` needs a stated zero-point prior.
- **`injection_limits(flux_bounds=None)` limits change** where 0.2.0
  returned the top of its search range (1000); both limit functions now
  find the first crossing from below. Bounded results agree with 0.2.0 to
  about 1e-4.
- **L-BFGS keeps 50 past steps** (`fit(lbfgs_memory=50)`, was optax's 10),
  so L-BFGS fits, regularized images especially, change slightly.
- **`total_mass` and `distance_pc` use GM☉** instead of a³/P² in Julian
  years, which raises masses by 3.8e-5 (relative).
- **`Tabulated` is deprecated** in favour of `virgil.spectra.Nodes`: it
  now raises a `DeprecationWarning`, behaviour is unchanged, and removal is
  planned for 0.4.0.

### Added

- **Per-observable error scales.** `error_scale(model, data,
  by_observable=True)` re-estimates MacKay's noise scale separately for each
  kind of observable (`"vis"`, `"phi"`, and each kind in `extras`, pooled over
  datasets), solving the coupled equations `1/β_b = χ²_b/(N_b − γ_b)` with
  `γ_b` from the hat matrix. Use it when blocks are mis-calibrated by
  different factors (in a MATISSE N-band contest file, V² gave χ² per point
  0.005 and closure phases 0.49). `OIData.with_error_scale` accepts the
  returned dictionary of factors per kind. Data whose covariance includes
  nuisance terms that do not scale with the quoted errors (gains, closure
  offsets, marginalized flux scales, differential phases with a finite
  `prior_width`) raise a `ValueError`. The default single scale is unchanged.

- **Hierarchical error scales and tied `noise=` terms.** A `noise=` entry of
  `fit` and `numpyro_model` may now be a function of the sampled parameters
  instead of a prior (`likelihood.is_tied`), recorded as a deterministic site;
  a tied term's `log_prior(values)`, if any, is added once
  (`likelihood.tied_log_prior`). `priors.hierarchical_scales(name, n,
  median=, spread=, centred=True)` builds n scales (e.g. one closure-phase
  error scale per epoch) from a log-normal population with log-uniform
  hyperpriors: centred (log s sampled, for well-measured members; the
  non-centred form diverged on such members in a test) or non-centred.

- **`fit` reports `info["grad_norm"]`**, the infinity norm of the gradient
  of the loss per data point at the result, in the unconstrained
  coordinates of the convergence test, for every method (LM, L-BFGS, Adam).

- **`imaging.laplace_samples`** draws Gaussian-field images from the
  Laplace (Gauss–Newton) posterior of the whitened latents about the MAP,
  `N(z_MAP, (I + JᵀJ)⁻¹)`, and optionally renders the whole model for each
  draw. Other fitted parameters are held at the MAP.

- **`clean(base_priors=...)`** fits named parameters of the base (e.g. a
  companion's position) together with the components' fluxes at every major
  cycle. CLEAN also ends with a final major cycle when it stops at the target
  or at `max_iterations`, so its fluxes are refitted even when the data start
  close to the target and no cycle has run.

- **Multi-scale CLEAN**: `clean(scales_mas=(0.0, ...))` adds components
  that are pixels convolved with circular Gaussians of the given FWHM
  (Cornwell 2008), so extended emission takes a few broad components
  instead of many points. The search and the major cycles run over every
  (scale, pixel). `CleanResult` gains `components_by_scale` and
  `scales_mas`; `components` is still the total image. The default
  `scales_mas=(0.0,)` gives exactly the previous point-only results.
  An optional `scale_bias` favours small scales; it is off by default.

- **`fit(lbfgs_memory=50)`.** L-BFGS now keeps 50 past steps (optax's default
  is 10). On regularized images, 10 stopped short of the optimum at many
  weights; 50 reached lower losses (lower χ² and lower penalty together) and
  converged in 2–3× fewer steps for StarletL1, at a higher cost per step.
  L-BFGS fits therefore change slightly from earlier versions.

- **Isotropic-orientation priors.** `virgil.priors` has
  `IsotropicInclination(low=0, high=180)` (degrees, density ∝ sin i, so cos i
  is uniform; use `(0, 90)` when only |cos i| is identifiable) and
  `IsotropicLatitude(low=-pi/2, high=pi/2)` (radians, density ∝ cos lat), as
  numpyro distributions for `fit` and `numpyro_model`. Docstrings and error
  text now recommend Jeffreys-consistent priors: `LogUniform` for scales
  (RV jitter, fluxes, diameters, separations), with `ruffio_upperlimit`
  documenting why its flat flux prior is deliberate.

- **Log-uniform (scale-invariant) detection prior for `linear_flux_grid`.**
  `prior=LogUniform(f_min, f_max)` puts the scale-invariant (Jeffreys, under
  the scaling group) prior, `1 / f`, on the companion flux ratio, and is now
  the documented recommendation. The flux ratio is a scale parameter
  spanning decades, so the prior is the invariant measure of the scaling
  group, not the root-Fisher prior of the linearised likelihood (which has
  constant Fisher information and would be flat). The evidence needs a proper prior, so both bounds are
  required, and the Bayes factor depends on them as it must (widening them
  changes `log_bayes_factor` by about `-Δ ln ln(f_max / f_min)`). The
  evidence, posterior mean and sd come from 256-node Gauss-Legendre
  quadrature in `ln f` over the part of the bounds the likelihood occupies,
  inside `jit` and `vmap`; they agree with a dense-grid quadrature to
  better than 1e-4. A Gaussian prior is `dist.Normal(mean, sd)` (a
  Gaussian-prior evidence, a computational approximation where `f` may go
  negative). With no `prior`, the posterior and Bayes-factor fields are
  `None`, whatever the prior kind.

- **Angle vectors (`virgil.angles.AngleVector`).** A prior for any angle
  (degrees), sampled as a 2-D vector at the site `"<path>_vec"` with the
  angle as the deterministic `"<path>"`, so there is no wall at 0°/360°.
  A ring prior on the radius keeps MAP fits off the origin; von Mises and
  axial von Mises priors are chords √κ (v̂ − m̂), the same form as the phase
  residuals, so `fit`'s Levenberg–Marquardt and `gauss_newton_mass` take
  them (`fit` used to reject `VonMises`). The density is normalized in the
  plane. For orbits, `orientation_priors` samples 2Ω and ϖ = Ω + ω (or Ω
  and ϖ with RVs), and `KeplerOrbit.from_varpi` builds the orbit. After
  Octofitter's `UniformCircular` and exoplanet's `Angle`.

- **`KeplerOrbit.from_position_angle`.** The position angle θ at `t_ref`
  as an alternative to `dt_peri`, for short arcs (after Thompson et al.
  2023). `position_angle_prior(orbit_fn)` is a `likelihoods=` term adding
  log|∂M/∂θ| (`position_angle_log_jacobian`), so a uniform θ gives the
  invariant prior, uniform in the time of periastron. Singular at i = 90°.

- **Spectro-interferometric observables (Stage 6a, PR B).**
  `read_oifits(..., extras=...)` and `OIData(path, extras=...)` read OI_FLUX
  (`"flux"` or `"nflux"`), T3AMP, VISAMP beside V² (absolute, or correlated
  flux as `AMPTYP` declares) and VISPHI as a differential phase beside
  closure phases. They are off by default, so existing analyses are
  unchanged. The new `virgil.observables` blocks follow the phases in the data
  vector. Spectra are predicted from `total_spectrum`, with the grey scale
  (and optionally a polynomial in λ) marginalized analytically under a broad
  Gaussian prior; `likelihood.flux_scale_posterior` reports it. Differential
  phases use the exact arg V, with the continuum normalization (offset and
  delay over continuum channels) applied as a linear operator and its
  propagated covariance N D Nᵀ. Beside closure phases, only their
  closure-free part in the line windows is used, so nothing is counted
  twice. New methods: `OIData.with_continuum`, `with_flux_scale` and
  `with_error_floor` (PMOIRED-style floors per observable). The writer
  gains OI_FLUX and the `AMPTYP`/`PHITYP`/`CALSTAT` keywords. Guide:
  "Spectro-interferometric observables".
- **`vis_error`, and `where=`/`combine=` in `inflated_errors`.** A fitted
  absolute visibility error term. Floors and fitted terms share one rule,
  `_utils.inflate_errors`.

- **Detection statistics for ROC curves (`virgil.detection`).**
  `detection_statistics(data, model, samples_dict)` returns, from one
  companion grid search, the profile likelihood ratio `delta_chi2` (flux >= 0),
  a grid-marginalized `log_bayes_factor` against no companion (trapezoid prior
  weights, uniform in position and following the flux axis's spacing), the
  largest flux/σ `max_snr`, and the best position and flux. It is traceable in
  the data, so it compiles once under `jax.lax.map` over simulated datasets;
  `local_nsigma` gives the single-position (Wilks) significance. A new tutorial,
  "Detection ROC curves", calibrates the statistics with simulations. Design
  and later stages in `design/detection_roc.md` (virgil#2).
- **Detection Monte Carlo (`virgil.detection`, stage 2 of virgil#2).**
  Null simulators `gaussian_null` (noise from the template's errors, with an
  `error_scale` for mis-estimated errors) and `bootstrap_null` (a sign-flip or
  resampling residual bootstrap about the null scene; correlated closure
  phases are whitened and re-coloured with the new `ClosureNoise.colour`),
  and `rescale_errors` (visibility and phase errors scaled separately so the
  null has χ²_r = 1). `injection_grid` lays out companions at random PAs, and
  `injection_recovery` runs the search on null and injected draws with one
  compiled kernel (`jax.lax.map`, chunked, with a progress bar). It returns a
  NumPy `DetectionMC` with empirical false-alarm probabilities and their
  Clopper–Pearson intervals, thresholds with bootstrap errors, ROC curves,
  AUC, completeness maps, contrast curves in the units of `absil_limits`, an
  optional `match_radius` (Cartesian `dra`/`ddec` or angular `sep`/`pa`
  grids), and `save`/`load`/`concatenate` for array jobs; `concatenate`
  compares fingerprints of the whole model, null scene and template (every
  field, static or not) and refuses runs that cannot be fingerprinted.
- **Detection plots (`virgil.plotting`, stage 3 of virgil#2).**
  `plot_null_distribution(mc, stat, observed=, fap=)` draws the empirical
  false-alarm probability of every threshold on a log scale, with the
  single-position reference (½χ²₁ for Δχ², the Gaussian tail for max SNR),
  the threshold at a FAP, and an observed value with its FAP and 95%
  interval; its legend sits outside the axes so it never hides the tail.
  `plot_roc(mc, stat, flux=, sep_bin=)` draws ROC curves for one or several
  statistics, fluxes or separation bins (distinct colours and line styles),
  on a log false-positive axis by default with the chance curve TPR = FPR
  drawn as a curve, and marks the one-position FAP of local 3σ and 5σ
  (0.135%, 2.9×10⁻⁷) next to where each curve's threshold equals them, to
  show the look-elsewhere effect. `plot_completeness(mc, stat, fap,
  units=)` draws the completeness map against separation and Δmag, contrast
  or flux, with the 50% and 90% `contrast_curve`s, on axes that
  `plot_contrast_curve` can draw Absil or Ruffio limits onto.

- **Gauss-Newton and a marginal-likelihood map in `linear_flux_grid`.**
  `n_iter=k` relinearises the whitened residuals at the current flux per pixel
  for `k` extra steps, removing the bias for bright companions (at f = 0.3,
  `n_iter=3` agrees with `optimized_flux_grid` to 3e-7 and with the Laplace
  `sigma_f` to 2e-5, where `n_iter=0` is 34% low). `prior=Gaussian(mean, sd)` puts a
  Gaussian prior on the flux and fills the `posterior_mean`,
  `posterior_sd` and `log_bayes_factor` fields (closed-form evidence ratio
  against f = 0, in the linearised model about the final point). The result
  is a `LinearFluxGrid` named tuple (those three fields are `None`
  without a prior), so unpack it by attribute, or take the first three
  with `flux, error, snr = res[:3]`; unpacking it into three names directly
  fails.

- **Fitted RV jitter.** `RVData.term(params, jitter="rv_jitter")` inflates the
  errors to `sqrt(d_rv² + s²)` with `s` a fitted value (km/s; give it a
  half-normal or log-uniform prior). The term protocol gains an optional
  `log_norm(values)`, the `Σ log σ_eff` that a fitted error makes
  non-constant: `fit` adds it to the loss, defaults to L-BFGS (as for
  `noise=`), and raises `TypeError` for `method="lm"`. `numpyro_model` needs
  no change, since the term's `loglike` is already normalized.

- **Marginalised RV zero points.** `RVData(..., instrument=labels)` and
  `RVData.term(params, marginalise_offsets=(mean, sd))` marginalize one
  velocity zero point per instrument analytically (Luger, Foreman-Mackey &
  Hogg 2017), in O(N k²) by the Woodbury identity and the matrix-determinant
  lemma, with the jitter-dependent log-determinant in `log_norm`.
  `term.posterior(values)` gives the zero points' conditional mean and
  covariance after a fit. Only a finite prior width is supported (`True` is
  N(0, 1000²) km/s and warns).

- **North-angle and plate-scale nuisances (#212).**
  `OIData.with_north_angle(angle)`, and the `noise=` term `north_angle`
  (degrees) in `fit`, `numpyro_model`, `model_loglike` and
  `whitened_residuals`, rotate a dataset's sky by a fitted angle: every
  position angle it measures is the true one plus the angle (North through
  East). The plate scale is the existing `wavel_scale`, as 1/m for a
  magnification m. For positions, `PositionData.model(orbit, north_angle=,
  plate_scale=)` and `PositionData.term(orbit, north_angle="path",
  plate_scale="path")` take the paths of fitted values. Both are off by
  default, there are no default widths, and a `uv_grid` is dropped when the
  rotation is applied. After Octofitter (Thompson et al. 2023).

- **Closure-phase offsets per frame** (Stage 6d).
  `OIData.with_closure_offsets(baseline=, triangle=, modes=)` adds closure-phase
  offsets common to a frame's channels (per baseline, as T·e; per triangle; or
  supplied modes), marginalized analytically on the whitened closure phases,
  with widths `phi_offset_baseline`, `phi_offset_triangle` and
  `phi_offset_modes`. Four or more telescopes; off by default.

- **`injection_limits`: injection-method detection limits.** Like
  `absil_limits`, with the same grid, inputs and return format, but each
  limit is the flux at which a companion injected into the data (Gallenne et
  al. 2015, section 3.2, as in CANDID's `detectionLimit(methods=["injection"])`)
  would be detected at the requested significance. It solves for the flux by
  bisection in log flux, vmapped over the grid. It agrees with CANDID's
  criterion to 1e-4; unlike CANDID it does not refit the primary's diameter
  to the injected data. It equals `absil_limits` on the data reflected about
  the null model, and neither is uniformly more sensitive.


- **`linear_flux_grid`.** A closed-form, fouriever-style (`lincmap`)
  linearised companion flux map for fast first-pass searches, beside
  `optimized_flux_grid`. The derivative of virgil's whitened residuals with
  respect to the flux at f = 0 is computed exactly with `jax.jvp`, so
  correlated closure phases are whitened as in the likelihood, and
  `f_hat = -(g . r0) / (g . g)`, `sigma_f = (g . g)**-0.5` and SNR are
  returned per pixel (f_hat unconstrained in sign). Valid only for
  f much smaller than 1: a bright companion (f ~ 0.3) is biased low.
- **Wavelength-scale nuisance** (Stage 6d). `OIData.with_wavelength_scale`
  evaluates models at scale·λ + offset, and the noise terms `wavel_scale` and
  `wavel_offset` fit or sample it (e.g. `Normal(1, 2e-4)` for GRAVITY).

- **Calibration gains correlated across channels** (Stage 6d).
  `OIData.with_gains(telescope=, baseline=, chromatic=, modes=)` adds gains
  on log |V| per frame: per telescope, per baseline, a chromatic coherence
  loss, or supplied modes such as a calibrator PCA's. The likelihood
  marginalizes them analytically (`virgil.gains`), and their widths can be
  fitted or sampled with the noise terms `vis_gain_telescope`,
  `vis_gain_baseline`, `vis_gain_chromatic` and `vis_gain_modes`.
  `OIData.stations` holds each sample's station pair, read from `STA_INDEX`.

- **Line and node spectra (Stage 6a, spectra).** `GaussianLine` and
  `LorentzianLine` (amplitude is the peak flux; negative for absorption),
  `Nodes` (linear or natural cubic spline through free fluxes, constant or a
  fixed `outside` value beyond the nodes, e.g. `outside=0.0` for an excess)
  and `Sum` (named parts, so continuum plus lines is one flux). Every
  spectrum's reference flux is its value at `wavel0`, and positivity is
  checked on the total. `Spectrum()` with no argument evaluates at `wavel0`.
  `SourceModel.total_spectrum(wavel)` (a `System`'s sum over its parts) and
  `OIData.select(wavel_min, wavel_max)`. `Tabulated` is deprecated in
  favour of `Nodes` (a `DeprecationWarning`; behaviour unchanged). Removal is
  planned for 0.4.0.

- **`numpyro_model(..., likelihoods=[...])`.** The extra data terms that `fit`
  takes (`PositionData.term`, `RVData.term`, or a callable returning whitened
  residuals) can now be sampled: term `i` is added as the site
  `likelihood_<i>`, and `data_obj` may be `()`. Only `PositionData` and
  `RVData` terms are fully normalized Gaussian log densities (like the OIData
  terms); a plain callable of whitened residuals adds `-0.5 * sum(r**2)` only.

- **`TruncatedCone`.** A thin, optically thin conical shell truncated near its
  apex (e.g. the dust cone of a colliding-wind binary), with an analytic
  visibility (a stack of projected rings), a tilt out of the sky plane, an
  optional elliptical cross-section, and a rendered image that matches its
  visibilities.
- **`EllipticalLimbDarkenedDisk(diam, ratio, pa, u)`.** A limb-darkened disk
  with an elliptical outline (minor/major axis `ratio`, major axis at `pa`
  North to East as for `EllipticalGaussian`), with the polynomial law and
  `u` of `LimbDarkenedDisk` and analytic visibilities.
- **Short-arc orbits.** `StateVectorOrbit` parameterises an orbit by the
  relative position and velocity at `t_ref` and the gravitational parameter,
  which a short arc constrains well where the elements are degenerate; it
  converts exactly to a `KeplerOrbit` (`to_kepler`, `from_kepler`).
- **Radial velocities and masses.** `RVData` fits radial velocities of either
  star (with the mass ratio, systemic velocity and distance), which fix the
  node that positions leave ambiguous by 180°; `total_mass` and `distance_pc`
  convert between them by Kepler's third law; `AxialVonMises` is a prior on an
  angle known only modulo 180°. `fit(..., likelihoods=[...])` adds such terms
  (`PositionData.term`, `RVData.term`) to a fit, with or without visibilities.

- **Orbit example.** `notebooks/mwe/mwe_orbit.ipynb`: a companion and an
  attached disc observed on six VLTI nights, recovered through per-night
  positions, Thiele–Innes starting orbits and a joint fit to all the
  visibilities. `coverage.vlti_oidata(nights_mjd=...)` gives synthetic
  coverage with times.

- **Simulation.** `virgil.simulate.simulate(scene, template)` observes a scene
  with a template's sampling, errors and times (each sample at its own time
  for a moving scene, with `shift_days` to move the epochs), and
  `bias_test` fits a model to many noise draws to show biases and spreads.

- **Times and frames.** `OIData` keeps each sample's time (`mjd`, stored as
  `dt` days since a float64 `t_ref`) and exposure (`frame`) from OIFITS, and
  dict input may give `mjd` and `frame`. A frame is the baselines that
  closure phases tie together; by default all its samples get the frame's
  mean time (`read_oifits(frame_mjd="row")` keeps each row's).
  `OIData.epochs(gap_days=0.5)` labels nights, and `split_by_epoch()` returns
  one `OIData` per night. This is the groundwork for orbits and per-frame
  calibration terms.
- **Orbits.** `KeplerOrbit` (period, time of periastron, eccentricity,
  inclination, the secondary's ω, the receding node's Ω, angular semimajor
  axis) gives the secondary's `relative` position and exact
  `relative_velocity` in virgil's sky conventions, and `ThieleInnesOrbit` the
  linear form used for starting orbits, with converters between them and to
  jaxoplanet, which solves Kepler's equation (the new `[orbits]` extra).
  `PositionData` holds measured positions with their covariances (or
  separations and position angles), and `starting_orbits` finds good
  starting orbits for them by an exact Thiele–Innes least-squares solve on a
  grid of period, eccentricity and time of periastron.
- **Scenes that move.** `SourceModel.at(mjd)` gives a model at a time, and
  `Attached(component, orbit, anchor, bind, offsets)` places a component on a
  binary's orbit and binds its angles to the binary frame (line of centres,
  line of nodes, inclination, the side facing the primary). `OIData.model`
  evaluates a time-dependent model at each sample's own time; static models
  keep their fast path.
- **A model per dataset in sampling.** `numpyro_model`, like `fit`, accepts
  a model function returning a list of models, one per dataset, sharing
  parameters (e.g. a binary at several epochs with one flux ratio).
  Regularisers act on the first model.

### Changed

- **API consistency before 0.3.0.**
  - `linear_flux_grid(prior=)` takes numpyro's `dist.LogUniform(low, high)`
    and `dist.Normal(mean, sd)` (scalar parameters), the same classes as
    `fit` and `noise=`, so passing the numpyro `LogUniform` no longer raises
    `TypeError`. The `virgil.grid_fit.LogUniform` and `Gaussian` named tuples
    still work. A bare `(mean, sd)` tuple, whose deprecation path never
    shipped, is now a `TypeError` with a message saying so.
  - `RVData.term(marginalise_offsets=True)` raises `ValueError` instead of
    silently using N(0, 1000²) km/s: state the zero-point prior as
    `(mean, sd)`.
  - `flux_param` and `batch_size` are keyword-only in `likelihood_grid`,
    `optimized_likelihood_grid`, `optimized_flux_grid`, `linear_flux_grid`
    and `laplace_flux_uncertainty_grid` (in the last, after `flux`).
    The first four are from 0.2.0, so a positional `flux_param` or
    `batch_size` there now raises `TypeError`.
  - `injection_recovery(draw_batch > 1)` divides the default grid
    `batch_size` by `draw_batch` and caps `draw_batch` at that default,
    so the working set stays within the documented bound, and by default
    the last chunk is compiled at its own length instead of being padded
    with discarded draws (an explicit `chunk_size` still pads, to reuse one
    compilation).

- **Frames are grouped by `MJD` and `TIME` (#203).** OIFITS v1 gives each row
  a `TIME` (UTC seconds) and an `MJD`, and some writers (OYSTER, the 2004
  Interferometry Beauty Contest files) set `MJD` to the night's date and put
  the snapshot in `TIME`. The reader grouped frames, and matched
  closure-phase legs to baselines, by `MJD` alone, so every snapshot of such
  a file became one frame and closure phases from different snapshots were
  whitened as one correlated group. Rows are now one frame, and a leg matches
  a baseline row, only if `MJD` and `TIME` both agree within the existing
  tolerance. **Behaviour change:** for the 2004 contest files the independent
  closure phases go from 10/130 to 130/130, and every χ² from them changes.
  Files with a constant `TIME` group as before, and epochs and model times
  still come from `MJD`.
- **`fit` optimizes each prior in its flat coordinate.** A prior that is
  uniform in some coordinate of its parameter, `LogUniform(a, b)` in
  `log x`, or any prior with a `flat_coordinate()` method (isotropic
  inclinations in `cos i`, latitudes in `sin(lat)`), is fitted in that
  coordinate, where it is constant and adds nothing to the loss. So
  Levenberg–Marquardt now runs, and is chosen automatically, with these
  Jeffreys priors (it used to fall back to L-BFGS, and `method="lm"`
  raised), and `gauss_newton_mass` accepts them. **Behaviour change:** a
  fit with a `LogUniform` prior (on a parameter or a `noise=` term) is now
  the maximum of the likelihood inside the prior's range, the MAP in
  `log x`; before, it was the mode of likelihood × `1/x` in `x`, which
  pulled scales towards small values. Uniform, Normal and other priors are
  unchanged. See "Priors and the MAP" in the conventions.
- `starting_image`'s internal fit uses `LogUniform` priors on the envelope
  width and flux (flux bounds 1e-4 to 100), so the starting point may differ
  slightly.

- **One home for analytic marginalization of linear parameters.**
  `virgil._linear` holds the shared algebra: `LinearMarginal(design,
  prior_mean, prior_sd | prior_cov, method)`, the successive rank-one and
  dense-Cholesky whitenings, and the conditional posterior. The gains,
  closure-phase offsets, VISPHI continuum terms, flux grey scales and RV
  zero points use it.
- **The OI_FLUX / correlated-flux grey-scale prior is stated, not taken from
  the data.** `with_flux_scale(scale=(mean, sd))` gives it in the
  data's units. It is required for `"flux"` and
  `"corrflux"`, and the likelihood raises until it is given; `"nflux"`
  defaults to `(1, 0.1)`. The Gaussian is documented as a proposal for the
  Jeffreys 1/k prior.

- **`TruncatedCone.n_rings` guidance.** The docstring now states the measured
  `1 / n_rings**2` error scale (about 8e-4 in |V| at the default 32 for a
  13.8 mas cone), and recommends doubling `n_rings` and checking Δχ² at the
  best fit; well-measured data may need 64 or more. A convergence test was
  added.

### Fixed

- **`clean` no longer hangs when `nnls` gives up in a major cycle.** The
  bounded least-squares fallback now uses BVLS: scipy's default TRF solver
  can loop forever in its line search once the step underflows, which stalled
  the py3.11 lowest-dependency CI job on scipy 1.13.
- **Contrast limits: one search, from below.** Found in the pre-0.3.0 review
  (`design/codebase_review_2026-10b.md`, B1, B2, S2 and S5).
  - `injection_limits(flux_bounds=None)` returned 1000, the top of its search
    range, at every position whose limit was above about 1e-3. For a
    normalized scene the significance falls again once the companion
    outshines the primary, and the bisection ended at that top. Both limit
    functions now find the *first* crossing from below: they step up by
    quarter decades (CANDID steps by 1.4) and then bisect the last step in
    log flux (one shared helper, `_grid.first_crossing`). Unbounded results
    now match bounded ones, for model classes and for `System` templates.
  - `injection_limits` raised `TypeError` on data with extra observables
    (T3AMP, VISAMP, VISPHI, OI_FLUX). The companion's signal is now injected
    into every block of the data vector, extras included. The companion
    model's chi-squared on the injected data is computed in full, since the
    OI_FLUX blocks and gains whiten with the model's own prediction.
  - Both functions raise a `ValueError` up front for a `sigma` beyond what
    `nsigma` can represent in the float type (about 12.95 in float32, 37 in
    float64). Before, `absil_limits` returned about 1e37 or silently clipped,
    with a stale "optimizer did not converge" warning. The new warnings say
    what happened: limits clipped to `flux_bounds`, or no crossing within 40
    decades of the start.
  - `flux_bounds` now means the same in both functions: the range searched,
    upward from its lower end, with limits outside it set to the nearer
    bound and a `RuntimeWarning`. With `flux_bounds=None`, the search starts
    at the flux axis's smallest positive value, and only then must the axis
    have one. `absil_limits` no longer evaluates the whole positions × fluxes
    loss grid to choose a start. Results agree with the old ones to about
    1e-4 relative or better.
  - `flux_param`, `flux_bounds` and `batch_size` are keyword-only in
    `absil_limits` and `injection_limits`, as in `grid_fit` and `detection`.

- **Orbits: GM☉ in `total_mass`, Ω range, face-on inclination (F14–F16).**
  `total_mass` and `distance_pc` now use Kepler's third law with the IAU 2015
  nominal GM☉, au and the 86400 s day instead of a³/P² with P in Julian years
  (masses were 3.8e-5 low). `ThieleInnesOrbit.to_kepler` no longer returns
  Ω = 180° exactly: Ω is in [0°, 180°) with ω paired to keep the sky orbit.
  `StateVectorOrbit.to_kepler` computes i, Ω and ω by atan2 from the orbit
  normal, so nearly face-on orbits keep their inclination to float64
  precision. `orientation_priors(..., inclination=True)` also returns
  `IsotropicInclination` under `"inc"`: the full Haar orientation prior in
  one call (default off, so existing callers are unchanged).

- **`log_evidence` with calibration gains or closure offsets** now includes
  the likelihood's model-dependent normalization, ½ log det of the
  marginalized nuisances' covariance factor (and, for extra observable
  blocks, their effective-error normalizer), exactly as `model_loglike`
  evaluates it. Before, it used the whitened χ² alone, so wider gains
  always looked better. Values for plain data are unchanged; the evidence
  still omits the data-only normalization (`-Σ log σ - ½ n log 2π`).

- **`log_evidence` and `clean` on high signal-to-noise data (#214).** The
  evidence takes `log det(I + JᵀJ)` from the singular values of the Jacobian,
  instead of a Cholesky factor of `I + J Jᵀ`, which was numerically
  indefinite and raised `LinAlgError` when `J Jᵀ` is rank deficient;
  `classic_maxent`'s curvature and `error_scale`'s λ use the squared singular
  values too. The CLEAN major-cycle refit survives an `nnls` failure: it
  gets a larger iteration limit and falls back to bounded `lsq_linear`, so
  no-base CLEAN runs on the 2004 contest data instead of raising.
- **Components build under `jax.jit` from concrete shape parameters.**
  `TruncatedCone` validated `tilt` with a `jax.numpy` call on the concrete
  array, which inside `jit` became a tracer and raised
  `TracerBoolConversionError` when a fit's model function built a cone from
  fixed shapes and a traced flux. The check now uses NumPy. The other
  concrete checks (components, spectra, orbits) were audited and need no
  change; a regression test builds each checked component inside `jit`.
  Found in a real-data OzSTAR run.
- **`absil_limits` with a far-off or single-value flux axis.** The
  significance saturates (about 37 sigma in float64) for bright companions,
  so starting the optimizer on such a flux, e.g. `flux=[0.01]`, gave a flat
  loss and returned the starting flux with a non-convergence warning. The
  flux axis now only gives a rough starting point: the limit is bracketed by
  decades and bisected in log flux, replacing the BFGS search, so the result
  no longer depends on the axis.
- `gauss_newton_mass(model, priors, (), values)` no longer raises a
  `ValueError` on an empty residual list: with `data=()` the curvature comes
  from the priors alone, as `fit` and `numpyro_model` already allow.

### Removed

- `examples/elr_pavo/`, the PAVO re-analysis scripts, moved to the private paper repository; the golden-fixture generator is now `scripts/make_elr_golden.py`.

### Docs

- **`GravityDarkenedStar` docstring: "Choosing `n_lat`"** gives the
  measured mesh error from an independent ELR11 reference (virgil-validation),
  the second-order convergence, and the doubling check (|Δχ²| ≳ 1 per
  dataset).
- **"Detection ROC curves" rewritten on `injection_recovery`** (Binaries,
  `notebooks/detection_roc.ipynb`). A candidate companion in a simulated
  NIRISS AMI observation, one Monte Carlo call (10⁴ null and 1280 injected
  searches), the look-elsewhere effect on the null distribution, the
  empirical threshold and FAP to quote for a detection, ROC curves of the
  three statistics, the completeness map and 50%/90% contrast curves to quote
  for a non-detection against Absil and Ruffio limits, and what wrong error
  bars do to a Gaussian null and how `rescale_errors` and the bootstrap fix
  it.
- **New tutorial: "Orbits from interferometric data"**, a joint fit to
  the interferometric data (Binaries, `notebooks/orbit_fitting.ipynb`).
  One `KeplerOrbit` model is fitted to the V² and closure phases of all
  eight simulated VLTI epochs at once, each sample at its own time, with
  per-epoch V² and closure-phase error scales drawn from fitted log-normal
  populations (`hierarchical_scales`). Priors are Jeffreys throughout
  (`IsotropicInclination`, `orientation_priors` and `AngleVector` angles,
  log-uniform scales, interim uniform e), with a no-data check.
  Initialisation is kept separate from the inference: coarse per-epoch grids
  seed `starting_orbits`, the candidates are ranked by the joint likelihood
  of all the data, and the best four are refined with `fit`. Outputs are NUTS
  diagnostics, a corner plot, an ensemble of posterior orbits on the sky with
  the implied per-epoch positions, posterior-predictive closure-phase
  checks, the inferred calibration, separation and position angle in time,
  and a comparison with the two-step posterior.
- **`plotting.plot_orbit_ensemble`.** Draws a batch of `KeplerOrbit`s on the
  sky (East left, North up) as thin lines, one period each, with measured
  `PositionData` positions coloured by epoch with their error ellipses, a
  reference orbit and the primary.
- **Conventions.** Dropped the stale "Not yet in this version" note from the
  orbit conventions: `virgil.orbits` is on main.
- The `virgil._linear` API page is labelled internal.
- Docs pages no longer point to the internal design notes.
- Docstrings no longer point to the internal design notes.
- **Closure-phase whitening** (`virgil._closure`): the docstring states that the correlation, built from the per-triangle errors, is exact only when the baseline phase errors are equal; with unequal errors the whitened χ² comes out slightly low (about 2% with one baseline three times noisier). OIFITS carries no per-baseline phase errors.

## 0.2.0 (2026-10-03)

### Renamed: drpangloss is now virgil

- The import package is `virgil` (`import virgil`) and the PyPI distribution
  is `virgil-astro` (`pip install virgil-astro`). `pip install virgil` installs
  an unrelated package.
- The GitHub repository is `benjaminpope/virgil` and the docs are at
  <https://benjaminpope.github.io/virgil/>.
- A final `drpangloss` 0.2.0 on PyPI depends on `virgil-astro` and forwards
  to it, with a `FutureWarning` on import. Replace `import drpangloss` with
  `import virgil` and `drpangloss.x` with `virgil.x`; nothing else was renamed.
- There is no `nufft` extra: the NUFFT backend was removed (see below).

### Changed: closure phases (every four-telescope chi-squared changes)

- Closure phases of triangles that share a baseline are correlated. For four or
  more telescopes, `virgil` now keeps only the independent combinations of the
  closure phases of each frame and channel (three of the four triangles for
  four telescopes) and whitens them with the covariance of Kammerer et al.
  (2020, A&A 644, A110): the reported errors on the diagonal and correlations
  of +/-1/3 between triangles that share a baseline. Before, all triangles were
  treated as independent, which counted the closure phases 4/3 times for four
  telescopes.
- As a result every chi-squared, likelihood, evidence, grid and fit that uses
  four or more telescopes changes, and the number of closure phases in the
  likelihood (`OIData.n_independent`) is smaller than the number stored.
  Three-telescope data (e.g. JWST/NIRISS AMI with a single triangle per
  frame) are unchanged. Numbers from earlier versions, including some logged in
  the design notes, are marked "pre-6.0" and have not been re-measured.
- Unprojected closure phases enter the likelihood as the chord 2 sin(delta/2)
  over sigma (a von Mises likelihood for independent phases), which is
  smooth across +/-pi.

### Added

- **Readers.** `read_oifits(insname=...)` selects an instrument's tables. A
  GRAVITY file holding both fringe-tracker (FT) and science (SC) tables must
  be read with `insname=`: the two are never merged. `PHITYP` is checked, so a
  differential VISPHI is never read as an absolute phase. Closure phases are
  matched to visibilities of the same exposure, within twice the longest
  `INT_TIME`, since GRAVITY's pipeline averages different frames for each
  table (before, 50 of 136 reads of archival GRAVITY files failed).
- **Image reconstruction.** An `Image` component (a pixel map in a `System`,
  with an exact DFT, and an exact matrix Fourier transform on the uv lattices of
  AMIGO DISCO data), `fit` (Levenberg-Marquardt, L-BFGS or Adam, in float64 by
  default), the regularizers `MaxEntropy`, `TSV`, `TV` and `Centroid`,
  `l_curve` with corner, discrepancy and classic MaxEnt weights,
  `GaussianField` Gaussian-process pixels, `log_evidence`, `error_scale`
  (MacKay's re-estimate of the error bars), `dirty_image`, `beam`,
  `convolve_beam`, `diagnose`, and a Gauss-Newton mass matrix for NUTS. See
  the imaging tutorials in the docs.
- **Synthetic data.** `virgil.coverage` (`ami_grid_record`, `nrm_oidata`,
  `vlti_oidata`, `mask_transfer`) and `virgil.scenes` for simulations and tests.
- **Chromatic scenes.** Wavelength-dependent fluxes (`PowerLaw`, `BlackBody`)
  following SPARCO, a `Resolved` background, and multi-channel and multi-file
  `OIData`.
- **Source models.** Composable components and `System`, explicit `flux`
  parameters, `FlaredDisk` (Blakely et al.), `ModulatedGaussianRim`,
  `UniformDisk`, `EllipticalGaussian` and `GaussianArc`, an anisotropic
  `GaussianField`, fitted error inflation (`noise=`), the rapid-rotator
  `GravityDarkenedStar` (ported from Shashank Dholakia's ELR model, grey and
  chromatic), and an interface to harmonix for spotted stars.
- **Packaging.** Bessel functions now come from the separate `jaxbessel`
  package. Releases are published to PyPI from GitHub releases.

### Fixed and improved

- One likelihood (`likelihood.whitened_residuals`) for fitting, grids, limits
  and sampling, with normalized Gaussian log-likelihoods.
- The image coordinate convention (East left, North up, position angle North
  to East) is applied everywhere, including a position-angle bug in
  `BinaryModelAngular` and the orientation of plots.
- Maximum-entropy L-BFGS fits no longer collapse onto a few pixels, and
  stalled image fits are capped.
- Solvers and curvature functions compile once rather than on every call, grid
  tools size their batches by data size, and the test suite is faster.

### Fixed after the October 2026 codebase review

- The rim's non-negativity check (`ModulatedGaussianRim`, `is_physical`) no
  longer passes negative brightness when the top azimuthal order is zero.
- Independent phases use the exact von Mises normalizer, so fitted phase
  errors are no longer biased at large sigma (a true 1.8 rad used to fit as
  1.3 rad). Small-sigma likelihoods are unchanged.
- Converting V^2 to amplitudes or log-amplitudes floors the data at their own
  error, so points near or below zero no longer get enormous or tiny errors.
- Levenberg-Marquardt and L-BFGS in float32 converge instead of running to
  `max_steps`: the gradient tolerance is floored at sqrt(eps) of the starting
  gradient.
- `laplace_cov` and `fisher` compute in float64, like `fit`.
- `error_scale` solves MacKay's self-consistent equation instead of
  evaluating it at beta = 1; `log_evidence`, `error_scale` and
  `classic_maxent` accept a `FitResult` and refuse fits with fitted noise
  terms or per-dataset model lists, which they cannot handle.
- Index arrays cached under one x64 mode no longer break the other on JAX
  0.10 (closure-phase noise and the gravity-darkened star mesh).
- Corrected citations: MacKay (1992) eq. 4.10; classic MaxEnt is Gull (1989)
  with Skilling (1989).
- `virgil.__version__`; true minimum dependency versions (`jax>=0.8`), tested
  in CI; pandas, ChainConsumer and astroquery moved to the `plots` and
  `legacy` extras; CI on macOS with Python 3.11, Python 3.13, the lowest
  versions, and a wheel build.

### Removed

- The experimental NUFFT backend (`backend="nufft"`, the `nufft` extra): on a
  GPU it was slower than the DFT. Its code is in the history of PR #71.
- `amigo.simulated_disco_record`, replaced by `coverage.ami_grid_record`.
- Before the first release, these names were removed from the public API:
  `GaussianDiskModel` (use `System(star=PointSource(), disk=GaussianDisk(...))`),
  `cvis_gaussian_disk`, `cvis_binary_angular` (now private),
  `loglike_nosignal`, `fisher_matrix` and `observed_information` (use
  `fisher` and `inference.hessian_matrix`), `plotting.plot_trace_panels`,
  `plotting.plot_recovery_residuals`, the `legacy.savefits` module and
  `legacy.oifits_implaneia.load_oifits`.
- `Tabulated` and `load_oi_data` are no longer top-level names: they are
  `virgil.spectra.Tabulated` (provisional, to be replaced) and
  `virgil.amigo.load_oi_data`. `pixel_offsets` and `HarmonixModel` are new
  top-level names.
- The unused `termcolor` dependency and `sampling` extra, and the unused
  `data/chi2_ppf*.npy` tables.
