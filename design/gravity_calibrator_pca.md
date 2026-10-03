# Plan: an empirical model of GRAVITY systematics from archival calibrators

Status: **plan**, 2026-10-03. Not started. This belongs to the **separate GRAVITY project**, which begins once the core of the package (renamed virgil) works (Ben, 2026-10-03). It answers open question 4 of [`gravity_calibration_review.md`](gravity_calibration_review.md).

## Goal
Learn the shapes of GRAVITY's calibration systematics from many observations of stars whose true visibilities are known, then use those shapes as nuisance modes when fitting science targets. The systematics in question are:
- chromatic phase errors in VISPHI from air dispersion in the delay lines and a stable instrumental phase ([GC20b] App. A builds a template of this kind);
- chromatic coherence loss in V², going as exp(−a/λ²) ([PM] eq. 58);
- any non-closing closure-phase errors (review §13).

A low-rank basis from principal component analysis (PCA), with widths from the spread of the calibrators, replaces guessed polynomial orders and guessed priors. It also answers two questions empirically:
- whether closure-phase offsets are needed at all (review §13);
- how large the wavelength-calibration error really is (review §6).

## Data
- **Selection.** Calibrator observations from the ESO archive, through the `eso-archive` skill (TAP at `https://archive.eso.org/tap_obs`).
  - Start with `ivoa.ObsCore` (`instrument_name LIKE 'GRAVITY%'`): pipeline-reduced OIFITS, 8–25 MB each, are far cheaper than raw frames.
  - Fall back to `dbo.raw` with `instrument = 'GRAVITY'` and `dp_cat = 'CALIB'`, reducing the raw frames ourselves with the ESO pipeline.
  - **[To check]** whether calibrator observations have reduced products in ObsCore, and which `dp_type` values mark calibrators in single- and dual-field mode. Count and summarise with `GROUP BY` before anything else.
- **Stratification.** Systematics depend on configuration, so the analysis is split by:
  - resolution (MEDIUM, HIGH);
  - polarisation (SPLIT, COMBINED);
  - single- or dual-field, including GRAVITY Wide;
  - telescopes (UTs or ATs, and AT configuration);
  - epoch, across pipeline versions and hardware changes such as GRAVITY+.

  Record the relevant headers for each file: `ESO INS SPEC RES`, `ESO INS POLA MODE`, `ESO ISS CONF STATION1-4`, the pipeline version, seeing and coherence time.
- **Calibrator truth.** Take each calibrator's diameter from the JSDC (with its error) and model it as a uniform disk, so the expected V², closure phase and differential phase are known. Exclude known binaries and stars with poor diameters.
- **Per-DIT products.** These come where available, for empirical covariances (open question 1 of the review).
- **Volume.** Expect a few thousand calibrator exposures per configuration, so hundreds of GB of reduced products, and far more if raw. Download in batches to OzSTAR `/fred`, never into a repository. Count before downloading, and confirm with Ben above about 5 GB.

## Method
1. **Residuals.** For each exposure, subtract the expected calibrator observables:
   - V² ratios to the uniform-disk model, in log space;
   - closure phases, whose expectation is zero;
   - VISPHI, after the same continuum basis the fits will use (review §2).
2. **Coordinates.** Express the residuals per baseline (or triangle) and per frame on a common grid in 1/λ. Interpolate onto it, recording that the pipeline's own interpolation correlates neighbouring channels.
3. **Weighted PCA.** Run it per configuration and observable, whitening by the reported errors first, so that components describe systematics rather than noise. Keep components until the explained excess over the noise is exhausted, judged by cross-validation on held-out nights.
4. **Structure.** Test whether components are telescope-based (shared by baselines with a common telescope, as in [K20]) or baseline-based. The answer decides how they enter the low-rank nuisance model (review §4c).
5. **Widths.** The spread of the component amplitudes across calibrators gives the priors on the nuisance widths, with any dependence on conditions (seeing, τ₀, airmass) modelled as a regression.

## Validation
- **Held-out calibrators.** Fit their amplitudes with the templates, and check that the whitened residuals are consistent with noise and that the uniform-disk diameters are unbiased.
- **Injection.** Add a synthetic companion or disk to calibrator data, fit with and without the templates, and compare recovery and error calibration.
- **Science check.** Re-fit one well-studied target (e.g. a published GRAVITY binary orbit) with and without the templates.

## Products and code
- A small, versioned file of templates per configuration (components, widths, provenance), distributed with the GRAVITY project, not in the core package.
- In the core package, a generic interface for externally supplied nuisance modes (Stage 6d's low-rank blocks), so that core code does not depend on GRAVITY.
- Scripts for selection, download and reduction, as OzSTAR jobs in `ozstar_scripts`, with manifests of `dp_id`, date, programme and configuration.

## Effort and risks
- **Effort:** about 3–5 agent-days, plus cluster time and Ben's review.
- **Risks:**
  - *Configuration drift:* templates may not transfer across pipeline versions or hardware changes, which is why the work is stratified and validated across epochs.
  - *Proprietary periods:* recent calibrators may be inaccessible.
  - *Data volume.*
  - *Calibrator diameters:* errors in them would bias V² templates. Use only small, well-measured calibrators for V².
- **First ask the GRAVITY team** whether the consortium already has such templates ([`questions_for_gravity_team.md`](questions_for_gravity_team.md)).
