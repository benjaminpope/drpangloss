# Questions for the GRAVITY team

We are writing a JAX-based code for modelling and imaging interferometric data (VLTI/PIONIER, GRAVITY and MATISSE, and JWST aperture masking). For GRAVITY we want to treat the data the way the instrument team does, and avoid inventing error models that the team already knows to be wrong. We have read:
- the pipeline manual (1.11.0);
- the instrument paper (2017);
- the Galactic Centre, ExoGRAVITY and BLR papers;
- Kammerer et al. (2020) on correlated errors.

The questions below are the ones that reading didn't settle. Short answers or pointers to papers would be enormously helpful.

## Wavelength calibration
1. The manual quotes an absolute spectral calibration of about 0.5 nm and about 0.1 nm between baselines. Are these current, and do they hold for MEDIUM and HIGH resolution? Is the error better described as a shift, a scale, or something per baseline?
2. We plan to fit a per-dataset wavelength nuisance with a prior of about 2×10⁻⁴ in scale. Is that sensible? How large should the baseline-to-baseline part be allowed to be?

## Errors and correlations
3. The pipeline's error bars are statistical, from bootstrapping over frames, with no calibration term. What does the team do in practice: rescale them, add a floor, or estimate a full covariance from the individual DITs? Which product is best for that (`P2VMRED` or `ASTROREDUCED`)?
4. Are the V² of the averaged product averaged coherently or incoherently between DITs?
5. The spectra are linearly re-interpolated and about twofold oversampled, so neighbouring channels are correlated. Is there a recommended way to account for this, or a typical correlation length?
6. GRAVITY writes all four closure phases of a four-telescope array, of which three are independent. We now keep only the independent combinations, with the ±1/3 correlation structure that baseline-phase noise implies. Is that what the team does, or is something else recommended?

## Calibration systematics
7. Are there known non-closing (baseline-based) closure-phase errors of practical size? Does anyone fit closure-phase offsets, and has anyone measured them on calibrators?
8. For V² calibration errors, the GRAVITY-RESOLVE paper puts a gain on |V| per exposure and baseline, with a 10% prior. Are the gains better described per telescope, and with a chromatic shape (coherence loss going as exp(−a/λ²))?
9. For dual-field observations of faint off-axis targets, how should V² be calibrated, given anisoplanatism and field-dependent injection? Are the "phase maps" for field-dependent aberrations available to users?

## Differential phase
10. In single-field products, the pipeline removes the mean phase and group delay over all channels, lines included. For line science (e.g. a BLR-like analysis), what normalisation does the team recommend: refitting the continuum outside the lines, or something else?
11. Is there a template for the chromatic phase from air dispersion and the instrument (as in the 2020 BLR paper's appendix), and is it available? Has the consortium built anything like a PCA of calibrator residuals?

## Fibre injection and field of view
12. What fibre coupling model should we use for extended or offset sources: a Gaussian of FWHM about λ/D convolved with tip-tilt jitter, and with what jitter on the UTs and the ATs? Does the injection differ appreciably between telescopes within an exposure?

## Conventions and smearing
13. Have the sign conventions of VISPHI and T3PHI, or the baseline orientations, changed between pipeline versions? Is there a reference binary the team uses to check them?
14. For bandwidth and time smearing at MEDIUM and HIGH resolution, what does the team do: average the coherent flux and the flux separately before dividing, and integrate over the DIT?
15. Can the two polarisations of split-polarisation data be treated as independent measurements, or are there known differential-polarisation systematics?
