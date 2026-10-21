# ImagingPipeline (provisional)

Status: **PR B built** (2026-10-21): stages `load`, `overview`, `start`,
`star`, `hyperparameters`, one-member `members`, pass-through `combine`
and `quicklook`; checks 1–11; the FITS writer; `Result.image`; CLI `image`.
PRs C (the ensemble: checks 12, 13, 15) and D (channels, check 14, the cube)
are not built; the text below is their design. Two options are stubs that
raise `NotImplementedError`: `anneal` and `star_model="ud"`.

`ImagingPipeline(data, model=None, *, output="run", **settings)`, CLI
`image`, `STABILITY = "provisional"`, schema `virgil-pipeline-run-v1`. It
is the recipe that the imaging-contest campaign in virgil-validation found
to work (`design/plan_imaging_contests.md` C2b–C2e,
`design/contest_convergence_diagnosis.md`, `scripts/contest_images.py`), and
what Ben's review of those images (`contests/fit_advice.csv`) showed it does
not do. The run folder, resume rules, `Result` and stability policy are
those of [`pipeline.md`](pipeline.md); the stages, settings, checks and some
files are new. It stays provisional until the GP-ensemble functions it calls
and the chromatic stage have been unchanged for a release.

## What it can and cannot promise

Read this before the rest. An automatic imaging pipeline can promise
much less than `StarPipeline` or `BinaryPipeline`, and the design is built
around saying so in `summary.json` rather than hiding it.

- **It can promise a reproducible, data-consistent image family.** Every
  image it keeps fits the data on the quoted errors to a stated χ²/N per
  observable, from a stated set of starts, at hyperparameters chosen by a
  stated rule, with the spread between reasonable choices mapped.
- **It cannot promise the morphology is right.** The campaign's own
  evidence: 2006, the 2008 AGB stars and the 2024 Obj1 spiral reached
  χ²/N ≈ 0.7–1.1, and Ben judged the images "absolutely terrible",
  "blobby" or "a chain of dots". Every data-fidelity check in this note
  passed on those runs. No check here measures whether an image looks
  like the source; the ensemble σ map measures sensitivity to the
  pipeline's own choices, not to choices it never made (another
  regulariser family, point sources, a two-scale field). The quicklook
  says this in its first cell, and the `family_agreement` check (opt-in)
  is the only partial guard.
- **Hyperparameters are chosen by evidence, which assumes the errors and
  the model family are right.** With biased data (2004 data2, 2010,
  2024 Obj2) the evidence still returns a best point, often on the grid
  edge. The pipeline cannot fix that; it can only detect the symptoms
  (error scale far from 1, short-baseline bias, edge optimum) and fail.
- **A grey image of chromatic data is not a fit** (2010: V² errors of
  1e-4 against a chromatic source, χ²/N ≈ 3000). The pipeline detects
  this and fails; per-channel imaging from a grey default is a mitigation,
  not a cure, and joint cross-wavelength regularisation is out of scope.
- **Cost.** Contest members took 2–8 h each on OzSTAR. The defaults are
  sized for a laptop on small data (≤ 128 pixels, 4 members, a 3×3
  hyperparameter grid); real VLTI data with many channels need the member
  stage run as an OzSTAR array (`run_member`, below). The pipeline does
  not choose compute for the user.

## Scope and non-goals

In scope: grey (or SPARCO-weighted) images from any observables `fit`
accepts (V², closure phases, kernel/DISCO phases, complex visibilities),
with an optional analytic central star; one dataset or a list fitted
jointly; per-channel images from a grey default as an opt-in stage.

Not in v1:

- image posterior sampling (NUTS over latents, `imaging_sampling`);
  `laplace_samples` maps may be added later as an additive stage;
- joint cross-wavelength regularisation (MiRA-3D style); chromatic
  morphology beyond per-channel images;
- free point sources beyond the central star, two-scale fields, harmonix
  spotted stars, parametric-only scenes (fit-plan arms `point-sources`,
  `two-scale`, `parametric-only`): each could become a model-registry
  entry later, with no new stage;
- RML families as the default (TV, MaxEnt, starlet stay available through
  the opt-in `families` setting, via the existing `virgil.ensemble`);
- calibration (error floors are the user's; the pipeline only reports);
- bandwidth smearing beyond the field ceiling `field_of_view` already
  applies.

## Library work first

The contract says a pipeline adds no science. Today the recipe lives in
virgil-validation's `contest_images.py` (`run_gp`, `grow_grid`,
`member_settings`, `residual_diagnostics`, `combine_ensemble.py`), not in
virgil. `virgil.ensemble` is L-curve/RML only: `combine` assumes every
member came from an `LCurve` group. So before the pipeline:

1. `imaging.chi2_by_observable(model, data)` (public, from the existing
   private `_observable_blocks`): raw χ² and independent count per kind
   (`vis`, `phi`, extras), on quoted errors.
2. `imaging.short_baseline_residual(model, data, quantile=0.2)`: mean
   whitened visibility residual on the shortest fraction of spatial
   frequencies, and its standard error (MACIM's test).
3. `imaging.residual_image(model, data, npix, pixel_scale_mas)`: the
   normalised χ² gradient per pixel, −g_p / |J e_p|, which is the residual
   dirty map for any observable (closure phases included). `clean`
   already computes it; this exposes it.
4. `diagnose` gains the keys `chi2_red_by_observable` and
   `short_baseline_mean` (additive).
5. `ensemble.evidence_grid(...)`: GP fits over σ × ℓ (ℓ in units of the
   beam's minor axis), isotropic and, when an elliptical-Gaussian fit to
   the data gives an axis ratio below 0.9, anisotropic; warm-started; the
   grid grows by factors of 2 while the optimum is on a movable edge
   (`grow_grid`), within limits. Returns the grid of `log_evidence`, the
   fits, and whether the optimum is interior.
6. `ensemble.gp_members(...)` and `combine` generalised to members with no
   L-curve (the window step is skipped when a group has no curve; the χ²
   filter, MAD cut, alignment and running-mean acceptance are unchanged).
   An `EnsembleSpec` field says which. The mean is still the exact mixture
   scene; `std` is unchanged in meaning.

Each comes with a known-truth unit test; see PRs A1 and A2.

## Stages

1. `load`: the `OIData` (or list), `wavel_range`, `error_floor`, flags.
   Records per observable: N, independent N, fraction flagged, whether
   closure phases exist (V²-only data cannot fix orientation), and the
   distribution of quoted errors (the `quoted_errors` check: a block whose
   errors are all identical, or zero, is reported — 2010's were all 1e-4).
2. `overview`: uv coverage, `beam`, `nyquist_pixel_scale`,
   `second_moment_fwhm`, `field_of_view` and the field from `image_field`
   (flagged samples excluded, as of #361), the pixel scale (Nyquist /
   `oversample`), npix capped at `max_npix` with the pixels coarsened and
   the field kept. Every later length default is stated in beam minor
   axes. The channel count decides whether the `channels` stage can run.
3. `start`: `starting_image` (envelope fit), then gradient `clean` on a
   grid capped at `clean_max_npix` (65) over the same field, relative to
   the star base when a star is a candidate, with `scales_mas` (0, ½ beam,
   1 beam) by default. The restored components (beam-convolved, resampled
   to the fine grid, floored) are the GP mean template. CLEAN's own image
   and components are written as a result in their own right
   (`images/clean.fits`), since for fields of point sources it may be the
   better answer (fit-plan arm `clean-final`).
4. `star`: with `star="auto"` (default), a GP fit at the grid's central
   (σ, ℓ) with and without an analytic star, **on the same grid and
   template**, and ΔlogZ from `log_evidence`. |ΔlogZ| ≥ `star_threshold`
   (5 nats) decides; otherwise the decision is `"both"` and the ensemble
   alternates star on/off members, as the campaign did. `star=True/False`
   skips the comparison. `star_model` is `"point"` (default) or `"ud"`
   (diameter fitted, log-uniform from 0.1 beam to ¼ field; 2004 data2).
   Without a star the gauge is fixed by `Centroid(0.1 beam)`; with one,
   by the star, and the support has a hole of half the beam's minor axis
   (`hole_mas`). The envelope's flux is always fitted (a log-uniform
   prior), which is what removes the spurious spot beside the star.
5. `hyperparameters`: `evidence_grid` from σ ∈ {1, 2, 4},
   ℓ ∈ {0.5, 1, 2} beam, isotropic and (if elongated) anisotropic, with
   up to `grid_growths` (2) growths. Then `error_scale(...,
   by_observable=True)` at the winner. If any block's scale is outside
   [1/`rescale_band`, `rescale_band`] (1.3), the data are rescaled with
   `with_error_scale` and the grid is evaluated **once more** on the
   rescaled data (MacKay: rescale before choosing hyperparameters); the
   second winner is kept. The raw χ² on quoted errors is what is reported
   either way; the rescaled χ² is stored, labelled, never checked.
6. `members`: `n_members` (4) CLEAN-started GP fits at the chosen
   hyperparameters, each with a seeded random field factor (1, 1.5, 2 ×),
   oversample (2, 3, 4), CLEAN gain (0.05, 0.1, 0.2) and, if the star
   decision was `"both"`, alternating star. Each member is written to
   `members/<k>/` and is resumable on its own: a resumed stage skips
   members whose files exist. `ImagingPipeline.run_member(k)` runs one
   member alone (an OzSTAR array task); the `combine` stage then reads
   the folder. `member_hyperparameters="fixed"` (default) or
   `"evidence"` (each member re-grids, as the campaign did, at
   9–36× the cost).
7. `combine`: the generalised `combine` (χ² filter per observable,
   MAD cut, alignment without a star, running-mean acceptance), giving
   the mean scene, the mean image on the common grid and the per-pixel
   σ. Then on the mean scene: `diagnose` (flip Δχ², edge flux, anchored),
   `chi2_by_observable`, `short_baseline_residual`, `residual_image`,
   MacKay γ. The best single member is kept alongside the mean
   (`models/best_member`).
8. `channels` (opt-in, `chromatic="channels"`; `"auto"` runs it when the
   `chromatic` check warns and there are ≥ `min_channels` channels): one
   GP fit per channel (or per `channel_bin` channels) with the grey mean
   as the GP mean template and the grey hyperparameters, one start each,
   star flux free per channel. Combined into a cube; a per-channel
   median of members is not attempted in v1. Skipped, with the reason in
   its report, otherwise.
9. `quicklook`: the executed notebook.

The preferred model is `models/best` (the ensemble mean scene, an exact
`System` of the kept members' images on their own grids, plus the star).

## Settings (defaults)

| Setting | Default | Meaning |
| --- | --- | --- |
| `wavel_range`, `error_floor` | `None` | as elsewhere |
| `oversample` | 3.0 | pixels per Nyquist pixel |
| `field_fwhms` | 6.0 | `image_field`'s factor |
| `field_mas` | `None` | override the data-chosen field |
| `max_npix` | 128 | grid cap (pixels coarsened, field kept) |
| `clean_max_npix` | 65 | CLEAN grid cap (setup cost ∝ npix⁴ N) |
| `clean_scales_beam` | (0, 0.5, 1) | multi-scale CLEAN scales, beam minor axes |
| `clean_iterations` | 3000 | CLEAN cap |
| `star` | `"auto"` | `"auto"`, `True`, `False` |
| `star_model` | `"point"` | `"point"` or `"ud"` |
| `star_threshold` | 5.0 | nats for a decisive star decision |
| `hole_beam` | 0.5 | support hole under the star, beam minor axes |
| `sigmas` | (1, 2, 4) | GP σ grid |
| `lengths_beam` | (0.5, 1, 2) | GP ℓ grid, beam minor axes |
| `anisotropic` | `"auto"` | `"auto"` (axis ratio < 0.9), `True`, `False` |
| `grid_growths` | 2 | edge growths of the grid |
| `rescale_band` | 1.3 | rescale and re-grid outside this factor |
| `n_members` | 4 | ensemble members |
| `member_hyperparameters` | `"fixed"` | `"fixed"` or `"evidence"` |
| `families` | `("gp",)` | add `"tv"`, `"maxent"`, `"tsv"`, `"starlet"` for RML members via `virgil.ensemble` |
| `chromatic` | `"grey"` | `"grey"`, `"sparco"` (star and image `PowerLaw`), `"channels"`, `"auto"` |
| `min_channels` | 4 | channels needed for `"auto"` to image per channel |
| `channel_bin` | 1 | channels per per-channel image |
| `max_steps` | `None` | optimiser cap (tests set it small) |
| `seed` | 0 | member draws |
| `quicklook_execute` | `True` | as elsewhere |

`"sparco"` follows the DiskPipeline rule only as an option, not
automatically: whether it should switch on automatically with several
channels is open question 3.

## Checks

In this order. The first is χ²/N on the quoted errors of the ensemble
mean scene; nothing here is computed on rescaled errors. The same run
always gives the same checks.

1. `chi2_reduced`: total χ²/N, quoted errors. warn > 2 or < 0.5, fail > 5
   (existing `chi2_reduced`).
2. `chi2_reduced_vis`: χ²/N of the visibility block alone (V², |V|,
   log-amplitudes); same thresholds. Omitted when there is no such block.
3. `chi2_reduced_phi`: χ²/N of the phase block (closure, kernel, DISCO
   phases), counted on independent combinations; same thresholds.
   Splitting matters: the 2024 MATISSE N file had 0.005 and 0.49, and a
   total hides both. Extras get `chi2_reduced_<kind>` appended after the
   fixed checks, only when present.
4. `quoted_errors`: warn when a block's quoted errors are all equal (to
   1e-6 relative), or any are zero or non-finite. A data check: such
   errors make every hyperparameter choice suspect.
5. `error_scale`: the per-observable MacKay scales s_b at the winner.
   warn when any s_b is outside [0.5, 2], fail above 5. Unlike the
   existing check it also warns below 0.5 (overstated errors
   over-regularise every evidence choice). The message gives each s_b and
   whether the grid was re-run on rescaled data.
6. `effective_parameters`: N − γ and γ/N. warn when N − γ < 0.2 N: the
   image has nearly as many measured parameters as data (2022 AMI:
   N = 56, γ ≈ 53), so a small χ²/N is by construction, not a good fit.
   The message suggests comparing with a parametric model
   (`BinaryPipeline`, `StarPipeline`).
7. `short_baselines`: mean whitened visibility residual on the shortest
   20% of spatial frequencies, in standard errors. warn above 3, fail
   above 5. A negative bias means over-resolved flux or a calibration
   bias (2004 data2: −8σ; 2024 Obj2: −4σ); the message names a halo and
   calibration, and says the evidence's hyperparameters are then not to
   be trusted.
8. `hyperparameter_edge`: warn when the evidence optimum is on an edge of
   the grid after growth (the limits were reached); the value is which
   edge. Combined with check 5 or 7 failing, it fails: hyperparameters
   chosen under biased data are not automated (below).
9. `star`: ΔlogZ (star − no star) on the same grid. pass when decisive
   either way, warn when |ΔlogZ| < `star_threshold` (both kept). The
   message says the Laplace evidence holds the star's flux at its MAP, so
   ΔlogZ lacks that Occam factor (about ½ log N nats), and is not a
   detection significance. Skipped (`pass`, value `None`) when `star` was
   set by hand.
10. `orientation`: the flip Δχ² of the mean scene (`diagnose`'s
    `flip_dchi2`). warn below 9 (orientation constrained at less than
    ~3σ). V²-only data always warn with value 0: the image and its
    180° rotation fit equally.
11. `edge_flux`: the largest edge-flux fraction of any kept member's
    image. warn above 1%: the field may be too small (remedy: `field_mas`
    or a halo).
12. `ensemble_kept`: kept / members. warn below 0.5, fail below
    `min_kept` (3) members, or 2 when `n_members` < 4. Few survivors mean
    the starts land in different minima.
13. `ensemble_spread`: the flux-weighted median over the mean image of
    σ/mean. warn above 0.3. The value is a choice-sensitivity, not an
    uncertainty, and the message says so. With one kept member it warns
    with value `None`.
14. `chromatic`: only with more than one channel. The grey mean scene's
    χ²/N per channel; value = slope of χ²_chan/N against wavelength in
    standard errors, plus the spread max/median. warn when the spread
    exceeds 3 or the trend exceeds 5σ; **fail** when, in addition,
    `chi2_reduced` fails: a grey image cannot describe these data. The
    message names `chromatic="sparco"` and `"channels"`.
15. `family_agreement` (only when `families` has more than one entry):
    NCC between the families' mean images after `align`. warn below 0.8:
    features that change between regularisers are suspect (the 2012
    organisers' rule).

Warnings captured inside stages (optimiser non-convergence, CLEAN
stalls) become checks appended after these, as for the other pipelines.

**Deliberately not a check:** χ²/N after rescaling (≈ 1 by construction),
any image-quality score against a truth (there is none), and a
"chain of dots" detector. Ben's advice names fragmentation as the most
common failure; a fragmentation metric (number of beam-sized islands
above a fraction of the peak outside the main source, in the
beam-convolved mean) is easy to compute but its thresholds would be tuned
on the contest images that it is meant to judge. Proposed as a reported
number (`combine.islands`), not a check, until it is calibrated
(open question 5).

## What is not automated, and fails loudly

- **Hyperparameters under biased data.** When `error_scale` fails, or
  `short_baselines` fails, or the optimum sits on the grid edge after
  growth together with either, the evidence's choice is not trusted:
  `hyperparameter_edge` fails, `summary.json` sets
  `hyperparameters.trusted = false`, and the quicklook's first cell says
  to choose by L-curve or by hand (the 2018 BSMEM practice). The
  pipeline does not switch to an L-curve on its own: an automatic
  fallback would make a choice the user should make.
- **Chromatic mismatch (the 2010 case).** The `chromatic` check fails;
  `chromatic="auto"` runs per-channel images only if the user asked for
  `"auto"`, and even then `chi2_reduced` and `chromatic` still report the
  grey fit's failure, since the per-channel images do not repair it.
- **Under-determined data (2022 AMI).** `effective_parameters` warns; the
  image is reported but the quicklook recommends a parametric model.
- **Stalled fits.** A member whose χ² is non-finite or whose optimiser
  did not converge is dropped and counted; if fewer than `min_kept`
  survive, `combine` raises and the run is `failed`, with the members'
  χ²/N in the report. No image is written as `models/best` from a failed
  ensemble.

## summary.json

Sections (keys additive within the provisional tier):

- `data`: `n_independent` and `n_by_observable`, `flagged_fraction`,
  `has_closure_phases`, `n_channels`, `wavel_range_m`, `quoted_errors`
  (per block: min, median, max, all-equal flag).
- `field`: `field_mas`, `field_source` (`"image_field"` or
  `"setting"`), `pixel_scale_mas`, `npix`, `nyquist_mas`,
  `beam` (`major_mas`, `minor_mas`, `pa_deg`), `fwhm_mas` (second
  moment), `ceiling_mas`.
- `start`: `clean_chi2_red`, `clean_stop`, `clean_flux_fraction`,
  `envelope_fwhm_mas`, `envelope_flux`.
- `star`: `mode` (`"auto"`/`"on"`/`"off"`), `decision`
  (`"on"`/`"off"`/`"both"`), `delta_log_z`, `star_model`, `flux_ratio`
  (median over kept members), `diam_mas` (for `"ud"`).
- `hyperparameters`: `sigma`, `length_mas`, `length_beam`, `variant`
  (`"iso"`/`"aniso"`), `axis_ratio`, `pa_deg`, `log_z`, `grid_shape`,
  `interior`, `rescaled` (bool), `trusted` (bool).
- `errors`: `scale_by_observable`, `gamma`, `n_minus_gamma`,
  `chi2_rescaled` (labelled; never checked).
- `fit`: `chi2`, `chi2_reduced`, `chi2_reduced_by_observable` (all on
  quoted errors, of the mean scene), `best_member_chi2_reduced`,
  `short_baseline_mean_sigma`, `flip_dchi2`, `edge_flux`.
- `ensemble`: `n_members`, `n_kept`, `dropped` (reason → count),
  `spread_median`, `islands`, `families`, `family_ncc` (when present).
- `channels`: `ran` (bool), `reason`, `n_images`, `chi2_reduced_by_channel`,
  `chromatic_trend_sigma`.
- plus `checks`, `worst_status`, `warnings`, `stability`, as the
  contract requires.

## Run folder

The contract's files, plus:

| Path | Contents |
| --- | --- |
| `images/mean.fits` | the ensemble mean image (common grid), with a WCS (RA---SIN/DEC--SIN about the phase centre, CDELT in degrees, East left), BUNIT relative flux, the beam (BMAJ/BMIN/BPA) and the star's flux in the header; extension `STD` the per-pixel σ, extension `NMEMBERS` |
| `images/best_member.fits`, `images/clean.fits` | the best member; CLEAN's restored image and its components |
| `images/cube.fits` | with `channels`: one plane per channel, wavelength axis in metres |
| `images.h5` | the same arrays with axis names and units as attributes, plus every kept member's image on its native grid, the residual images and the evidence grids (`evidence/<variant>`) |
| `members/<k>/` | per member: `manifest.json` + `values.npz` (as `models/`), `report.json` (settings drawn, χ² per observable, converged, wall time) |
| `models/best/`, `models/best_member/`, `models/clean/` | rebuildable models |
| `data/processed.oifits` | the data as fitted; `data/rescaled.oifits` when the grid was re-run on rescaled errors |
| `data/model.oifits` | the mean scene's model observables on the same layout, for any external viewer |
| `plots/` | uv coverage; CLEAN start; evidence grids; mean, σ and σ/mean maps with the beam; residual image (z-like, symmetric diverging scale); residuals against spatial frequency per observable, short baselines marked; star ΔlogZ; members thumbnails; per-channel panels |
| `quicklook.ipynb` | reloads the folder; first cell: what the run can and cannot claim (the section above, filled from the checks); then images, residuals, checks |

FITS is a new file type for the contract (an additive change: the
"stores everything as" list gains FITS images). The writer lives in
`pipeline/_io.py`; orientation is checked by a test that writes a known
off-centre point source and reads its RA offset back with astropy's WCS.

`Result` gains, additively, `image(name="mean")` (array, pixel scale,
beam), `image_std()`, `members()` and `cube()`; the existing
`model(name)` reads `best`, `best_member` and `clean`.

## Tests

Fast, tiny and deterministic; no test may take more than a few seconds
of JAX work, and the module's total stays under about a minute on CI
(float64 paths, `max_steps` small, npix ≤ 16, members ≤ 2, a 2×2 grid,
`quicklook_execute=False` except one test).

- **Library (PRs A1, A2), known truth:**
  - `chi2_by_observable` sums to `diagnose`'s χ²; blocks match
    `_observable_blocks`.
  - `short_baseline_residual` is ≈ 0 on data simulated from the model and
    negative by the expected amount when 10% of the flux is moved into an
    over-resolved halo absent from the model.
  - `residual_image` peaks at the position of a point source missing
    from the model, for V²+CP data (not only complex visibilities).
  - `evidence_grid` on data drawn from a GP with known σ recovers σ (as
    the existing evidence test), and grows the grid when the truth lies
    beyond its edge; `interior` is false when growth is capped.
  - generalised `combine` on curve-less members: identical members give
    zero spread; a badly fitting member is dropped; the mixture model
    equals the mean of the members' scenes (existing tests, re-run with
    curve-less members).
- **Pipeline (PRs B–D):**
  - snapshot tests of the constructor and `run` signatures, stage names,
    setting defaults, summary keys and check order (provisional, but the
    snapshot makes changes deliberate);
  - **known-truth recovery**: star (flux 0.6) + inclined Gaussian ring on
    a synthetic 4-telescope coverage, V² + CP, npix 16. After `align`,
    NCC of the mean image with the truth > 0.8; star decision `"on"`;
    `chi2_reduced` passes. The same ring without a star: decision `"off"`.
  - each check fires on a constructed case: identical quoted errors;
    errors overstated 3× (`error_scale` warns low, re-grid runs); a halo
    left out (`short_baselines`); V²-only data (`orientation` warns with
    value 0); a 2-channel scene whose ring has a different spectral index
    from the star, fitted grey (`chromatic` warns; fails when the
    contrast is large enough that `chi2_reduced` fails);
  - the 180° flip: a point-symmetric truth gives a small flip Δχ², an
    asymmetric one a large one;
  - resume: a run interrupted after member 0 skips it on resume;
    `run_member(1)` in a second process then `run()` gives the same
    summary as one `run()`; changing a setting raises
    `ConfigMismatchError`;
  - FITS orientation round trip (above); `validate` passes on the folder;
    `load` round-trips `image()`, `image_std()`, `model("best")`;
  - one test executes the quicklook (marked like the other pipelines'
    quicklook tests);
  - CLI: `virgil-pipeline image data.oifits --set n_members=2 ...`.
- **Not in CI:** the full-size run on the `imaging_ensemble` tutorial's
  scene, executed on OzSTAR with the tutorial (below), and a
  virgil-validation bench re-running the contest datasets through the
  pipeline against the campaign's numbers (`ensemble_8562a39_3018a9f`).

## Relation to the tutorials

`imaging_ensemble` (or `imaging_gp`) gets a closing "As a pipeline"
section running `ImagingPipeline` on its simulated scene, asserting against
the truth (NCC after alignment, the star decision, χ²/N on quoted errors
near 1 per observable), not against the tutorial's own numbers, since the
pipeline draws its own starts. Executed on OzSTAR like the other
tutorials.

## Decisions (Ben, 2026-10-21)

These settle the questions the first draft left open.

1. **Name.** `ImagingPipeline`, CLI `image`.
2. **Default prior.** The Gaussian process, with σ and ℓ chosen by evidence;
   classical regularizers (`regularizers=[...]`) are chosen by the L-curve
   corner instead. PR B takes one regularizer at a time, and only with
   `star=True` or `False` (not `"auto"`, whose evidence needs the GP).
3. **SPARCO.** Switches on automatically with several channels and a star
   (PR D).
4. **No re-gridding.** There is no second hyperparameter grid on rescaled
   errors; the rescaled χ² is reported beside the raw one and never checked.
5. **Star threshold.** 5 nats of ΔlogZ for v1.
6. **Hyperparameters are shared** across the members: one evidence search up
   front, the members vary only the start. The summary says that the
   ensemble's sigma map then excludes hyperparameter uncertainty.
7. **Anneal.** An opt-in sub-beam Gaussian convolution at each optimizer step
   (width in beam fractions), recorded in `run.json`, not a check, to anneal
   out a chain of dots (Max Charles's dorito work). Needs a design check on
   its interaction with the L-BFGS stall. **A stub** in PR B.
8. **Fragmentation.** The `islands` number only, no check, in v1.
9. **Caveat.** The quicklook's first cell says that data-fidelity checks
   alone passed contest fits that Ben judged terrible, so the morphology is
   not promised.

## Implementation: PR split

Sonnet workers, each its own PR from `main`, through the `pr-worker`
skill. Laptop: tests only, JAX threads capped at 2, one process; the
tutorial execution goes to OzSTAR.

| PR | Scope | Files owned | Depends on | Merge |
| --- | --- | --- | --- | --- |
| **A1** imaging diagnostics | `chi2_by_observable`, `short_baseline_residual`, `residual_image`; `diagnose` additive keys; API docs | `src/virgil/imaging.py` (new functions, `diagnose`), `tests/test_diagnose.py`, `tests/test_imaging.py` (new tests only), `docs/api/imaging.md` | none | 1st |
| **A2** GP ensembles | `evidence_grid` (iso/aniso, growth), `gp_members`, curve-less `combine`, `EnsembleSpec` field | `src/virgil/ensemble.py`, `tests/test_ensemble.py`, `docs/api/ensemble.md` | none (parallel with A1) | 2nd |
| **B** pipeline skeleton | `ImagingPipeline` with `load`, `overview`, `start`, `star`, `hyperparameters`, a single-member `members` and a pass-through `combine`, `quicklook` (basic cells); checks 1–11; FITS writer; `Result.image`; CLI; lazy export | `src/virgil/pipeline/image.py` (new), `_checks.py` (appended functions), `_io.py` (FITS writer, `Result` additions), `_quicklook.py` (image cells), `__init__.py`, `cli.py`, `tests/test_pipeline_image.py` (new) | A1 | 3rd |
| **C** ensemble stage | `n_members`, resumable members, `run_member`, generalised `combine`, `families`; checks 12, 13, 15; `images.h5`, `members/`, `image_std`, `members()` | `pipeline/image.py`, `_checks.py` (appended), `_io.py` (`Result` additions), `tests/test_pipeline_image.py` (new tests) | A2, B | 4th |
| **D** chromatic | `chromatic` setting, `channels` stage, check 14, `images/cube.fits`, `Result.cube` | `pipeline/image.py`, `_checks.py` (appended), `_io.py`, `tests/test_pipeline_image.py` | C | 5th |

Docs and tutorial (the "As a pipeline" section, `docs/pipeline.md` row,
`design/pipeline.md` status) ride with D, or as a small sixth PR if D is
already large; the notebook execution is an OzSTAR job either way.

B, C and D all touch `pipeline/image.py` and append to `_checks.py`, so
they are strictly sequential (each branches from `main` after its
predecessor merges); A1 and A2 can run in parallel. Each worker's brief
should point at this section, at `scripts/contest_images.py` (`run_gp`,
`grow_grid`, `member_settings`, `residual_diagnostics`) as the reference
behaviour, and at the existing `StarPipeline` tests as the template for
snapshot and resume tests.
