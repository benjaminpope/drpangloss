# ImagingPipeline (provisional)

Status: **PRs B and C built** (2026-10-21, B cut back twice at Ben's
request). This note describes what the code does and, at the end, what is
not built. It
replaces a 491-line first design; the cut material (CLEAN starts, classical
regularizers by L-curve, `anneal`, the ensemble checks, the chromatic stage)
is in the git history of this file and is not in the code, with no stub
options for it.

`ImagingPipeline(data, model=None, *, output="run", **settings)`, CLI
`image`, `STABILITY = "provisional"`. It follows the imaging-contest
recipe of `design/plan_imaging_contests.md` and
`scripts/contest_images.py`. The run folder, resume rules and `Result` are
those of [`pipeline.md`](pipeline.md).

## What it can and cannot promise

It promises a reproducible, data-consistent image: it fits the data on the
**quoted** errors to a stated χ²/N per observable, at hyperparameters chosen
by a stated rule (the Laplace evidence). It does **not** promise that the
morphology is right: on contest data (2006, the 2008 AGB stars, the 2024
Obj1 spiral) reconstructions with χ²/N ≈ 0.7–1.1 were judged "terrible",
"blobby" or "a chain of dots", and every check below passed. No check
measures whether an image looks like the source. The quicklook says this in
its first cell. A grey image of chromatic data is not a fit, and
biased data (2004 data2, 2010) give an evidence optimum that the checks
can only flag, not fix.

## Stages

1. `load`: the processed data, N per observable, and the quoted-error
   statistics (all equal, or not positive and finite, per block).
2. `overview`: uv plots, the beam, the field (`image_field`, or
   `field_mas`), the pixel scale (Nyquist / `oversample`) and `npix`
   capped at `max_npix`.
3. `star`: `star="auto"` fits the grid's central (σ, ℓ) with and without an
   analytic star on one grid and compares `log_evidence`; |ΔlogZ| ≥
   `star_threshold` (5 nats) decides, else the larger evidence is kept
   (decision `"both"`). `star=True/False` skips the comparison.
4. `hyperparameters`: `ensemble.evidence_grid` over σ × ℓ (ℓ in beam
   minor axes; isotropic, plus an elliptical variant when the beam axis
   ratio is below 0.9; the grid grows `grid_growths` times while the
   optimum is on an edge). Saves the best fit as `models/best` and the grid
   as `grids.h5`; reports MacKay's per-observable error scales and γ.
5. `members`: `members` fits (default 8), each a Gaussian-process image at
   the **shared** (σ, ℓ) of step 4, varying only the start (`moments` or
   `flat`), the pixel scale (`oversample` or 1.5 × it) and the field (1 or 1.5 × `image_field`), with the star decision of step 3. They
   are saved in `members/NNN/` and resumable: `run_member(k)` fits one, so
   cluster array tasks can each run one member before `run()` finishes.
   `member_grids=True` gives each member its own evidence grid instead (for
   cluster runs).
6. `combine`: `ensemble.combine` on the members, with V² and closure phases
   as separate datasets: a member is dropped if either fits worse than the
   best by the ensemble's χ² rules, and a member joins the mean only if the
   mean's χ² stays within the noise on both. Writes `images/mean.fits` (row 0
   North, stored flipped, WCS `RA---SIN`/`DEC--SIN`, `STARFRAC` = the star's
   flux share) and `images/sigma.fits` (the members' standard deviation), and
   the diagnostics of `virgil.imaging` on the mean image as displayed
   (resampled to a common grid, so it fits slightly worse than the exact
   mixture). The sigma map excludes hyperparameter uncertainty when the
   hyperparameters are shared, and any error every member shares.
7. `quicklook`: the notebook, with the caveat first and the image.

`Result.image()` returns a `SkyImage` (pixels, pixel scale, centre offset
`dra`/`ddec`, header).

## Checks

In `CHECKS` order; every χ²/N is on the quoted errors and nothing is
computed on rescaled errors (that is ≈ 1 by construction).

1. `chi2_reduced`: total χ²/N (existing thresholds).
2. `chi2_reduced_vis` and 3. `chi2_reduced_phi`: the same on the visibility
   and phase blocks; a total can hide a block at 0.005 and another at 0.5.
4. `quoted_errors`: warn when a block's errors are all equal or not
   positive and finite (2010's were all 1e-4).
5. `error_scale`: MacKay's s_b per observable; warn outside [0.5, 2], fail
   above 5.
6. `effective_parameters`: warn when N − γ < 0.2 N (2022 AMI: N = 56, γ ≈
   53): a small χ²/N is then close to automatic.
7. `short_baselines`: mean whitened residual on the shortest baselines; warn
   above 3σ, fail above 5σ (2004 data2: −8σ).
8. `star`: ΔlogZ against the threshold; the message says that ΔlogZ lacks
   the star flux's Occam factor and is not a detection significance.
9. `orientation`: the 180° flip Δχ²; warn below 9 (V²-only data always warn:
   the image and its rotation fit equally).
10. `edges`: warn when the evidence optimum is on an edge of the
    hyperparameter grid, or more than 1% of the flux is in the outer two
    pixels (the field may be too small).

11. `ensemble_kept`: warn when fewer than 3, or under half, of the members
    are kept.
12. `ensemble_spread`: the members' summed pixel σ over their summed mean;
    warn above 0.5 (a convention, not calibrated on truth) or with fewer
    than two members. The message states what the spread excludes.

Not checks, deliberately: rescaled χ², any image-quality score against a
truth, and a "chain of dots" detector (a fragmentation number would be
tuned on the contest images it is meant to judge).

## Decisions (Ben, 2026-10-21)

Name `ImagingPipeline`, CLI `image`. Default prior: the Gaussian process with
σ, ℓ by evidence. No re-gridding on rescaled errors. Star threshold 5 nats.
The quicklook's first cell states the caveat above.

## Not built (PRs C and D)

- **C, left out**: CLEAN-gain starts (the ensemble API has `moments`,
  `dirty`, `flat`), star on/off as a member variation (`combine` takes one
  star setting; the star stage decides it), classical regularizers by
  L-curve and the `family_agreement` check (B has no `regularisers=`), and
  `anneal`.
- **D**: per-channel imaging, SPARCO, the `chromatic` check and a cube.

Each branches from `main` after its predecessor merges. They call
`ensemble.gp_members` and the curve-less `combine` of A2 (#371).
