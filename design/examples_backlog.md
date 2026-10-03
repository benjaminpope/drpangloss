# Examples: what the docs cover, and what is missing

Status: 2026-10-03. The docs group the worked tutorials into the tabs Data Handling, Sources, Binaries and Imaging. Each is a notebook in `notebooks/`, synced to `docs/` by `scripts/sync_tutorial_docs.py`. This note lists the important capabilities that have no example yet, so the gaps get filled before each feature is called finished. New examples follow the five-cell style: a markdown cell before each short code cell, one plot or one printed result per cell, and simulated or bundled data that run in minutes.

## Covered

| Section | Example | Covers |
|---|---|---|
| Data Handling | Reading data: OIFITS and OIData | `read_oifits`, `OIData`, observables |
| | AMIGO DISCO data | `amigo.load_oi_data`, DISCOs |
| Sources | Visibility models | analytic components, model syntax |
| | Extended source models | disks, rims, extended components |
| | Composing models | `System`, fluxes, composition |
| | Spotted stars with harmonix | `HarmonixModel` |
| Binaries | Binary search | grid searches |
| | Contrast limits | Absil and Ruffio limits |
| | Hierarchical inference across filters | a shared binary geometry with per-filter fluxes, sampled with numpyro |
| Imaging, parts 1–5 | Simulating data; regularised maximum likelihood; Gaussian-process priors; a ring around a binary; sampling the posterior | `Image`, regularisers, `l_curve`, `GaussianField`, the evidence, `error_scale`, `gauss_newton_mass` |

## Missing, in priority order

| # | Example to write | Covers | Today | Notes |
|---|---|---|---|---|
| 1 | **Chromatic scenes with SPARCO** on simulated VLTI data | `spectra.PowerLaw`, `BlackBody`, `System` with stars, a `Resolved` background and an image; spectral indices relative to the star | Only `notebooks/mwe/mwe_vlti` and the nuHor real-data notebooks | The most-used real-data workflow (PIONIER post-AGB stars). Explain the degeneracy of a smooth image with a resolved background, and of indices with the star's spectrum |
| 2 | **Fitting a parametric model and its uncertainties** | `fit` (LM, L-BFGS), Laplace and Fisher errors (`inference`), error rescaling | `notebooks/mwe/mwe_fit` | Under Sources or Binaries; the bridge between the introductory pages and the advanced ones |
| 3 | **Correlated and independent closure phases** | Why four telescopes give three independent closure phases per frame; `OIData.cp_noise`, `n_independent`; `read_oifits(insname=...)` | None (Stage 6.0) | Short, and it explains a change in every four-telescope result |
| 4 | **Simulating VLTI and aperture-masking coverage** | `coverage.vlti_oidata`, `nrm_oidata`, `OIData.with_model`, noise | Only AMI, in Imaging part 1 | Useful for planning observations, and as the source of the other examples' data |
| 5 | **Multi-epoch fits** | `Rotated`, model functions returning one model per dataset | `notebooks/mwe/mwe_rotating_epochs` | Becomes part of the orbit examples (6a.1) |
| 6 | **Disks and rims** | `ModulatedGaussianRim`, `FlaredDisk` (Blakely et al. 2024) | The PDS 70 notebooks on the unmerged branch `pds70-disk` | Needs `pds70-disk` reviewed and merged first |
| 7 | **Kernel-phase data** | kernel phases through `phi_mat` | None | Waits on a kernel-phase reader (`pmoired_parity.md`, KPFITS) |
| 8 | **Gravity-darkened stars** | `GravityDarkenedStar` | Arrives with the ELR stack (#99) | Owned by the ELR session |
| 9 | **Orbits** | `KeplerOrbit`, starting orbits, radial velocities, `Attached` | None | Written with Stage 6a.1 |
| 10 | **Spectro-interferometry** | lines, node spectra, differential phase, NFLUX | None | Written with Stage 6a (the Brγ disk MWE) |
| 11 | **Correlated calibration errors** | Stage 6d's low-rank nuisances | None | Written with Stage 6d |

Items 1–4 need no new library code and should be written before the 0.2.0 release if time allows. Items 5–11 come with the stages and branches named.

## Also worth doing
- Promote the best developer demos in `notebooks/mwe/` (`mwe_fit`, `mwe_vlti`, `mwe_rotating_epochs`, `mwe_uv_lattice_mft`) into examples or into the items above, rather than keeping two parallel sets.
- A "choosing a regulariser and prior" guide under Background (Stage 7).
