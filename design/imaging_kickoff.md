# Imaging project: kickoff brief for the implementing session

Written 2026-09-30 by the planning session. Read this file first, then `design/imaging_plan.md`, which is the authoritative staged plan (revision 4 plus the orchestration section). This brief holds the context the plan relies on but does not spell out: facts verified in the code, measurements, research conclusions, and the user's standing preferences.

## 0. Start here: Stage 0 checklist
1. You are on branch **`imaging`**, created from `chromatic-scenes` at `38569d3`. It has **not** been pushed; ask before pushing.
   - `uv.lock` has a one-line modification that predates this project. Leave it alone unless the user says otherwise.
   - `design/imaging_plan.md` and this file are untracked. Commit them as the first commit of Stage 0, with the user's OK.
2. Stage 0 is a set of PR-sized chunks; see "Implementation orchestration" in the plan:
   - **C0a:** `design/image_reconstruction.md` (the decision record). Sections 3–6 below are the source material.
   - **C0b:** `likelihood.whitened_residuals` and a redefined `model_loglike`, with regression tests. **Opus**, because this changes existing behaviour.
   - `_precision.py` and the coverage fixture/generator were moved to Stages 3 and 4, where they are first used.
   - Also update `AGENTS.md`: the precision policy, the new modules and the import DAG.
3. Stage-0 working example: re-run the existing tutorials and show that the results are unchanged. Make a figure of the old and new closure-phase χ² along a slice through a phase-wrap region, showing the kink is gone.
4. **Checkpoint:** stop and show the user before starting Stage 1. Each stage ends with a PR from `imaging-sN-<name>` into `imaging` and a feedback pause.

## 1. User preferences and constraints (standing)
- **No ssh to OzSTAR** (the Swinburne HPC cluster) from Claude, ever. Give the user commands to run instead. The one deferred cluster task, a GPU benchmark and TF32 check, is specified in `design/imaging_plan.md` under "Deferred OzSTAR GPU test".
- **Precision:** every FT and matmul uses `precision=lax.Precision.HIGHEST`. Fitting and sampling entry points default to float64 via a local `jax.enable_x64(True)` context, with float32 as an option. Never enable x64 globally; AGENTS.md forbids it in tests at import time. Forward-model code must pass in both precisions.
- **Test data: simulate only, against ground truth.** Do not use WR 137, NGC 1068 or other real DISCO products. Borrow realism from ν Hor: its baselines and uv coverage, and its per-point σ for SNRs.
- **Minimise technical debt:** one likelihood, one parameter currency, no sampler wrappers, no bespoke transform library. Users are new PhD students and sceptical astronomers, so readability matters.
- Ask before adding runtime dependencies (AGENTS.md). **Already approved:**
  - declare `optax` as a runtime dependency;
  - optional extras `drpangloss[nufft]` (`jax-finufft>=1.3.1`) and `drpangloss[sampling]` (`blackjax`).
- zodiax stays `>=0.4`. Use the `evaluate(pixel_scale_mas=...)` protocol so that a later move to zodiax 0.6 `Expression`s is only an import change.
- Commit or push only when asked. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 2. Repo facts verified during planning (drpangloss at `38569d3`)
- **Modules and import DAG** (`AGENTS.md`): `_utils`/`_geometry`/`bessel` → `oifits`/`amigo`/`oidata` → `models` → `likelihood` → `inference` → `_grid` → (`grid_fit`, `limits`) → `plotting`.
  - `spectra.py` (`PowerLaw`, a component's `flux` can be a `Spectrum`) is the precedent for the new parameterisation modules.
  - The rule "new model code goes in models.py" applies to `Image`.
- **Image convention** (`_geometry.py:16` `image_coordinates`): x = −(col − c)·s (East left), y = (c − row)·s (North up), with c = (N−1)/2.
  - `offset_phase` (`_geometry.py:27`) computes exp(−2πi·mas2rad·(uu·dra + vv·ddec)), with uu = u/λ.
  - The image-to-uv DFT sign is pinned by `tests/test_models_sources.py:376`.
  - AGENTS.md requires an orientation regression test for any new model.
- **`Component`** (`models.py:204`): subclasses implement `_centred_cvis(uu, vv)` and `_centred_image(xx, yy, pixel_scale_mas)`; offsets and `System` mixing come for free.
  - `System` (`models.py:580`) computes V = Σ fᵢ(λ)Vᵢ / Σ fᵢ(λ). Component paths look like `"env.flux"`, via zodiax `get`/`set`.
- **Data** (`oidata.py`): `OIData.model(model)` calls `model.model(u, v, wavel)` and then `standardize_model` (:478).
  - `mixed_log_complex` (AMIGO DISCO) computes `vis_mat @ log(cvis).real + phi_mat @ log(cvis).imag`.
  - `to_phases` (:504) applies `phi_mat` to `np.angle(cvis)` or to closure phases.
  - `residuals` (:451) wraps **only unprojected** phases (`_phases_wrap`).
  - ⇒ **Hazard:** projected (kernel/DISCO) phases jump if a model visibility crosses the negative real axis. Binaries with flux < 1 never do; images and uniform disks beyond their null can. The plan's `diagnose` regime check covers this.
  - AMIGO loader: `amigo.py` (`load_oi_data`, `mixed_disco_fields`, which negates u,v). `OIData.with_model(model, key, noise_scale)` simulates noisy data while keeping the operators; use it for every simulation.
- **Likelihood** (`likelihood.py`):
  - `model_loglike` (:61) is a sum of `norm.logpdf` of the wrapped residuals, with `inflated_errors` (:24; `vis_error_rel`, `phi_error`).
  - `build_model` (:141) is `template.set(paths, values)`, and it accepts array values.
  - `numpyro_model(model, priors, data)` (:234) samples every key of `priors` as a path, with a positivity check on flux priors.
- **Optimisers:** only `optx.compat.minimize(... "BFGS")`, for 1-D flux refinement (`grid_fit.py:94`, `limits.py:446`).
- **Installed** (`uv.lock`): jax/jaxlib 0.9.1, equinox 0.13.4, zodiax 0.4.1, optimistix 0.1.0 (has `LBFGS`, `LevenbergMarquardt`, `Dogleg`), lineax 0.1.0 (`Normal`, `CG`, `LSMR`; `NormalCG` is deprecated), numpyro 0.20.0, optax 0.2.7 (transitive).
  - `jax.scipy.fft` has `dctn`/`idctn`.
  - `jax.nn.softmax` accepts `where=`.
  - Python venv: `/Users/benpope/code/drpangloss/.venv/bin/python`.
- **In-repo data** (`data/`): `NuHor_F480M.oifits` (AMI), `calibrated_visibility.npy` (AMIGO mixed-DISCO record; use it for AMI coverage and operators), `calibrated_visibility_private.npy`, and `chi2_ppf*.npy`.
- `HarmonixModel.source` is a static field that may hold arrays. zodiax 0.6's new check would reject that; note it for the later migration.

## 3. Measurements made during planning (laptop, Apple arm64, CPU, jax 0.9.1)

**Separable DFT, jitted value+grad** of `einsum('kr,rc,kc->k', B, I, A)`, in ms:

| npix \ points | 10³ | 10⁴ | 10⁵ |
|---|---|---|---|
| 64 | 0.36 | 2.3 | 23 |
| 128 | 0.90 | 7.7 | 75 |
| 256 | 2.5 | 23 | 226 |
| 512 | 8.1 | 78 | 817 |

**Accuracy** on a 129² resolved scene (star plus six Gaussian blobs), with 2×10⁴ random frequencies |f·s| < 0.4, measured against a float64 DFT. The lowest-10% |V| is about 0.009.

| Method | Max \|ΔV\| | Max phase error on low-\|V\| points |
|---|---|---|
| float32 separable DFT | 4.7e-7 | 1.6e-5 rad |
| Padded FFT ×2, bilinear | 2.6e-2 | 0.41 rad |
| Padded FFT ×4, bilinear | 5.4e-3 | 0.11 rad |
| Padded FFT ×4, cubic | 7.6e-6 | 2.2e-4 rad |
| Padded FFT ×8, cubic | 4.4e-7 | 1.4e-5 rad |

The padded FFT was run in **float64**, which flatters it.

**Lesson from the first attempt:** with even N, putting the image centre at (N−1)/2 inside an FFT grid centred on N/2 gives a half-pixel offset and wrong answers. Use odd N or apply an explicit phase factor. The same trap applies to the NUFFT convention mapping.

## 4. Research conclusions, with sources (for the design note)

**dorito** (github.com/maxecharles/dorito; paper arXiv:2510.10924, Charles et al., PASA):
- **What to reuse:** only the pure pieces.
  - `stats.py` regularisers: TV = Σ√(dx²+dy²+ε²), TSV = Σ(dx²+dy²), ME = Σ P log P, L1, L2, with zero-padded differences.
  - The `reg_dict` weighted sum.
  - The linear basis `LinearBasis` (formerly `ImageBasis`), and PR #32's `LatentBasis`.
  - The idea of a phase-centre (centroid) prior.
- **What not to take:**
  - The forward model: an MFT onto the gridded 51×51 AMI half-plane, then downsample by 2, then divide by `base_uv`.
  - Its image rotation by parallactic angle with linear interpolation.
  - The amigo coupling.
- **PR #32** (`frito_overhaul`) is an open draft with bugs:
  - `eqx` is used without being imported;
  - there is a `contrast` keyword mismatch;
  - `log_dist` is now inconsistently linear.
- **Strategic note:** drpangloss's `amigo.py` already reads PR #32's mixed-DISCO format (the same keys) and builds the same observable. Its HMC notebook `hmc_fitting.ipynb` uses a Dirichlet pixel prior.
- **dorito's recipe for AMI DISCO imaging** (from dorito_notebooks `wr137_disco.py`), which the Stage-3 simulated working example imitates:
  - `ResolvedDiscoModel(..., distribution=ones((151,151)), oversample=6.0)`;
  - `{"ME": (1e5, dorito.stats.ME)}`;
  - Adam(1e-3) for 20k epochs, then optimistix BFGS;
  - a phase-centre prior of width 1e-3, and a sum-to-one prior.

**Gauge freedoms:**
- Closure, kernel and DISCO phases are invariant to image translation. V² data alone also leaves a 180° flip ambiguity. Normalised visibilities remove the flux scale.
- The gradient of an invariant objective has no component along the gauge direction. Drift, such as MiRA's "travelling modes", comes from the symmetry-breaking terms: edges, positivity and the pixel washboard.
- How other codes fix it:
  - a centred default image: BSMEM, eht-imaging;
  - a centroid penalty: eht-imaging `cm`, SMILI, SQUEEZE, OITOOLS `centering`, DPI;
  - compactness: MiRA, WISARD;
  - hard reparameterisation: Comrade `shifted(m, -centroid)`, SPARCO's star at the origin;
  - integer recentring between runs: MiRA `-recenter`.
- Key reference: Thiébaut & Young 2017, arXiv:1708.08390.
- **Design rule:** something must fix the origin. That is an analytic star, a `Centroid` prior (a Gaussian on the centroid, which is legitimately probabilistic), or a centred GP mean.

**IFT and NIFTy:** mostly Gaussian-process and hierarchical-Bayes machinery under new names. Key sources: Enßlin et al. 2009 (arXiv:0806.3474), the correlated-field model (Arras et al. 2021, arXiv:2008.11435), and NIFTy.re (Edenhofer et al. 2024, arXiv:2402.16683). What to take for drpangloss:
- The whitened-latent idea.
- A DCT-based Matérn-like field:
  - S ∝ (κ²+λ_jk)^(−order), with λ_jk = 4sin²(πj/2N) + 4sin²(πk/2N) and S₀₀ = 0;
  - normalised to a mean marginal variance of σ²;
  - `order=1` is exactly TSV+L2 on log-brightness (Comrade/HIBI GMRF, arXiv:2511.17706).
- **Do not depend on nifty.re:**
  - its licence metadata is ambiguous;
  - it churns between versions, and its demos assume x64;
  - `amend` makes callables static;
  - J-UBIK forces x64 globally.
- **Pitfall:** never jointly MAP the image and its GP hyperparameters.

**Optimisers** (verified in the installed sources):
- **Levenberg–Marquardt:** `optx.LevenbergMarquardt(..., linear_solver=lx.Normal(lx.CG(rtol=0., atol=0., max_steps=k)))` or `lx.LSMR(0., 0., max_steps=k)`, keeping `jac="fwd"` (the default).
  - Its default `lx.QR()` and `jac="bwd"` materialise the Jacobian.
  - An inner solve with non-zero tolerances that hits `max_steps` **aborts the outer solve**; use fixed iterations.
  - Plain `GaussNewton` has no globalisation.
  - optimistix does not pass the linear solver a preconditioner. Whitened GP latents give a prior Hessian of I, which makes this exactly MGVI's metric JᵀN⁻¹J + 1.
- **Toy test** (planning agent; 300² plus a parameter; V² + CP + L2 + TSV residuals):
  - LM with Normal(CG, 20 steps): loss 1.8e10 → 2750 in 30 steps (about 1 s);
  - `optax.lbfgs`: about 2.9e4 after 300 iterations;
  - Adam: about 7e4.
- **Phase residual:** r = 2 sin(Δ/2)/σ, so r² = 2(1−cos Δ)/σ², which is von Mises with κ = 1/σ². Use Δ/σ for projected phases.
- **Starting points:** `jnp.angle` has NaN or meaningless gradients at V ≈ 0, and a flat start image failed in the toy. Initialise from a parametric fit (`Image.from_model`).
- **L-BFGS:** `optx.LBFGS` (a declared dependency) for MEM and TV. `optax.lbfgs` needs `value_fn` at update time.
- **Adam** is invariant to diagonal gradient scaling, so amigo-style Fisher preconditioning (`-1/diag(F)`) only matters for SGD.
- **Matrix-free Gauss–Newton diagonal** (GNB): E[(Jᵀε)²] with Rademacher ε, one VJP per probe.

**Sampling:**
- BlackJAX 1.6.2 (Apache-2.0; `jax>=0.9`; all its dependencies are already in the venv).
  - Pytree positions work.
  - No automatic constraints: add log-Jacobians yourself.
  - `blackjax.window_adaptation(blackjax.nuts, logdensity)`; MCLMC via `blackjax.mclmc_find_L_and_step_size(...)`, and MAMS via `adjusted_mclmc`.
  - Progress bars need jax ≥ 0.10, so they are unavailable here.
  - The API moves fast, so do not wrap it; show it in tutorials.
- numpyro `NUTS(potential_fn=...)` is the dependency-free alternative.
- Float32 log-densities of 10⁴–10⁵ carry about 0.1 absolute error, so run samplers under x64.

**Fourier transforms:**
- **Martinache HDR 2018** (https://frantzmartinache.eu/static/share/hdr_no_papers.pdf, pp. ~73–80) tested nearest-pixel FFT sampling, not interpolation, against his LDFT: about 50× worse, while the LDFT has a phase bias below 4e-6 rad. XARA builds its DFT matrices in complex128.
- **FINUFFT:**
  - Per-point bound: |ΔV_j| ≤ ε‖I‖₁.
  - The float32 floor is about 1–3e-5 (upsampfac 2), and eps < 1e-6 is impossible in single precision.
  - That is fine for VLTI (0.2–1°) and for measured AMI performance (0.14° CP), but **not** for AMI's 1e-4-contrast requirement (σ_CP ~ 2e-5 rad). There, use the DFT or float64 NUFFT.
- **jax-finufft 1.3.1:** `nufft2(f, x, y, iflag=-1, eps=1e-6, opts=None)`.
  - `f` must be complex. Axis 0 pairs with `x`. Modes run −N/2…N/2−1, so the centre is at index N/2.
  - It re-plans on every call (issue #157).
  - It uses private jax internals (jax 0.9.1 is handled; vmap broke on 0.10.2).
  - Its drpangloss mapping: `nufft2(I, Y, X, iflag=+1)` with X = 2π·mas2rad·s·uu and Y = 2π·mas2rad·s·vv, times exp(+i(X+Y)/2) for even N. **Verify against the DFT; do not trust this derivation blindly.**
- **GPU:** on A100/H100, JAX's default matmul precision is TF32 (about 1e-3), which is why `HIGHEST` is required.

**zodiax and dLux** (for later; do not depend on them now):
- zodiax 0.5 (released) has `map_optimisers` (per-path optax over a flat dict), `jacobian`/`hessian` with batching, and dict-style `set`.
- zodiax 0.6 (`numerics` branch, PR #89, unreleased) has `Expression.evaluate(**context)`, `Transform` (`fwd`/`inv`/`initialise`), `Map`, `Mask`, `Exp`, and a matrix-free `gauss_newton`.
- dLux's `restrucutre` (sic) and `overhaul` branches are superseded or CI-less rewrites. Their parametric ideas now live in zodiax 0.6.

## 5. Reproduction snippets
```python
# Separable DFT in drpangloss conventions (reference/oracle), frequencies in cycles/mas
import jax.numpy as jnp
from jax import lax
def dft(I, fu, fv, x, y):   # I (N,N); fu,fv (M,) = uu*mas2rad; x,y (N,) from image_coordinates
    A = jnp.exp(-2j*jnp.pi*jnp.outer(fu, x))   # (M, Ncol)
    B = jnp.exp(-2j*jnp.pi*jnp.outer(fv, y))   # (M, Nrow)
    return jnp.einsum('kr,rc,kc->k', B, I.astype(A.dtype), A, precision=lax.Precision.HIGHEST)
```
The benchmark and accuracy scripts from planning were ad hoc. Regenerate them as `scripts/bench_ft.py` in Stage 2, using the tables in section 3 as the expected ballpark.

## 6. External references (for the design note)
- Thiébaut & Young 2017, arXiv:1708.08390 (image reconstruction tutorial; gauges; quadratic regularisers as Gaussian priors).
- Chael et al. 2018, arXiv:1803.07088 (closure-only imaging; the `cm` regulariser).
- Kluska et al. 2014, arXiv:1403.3343 (SPARCO).
- Tiede et al., HIBI, arXiv:2511.17706 (GMRF priors, visibility-domain recentring, NUTS at 64²).
- Arras et al. 2022 M87*, arXiv:2002.05218 (resolve; "source teleportation"; centred initialisation).
- Barnett, Magland & af Klinteberg 2019, arXiv:1808.06736 (FINUFFT error analysis).
- Knollmüller & Enßlin, arXiv:1901.11033 (MGVI); Frank et al. 2021, arXiv:2105.10470 (geoVI).
- Ireland 2013, arXiv:1301.6205, and Martinache 2010 (kernel phase).
- Sivaramakrishnan et al. 2023, arXiv:2210.17434 (AMI requirements); Sallum et al. 2024, arXiv:2310.11499 (AMI on-sky CP of 0.14°).
- Code: github.com/maxecharles/dorito (and PR #32), dorito_notebooks, github.com/LouisDesdoigts/amigo, github.com/flatironinstitute/jax-finufft, github.com/blackjax-devs/blackjax, github.com/LouisDesdoigts/zodiax (the `numerics` branch).

## 7. Data for simulations
- **AMI:** `data/calibrated_visibility.npy` and `data/NuHor_F480M.oifits`.
  - The `.npy` file is a dict of AMIGO mixed-DISCO records keyed by filter (F380M, F430M, F480M). Each record has 47 (u,v) points, 206 DISCO coefficients with σ and covariance, and (206×47) log-amplitude and phase operators, plus `wavelength_m`, `parang_mean_deg`, and a `synthetic_truth` field (inspect it: it may already be a simulated product).
  - Use its coverage, operators and σ; replace the coefficients with `OIData.with_model` simulations.
  - Having three filters makes it a ready-made test bed for the Stage-6 joint multi-filter work.
  - The AMI uv set is small, 47 points, so the DFT is trivially cheap for AMI.
- **Long baseline (MATISSE LM LOW):** three ν Hor calibrated OIFITS files in `/Users/benpope/code/nuHor/data/`, git-ignored there:
  - `2023-12-12T004705_…_oifits_0.fits`
  - `2023-12-12T020414_…`
  - `2023-12-13T010931_…`

  C4a (Stage 4) extracts **only geometry and noise** (u, v, wavelengths, triangle indices, flags, per-point σ) into `data/coverage_nuhor_matisse.npz`. **Confirm with the user before committing it.**
- **Synthetic coverage:** also write a simple generator (`tests/_coverage.py`: N telescopes, Earth-rotation tracks, channels, σ drawn from ν Hor statistics), so tests don't depend on the fixture.
- **No real-data imaging anywhere in this project.**

## 8. Execution advice
- Orchestrate one stage per session from the Claude Code CLI or desktop app. Use sub-agents with `isolation: "worktree"` for the parallel chunks.
- Model tiers are in the plan: Opus for conventions, likelihood, `Problem`/`fit` and GP mathematics; Sonnet for well-specified implementation, tests and notebooks; Haiku for mechanical runs.
- The orchestrator alone edits the shared files (`models.py`, `__init__.py`, `AGENTS.md`, `pyproject.toml`).
- Before each checkpoint, run `uv run pytest` (in float32 and x64), `ruff check` (0.11.0), the tutorial-sync test, and the docs build. Report the results honestly.
