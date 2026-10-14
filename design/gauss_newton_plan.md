# Gauss–Newton and Levenberg–Marquardt across virgil: implementation plan

Status: **plan, reviewed, not built** (2026-10-14). Ben accepted the six
recommendations of the Gauss–Newton audit (coordinating session,
2026-10-14; experiments `e1`–`e9`). This note turns them into PR-sized
pieces that Sonnet sessions can build without further design decisions.
It supersedes V19 of `visibility_orbits.md` and the MEM-preconditioning
item of `imaging_plan.md` Stage 7 (the "least-squares form of the entropy
term" option).

Contents: §1 the problem; §2 the design in one page; §3 the explanation
for the docs (the draft Ben asked for); §4 the pieces; §5 where
Gauss–Newton must not be trusted; §6 the split with the orbit solver;
§7 open decisions; §8 review and responses.

## 1. The problem

virgil has three pieces of Gauss–Newton (GN) machinery, built for
imaging:

- `fit(method="lm")`: optimistix's Levenberg–Marquardt (LM) on
  `_Objective.residuals`, dense QR up to 200 coordinates and matrix-free
  CG above (`fitting.py:1216`);
- `gauss_newton_mass`: (JᵀJ)⁻¹ as a fixed dense NUTS mass matrix, with a
  Woodbury branch when parameters outnumber data (`fitting.py:881`,
  `_gauss_newton_covariance` at `:998`);
- `aliases._newton_polish`: a separate damped Newton on the exact
  Hessian, run only after L-BFGS-B with restarts (`aliases.py:502-580`).

Every orbit path, Apep and both pipelines bypass the first two, because
`_Objective._check_least_squares` (`fitting.py:339-368`) refuses fitted
`noise=` terms, data with gains (or other model-dependent covariance) and
likelihood terms with a `log_norm`: none of these is a plain sum of
squares. They fall back to L-BFGS and to adapted NUTS. The audit measured
what that costs:

| experiment | finding |
| --- | --- |
| e3 (8-parameter orbit, NUTS) | adapted dense: 87 leapfrog steps per draw; fixed GN mass: 6.3 — about 50× the ESS per gradient |
| e2 (orbit fits, near starts) | LM 88 % success in 9 steps; L-BFGS-B 75 % in 126 |
| e9 (alias objective) | damped Newton from the start: same optima in 82 vs 230 evaluations (near), 37 vs 341 (mid); 3–9× fewer |
| e8 (fitted error scales) | Fisher block `blockdiag(JᵀJ, 2N_b)` matches the exact Hessian at the MAP; cross terms small |
| e4b, e4d (gains, misspecification) | JᵀJ through the gain-whitened residuals is a good curvature with modest gains; degrades as χ²/N grows |
| e6 (e ≈ 0.03 orbit) | GN fixed mass: 35–59 divergences; `AngleVector` on ω does not help (not a wrap problem) |
| e7, e7b (images) | dense covariance is n²: 256² needs ~34 GB; in float32 the Woodbury result went non-PD while `ok` stayed True |

## 2. The design in one page

**One idea.** Every objective virgil optimizes or samples is a negative
log posterior Φ(z) whose curvature has a Gauss–Newton form

    F(z) = J(z)ᵀ J(z) + D(z),

where J is the Jacobian of a vector of *curvature rows* ρ(z) and D a
non-negative diagonal from elementwise priors and coordinate transforms.
F is positive semi-definite by construction. The gradient is always the
exact ∇Φ from autodiff; only the curvature is approximated. So a damped
Newton step with F converges to the exact optimum of Φ, whatever F's
error, and F⁻¹ at the optimum is the NUTS mass matrix.

The curvature rows are:

1. **Mean rows**: the whitened residuals of `whitened_residuals`, with
   everything that sets the errors or the whitening (inflated errors, gain
   and closure-offset Jacobians and widths, extra-block scales) frozen with
   `jax.lax.stop_gradient`, so only the model prediction moves. With no
   fitted noise and no gains these are today's residuals exactly.
2. **Normalizer rows**: √2 · log σ_k for each observable's diagonal error
   (after `inflated_errors`), and √2 · ½ log det for each marginalized
   linear-nuisance block (a gain block, a closure-offset block, an extra
   observable block). These are the covariance part of the Fisher
   information of a Gaussian (§3). Rows that do not depend on z have zero
   Jacobian and add nothing.
3. **Block weights** for scale-marginalized objectives
   (Φ = Σ_b (fν_b/2) ln χ²_b and its bounded-integral variant): each
   block's rows are multiplied by √w_b with w_b = 2 ∂Φ/∂χ²_b, frozen. For
   the plain marginal w_b = fν_b/χ²_b = 1/ŝ_b², the profile scale: this is
   iteratively reweighted least squares (IRLS).
4. **Prior rows**: the residuals of `AngleVector` priors and of likelihood
   terms (`PositionData.term`, `RVData.term`), as now.

D holds, per coordinate, max(0, ∂²/∂z² of that coordinate's own prior
term), with the log-Jacobian of the bijection included when the matrix is
for NUTS (whose potential includes it) and excluded for `fit` (which, by
its documented convention, evaluates priors without it).

**One damped-Newton kernel** (`src/virgil/_gauss_newton.py`, private,
imports nothing from virgil): dense, for up to `_DENSE_LM` = 200
coordinates, with optional box bounds, an acceptance test on Φ itself,
pure `init`/`step` functions so that callers can `vmap` and `scan` it.
It serves `fit` (objectives that are not least squares), `aliases`, the
static per-epoch search in `epochs`, and the orbit solver's
`fit_multistart`.

**What stays.** optimistix LM stays the solver for objectives that are
already sums of squares (all of today's `method="lm"` fits, including
large images with CG): their paths and results do not change. L-BFGS stays
for regularizers without a residual form and for more than 200 coordinates
that are not least squares. Folding optimistix's dense case into the
kernel is a later, separate decision (§7, D3).

**What goes.** The restart loop of `aliases._optimize`
(`_MAX_RESTARTS`, `_RESTART_GAIN`), `aliases._newton_polish`, the scipy
L-BFGS-B call in `epochs._refine_marginal` (except as a fallback in
`aliases`), the noise/gains/`log_norm` refusals in
`_check_least_squares` (their only remaining use is choosing optimistix
vs the kernel), and `_prior_residuals`'s role in curvature (replaced by
the D rule above, which agrees with it on every prior it accepted).

**Public API changes** (all keyword-only, consistent with
`design/api_argument_order.md`):

- `gauss_newton_mass(model, priors, data, values, *, likelihoods=(),
  noise=None, flat_coordinates=False)`: new `noise=`, as `fit` and
  `numpyro_model` take it. It also accepts data with gains, any prior, and
  returns a diagonal mass above `_DENSE_MASS_MAX` coordinates.
- `fit`: no new arguments; `method=None` now picks the kernel ("lm") for
  objectives that are not least squares with at most 200 coordinates.
  `info["method"]` stays `"lm"`; `info["solver"]` says `"optimistix"` or
  `"gauss-newton"`.
- Pipelines: a `mass` setting, `"adapted"` (default, today's behaviour)
  or `"gauss_newton"`.
- `MaxEntropy` and `TV` gain `residuals` (used with `method="lm"`; the
  default stays L-BFGS for them until the Stage 7 success test passes).

## 3. The explanation (draft for the docs)

This is the main text of the new Background page `docs/gauss_newton.md`
("Gauss–Newton fits and sampling"), written for an astronomer who knows
χ² but not optimization. It must stand alone: no links to `design/`.

> ### Fitting with Gauss–Newton and Levenberg–Marquardt
>
> **Residuals and their Jacobian.** virgil compares a model with data
> through one vector of whitened residuals, r = (model − data)/σ, one
> number per independent observable. The χ² is the sum of their squares,
> and the fit minimizes ½χ² minus the log prior. The Jacobian J is the
> matrix of derivatives ∂r_k/∂θ_j: row k says how residual k moves when
> each parameter moves.
>
> **Why JᵀJ is the curvature.** Near the best fit the model is close to
> linear in the parameters, r(θ + δ) ≈ r(θ) + Jδ, so ½χ² is a quadratic
> bowl whose curvature (Hessian) is JᵀJ. That is the Gauss–Newton
> approximation. JᵀJ is always positive semi-definite: for any direction
> δ, δᵀJᵀJδ = |Jδ|² ≥ 0, a sum of squares. It is positive definite when
> every direction changes at least one residual, that is, when the data
> (or a prior) constrain every parameter. So a Gauss–Newton step,
> δ = −(JᵀJ)⁻¹Jᵀr, always points downhill, which the exact Hessian does
> not guarantee away from the minimum.
>
> **What Gauss–Newton drops.** The exact Hessian of ½χ² is
> JᵀJ + Σ_k r_k ∇²r_k. The second term is each residual times the
> curvature of the model. It is small when the residuals are noise-like
> (they average out, |r_k| ≈ 1 with random signs) or the model is nearly
> linear over the error bars. It matters when the fit is poor (χ²/N ≫ 1,
> a misspecified model) or the model curves sharply on the scale of the
> errors (a companion near a closure-phase wrap, an orbit near e = 0).
> There Gauss–Newton is only an approximation to the curvature: still a
> safe direction to step in, but a poorer guide to the error bars.
>
> **Levenberg–Marquardt.** When the model is far from linear, a full
> Gauss–Newton step can overshoot. Levenberg–Marquardt solves
> (JᵀJ + λD)δ = −g instead, where g is the gradient and D the diagonal of
> JᵀJ. With λ → 0 this is the Gauss–Newton step; with λ large it is a
> short step straight down the gradient, each parameter scaled by its own
> curvature. After each step the loss is checked: if it fell, the step is
> kept and λ is reduced (trust the quadratic more); if not, the step is
> rejected and λ is increased. So the method takes fast Newton-like steps
> near the minimum and cautious gradient steps far from it.
>
> **Noise scales and gains.** When the error bars themselves are fitted
> (`noise={"vis_scale": ...}`), the likelihood has a second term,
> Σ log σ_k, that is not a sum of squares. Its curvature is still known:
> for Gaussian data, each parameter that changes log σ_k adds
> 2 (∂ log σ_k/∂θ)² to the curvature. A scale s multiplying one block's N
> errors therefore adds 2N in ln s, independent of the model, so the
> matrix becomes blockdiag(JᵀJ, 2N). virgil builds this from the same
> machinery, with extra rows √2 log σ_k next to the residuals. This is the
> Fisher information, the expected curvature. When the scales are instead
> integrated out (the orbit and alias searches, where each block's
> likelihood becomes −(ν/2) ln χ²), each block's residuals are weighted by
> 1/ŝ², with ŝ² = χ²/ν the scale the block's own residuals imply: the
> classical iteratively reweighted least squares. Gains correlated across
> channels are integrated out analytically; their covariance whitens the
> residuals, and its log-determinant adds rows of the same kind as log σ.
>
> **Why (JᵀJ)⁻¹ makes a good NUTS mass matrix.** Near the best fit the
> posterior is approximately Gaussian with covariance (JᵀJ)⁻¹ (in the
> coordinates the sampler uses, including the priors). NUTS moves fastest
> when the posterior it sees is round: given that covariance as its
> inverse mass matrix, it sees a unit sphere, so one step size suits every
> direction. Without it, the step size shrinks to suit the most tightly
> constrained combination of parameters, and every draw needs hundreds of
> steps. For an eight-parameter orbit, this cut the cost from 87 to about
> 6 steps per draw. Because the matrix is computed at the fit rather than
> learned during warm-up, it is fixed: NUTS does not adapt it further.
>
> **Relation to zodiax.** zodiax's optimisation tools compute the Hessian
> of the loss with `jax.hessian` (calling it the Fisher matrix), step
> along the gradient preconditioned by its inverse (a natural-gradient
> step), and sample in the parameter's eigenbasis ("eigen-projection").
> virgil does the same with JᵀJ in place of the Hessian. At a good fit
> the two agree. JᵀJ is always positive semi-definite, needs only first
> derivatives, and is the expected Fisher information. A dense NUTS mass
> matrix is equivalent to sampling in the eigenbasis. The Hessian of the
> loss can be indefinite on a ridge or between modes, and costs second
> derivatives. Levenberg–Marquardt with λ → 0 is the natural-gradient
> step.
>
> **When not to use it.** The matrix describes one peak. Do not use a
> fixed Gauss–Newton mass when the posterior has several modes (orbit
> aliases, the 180° flip of a symmetric binary): sample each mode
> separately and combine them by their evidences. Do not use it for
> nearly circular orbits (e ≲ 0.05), where ω and the time of periastron
> become degenerate and the posterior curves away from any fixed metric;
> NUTS then reports divergences. Do not use it for companions too faint
> to be detected, whose posterior fills the prior. Do not use it where a
> companion can cross a closure-phase wall. virgil's pipelines check the
> matrix and the first draws, and fall back to an adapted mass when either
> check fails.

The page also carries (written by the implementers, §4): a code example
(fit → `gauss_newton_mass` → NUTS), the options (`noise=`, gains,
`flat_coordinates`), the size limit and diagonal fallback, and the
pipeline setting.

**Docstrings that carry a short form of this text** (three to six
sentences each, linking to the page):

| object | adds |
| --- | --- |
| `fit` (the `method` entry) | what LM does, the two solvers, the kernel's acceptance test, when L-BFGS is used |
| `gauss_newton_mass` | JᵀJ and why it is PSD; the noise and gains rows; the D term; what is dropped; the size guard; "when not to use" one-liners |
| `_gauss_newton.py` module docstring | the kernel's step, damping and bounds rule, for maintainers |
| `_Objective.curvature_rows` | the four kinds of row (§2) |
| `epochs.marginal_loglike` | the IRLS weight 1/ŝ_b² and its use in the search |
| `aliases.fit_orbit_aliases` (Notes) | the optimizer: damped Newton/IRLS from every start, L-BFGS-B as fallback |
| `MaxEntropy`, `TV` | the residual form and that LM is opt-in |
| `docs/pipeline.md` | the `mass` setting, its checks and fallback |

## 4. The pieces

Seven PRs. GN1 comes first; GN2 and GN4 depend only on GN1; GN3 on GN1
and GN2; GN5 on GN2; GN6 is independent; GN7 (docs page) lands with GN3
and is extended by GN5. Each is one Sonnet `pr-worker` session.

```
GN1 kernel + _marginal ─┬─ GN2 curvature rows, V19 mass ─┬─ GN3 fit path (+ docs page)
                        │                                └─ GN5 pipelines
                        └─ GN4 aliases + epochs search
GN6 MEM/TV residual forms (independent)
```

General rules for every piece:

- float64 by default, as `fit` and `gauss_newton_mass` already run; every
  new test also runs in float32 where the entry point supports it.
- No new module-level imports outside the order in AGENTS.md;
  `_gauss_newton.py` and `_marginal.py` go into AGENTS.md's package table
  and import map in the PR that adds them.
- Local runs are tiny (a few seconds to a minute, one process). Anything
  larger is an OzSTAR job via `~/code/ozstar_scripts`, which Ben runs;
  agents never touch the cluster.
- Regression tests compare with values computed on `main` before the
  change and stored in the test (not recomputed by the new code).

### GN1. The damped-Newton kernel, and the marginal surface moved down

**Files.** New `src/virgil/_gauss_newton.py`; new `src/virgil/_marginal.py`
(moved from `epochs.py`); `epochs.py` imports from it.

**Behaviour: `_gauss_newton.py`.** A problem is any object (an
`eqx.Module`, or a `functools.partial` closure) with three pure
functions of a flat 1-D vector z:

- `loss(z)`: the scalar Φ;
- `rows(z)`: the curvature rows ρ (any length, possibly 0);
- `diag(z)`: the non-negative diagonal D (length n), or zeros.

```python
class State(NamedTuple):
    z; value; grad; curvature; lam; steps; rejects; stop  # stop: int code

def init(problem, z0, lower, upper) -> State
def step(problem, state, lower, upper, gtol, dtol) -> State
def solve(problem, z0, *, lower=None, upper=None, gtol=0.0, dtol=0.0,
          max_steps=200) -> State
def curvature(problem, z) -> (n, n) array   # JᵀJ + diag, J by jax.jacfwd
```

- `curvature`: J = `jax.jacfwd(problem.rows)(z)` (n ≤ 200, so forward
  mode), F = JᵀJ + diag(D), symmetrized.
- `step`: the projected gradient sets g_i = 0 for a coordinate on a bound
  whose gradient pushes it outward (z_i ≤ lower_i and g_i > 0, or
  z_i ≥ upper_i and g_i < 0); those coordinates are held (their rows and
  columns of the system replaced by the identity). Solve
  (F + λ·M)δ = −g by Cholesky, M = diag(max(F_ii, 1e-12·max_j F_jj, tiny)).
  Trial z′ = clip(z + δ, lower, upper). Accept if Φ(z′) is finite and
  Φ(z′) < Φ(z): then λ ← max(λ/3, 1e-12), recompute value, gradient
  and F at z′. Otherwise λ ← 4λ. λ starts at 1e-3.
- Stop codes: `converged` when the projected gradient's infinity norm is
  ≤ `gtol` or the Newton decrement ½ gᵀF⁻¹g (on the free coordinates) is
  ≤ `dtol`; `stalled` when λ > 1e10; `limit` at `max_steps` (accepted
  plus rejected); `non-finite` when the starting value or gradient is
  not finite. A caller passes the test it needs (`fit` passes `gtol`, the
  orbit searches `dtol`).
- `solve` is `jax.lax.while_loop` over `step`, jitted with `eqx.filter_jit`
  and the problem as an argument (so repeated problems of one structure
  reuse a compilation, as `fit`'s solvers do). `step` is pure, so
  `jax.vmap(lambda z0: solve(problem, z0, ...))` and a `lax.scan` over
  `step` both work: this is what the orbit solver's `fit_multistart`
  needs.
- `lower`/`upper` default to ±inf (no bounds).

**Behaviour: `_marginal.py`.** Move `epochs._Surface`, `_blocks`,
`_check_marginal_data` and `_log_integral` (and only what they need)
unchanged into `_marginal.py`, which imports `likelihood` and nothing
else from virgil, so that `fitting` can use it without a cycle. `epochs`
re-imports them; no names change. Add one method:

```python
def weights(self, chi2, dof, data):
    """IRLS weights w_b = 2 ∂(−score)/∂χ²_b, by autodiff of score."""
```

w_b ≥ 0 because the score falls as any χ²_b rises; for `s_max=None` it
is dof·ν_b/χ²_b.

**Deleted.** Nothing yet (the move leaves no copy behind).

**Tests** (`tests/test_gauss_newton.py`, new):
- a linear least-squares problem converges in one accepted step to
  `onp.linalg.lstsq`'s answer (1e-10, float64);
- the Rosenbrock function as residuals (10(y−x²), 1−x) from (−1.2, 1)
  reaches (1, 1) to 1e-8 in ≤ 50 steps;
- a non-least-squares Φ (ln of a sum of squares, the marginal form) with
  IRLS rows reaches the minimum found by `scipy.optimize.minimize`
  (BFGS, tight tolerances) to 1e-8;
- bounds: a quadratic whose minimum lies outside the box stops on the
  bound with the right coordinates held, and a coordinate that starts on
  a bound with an inward gradient leaves it (the case
  `_newton_polish` gets wrong today);
- `jax.vmap` over 8 starts equals a Python loop over them, bit for bit;
- a `scan` of 20 `step`s equals `solve(max_steps=20)`;
- every accepted step lowers Φ (monotonicity, recorded over a run);
- non-finite start gives `non-finite`; a flat direction with D = 0 does
  not produce NaNs (the M floor);
- `_Surface.weights` equals ν_b/χ²_b for `s_max=None`, and equals finite
  differences of the score for `s_max=3` and `dof=0.5`;
- `test_epochs.py` passes unchanged (the move).

**Local benchmark.** None beyond the tests.

**Docs.** AGENTS.md (package table, import map); module docstring of
`_gauss_newton.py` (the §2 kernel paragraph, with the damping rule).

### GN2. Curvature rows and V19: `gauss_newton_mass(noise=…)`, gains, guards

**Files.** `likelihood.py` (`_whiten_with_log_norm`, a new
`_curvature_rows`), `_linear.py` (`whiten_blocks` returns per-block log
dets), `gains.py` (pass them through), `observables.py` (each extra
block's log normalizer, already computed), `fitting.py`
(`_Objective.curvature_rows`, `_Objective.curvature_diag`,
`_gauss_newton_covariance`, `gauss_newton_mass`), `svi.py`.

**Behaviour.**

1. `likelihood._curvature_rows(model, data, noise)` returns the rows of
   §2 for one dataset:
   - mean rows: `_whiten_with_log_norm(..., frozen=True)`, where `frozen`
     applies `jax.lax.stop_gradient` to the inflated errors, to the
     prediction used for the gain Jacobian (`_gain_jacobian`), to the
     gain and offset widths, and to the extra blocks' marginalized
     scales, while `resid = prediction − reference` stays live. Mean
     terms (`wavel_scale`, `wavel_offset`, `north_angle`) act through
     the prediction and so stay live too.
   - normalizer rows: √2·log(σ_k) for the diagonal errors from
     `inflated_errors` (for uncorrelated wrapped phases, √2 times the log
     of the von Mises effective error; for correlated closure phases, the
     whitened rows' effective errors; the penalty rows' constant errors
     drop out), then √2·(½ log det) per marginalized block: per gain block
     and the spanning block from `whiten_blocks` (new third return value,
     the per-block `0.5 * sum(logdets)`), per closure-offset block, per
     extra observable block.
2. `_Objective.curvature_rows(z)`: the concatenation over datasets of (1),
   then the likelihood terms' residuals (not frozen: their errors are not
   exposed; see below), then the `AngleVector` priors' residuals. Its
   coordinates are the parameters *and the noise sites* (not tied ones),
   in the order `problem.sites + noise sites`.
3. `_Objective.curvature_diag(z, jacobian)`: per elementwise coordinate,
   `max(0, d²/dz²[−log p(x(z)) − (log|dx/dz| if jacobian else 0)])`,
   computed by a `jax.jvp` of `jax.grad` with a ones tangent (the terms
   are separable). With `jacobian=False`, priors flat in the fitted
   coordinate give 0 (as now). For a Normal prior on the real line this
   is 1/σ², what `_prior_residuals` gives today.
4. `_gauss_newton_covariance(problem, z)` builds F from (2) and (3) with
   `jacobian=True`, over parameters and noise sites, keeping both
   branches (Cholesky when rows ≥ coordinates, Woodbury otherwise). It
   then checks the result: symmetrize, Cholesky-factor the covariance
   itself; `ok` is false if that fails or any entry is non-finite. This is
   the PSD check that e7b showed was missing. Return also the condition
   estimate (max/min of the Cholesky diagonal, squared).
5. `gauss_newton_mass(..., noise=None)`:
   - the matrix covers the noise sites too, keyed by their numpyro site
     names (`"noise.vis_scale"`, `"noise[1].phi_scale"`), in the
     coordinates `numpyro_model` samples (bijections from `_Objective`,
     as for parameters);
   - accepts data with gains, any prior, and likelihood terms with a
     `log_norm` (curvature left out, as documented now);
   - **size guard**: if n > `_DENSE_MASS_MAX` = 8192 coordinates, it
     returns `dense_mass=False` with the diagonal of the GN covariance
     (Woodbury branch: `w²·(1 − Σ_j (L⁻¹K)_{ji}²)` plus the flat-prior
     correction, never forming n×n; Cholesky branch: 1/diag(F)), and a
     `UserWarning` naming the size. 128² images and above get this;
   - raises `SingularCurvature` when `ok` is false, with the condition
     estimate in the message; records `out.condition`.
6. `svi._gauss_newton_start` (the function at `svi.py:210-245`): noise
   sites now come in the GN matrix; delete the comment and the branch
   that gave them `init_scale` widths only when they were absent (keep
   the fallback for sites still absent, e.g. tied terms).

The likelihood terms' rows are not frozen, so for a fitted jitter they
give about half the Fisher curvature (r²(∂ℓ)² rather than 2(∂ℓ)²). That
is safe (PSD; a wider mass in that direction) and documented. Exposing
per-row log σ from `PositionData.term`/`RVData.term` is a follow-up if a
benchmark shows it matters.

**Deleted.** In `gauss_newton_mass`, the `_check_least_squares` call;
in `_gauss_newton_covariance`, `prior_curvature` (replaced by
`curvature_diag`). `_prior_residuals` stays for `_Objective.residuals`
(optimistix's path).

**Tests** (`tests/test_fitting.py`, `tests/test_sampling.py`):
- *Unchanged where it should be.* For the existing `gauss_newton_mass`
  fixtures with Normal and ImproperUniform priors and no noise, the
  covariance equals a stored copy computed on `main` to 1e-10. For
  Uniform and LogUniform priors it differs from it by exactly the
  transform diagonal 2σ(z)(1−σ(z)) in the precision (test against the
  stored copy plus that term).
- *V19 block.* Binary with `noise={"vis_scale": LogUniform(0.1, 10),
  "phi_scale": LogUniform(0.1, 10)}`, at the MAP (from `fit`, L-BFGS):
  the noise block of F equals 2N_b·(d ln s/dz)² to 1e-6 (N_b the block's
  independent observables), the θ–noise cross block of F is 0, and eig(F⁻¹H) with H the exact Hessian of
  the negative log posterior (`jax.hessian` in the same coordinates) lies
  in [0.8, 1.25] (e8's check, as a test).
- *Gains.* Binary with `with_gains(telescope=0.03, baseline=0.02)`: the
  mass is computed, PSD, and eig(F⁻¹H) in [0.5, 2] (e4b's check). With a
  fitted `vis_gain_<group>`, its diagonal entry is within a factor 3 of
  the exact Hessian's.
- *Added errors.* `vis_error_rel` fitted: F has a non-zero θ–noise
  block, PSD.
- *PSD guard.* A forced near-singular image case in float32 (e7b's
  16², errors ×0.001) raises `SingularCurvature`, not a silently non-PD
  matrix.
- *Size guard.* With `_DENSE_MASS_MAX` monkeypatched to 10, a 16-pixel
  image returns a diagonal whose entries equal the diagonal of the dense
  result to 1e-10.
- `test_svi.py`, `test_angles.py`, `test_flat_coordinates.py`,
  `test_sampler_flat_coordinates.py`, `test_no_data_priors.py` pass
  (with the stored-copy updates above where a fixture uses bounded
  priors).
- A NUTS smoke test: the V19 mass on the binary with noise runs 50 warmup
  and 50 draws with no divergences and mean steps per draw < 15.

**Local benchmark** (one process, a few minutes): e3 on the orbit with
`noise=` scales added (8 + 2 parameters), 1 chain, 300 warmup, 300 draws:
fixed V19 mass vs adapted dense. Report steps per draw, divergences,
min ESS per gradient. Pass: ≥ 10× ESS per gradient, no divergences.

**Docs.** `gauss_newton_mass` docstring (the table in §3); the
`docs/api/fitting.md` blurb ("for sampling large images" becomes "for
sampling any posterior with one peak; see Gauss–Newton fits and
sampling").

### GN3. `fit` on the kernel for objectives that are not least squares

**Files.** `fitting.py` (`fit`, `_Objective`, a new `_gn` solver beside
`_lm`), `docs/gauss_newton.md` (new), `mkdocs.yml` (Background nav),
`CHANGELOG.md`.

**Behaviour.**
- The method rule in `fit` (default `method=None`):
  1. the whole objective is a sum of squares (`_has_residuals`, as now)
     → `"lm"` on optimistix, exactly as today;
  2. otherwise, if no regularizer lacks `residuals` and the number of
     unconstrained coordinates (parameters and noise sites) is
     ≤ `_DENSE_LM` → `"lm"` on the kernel;
  3. otherwise `"lbfgs"`.
  `method="lm"` given explicitly follows 1 then 2, and raises
  `NoLeastSquaresForm` only when neither applies (a regularizer without
  `residuals`, or > 200 coordinates without a least-squares form).
- `_gn(problem, z0, scale, max_steps, gtol, budget)`: flattens z with
  `jax.flatten_util.ravel_pytree`, builds a kernel problem with
  `loss = problem.loss / scale`, `rows = problem.curvature_rows / √scale`,
  `diag = problem.curvature_diag(jacobian=False) / scale`, tolerance from
  `_lm_tolerance` (so `gtol` means what it means for the other methods),
  and runs `_gauss_newton.solve` in chunks of `_LM_CHUNK` with the whole
  `State` carried between chunks (so a time limit does not change the
  path, unlike optimistix's chunks). Default `max_steps` 1000, as for LM.
- **Fallback**: if the kernel stops `stalled` or `non-finite`, `fit`
  continues with L-BFGS from the kernel's last finite point and reports
  `info["solver"] = "gauss-newton+lbfgs"`; the non-convergence warning
  is then L-BFGS's.
- `info["solver"]`: `"optimistix"`, `"gauss-newton"` or
  `"gauss-newton+lbfgs"`; `info["method"]` stays `"lm"`.
- **Marginalized scales (private, for the orbit solver).**
  `_Objective(..., marginal=None)`; `marginal` is a dof value or a list
  (per dataset) of per-block arrays f. With it, each dataset's data term
  in `loss` is `−_Surface(d, s_max).score(chi2_blocks, f, d)` instead of
  ½χ² (no noise sites allowed with it: `ValueError`), and
  `curvature_rows` weights each block's mean rows by
  √(`_Surface.weights`), frozen with `stop_gradient`. `fit` does not
  expose it; the orbit solver's `fit_multistart` constructs `_Objective`
  directly. `numpyro_model` is unchanged.

**What changes for existing users.** Fits with fitted `noise=`, with
gains, with priors such as HalfNormal or Beta, or with `log_norm` terms,
and ≤ 200 coordinates, now run LM on the kernel by default instead of
L-BFGS. They reach the same optimum (both use the exact gradient of the
same loss) to the fit tolerance, in fewer steps. A multimodal problem can
end in a different local optimum from the same start: the regression
tests below pin this on every existing fixture, and the CHANGELOG says so,
with `method="lbfgs"` as the way back.

**Deleted.** From `_check_least_squares`, nothing is removed (it still
decides optimistix vs the kernel), but its messages no longer tell the
user to switch to L-BFGS for noise and gains; the `fit` docstring's
"so the default method is L-BFGS" sentences for noise and gains go.

**Tests.**
- *Regression.* For every test in `tests/` that fits with `noise=`,
  gains or a `log_norm` term (find them with
  `grep -ln "noise=\|with_gains\|RVData\|jitter" tests/*.py`), add a
  parametrized check that the default fit and `method="lbfgs"` give
  values within 1e-4 relative (or 1e-3 of the Laplace σ, whichever is
  larger) and losses within 1e-6·N. Existing assertions on
  `info["method"] == "lbfgs"` for these fits are updated to `"lm"`, each
  such edit listed in the PR description.
- Pure least-squares fits: `info["solver"] == "optimistix"` and values
  identical to stored `main` values to 1e-12 (the path is untouched).
- `time_limit` chunking: a chunked kernel run equals an unchunked one
  bit for bit.
- Fallback: a problem engineered to stall (a loss with a kink) ends on
  L-BFGS with `info["solver"] == "gauss-newton+lbfgs"`.
- `marginal`: on the `test_aliases` synthetic orbit, `_Objective(...,
  marginal=1.0)` minimized by the kernel reaches the same optimum as
  `epochs.marginal_loglike` maximized by scipy BFGS, to 1e-6 nats; its
  `curvature` at the optimum, inverted, is PSD and within a factor 2 of
  the exact Hessian's inverse in every eigen-direction.
- float32: the binary-with-noise fit converges with `dtype="float32"`.

**Local benchmark** (few minutes, one process): e2 on the orbit with
`noise=` scales, 8 near and 8 mid starts: kernel vs L-BFGS. Report
success (within 1 of the best χ²), median steps and time. Pass: success
rate ≥ L-BFGS's and median steps ≤ ½ of L-BFGS's.

**OzSTAR** (Ben runs; job written by the worker in
`~/code/ozstar_scripts`, branch + PR there): the Apep fit with the
default solver vs `method="lbfgs"` on the current Apep branch, comparing
values and losses. Required before the CHANGELOG calls the change safe
for real data.

**Docs.** New `docs/gauss_newton.md` (§3 text plus the example and
options); nav entry under Background after "Conventions"; `fit`
docstring (`method`, `noise`); CHANGELOG entry (behaviour change, how to
get the old one).

### GN4. `aliases._optimize` and the static per-epoch search on the kernel

This is recommendation 2. It belongs here, not in the orbit solver
(agreed split, §6): the solver will replace `aliases` as a pipeline later,
but `fit_orbit_aliases` stays public and its users get the speed-up now.

**Files.** `aliases.py` (`_Problem`, `_optimize`), `epochs.py`
(`_refine_marginal`, `_negative_marginal_and_grad`'s caller).

**Behaviour.**
- `aliases._Problem` gets `rows(x)`: per epoch, the block residuals of
  `_epoch_chi2` (with the profiled V² amplitude, `vis_amp`, applied to
  block 0, as in `_epoch_chi2`), each multiplied by √w_b from
  `surface.weights(chi2, 1.0, d)` under `stop_gradient`. `loss(x)` is
  `−_loglike`. `diag` is zero (flat priors in these coordinates).
  Jitted at module level like `_value_and_grad`.
- `_optimize(problem, bounds, x0, max_steps)`:
  1. `_gauss_newton.solve` from `x0` with the bounds (±inf where none),
     `dtol = 1e-9` nats, `max_steps = 200`;
  2. if the stop code is not `converged`: one scipy L-BFGS-B run from the
     kernel's point with today's options, then the kernel again from
     L-BFGS-B's point (its old polishing role);
  3. `_fold` the result, as now. Return `(x, value)`.
  The multistart (`_candidates`, `_init_starts`, `_refine`) is unchanged.
- `epochs._refine_marginal`: the kernel on (dra, ddec, flux) with the
  same bounds, `dtol = 1e-12`, rows from the surface's blocks weighted by
  `weights(chi2, dof, d)`; no scipy.

**Deleted.** `_MAX_RESTARTS`, `_RESTART_GAIN`, the restart loop and its
comment block (`aliases.py:494-531`), `_newton_polish` (`:534-580`), the
scipy import in `epochs._refine_marginal`. `_hessian` stays (the Laplace
evidence uses the exact Hessian, correctly).

**Tests.**
- `test_aliases.py` and `test_epochs.py` pass unchanged (band
  probabilities, recovered orbits, peaks).
- New: on `test_aliases._synthetic(12.1)`, from the e9 near starts (8 of
  them, fixed seed), new `_optimize` reaches within 0.01 nats of the best
  of old-`_optimize` values (stored) for every start where the old one
  did, with fewer `value_and_grad` calls in the median (counted with a
  wrapper).
- New: a start on the e = 0 bound with an inward gradient leaves the
  bound.
- New: `epoch_positions` peaks equal stored `main` values to 1e-4 mas and
  1e-4 in flux.
- Laplace evidences per band within 0.05 nats of stored `main` values.

**Local benchmark.** e9 (12 near, 12 mid starts): calls and success,
old vs new. Pass: median calls ≤ ⅓ of the old, success ≥ old.

**Docs.** `fit_orbit_aliases` Notes paragraph; `docs/orbit_fitting.md`
sentence on the optimizer if it names L-BFGS-B.

### GN5. Pipelines: `mass="gauss_newton"`, guarded

This is recommendation 3. The default stays what each pipeline does now
(binary: adapted diagonal; star: adapted dense). Nothing changes unless the
setting is chosen.

**Files.** `pipeline/_common.py` (`_mcmc`, a new `_gn_mass`),
`pipeline/binary.py` (`_run_nuts`, settings), `pipeline/star.py`
(`_nuts`, settings), `pipeline/_core.py` (settings default for old
`run.json` files), `docs/pipeline.md`, `docs/gauss_newton.md`.

**Behaviour.**
- New setting `mass`: `"adapted"` (default) or `"gauss_newton"`. A
  `run.json` without the key reads as `"adapted"`, so resuming old runs
  does not raise `ConfigMismatchError`.
- With `"gauss_newton"`:
  1. *Before sampling*: `gauss_newton_mass(model, priors, data, start,
     noise=p._noise())` at the fit (`flat_coordinates` as the pipeline's
     `numpyro_model`). If it raises (`SingularCurvature`, size guard
     returning a diagonal is allowed) or `out.condition > 1e12`, fall
     back to `"adapted"` with reason `"conditioning"`.
  2. *Warm-up check*: run `mcmc.warmup(..., collect_warmup=True,
     extra_fields=("diverging", "num_steps"))` with the fixed mass
     (`adapt_mass_matrix=False`; step size still adapts). If more than 1 %
     of the second half of warm-up transitions diverge, discard the run
     and start again with `"adapted"`, reason `"divergences"`. Otherwise
     `mcmc.run(mcmc.post_warmup_state.rng_key, ...)` continues from the
     warm-up state.
  3. Never re-adapt a GN mass (no GN-then-adapt hybrid).
- `summary.json` gets `"mass": {"requested", "used", "fallback_reason",
  "condition"}`; the quicklook notebook prints it. A fallback is a
  `Check` with level "warning".
- `_mcmc(posterior, settings, *, dense_mass, mass_kwargs=None)`: the
  GN kwargs replace `dense_mass` when given.

**Deleted.** Nothing.

**Tests** (`test_pipeline_binary.py`, `test_pipeline_star.py`):
- default settings: samples identical to stored `main` values for the
  fixture seed (nothing changed);
- `mass="gauss_newton"` on the binary fixture: runs, `used ==
  "gauss_newton"`, mean `num_steps` lower than the adapted run's;
- a forced fallback (monkeypatch `gauss_newton_mass` to raise
  `SingularCurvature`): `used == "adapted"`, reason recorded, check raised;
- a divergence fallback (monkeypatch the warm-up divergence fraction):
  reason `"divergences"`;
- an old `run.json` without `mass` resumes.

**Local benchmark.** The binary pipeline on its test fixture with the
smallest sampler settings, both masses: steps per draw and wall time.

**OzSTAR.** One real binary dataset and one star dataset through both
masses (Ben runs): ESS per second, fallbacks.

**Docs.** `docs/pipeline.md` (the setting, the two checks, the summary
fields); the "When not to use it" section of `docs/gauss_newton.md`
mentions the star pipeline's closure-phase walls.

### GN6. Residual forms of maximum entropy and total variation

This is recommendation 6 (the Stage 7 "LM form of the entropy term").

**Files.** `imaging.py` (`MaxEntropy.residuals`, `TV.residuals`, the
default-method flag), `fitting.py` (one line in the method rule).

**Behaviour.**
- `TV.residuals(model)`: for each of the four flips, √(2·weight/4 · t_i)
  with t_i = √(dx_i² + dy_i² + ε²) (≥ ε > 0, so smooth), concatenated.
  `value` is unchanged and equals ½Σr² to rounding.
- `MaxEntropy.residuals(model)`: per pixel, with b and q the unit-sum
  image and default, φ_i = b_i ln(b_i/q_i) − b_i + q_i ≥ 0 (the
  generalized relative entropy; Σφ equals today's value because Σb = Σq
  = 1), and r_i = sign(b_i − q_i)·√(2·weight·φ_i). Near b = q compute it
  as (b_i − q_i)·√(2·weight·h(b_i/q_i)/q_i) with h(t) = (t ln t − t +
  1)/(t − 1)², h(1) = ½, using a series for |t − 1| < 1e-3 and the
  double-`where` trick so that gradients are finite. Outside the support
  (q = 0) behave as now. Its GN curvature near b = q is weight/q, the
  entropy metric of Skilling & Bryan (1984): LM on this form is the
  preconditioning Stage 7 asked for.
- Both classes set `lm_default = False`; rule 1 of GN3 then picks LM only
  if every regularizer present has `lm_default` true (all others: true,
  by a default on `_ImageRegulariser`). So default fits with MEM/TV stay
  on L-BFGS; `method="lm"` uses the residuals.

**Deleted.** Nothing now. If the success test passes, a follow-up flips
`lm_default` (Ben's decision, §7 D4).

**Tests** (`test_imaging.py`):
- `value == 0.5 * sum(residuals**2)` to 1e-12 relative (float64) for
  random images, at b = q exactly, and with a support mask;
- gradients of ½Σr² equal `jax.grad(value)` to 1e-10, including at b = q
  (no NaN);
- default `fit` with MEM or TV still uses L-BFGS (`info["method"]`);
- `method="lm"` on a 16² MEM fit reaches the L-BFGS loss to 1e-6
  relative.

**Local benchmark.** A 32² MEM L-curve at 5 weights (0.1, 1, 10, 100,
1e4), float64, `method="lm"` vs default: steps, converged, loss.

**OzSTAR.** The Stage 7 success test: an L-curve from w = 0.1 to 1e4 on
the `mwe_gaussian_field` setup, float32 and float64, both methods.

**Docs.** `MaxEntropy`, `TV` docstrings; `docs/imaging_rml.md` one
paragraph on `method="lm"`.

### GN7. The docs page

Lands inside GN3 (the page needs both the mass and the fit path to
exist) and is extended by GN5. Listed separately so that a reviewer can
check it against §3: the page's first sections are §3's text, edited only
for links; then a worked example, options and the size limit.
`zensical build --clean` must pass; no `design/` links.

## 5. Where Gauss–Newton must not be trusted

These go into the docs page (§3, "When not to use it") and the
`gauss_newton_mass` docstring, and the pipeline checks in GN5 enforce
the last line of defence. They are the measured failure modes:

| case | what goes wrong | what to do |
| --- | --- | --- |
| multimodal and alias posteriors | the matrix describes one mode; NUTS with it cannot cross to another | sample each mode separately with its own matrix; weight modes by evidence (the orbit solver's S6) |
| near-circular orbits (e ≲ 0.05) | ω and the periastron time decouple from the data; the posterior curves away from any fixed metric: 35–59 divergences in e6, unchanged with `AngleVector` on ω | adapted mass, or regular coordinates (√e cos ω, √e sin ω; orbit solver §7.1) |
| faint, non-identifiable companions | the posterior fills the prior; JᵀJ is near-singular and the D term dominates | adapted mass; report an upper limit instead |
| closure-phase walls (star pipeline) | a companion crossing a wrap sees a curvature change no local matrix knows | the GN5 warm-up check falls back |
| χ²/N ≫ 1 (misspecified model) | the dropped Σ r ∇²r term is no longer small (e4d) | fix the model or fit the noise scales first; the GN matrix is then still a safe preconditioner for the optimizer |
| images ≥ 128² | n² memory | diagonal mass (GN2 size guard) |

## 6. The split with the orbit solver

The orbit-solver session (virgil-validation, `design/orbit_solver_v2.md`
§6.4 and §11, branch `claude/orbit-solver-v2-design-5dc46b`) proposed a
split, which this plan accepts:

- **This plan (the GN layer) owns:** the kernel (`_gauss_newton.init`,
  `step`, `solve`, `curvature`; GN1), the IRLS weights
  (`_Surface.weights`; GN1), block-weighted curvature rows and the
  private `_Objective(marginal=f)` with per-block f defaulting to 1 (GN3),
  V19 `gauss_newton_mass` with noise and gains, the PSD check and size
  guard (GN2), the guarded pipeline mass (GN5), and the
  `aliases._optimize` and `epochs._refine_marginal` rewrites (GN4).
- **The orbit solver owns:** `fit_multistart` (a `vmap` of
  `_gauss_newton.solve` or a `scan` of `step` over starts, on an
  `_Objective`), the Thiele–Innes coordinates, catalogues, screen, mode
  merging, evidences, and per-mode NUTS.

Interfaces the solver can rely on:
- `_gauss_newton.solve(problem, z0, lower=, upper=, gtol=, dtol=,
  max_steps=)` and `step`, pure and vmappable, with `State.stop` codes;
- a kernel problem from an `_Objective`: `loss`, `curvature_rows`,
  `curvature_diag` on a flat vector (GN3 adds the small adapter
  `_Objective.flat_problem(scale)` that `fit` itself uses, returning the
  kernel problem and the `unravel` function);
- `_gauss_newton_covariance(problem, z)` on a marginal `_Objective` gives
  the block-weighted metric for S6 (their objection: unit weights are
  5–900× too stiff).

Two points where this plan differs from the solver's §6.4 text, sent to
that session: (1) its claim that optimistix cannot carry IRLS is right for
weights refreshed at every step, which is why the kernel is virgil's own;
(2) the prior and transform curvature is added always, through D (GN2),
not only above a condition-number threshold. That removes a discontinuous
rule and agrees with the old matrix wherever the transform is the
identity.

## 7. Open decisions for Ben

- **D1. Default `fit` solver for noise fits (GN3).** The plan switches
  the default to the kernel for objectives that are not least squares
  with ≤ 200 coordinates. The alternative is opt-in (`method="lm"`) until
  the Apep OzSTAR comparison is back. The plan's choice: switch, with the
  regression tests and the L-BFGS fallback, and land the CHANGELOG note.
- **D2. Gain-width curvature.** The per-block ½ log det row is exact for
  one mode per block and overstates the curvature by up to the number of
  modes in a block (typically 1–3). Good enough for a preconditioner;
  say if gain widths are a target of precise NUTS work.
- **D3. One LM or two.** optimistix stays for pure least squares (no
  path changes). Moving its dense (≤ 200) case onto the kernel would leave
  one dense LM in virgil but change the path of every existing small LM
  fit. The plan does not do it; revisit after GN3 has been used for a
  release.
- **D4. MEM/TV default (GN6).** Flip `lm_default` only if the Stage 7
  L-curve success test passes on OzSTAR.
- **D5. Pipeline default (GN5).** Kept at adapted. Flip to
  `"gauss_newton"` after the OzSTAR comparison if fallbacks are rare.

## 8. Review and responses

(Filled in after the adversarial review.)
