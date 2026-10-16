# Gauss–Newton and Levenberg–Marquardt across virgil: implementation plan

Status: **plan, adversarially reviewed, decisions taken, not built**
(2026-10-14). Ben
accepted the six recommendations of the Gauss–Newton audit (coordinating
session, 2026-10-14; experiments `e1`–`e9`). This note turns them into
PR-sized pieces that Sonnet sessions can build without further design
decisions. It supersedes V19 of `visibility_orbits.md` and the
"least-squares form of the entropy term" option of `imaging_plan.md`
Stage 7.

Contents: §1 the problem; §2 the design in one page; §3 the explanation
for the docs (the draft Ben asked for); §4 the pieces; §5 where
Gauss–Newton must not be trusted; §6 the split with the orbit solver;
§7 Ben's decisions; §8 review and responses.

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
  Hessian, run only after L-BFGS-B with restarts (`aliases.py:494-580`).

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
| e9 (alias objective) | damped Newton **with the exact Hessian** from the start: same optima in 82 vs 230 value-and-gradient calls (near), 37 vs 341 (mid); Hessian calls not counted |
| e8 (fitted error scales) | Fisher block `blockdiag(JᵀJ, 2N_b)` matches the exact Hessian at the MAP; cross terms small |
| e4b, e4d (gains, misspecification) | JᵀJ of the gain-whitened residuals is a good curvature with modest gains; degrades as χ²/N grows |
| e6 (e ≈ 0.03 orbit) | GN fixed mass: 35–59 divergences; `AngleVector` on ω does not help (not a wrap problem) |
| e7, e7b (images) | dense covariance is n²: 256² needs ~34 GB; in float32 the Woodbury result went non-PD while `ok` stayed True |

## 2. The design in one page

**One idea.** Every objective virgil optimizes or samples is a negative
log posterior Φ(z) whose curvature can be approximated as

    F(z) = J(z)ᵀ J(z) + P(z),

where J is the Jacobian of a vector of *curvature rows* ρ(z) and P a
non-negative diagonal from elementwise priors and coordinate transforms.
F is positive semi-definite by construction. The gradient is always the
exact ∇Φ from autodiff; only the curvature is approximated. So a damped
Newton step with F converges to the exact optimum of Φ whatever F's
error (errors cost speed, not correctness), and F⁻¹ at the optimum is the
NUTS mass matrix.

The curvature rows are:

1. **Mean rows**: the whitened residuals of `whitened_residuals`, with the
   *diagonal* errors (those from `inflated_errors`) frozen by
   `jax.lax.stop_gradient`. Everything else stays live, in particular the
   whitening by marginalized gains and closure offsets (the audit's "gains
   Jacobian through `whitened_residuals`", measured in e4b) and the extra
   observable blocks. With no fitted noise these are today's residuals.
2. **Normalizer rows**: √2 · log σ_k for each visibility and phase
   observable's diagonal error (for correlated closure phases, the
   whitened rows' effective errors; the penalty rows' constant errors
   drop out). For Gaussian data with diagonal errors, mean rows plus
   these rows give exactly the Fisher information; for one scale s per
   block, blockdiag(JᵀJ, 2N_b) in ln s (V19).
3. **Block weights** for scale-marginalized objectives
   (Φ = Σ_b (fν_b/2) ln χ²_b and its bounded-integral variant): each
   block's rows are multiplied by √w_b with w_b = 2 ∂Φ/∂χ²_b, frozen. For
   the plain marginal, w_b = fν_b/χ²_b = f/ŝ_b², with ŝ_b the profile
   scale: iteratively reweighted least squares (IRLS).
4. **Term rows**: the residuals of `AngleVector` priors and of likelihood
   terms (`PositionData.term`, `RVData.term`), live, as now.

Gain widths, closure-offset widths and the scale-like terms of extra
observable blocks therefore get curvature only through live rows (an
"observed" GN curvature, about half the Fisher information for a scale
and not exact for widths). That is a PSD heuristic, documented as such;
§8 objection 1 explains why the exact form is not worth its plumbing.

P holds, per elementwise coordinate, max(0, ∂²/∂z² of that coordinate's
own prior term), with the log-Jacobian of the bijection included when the
matrix is for NUTS (whose potential includes it) and excluded for `fit`
(which, by its documented convention, evaluates priors without it).

**One dense damped-Newton kernel** (`src/virgil/_gauss_newton.py`,
private, imports nothing from virgil): for up to 200 coordinates, with
optional box bounds, acceptance on Φ itself, and pure `init`/`step`
functions so that callers can `vmap` and `scan` it. It serves `fit`
(objectives that are not least squares), `aliases`, the static
per-epoch search in `epochs`, and the orbit solver's `fit_multistart`.

**What stays.** optimistix LM stays the solver for objectives that are
already sums of squares (all of today's default `method="lm"` fits,
including large images with CG): their paths do not change. L-BFGS stays
for regularizers without a residual form, images, and problems too large
for a dense kernel. Folding optimistix's dense case into the kernel is a
later decision (§7, D3: after a release).

**What goes.** The restart loop of `aliases._optimize`
(`_MAX_RESTARTS`, `_RESTART_GAIN`), `aliases._newton_polish`, the scipy
L-BFGS-B call in `epochs._refine_marginal` and its now-dead
`_negative_marginal_and_grad` (L-BFGS-B survives only as `aliases`'
fallback), the noise/gains/`log_norm` refusals in `gauss_newton_mass`,
and `_prior_residuals`' role in curvature (replaced by P, which agrees
with it on every prior it accepted).

**Public API changes** (all keyword-only, consistent with
`design/api_argument_order.md`):

- `gauss_newton_mass(model, priors, data, values, *, likelihoods=(),
  noise=None, flat_coordinates=False)`: new `noise=`, as `fit` and
  `numpyro_model` take it. It also accepts data with gains and any
  elementwise prior, and returns a diagonal mass above `_DENSE_MASS_MAX` coordinates (4096 since 0.4.0, 8192 before).
  `SingularCurvature` now means a direction constrained by neither the
  data nor a proper prior (an `ImproperUniform` or flat-in-coordinate
  parameter the data do not fix); a parameter under a bounded prior gets
  the transform's curvature instead of raising.
- `fit`: no new arguments; `method=None` now picks the kernel ("lm") for
  objectives that are not least squares with at most 50 coordinates
  (`method="lm"` explicitly: up to 200). `info["method"]` stays `"lm"`;
  a new `info["solver"]` says `"optimistix"`, `"gauss-newton"` or
  `"gauss-newton+lbfgs"`.
- Pipelines: a `mass` setting, `"adapted"` (default, today's behaviour)
  or `"gauss_newton"`.
- `MaxEntropy` gains `residuals` (used with `method="lm"`; its default
  stays L-BFGS until the Stage 7 success test passes).

## 3. The explanation (draft for the docs)

This is the main text of the new Background page `docs/gauss_newton.md`
("Gauss–Newton fits and sampling"), written for an astronomer who knows
χ² but not optimization. It must stand alone: no links to `design/`.

> ### Fitting with Gauss–Newton and Levenberg–Marquardt
>
> **Residuals and their Jacobian.** virgil compares a model with data
> through one vector of whitened residuals, roughly
> r = (model − data)/σ for each observable. (Correlated closure phases
> are first combined into independent ones.) The χ² is the sum of their
> squares, and a fit minimizes ½χ² minus the log prior. The Jacobian J
> is the table of derivatives ∂r_k/∂θ_j: row k says how residual k moves
> when each parameter moves.
>
> **Why JᵀJ is the curvature.** Near the best fit the model is close to
> linear in the parameters, r(θ + δ) ≈ r(θ) + Jδ, so ½χ² is a quadratic
> bowl whose curvature (its Hessian, the matrix of second derivatives) is
> JᵀJ. That is the Gauss–Newton approximation. JᵀJ is never negative in
> any direction: for any change δ, δᵀJᵀJδ = |Jδ|², a sum of squares. It
> is strictly positive when every direction changes at least one residual,
> that is, when the data or a prior constrain every parameter. So the
> Gauss–Newton step, δ = −(JᵀJ)⁻¹Jᵀr, always points downhill, which the
> exact Hessian does not guarantee away from the minimum.
>
> **What Gauss–Newton drops.** The exact Hessian of ½χ² is
> JᵀJ + Σ_k r_k ∇²r_k. The second term is each residual times the
> curvature of the model. It is small when the residuals are noise-like
> (|r_k| ≈ 1 with random signs, so they average out) or when the model is
> nearly linear over the error bars. It matters when the fit is poor
> (χ²/N ≫ 1, a model that is missing something) or when the model bends
> sharply on the scale of the errors (an orbit near e = 0, a phase near
> ±180°). There JᵀJ is still a safe direction to step in, but a poorer
> guide to the error bars.
>
> **Levenberg–Marquardt.** When the model is far from linear, a full
> Gauss–Newton step can overshoot. Levenberg–Marquardt solves
> (JᵀJ + λM)δ = −g instead, where g is the gradient and M the diagonal of
> JᵀJ. With λ → 0 this is the Gauss–Newton step. With λ large it is a
> short step straight down the gradient, each parameter scaled by its own
> curvature. After each step the loss is checked: if it fell, the step is
> kept and λ is reduced (trust the quadratic more); if not, the step is
> rejected and λ is increased. So the method takes fast Newton-like steps
> near the minimum and cautious gradient steps far from it.
>
> **Noise scales.** When the error bars themselves are fitted
> (`noise={"vis_scale": ...}`), the likelihood has a second term,
> Σ log σ_k, that is not a sum of squares. Its curvature is still known:
> for Gaussian data, a parameter that changes log σ_k adds
> 2 (∂ log σ_k/∂θ)² on average. A scale s multiplying one block's N
> errors therefore adds 2N in ln s, whatever the model, and the matrix is
> blockdiag(JᵀJ, 2N). virgil builds it with extra rows √2 log σ_k next to
> the residuals. This is the Fisher information, the expected curvature.
> (For phases with errors above about 30°, whose likelihood is a von Mises
> rather than a Gaussian, it overstates the curvature in s by up to a few
> times. That only makes the matrix a little less efficient.)
>
> When the scales are instead integrated out (the orbit and alias
> searches, where each block's likelihood becomes −(ν/2) ln χ²), each
> block's residuals are weighted by 1/ŝ², with ŝ² = χ²/ν the scale the
> block's own residuals imply. This is the classical iteratively
> reweighted least squares.
>
> **Gains.** Gains correlated across channels are integrated out
> analytically: their covariance whitens the residuals, and virgil
> differentiates through that whitening. That gives a good curvature for
> the model parameters, measured against the exact Hessian. For fitted
> gain widths it is only approximate, which again affects efficiency,
> not the answer.
>
> **Why the inverse of this matrix makes a good NUTS mass matrix.** Near
> the best fit the posterior is approximately Gaussian, with covariance
> F⁻¹ (in the coordinates the sampler uses, priors included). NUTS moves
> fastest when the posterior it sees is round. Given that covariance as
> its inverse mass matrix, it sees a unit sphere, so one step size suits
> every direction. NUTS can also learn a mass matrix during warm-up, but
> from a few hundred correlated draws the estimate is noisy, and poor
> for long, thin posteriors like an orbit's. For an eight-parameter
> orbit, the Gauss–Newton matrix cut the cost from 87 to about 6
> leapfrog steps per draw. Because the matrix is computed at the fit
> rather than learned, it is fixed: NUTS does not adapt it further.
>
> **Relation to zodiax.** zodiax's optimisation tools compute the Hessian
> of the loss with `jax.hessian` and call it the Fisher matrix. They
> precondition gradient steps by its inverse (a natural-gradient step).
> They sample in its eigenbasis, scaled by the square roots of its
> eigenvalues ("eigen-projection"). virgil does the same with JᵀJ in
> place of the Hessian:
> - JᵀJ is the expected Fisher information of Gaussian data, so the
>   Levenberg–Marquardt step with λ → 0 is the natural-gradient step.
> - A dense NUTS mass matrix is equivalent to sampling in the scaled
>   eigenbasis.
> - At a good fit, JᵀJ and the Hessian agree. JᵀJ needs only first
>   derivatives and is never negative in any direction. The Hessian can
>   be, on a ridge or between modes.
>
> **When not to use it.** The matrix describes one peak of the
> posterior.
> - Do not use a fixed Gauss–Newton mass when the posterior has several
>   modes: orbit aliases, or the 180° flip of a symmetric binary. Sample
>   each mode separately and combine them by their evidences.
> - Do not use it for nearly circular orbits (e ≲ 0.05). There ω and the
>   time of periastron become degenerate, and the posterior curves away
>   from any fixed metric; NUTS reports divergences.
> - Do not use it for companions too faint to detect, whose posterior
>   fills the prior.
> - Do not use it where a companion's closure phases can pass through
>   ±180°. The likelihood then has sharp walls between regions, which no
>   matrix taken at one point describes.
>
> virgil's pipelines check the matrix before sampling and the
> divergences after it. If either check fails, they sample again with
> an adapted mass.

The page also carries (written by the implementers, §4): a code example
(fit → `gauss_newton_mass` → NUTS), the options (`noise=`, gains,
`flat_coordinates`), the size limit and diagonal fallback, and the
pipeline setting.

**Docstrings that carry a short form of this text** (three to six
sentences each, linking to the page):

| object | adds |
| --- | --- |
| `fit` (the `method` entry) | what LM does, the two solvers, the kernel's acceptance test, the 50/200 coordinate limits, when L-BFGS is used |
| `gauss_newton_mass` | JᵀJ and why it is PSD; the noise rows; gains; P; what is dropped; the size guard; `SingularCurvature`'s meaning; "when not to use" one-liners |
| `_gauss_newton.py` module docstring | the kernel's step, damping, bounds rule and stop codes, for maintainers |
| `_Objective.curvature_rows` | the four kinds of row (§2) |
| `epochs.marginal_loglike` | the IRLS weight f/ŝ_b² and its use in the search |
| `aliases.fit_orbit_aliases` (Notes) | the optimizer: damped Newton/IRLS from every start, L-BFGS-B as fallback |
| `MaxEntropy` | the residual form, that LM is opt-in, and that its GN curvature is the entropy metric |
| `docs/pipeline.md` | the `mass` setting, its checks and fallback |

## 4. The pieces

Six PRs. GN1 comes first; GN2 and GN4 depend only on GN1; GN3 on GN1
and GN2; GN5 on GN2; GN6 is independent. The docs page lands in GN3 and
is extended by GN5. Each piece is one Sonnet `pr-worker` session.

```
GN1 kernel ─┬─ GN2 curvature rows, V19 mass ─┬─ GN3 fit path (+ docs page)
            │                                └─ GN5 pipelines
            └─ GN4 aliases + epochs search
GN6 MaxEntropy residual form (independent)
```

Rules for every piece:

- float64 by default, as `fit` and `gauss_newton_mass` already run; new
  tests also run in float32 where the entry point supports it.
- No new module-level imports outside the order in AGENTS.md;
  `_gauss_newton.py` goes into AGENTS.md's package table and import map
  in GN1.
- **Compute.** Local runs are unit-test sized (seconds, one process).
  Each piece has a *local smoke benchmark* of that size, run before any
  OzSTAR job, to catch a broken implementation cheaply. The benchmarks
  that decide anything are OzSTAR jobs, written by the worker in
  `~/code/ozstar_scripts` (branch and PR there) and run by Ben; agents
  never touch the cluster.
- **Regression tests** assert that the code path is unchanged (e.g.
  `info["solver"] == "optimistix"`) and compare results at the fit
  tolerance, not to stored numbers at 1e-12, which would break across
  JAX versions.
- **Cost** is compared as wall time after compilation, and as
  gradient-equivalents: a kernel step costs one gradient plus n
  forward-mode products (n coordinates), counted as 1 + n/3.

### GN1. The damped-Newton kernel

**Files.** New `src/virgil/_gauss_newton.py`; `tests/test_gauss_newton.py`.

**Behaviour.** A *kernel problem* is an `eqx.Module` with three methods
of a flat 1-D float vector z: `loss(z)` (scalar Φ), `rows(z)` (curvature
rows, any length, possibly 0) and `diag(z)` (P, length n, ≥ 0). Data and
other arrays are module fields, so they are traced; layouts and
configuration that must be static are `eqx.field(static=True)`. (A
`functools.partial` or plain class would be static to `eqx.filter_jit`
and either fail to hash or recompile per dataset.)

```python
RUNNING, CONVERGED, STALLED, LIMIT, NON_FINITE = 0, 1, 2, 3, 4

class State(eqx.Module):
    z; value; grad; curvature; lam; steps; rejects; stop   # stop: int32

def curvature(problem, z)                 # JᵀJ + diag(P), J by jax.jacfwd
def init(problem, z0, lower, upper)        # -> State (stop NON_FINITE if Φ, g or F not finite)
def step(problem, state, lower, upper, gtol, dtol, max_steps) -> State
def solve(problem, start, *, lower=None, upper=None, gtol=0.0, dtol=0.0,
          max_steps=200) -> State          # start: z0 or a State
```

- `curvature`: J = `jax.jacfwd(problem.rows)(z)`, F = JᵀJ + diag(P),
  symmetrized.
- **Active set** (ε-active, Bertsekas 1982): with
  ε = max(1e-10, 10·eps)·(1 + |bound|), eps the dtype's machine epsilon
  (so that ε exceeds rounding in float32; review of PR #349),
  coordinate i is *held* if z_i ≤ lower_i + ε and g_i > 0, or
  z_i ≥ upper_i − ε and g_i < 0. Held coordinates get δ_i = 0: their
  rows and columns of the system are replaced by the identity and their
  gradient by 0. The projected gradient is g with held entries zeroed.
- **Step.** M = diag(max(F_ii, 1e-12·max_j F_jj, tiny)) on the free
  coordinates. Solve (F + λM)δ = −g by Cholesky. Trial z′ = clip(z + δ,
  lower, upper). The trial is accepted if δ, Φ(z′), ∇Φ(z′) and F(z′)
  are all finite and Φ(z′) < Φ(z) (a Cholesky failure gives NaNs in JAX,
  not an exception, so it counts as a rejection). On acceptance
  λ ← max(λ/3, 1e-12) and the state moves to z′. On rejection λ ← 4λ.
  λ starts at 1e-3. `steps` counts accepted and rejected iterations.
- **Stop codes**, tested after each step:
  - CONVERGED if the projected gradient's infinity norm is ≤ `gtol`, or
    the Newton decrement ½gᵀF⁻¹g on the free coordinates is ≤ `dtol`;
  - when λ > 1e10: CONVERGED if the decrement is
    ≤ 10·eps·max(|Φ|, 1), so that the rounding floor does not read as a
    failure (e.g. in float32); STALLED otherwise;
  - LIMIT at `max_steps`.
- `step` returns the state unchanged once `stop != RUNNING`, so a `scan`
  of k steps equals `solve(max_steps=k)`, and `vmap` over starts is
  well defined.
- `solve` is a `jax.lax.while_loop` over `step`, under `eqx.filter_jit`
  with the problem as an argument (repeated problems of one structure
  reuse one compilation, as `fit`'s solvers do). Given a `State` it
  resumes: `stop` is reset to RUNNING and `max_steps` counts from
  `state.steps`. That is how `fit` chunks a run under a time limit with
  λ carried over.
- `lower`/`upper` default to ±inf.

**Rationale** (moved from the `_gauss_newton` module docstring).

- F is positive semi-definite by construction. Because a step is accepted
  only if Φ itself falls, an error in F costs speed, or a `STALLED` stop
  that a caller answers with a fallback optimizer; it never gives a wrong
  answer. The way to get `STALLED` is a coordinate with a non-zero
  gradient but no curvature in F (a term of Φ missing from `rows` and
  `diag`): its step is ~1/(λ·1e-12) long, so every trial is rejected.
- Small λ is the Gauss–Newton step; large λ is a short step down the
  gradient with each coordinate scaled by its own curvature. M is
  Marquardt's scaling, floored so that a coordinate the rows ignore
  (F_ii = 0) still gives a positive-definite system and no NaN.
- Holding only coordinates exactly on a bound makes a coordinate that
  sits just inside one clip and be rejected over and over, hence the
  ε-active set. A coordinate on a bound whose gradient points inward is
  free, so it leaves.
- The default tolerances (0) rely on the rounding-floor rule; pass `gtol`
  or `dtol` to stop earlier.

**Deleted.** Nothing yet.

**Tests** (`tests/test_gauss_newton.py`):
- a linear least-squares problem reaches `onp.linalg.lstsq`'s answer to
  1e-10 (float64) in ≤ 5 steps;
- the Rosenbrock function as residuals (10(y − x²), 1 − x) from (−1.2, 1)
  reaches (1, 1) to 1e-8 in ≤ 50 steps;
- a non-least-squares Φ = Σ_b (ν_b/2) ln χ²_b with IRLS rows reaches the
  optimum found by `scipy.optimize.minimize` (BFGS, tight tolerances) to
  1e-8;
- bounds: a quadratic whose minimum lies outside the box stops on the
  bound with the right coordinates held; a coordinate that starts on a
  bound with an inward gradient leaves it (the case `_newton_polish` gets
  wrong); two strongly coupled coordinates with one at a bound converge
  without clip–reject cycling (≤ 20 steps);
- `jax.vmap` over 8 starts equals a Python loop to 1e-12;
- a `scan` of 20 `step`s equals `solve(max_steps=20)`; a solve resumed
  from a `State` after 7 steps equals one uninterrupted solve;
- Φ never increases over a recorded run;
- a non-finite start gives NON_FINITE; a zero direction (rows ignore one
  coordinate, P = 0) gives no NaNs (the M floor) and converges in the
  others;
- float32: the Rosenbrock case ends CONVERGED, not STALLED.

**Local smoke benchmark.** The tests themselves.

**Docs.** AGENTS.md (package table, import map); the module docstring
(step, damping, active set, stop codes).

### GN2. Curvature rows and V19: `gauss_newton_mass(noise=…)`, gains, guards

**Files.** `likelihood.py` (`_whiten_with_log_norm` gains a `frozen`
flag; new `_curvature_rows`), `fitting.py` (`_Objective.curvature_rows`,
`_Objective.curvature_diag`, `_gauss_newton_covariance`,
`gauss_newton_mass`), `svi.py`.

**Behaviour.**

1. `likelihood._curvature_rows(model, data, noise)` returns, for one
   dataset, the concatenation of:
   - mean rows: `_whitened_errors_and_log_norm(...)`'s whitened residuals
     computed with `frozen=True`, which wraps the diagonal `errors` from
     `inflated_errors` (and nothing else) in `jax.lax.stop_gradient`
     before `_whiten_with_log_norm`. Gains, offsets, extra blocks and the
     mean terms (`wavel_scale`, `wavel_offset`, `north_angle`, which act
     through the prediction) stay live;
   - normalizer rows: √2·log σ_k with σ_k the unfrozen diagonal errors of
     the visibility and phase observables. For uncorrelated wrapped
     phases σ_k is the Gaussian σ (not the von Mises effective error;
     see §3 on weak phases). For correlated closure phases, the
     whitened rows' effective errors from `cp_noise.whiten`; penalty rows
     are left out. Extra observable blocks get no normalizer rows: their
     `whiten` has its own error model (`observables.py`), and their live
     rows already carry a PSD curvature.
2. `_Objective.curvature_rows(z)`: per dataset (1), then each likelihood
   term's residuals (live; `term.log_norm` curvature left out, as now),
   then the `AngleVector` priors' residuals. The coordinates are
   `problem.sites` followed by the non-tied noise sites in `self.noise`
   order; the flat vector and its `unravel` come from
   `jax.flatten_util.ravel_pytree` over a dict in that order.
3. `_Objective.curvature_diag(z, jacobian)`: for each coordinate whose
   prior has `event_shape == ()` (elementwise, including array-valued
   parameters with an elementwise prior):
   `max(0, ∂²/∂z²[−log p(x(z)) − (log|dx/dz| if jacobian else 0)])`,
   computed as a `jax.jvp` of `jax.grad` with a ones tangent, which is the
   diagonal because the terms are separable. With `jacobian=False`,
   priors flat in the fitted coordinate give 0, as now. Other priors
   (`AngleVector`, which has rows; multivariate priors; tied noise
   terms) give 0 there. For a Normal prior on the real line this is
   1/σ², what `_prior_residuals` gives today.
4. `_gauss_newton_covariance(problem, z)`: F from (2) and (3) with
   `jacobian=True`, over parameters and noise sites, keeping both
   branches (Cholesky when rows ≥ coordinates, Woodbury otherwise). In
   the Woodbury branch the zero-P set must be known when tracing: it is
   the coordinates whose prior is `ImproperUniform` or (with
   `jacobian=False`) flat in its coordinate, decided by prior type as
   `flat_prior` is now; every other P is floored at 1e-12·max(P) so that
   W = P^-½ is finite. Then the result is checked: symmetrize,
   Cholesky-factor the covariance itself; `ok` is false if that fails or
   anything is non-finite (the PSD check e7b showed was missing). It also
   returns the condition number of the *correlation* matrix (covariance
   scaled to unit diagonal), from `onp.linalg.eigvalsh` on the dense
   path; this does not depend on the parameters' units.
5. `gauss_newton_mass(..., noise=None)`:
   - the matrix covers the noise sites too, keyed by their numpyro site
     names (`"noise.vis_scale"`, `"noise[1].phi_scale"`) in the
     coordinates `numpyro_model` samples (the bijections of
     `_Objective`, flat or not as told);
   - accepts data with gains, any elementwise prior, and likelihood terms
     with a `log_norm`;
   - **size guard**: with n > `_DENSE_MASS_MAX` = 4096 coordinates (8192 before 0.4.0) it
     returns `dense_mass=False` with the diagonal of the GN covariance
     (Woodbury branch: `w_i²·(1 − Σ_j (L⁻¹K)_{ji}²)` plus the zero-P
     correction, never forming n×n; Cholesky branch: 1/diag(F)), and a
     `UserWarning` naming the size. 128² images and above get this;
   - raises `SingularCurvature` when `ok` is false; sets
     `out.condition` (dense) or `None` (diagonal).
6. `svi._start_scale` (`svi.py:210-250`):
   - pass `noise=noise` to `gauss_newton_mass`, so error terms get GN
     widths instead of `init_scale`;
   - handle a diagonal mass (place its entries on the diagonal) instead
     of indexing it as dense;
   - drop `NoLeastSquaresForm` from the except clause (no longer raised
     by `gauss_newton_mass`);
   - note in the docstring and CHANGELOG that data with gains, and priors
     without a least-squares form, now get a GN start rather than
     `init_scale`.

**Deleted.** In `gauss_newton_mass`, the `_check_least_squares` call; in
`_gauss_newton_covariance`, `prior_curvature` (replaced by
`curvature_diag`). `_prior_residuals` stays for `_Objective.residuals`
(optimistix's path).

**Tests** (`tests/test_fitting.py`, `tests/test_sampling.py`):
- *Unchanged where it should be.* For fixtures with Normal and
  ImproperUniform priors and no noise, the new covariance equals the
  old formula (JᵀJ + diag of prior residual curvature, recomputed in the
  test from the residuals) to 1e-10.
- *Bounded priors.* With `flat_coordinates=True` and LogUniform, P is
  exactly 2σ(z)(1 − σ(z)) per coordinate. Under numpyro's interval
  bijection P equals the analytic
  max(0, (log x)″ + 2σ(1 − σ)) (test against `jax.hessian` of the 1-D
  potential).
- *`SingularCurvature` semantics.* `tests/test_sampling.py:146` (a
  Uniform(−10, 10) dra the data do not constrain) no longer raises:
  change it to assert a finite matrix whose dra variance is the
  transform's, and add a case that still raises (an `ImproperUniform`
  parameter with no data).
- *V19 block.* Binary with `noise={"vis_scale": LogUniform(0.1, 10),
  "phi_scale": LogUniform(0.1, 10)}`, unwrapped or strong phases
  (σ ≤ 0.3 rad), at a MAP from `fit(method="lbfgs")`: the noise block of
  JᵀJ (without P) equals 2N_b·(d ln s/dz)² to 1e-6; its θ–noise block is
  0; eig(F⁻¹H), with H the exact Hessian of the negative log posterior
  in the same coordinates, lies in [0.8, 1.25] (e8's check).
- *Gains.* Binary with `with_gains(telescope=0.03, baseline=0.02)`:
  PSD, eig(F⁻¹H) in [0.5, 2] over θ (e4b's check). With `vis_scale`
  *and* a `vis_gain_<group>` both fitted: PSD, and the measured eig
  range is asserted with the bound the PR reports (record it; it is
  expected to be wider, e.g. [0.3, 3]).
- *Added errors.* `vis_error_rel` fitted: F has a non-zero θ–noise
  block, PSD.
- *Site keys.* List-form noise (`noise[1].phi_scale`), tied noise (no
  row, no key), `flat_coordinates=True` with noise, and an `AngleVector`
  with noise: the mass's sites equal the numpyro model's latent sites
  (check with `chain_init_params(..., mass=)`, which raises on mismatch).
- *PSD guard.* `_gauss_newton_covariance` on e7b's 16² image cast to
  float32 with errors ×0.001 returns `ok == False` (the old code returned
  True); `gauss_newton_mass` itself runs in float64.
- *Size guard.* With `_DENSE_MASS_MAX` monkeypatched to 10, a 16-pixel
  image returns a diagonal equal to the dense result's diagonal to 1e-10.
- *Gains and offsets in float32*: `_curvature_rows` is finite.
- `test_svi.py`, `test_angles.py`, `test_flat_coordinates.py`,
  `test_sampler_flat_coordinates.py`, `test_no_data_priors.py` pass,
  with fixture edits listed in the PR where a bounded prior's matrix
  changed by P.
- NUTS smoke test: the V19 mass on the binary with noise, 30 warmup and
  30 draws, no divergences.

**Local smoke benchmark.** The V19 and gains tests (seconds).

**OzSTAR.** e3 on the orbit with `noise=` scales added (8 + 2
parameters), 4 chains, 400 warmup, 600 draws: fixed V19 mass vs adapted
dense. Report steps per draw, divergences and min ESS per gradient. Pass:
≥ 10× ESS per gradient, no divergences.

**Docs.** `gauss_newton_mass` docstring (the table in §3);
`docs/api/fitting.md` blurb ("for sampling large images" becomes "for
sampling any posterior with one peak").

### GN3. `fit` on the kernel, and the marginal objective

**Files.** `fitting.py` (`fit`, `_Objective.flat_problem`, a `_gn`
solver beside `_lm`), `epochs.py` (`_MarginalObjective`; per-block dof in
`_Surface.score` and `_Surface.weights`), `docs/gauss_newton.md` (new), `mkdocs.yml`
(Background nav, after "Conventions"), `CHANGELOG.md`.

**Behaviour.**
- `_Objective.flat_problem(scale, jacobian=False)` returns
  `(kernel_problem, z_flat, unravel)`: an `eqx.Module` wrapping the
  objective, with `loss = self.loss(unravel(z)) / scale`,
  `rows = self.curvature_rows(unravel(z)) / √scale`,
  `diag = self.curvature_diag(unravel(z), jacobian) / scale`.
- **Method rule** in `fit`, with `method=None`:
  1. the whole objective is a sum of squares (`_has_residuals`, as now)
     and every regularizer has `lm_default` true → `"lm"` on optimistix,
     exactly as today;
  2. otherwise, if every regularizer has `residuals` and `lm_default`,
     no prior is an array-valued `ImproperUniform` (an image), and the
     number of coordinates (parameters and noise sites) is
     ≤ `_GN_DEFAULT` = 50 → `"lm"` on the kernel;
  3. otherwise `"lbfgs"`.
  With `method="lm"` given explicitly, `lm_default` is ignored: rule 1
  applies when the objective is a sum of squares, else the kernel up to
  `_DENSE_LM` = 200 coordinates, else `NoLeastSquaresForm`.
  `_has_regularizer_residuals` and `lm_default` are read with
  `getattr(r, "lm_default", True)`.
- `_gn(problem, z0, scale, max_steps, gtol, budget)`: builds the kernel
  problem with `flat_problem`, the tolerance with `_lm_tolerance` (so
  `gtol` means what it means for the other methods), and runs
  `_gauss_newton.solve` in chunks of `_LM_CHUNK` steps when the budget is
  chunked, carrying the `State` between chunks. Default `max_steps` 1000.
- **Fallback.** If the kernel stops STALLED or NON_FINITE (after a finite
  start), `fit` runs L-BFGS from the kernel's last point;
  `info["solver"] = "gauss-newton+lbfgs"`, and any non-convergence
  warning is L-BFGS's.
- `info["solver"]`: `"optimistix"`, `"gauss-newton"` or
  `"gauss-newton+lbfgs"` (L-BFGS and Adam: `None`). `info["method"]`
  stays `"lm"`.
- **Marginal objective (private; for `epochs`, `aliases` and the orbit
  solver).** In `epochs.py`, next to `_Surface`:

  ```python
  class _MarginalObjective(eqx.Module):
      objective: _Objective        # supplies priors, bijections, build
      surfaces: tuple = eqx.field(static=True)   # one _Surface per dataset
      dof: tuple                   # per dataset: scalar or per-block array
  ```

  Its kernel-problem methods use the objective's coordinates; `loss` is
  `−Σ_d surface_d.score(chi2_blocks_d, dof_d, d) − log prior` (log prior
  as `_Objective.loss` computes it), where `chi2_blocks_d` comes from
  `surface_d.chi2`; `rows` are each block's mean rows times
  √(stop_gradient(`surface_d.weights(...)`)), then the term rows; `diag`
  is the objective's. `s_max` comes from each `_Surface`. Noise sites
  that scale or add to the errors raise `ValueError` (the scales are
  integrated out); mean terms (`wavel_*`, `north_angle`) are allowed.
  `fit` does not expose it, and `numpyro_model` is unchanged. Also in
  `epochs.py`:
  - `_Surface.score` accepts `dof` as a scalar or a per-block array
    (indexed `dof[k]` in the `s_max` loop);
  - `_Surface.weights(chi2, dof, data)` (2 × `jax.grad` of −score with
    respect to `chi2`, w_b ≥ 0 because the score falls as any χ²_b rises;
    for `s_max=None` it is dof_b·ν_b/χ²_b) already exists: GN4 added it
    with a scalar `dof`, because the alias kernel needs it. GN3 extends
    `score` and `weights` to a per-block `dof` array.

**Default switch (D1).** GN3 merges with the kernel reached by `method="lm"` only (rule 2 behind a module flag `_GN_DEFAULT_ON = False`). When the OzSTAR comparison below shows the kernel beating L-BFGS, a follow-up sets the flag, updates the regression tests' expected solver and adds the CHANGELOG entry.

**What changes for existing users.** Default fits with fitted `noise=`,
with gains, with priors such as HalfNormal or Beta, or with `log_norm`
terms, and ≤ 50 coordinates, now use LM on the kernel instead of L-BFGS.
Both use the exact gradient of the same loss, so they reach the same
optimum to the fit tolerance from a start in its basin. A multimodal
problem can end in a different local optimum from the same start. The
regression tests pin the existing fixtures, and the CHANGELOG says so,
with `method="lbfgs"` as the way back.

**Deleted.** The `fit` docstring sentences "so the default method is
L-BFGS. So it is for data with gains…"; `_check_least_squares`'
messages stop recommending L-BFGS for noise and gains (it still decides
optimistix vs the kernel).

**Tests.**
- *Regression* (`tests/test_fitting_gn.py`, new, ~6 curated fixtures:
  binary with `vis_scale`/`phi_scale`; binary with gains; binary with
  `vis_error_rel`; a list-form noise over two datasets; an orbit with
  `RVData` jitter (`log_norm`); a HalfNormal prior): with `_GN_DEFAULT_ON`
  monkeypatched to True, the default fit uses `solver == "gauss-newton"`
  (with the flag off, `"lbfgs"` as now); values agree with
  `method="lbfgs"` within 1e-4 relative or 1e-3 of the Laplace σ,
  whichever is larger; losses within 1e-6·N. Existing assertions on
  `info["method"] == "lbfgs"` for such fits elsewhere are updated, each
  listed in the PR.
- Pure least-squares fits keep `solver == "optimistix"` (path pinned).
- An image with fitted noise keeps L-BFGS (rule 2's image exclusion);
  60 coordinates without a least-squares form keep L-BFGS by default and
  use the kernel with `method="lm"`.
- `time_limit` chunking equals an unchunked kernel run to 1e-12.
- Fallback: a loss with a kink ends `"gauss-newton+lbfgs"`.
- `_MarginalObjective` on the `test_aliases` synthetic orbit: the kernel
  reaches the optimum of `epochs.marginal_loglike` found by scipy BFGS to
  1e-6 nats; with `s_max=3` and per-block dof `[0.5, 1.0]`,
  `weights` equals finite differences of the score.
- float32: the binary-with-noise fit converges with `dtype="float32"`.

**Local smoke benchmark.** The regression fixtures, timed after
compilation (seconds).

**OzSTAR** (Ben runs):
- e2 on the orbit with `noise=` scales, 16 near and 16 mid starts:
  kernel vs L-BFGS, success (within 1 of the best χ²), median wall time
  and gradient-equivalents. Pass: success ≥ L-BFGS's, median wall time
  ≤ ½ of L-BFGS's.
- A size sweep at n ≈ 20, 50, 100, 200 (a binary plus a `Nodes`
  spectrum with noise) to decide whether `_GN_DEFAULT` can rise (§7 D1).
- The Apep fit, default vs `method="lbfgs"` on the current Apep branch:
  values and losses. Required before the CHANGELOG calls the change safe
  for real data.

**Docs.** New `docs/gauss_newton.md` (§3 text plus the example and
options); nav entry; `fit` docstring (`method`, `noise`); CHANGELOG
entry (the behaviour change and how to get the old one).

### GN4. `aliases._optimize` and the static per-epoch search on the kernel

This is recommendation 2. It belongs here, not in the orbit solver (agreed
split, §6). The solver will replace the alias search as a pipeline
later, but `fit_orbit_aliases` stays public, and its users get the
speed-up now.

**Files.** `aliases.py` (`_Problem`, `_optimize`), `epochs.py`
(`_refine_marginal`, and `_Surface.weights` and `weighted_rows`, moved
here from GN3 with a scalar `dof`; GN3 extends them to per-block `dof`).

**Behaviour.**
- `_Surface.weights(chi2, dof, data)` with scalar `dof`, as GN3 specifies
  it, tested against dof·ν_b/χ²_b (no `s_max`) and finite differences of
  the score (`s_max=3`); `_Surface.weighted_rows(r, dof, data)` multiplies
  each block of the whitened residuals by √(stop_gradient w_b).
- New `aliases._KernelProblem(eqx.Module)`: fields `data` (traced, the
  epochs' OIData tuple); static `surfaces`, `cfg`. `loss(x)` =
  `−_loglike(surfaces, x, data, cfg)`. `rows(x)`: per epoch, the block
  residuals of `_epoch_chi2`, with the profiled V² amplitude live
  (variable projection, clipped as now), each block multiplied by
  √(stop_gradient(`surface.weights(chi2, 1.0, d)`)). `diag` is zero (flat
  priors in these coordinates). `_Problem` builds it once and keeps its
  other methods.
- `_optimize(problem, bounds, x0, max_steps)`:
  1. `_gauss_newton.solve` from `x0` with the bounds (±inf where none),
     `dtol = 1e-9` nats, `max_steps = 200`;
  2. if the stop code is not CONVERGED: one scipy L-BFGS-B run from the
     kernel's point with today's options, then the kernel again from
     L-BFGS-B's point;
  3. `_fold` the result, as now. Return `(x, value)`.
  The multistart (`_candidates`, `_init_starts`, `_refine`) is unchanged.
- `epochs._refine_marginal(x0, dof, d64, surface, bounds)`: the kernel on
  (dra, ddec, flux) with the same bounds, `dtol = 1e-12`, through a small
  `eqx.Module` built like `_KernelProblem` (`rows` from the surface's
  blocks weighted by `weights(chi2, dof, d)`); no scipy.

**Deleted.**
- `_MAX_RESTARTS`, `_RESTART_GAIN`, the restart loop and its comment
  block (`aliases.py:494-531`);
- `_newton_polish` (`:534-580`) and its direct tests
  (`tests/test_aliases.py:665-694`), replaced by the bound-leaving and
  coupled-bound tests of GN1 and one `_optimize` test here;
- `epochs._negative_marginal_and_grad` (dead after this), with
  `tests/test_orbit_positions.py:182`'s cache check rewritten to check
  that repeated `_refine_marginal` calls of one shape reuse the kernel's
  compilation.
- `_hessian` stays: the Laplace evidence correctly uses the exact
  Hessian.

**Tests.**
- `test_aliases.py` and `test_epochs.py` otherwise pass unchanged (band
  probabilities, recovered orbits, peaks).
- On `test_aliases._synthetic(12.1)` from 6 near starts (fixed seed):
  every start reaches an m at least as high as the old `_optimize`'s
  (stored in the test) minus 0.01 nats.
- `epoch_positions` peaks agree with `main`'s to the refinement's
  tolerance (1e-4 mas, 1e-4 in flux).
- Laplace evidences per band ≥ `main`'s − 0.05 nats (a better-converged
  mode may raise them).

**Local smoke benchmark.** The new `_optimize` test, timed after
compilation.

**OzSTAR.** e9 rerun with the kernel and IRLS rows (the audit used the
exact Hessian, which is untested here): 12 near and 12 mid starts, old vs
new, success and wall time after compilation. Pass: wall time ≤ ½ of
the old median, success ≥ old. If IRLS-GN needs more than 1.5× the
exact-Hessian kernel's steps, report back before merging (do not add a
Hessian option without a decision).

**Docs.** `fit_orbit_aliases` Notes paragraph; `docs/orbit_fitting.md`,
if it names L-BFGS-B.

### GN5. Pipelines: `mass="gauss_newton"`, guarded

This is recommendation 3. The default stays what each pipeline does now
(binary: adapted diagonal; star: adapted dense), on a single
`mcmc.run` as now. Nothing changes unless the setting is chosen.

**Files.** `pipeline/_common.py` (`_mcmc`, new `_gn_mass`),
`pipeline/binary.py` (`_run_nuts`, settings), `pipeline/star.py`
(`_nuts`, settings), `pipeline/_core.py` (default for old `run.json`
files), `docs/pipeline.md`, `docs/gauss_newton.md`.

**Behaviour.**
- New setting `mass`: `"adapted"` (default) or `"gauss_newton"`. A
  `run.json` without the key reads as `"adapted"`, so resuming old runs
  does not raise `ConfigMismatchError`.
- With `"gauss_newton"`:
  1. *Before sampling*: `gauss_newton_mass(model, priors, data, start,
     noise=p._noise(), flat_coordinates=...)` at the fit, with `start`
     including the fitted `noise.*` values. (`binary._run_nuts` currently
     filters `start` to `k in priors`, `binary.py:803-807`; keep the
     noise sites too.) If it raises, or `out.condition` (of the
     correlation matrix) exceeds 1e10, use `"adapted"` with reason
     `"conditioning"`. A diagonal mass from the size guard is used as is.
  2. *Sampling*: one `mcmc.run` with the GN mass (`adapt_mass_matrix=False`;
     the step size still adapts in warm-up), `init_params` from
     `chain_init_params(posterior, starts, key=..., mass=mass)`.
  3. *Divergence check*: if more than 1 % of the post-warm-up draws
     diverge, run again from the same starts with `"adapted"`, reason
     `"divergences"`, and report the second run.
  4. A GN mass is never re-adapted.
- `summary.json` gets `"mass": {"requested", "used", "fallback_reason",
  "condition"}`; the quicklook notebook prints it; a fallback adds a
  `Check` at warning level.
- `_mcmc(posterior, settings, *, dense_mass, mass_kwargs=None)`: when
  given, `mass_kwargs` replace `dense_mass`.

**Deleted.** Nothing.

**Tests** (`test_pipeline_binary.py`, `test_pipeline_star.py`):
- default settings: one `mcmc.run`, `"mass"` reports `"adapted"`, and the
  existing assertions pass;
- `mass="gauss_newton"` on the binary fixture: `used ==
  "gauss_newton"`, the noise sites are in the mass, mean `num_steps`
  below the adapted run's;
- forced fallbacks: `gauss_newton_mass` monkeypatched to raise
  (`"conditioning"`); the divergence fraction monkeypatched above 1 %
  (`"divergences"`); both add the check;
- an old `run.json` without `mass` resumes.

**Local smoke benchmark.** The binary pipeline test with the smallest
sampler settings, both masses.

**OzSTAR.** One real binary dataset and one star dataset through both
masses: ESS per second and fallback rate (for §7 D5).

**Docs.** `docs/pipeline.md` (the setting, the two checks, the summary
fields); the "When not to use it" section of `docs/gauss_newton.md`
names the star pipeline's closure-phase case.

### GN6. A residual form of maximum entropy

This is recommendation 6 (the Stage 7 "LM form of the entropy term").
Total variation is left out: see §8 objection 3.

**Files.** `imaging.py` (`MaxEntropy.residuals`, `lm_default`),
`fitting.py` (the method rule reads `lm_default`; done in GN3 if GN3
lands first, else here).

**Behaviour.**
- `MaxEntropy.residuals(model)`: with b and q the unit-sum image and
  default (as `value` computes them), per pixel
  φ_i = b_i ln(b_i/q_i) − b_i + q_i ≥ 0, the generalized relative
  entropy (Σφ equals today's value because Σb = Σq = 1), and
  r_i = sign(b_i − q_i)·√(2·weight·φ_i). Computed as
  (b_i − q_i)·√(2·weight·h(b_i/q_i)/q_i) with
  h(t) = (t ln t − t + 1)/(t − 1)², h(1) = ½: a Taylor series
  h ≈ ½ − (t−1)/6 + (t−1)²/12 for |t − 1| < 1e-3, and the double-`where`
  trick so that gradients are finite at b = q. Outside the support
  (q = 0) behave as `value` does now. Near b = q its GN curvature is
  weight/q, the entropy metric of Skilling & Bryan (1984). LM on this form
  is the preconditioning Stage 7 asked for.
- `MaxEntropy.lm_default = False`, so default fits stay on L-BFGS;
  `method="lm"` uses the residuals (optimistix, with CG for large images).

**Deleted.** Nothing now. If the success test passes, a follow-up flips
`lm_default` (§7 D4).

**Tests** (`test_imaging.py`):
- `value == 0.5 * sum(residuals**2)` to 1e-12 relative (float64) for
  random images, at b = q exactly, and with a support mask;
- the gradient of ½Σr² equals `jax.grad(value)` to 1e-10, at b = q too
  (no NaN);
- a default `fit` with MEM still uses L-BFGS;
- `method="lm"` on a 16² MEM fit reaches the L-BFGS loss to 1e-6
  relative.

**Local smoke benchmark.** The 16² test, timed.

**OzSTAR.** The Stage 7 success test: an L-curve from w = 0.1 to 1e4 on
the `mwe_gaussian_field` setup, float32 and float64, `method="lm"` vs
L-BFGS.

**Docs.** `MaxEntropy` docstring; one paragraph in
`docs/imaging_rml.md` on `method="lm"`.

## 5. Where Gauss–Newton must not be trusted

These go into the docs page (§3, "When not to use it") and the
`gauss_newton_mass` docstring; GN5's checks are the last line of
defence. They are the measured failure modes:

| case | what goes wrong | what to do |
| --- | --- | --- |
| multimodal and alias posteriors | the matrix describes one mode; NUTS with it cannot cross to another | sample each mode with its own matrix; weight modes by evidence (the orbit solver's S6) |
| near-circular orbits (e ≲ 0.05) | ω and the periastron time decouple from the data; the posterior curves away from any fixed metric: 35–59 divergences in e6, unchanged with `AngleVector` on ω | adapted mass, or regular coordinates (√e cos ω, √e sin ω; orbit solver §7.1) |
| faint, non-identifiable companions | the posterior fills the prior; JᵀJ is near-singular and P dominates | adapted mass; report an upper limit |
| closure phases near ±180° (star pipeline) | a companion's phases crossing the wrap change the likelihood abruptly, which no local matrix describes | GN5's divergence check falls back |
| χ²/N ≫ 1 (misspecified model) | the dropped Σ r ∇²r term is no longer small (e4d) | fix the model or fit the noise scales first; the matrix remains a safe preconditioner for the optimizer |
| weak phases (σ ≳ 30°) with a fitted `phi_scale` | the Gaussian normalizer rows overstate the curvature in s (×1.7 at σ = 1 rad) | none needed; efficiency only |
| images ≥ 128² | n² memory | diagonal mass (GN2 size guard) |

## 6. The split with the orbit solver

The orbit-solver session (virgil-validation `design/orbit_solver_v2.md`
§6.4 and §11, branch `claude/orbit-solver-v2-design-5dc46b`) proposed a
split. This plan accepts it, and both sessions confirmed it on
2026-10-14:

- **This plan (the GN layer) owns:**
  - the kernel (`_gauss_newton.init`, `step`, `solve`, `curvature`;
    GN1);
  - the IRLS weights (`_Surface.weights`) and `_MarginalObjective`,
    with a per-block f defaulting to 1 (GN3);
  - V19 `gauss_newton_mass` with noise and gains, the PSD check and the
    size guard (GN2);
  - the guarded pipeline mass (GN5);
  - the `aliases._optimize` and `epochs._refine_marginal` rewrites (GN4).
- **The orbit solver owns:**
  - `fit_multistart`: a `vmap` of `_gauss_newton.solve`, or a `scan` of
    `step`, over starts, on a kernel problem;
  - the Thiele–Innes coordinates, catalogues, screen, mode merging,
    evidences, and per-mode NUTS.

Interfaces the solver can rely on:

- `_gauss_newton.solve(problem, start, lower=, upper=, gtol=, dtol=,
  max_steps=)` and `step`, which are pure and vmappable, with integer
  stop codes, and identity once stopped;
- kernel problems from `_Objective.flat_problem(scale)` or from
  `_MarginalObjective`, each an `eqx.Module` with `loss`, `rows` and
  `diag` on a flat vector;
- `_gauss_newton_covariance` and `_gauss_newton.curvature` on a
  marginal problem give the block-weighted metric for S6 (the solver's
  point: unit weights are 5–900× too stiff).

The solver's conjugate gain prior (its §4.3, g ~ N(ḡ, s²τ²), owned by
its P1) plugs into these interfaces without changes here:

- The block's Q is the squared norm of the variable-projection residual
  [W^½(y − mĝ), (ĝ − ḡ)/τ] at the profiled ĝ. It enters the score as that
  block's χ².
- The model-dependent −½ ln(1 + τ² mᵀWm) goes into `loss`, so its
  gradient is exact. Its curvature comes from the live rows.

Where this plan differs from the solver's §6.4 text:

1. The solver's claim that optimistix cannot carry IRLS is right for
   weights refreshed at every step. That is why the kernel is virgil's
   own.
2. Prior and transform curvature is always added, through P, rather than
   only above a condition-number threshold. This avoids a discontinuous
   rule, and agrees with today's matrix wherever the transform is the
   identity.

## 7. Decisions (Ben, 2026-10-14)

- **D1. The `fit` default for objectives that are not least squares
  (GN3): the kernel, if it beats L-BFGS.** GN3 switches the default to
  the kernel up to 50 coordinates and keeps `method="lm"` up to 200, with
  L-BFGS as the fallback when the kernel stalls. The switch is gated on
  GN3's OzSTAR comparison (e2 with noise scales, and the Apep fit):
  the kernel must succeed at least as often as L-BFGS, be faster in
  median wall time, and reach the same values and losses on Apep. Until
  that report is back, GN3 merges with the kernel opt-in
  (`method="lm"`); a follow-up of a few lines flips the default. The
  size sweep decides whether the 50 can rise.
- **D2. Fitted gain widths: the heuristic is accepted.** Their curvature
  comes from the live rows (PSD, not the Fisher information). An exact
  per-block form is not planned.
- **D3. Two LM paths for now.** optimistix stays for pure least squares;
  the kernel takes the rest. Revisit merging them after GN3 has shipped
  in a release.
- **D4. MEM default (GN6).** Flip `lm_default` only if the Stage 7
  L-curve success test passes on OzSTAR.
- **D5. Pipeline default (GN5).** It stays adapted. Flip it to
  `"gauss_newton"` only after the OzSTAR comparison, if fallbacks are
  rare.
- **D6. Benchmarks.** Every benchmark that decides anything is an OzSTAR
  job that Ben runs. The local step before it is the seconds-long smoke
  runs of each piece.

## 8. Review and responses

A fresh Opus agent reviewed the draft (commit 43b0202) adversarially.
It read the code and the orbit-solver design, and ran small numpy checks
of the algebra. It raised 22 objections (13 major, 9 minor, no
blockers). All were accepted. Several made the plan simpler.

| # | severity | objection | response |
| --- | --- | --- | --- |
| 1 | major | The per-block ½ log det rows for gains and offsets are not the Fisher information. With a scale and a width fitted together, the θ–width cross term has the wrong sign. eig(F⁻¹H) fell to 0.37. The θ dependence of C through the gain Jacobian can be understated, down to zero. | **Simplified.** The normalizer rows are kept only for diagonal errors, where they are exact. Gain and offset whitening stays live in the mean rows. That is the audit's measured "Jacobian through `whitened_residuals`", and it needs no `whiten_blocks` or `gains.py` changes. The width curvature is called a PSD heuristic (§2, §3, D2). A joint scale + width test records the measured bound. |
| 2 | major | von Mises rows overstate the phi_scale curvature 1.7× at κ = 1 and 3× at κ = 0.3. | Normalizer rows use the Gaussian σ. The overstatement for weak phases is documented (§3, §5). The V19 test uses σ ≤ 0.3 rad. |
| 3 | major | The TV residual √t has zero GN curvature in flat regions, where TV's curvature is largest. | **TV dropped from GN6.** Its correct GN form (lagged diffusivity, t frozen) is not a sum of squares for optimistix. The dense kernel cannot take images. A matrix-free kernel only for TV is not justified now. MaxEntropy's √φ form is sound and stays. |
| 4 | major | "Passes unchanged" was false for `test_sampling.py:146` (`SingularCurvature`), `test_aliases.py:665-694` (`_newton_polish`) and `test_orbit_positions.py:182`. | Each is listed with its replacement (GN2, GN4). `SingularCurvature`'s new meaning is stated (§2). `_negative_marginal_and_grad` is deleted. |
| 5 | major | Four test specifications could not pass as written: the one-step linear solve, the LogUniform P, the noise block of F including P, and a float32 test of a float64 entry point. | Rewritten: ≤ 5 steps; the analytic formula under each bijection; JᵀJ without P; the float32 case tests `_gauss_newton_covariance` directly. |
| 6 | major | A `functools.partial` problem breaks `eqx.filter_jit` (it is static, so it fails to hash or recompiles). | Kernel problems are `eqx.Module`s with traced data and static layouts (GN1, GN4). |
| 7 | major | Scan, vmap and chunking needed unstated behaviour. | Integer stop codes; `step` is the identity once stopped; `solve` resumes from a `State` with relative limits; vmap agrees with a loop to 1e-12. |
| 8 | major | False stalls at the rounding floor (float32) would trigger a pointless fallback. A non-finite trial must count as a rejection. | A stall whose decrement is at the rounding floor is CONVERGED. Any non-finite δ, Φ, ∇Φ or F rejects the step (GN1). |
| 9 | major | Woodbury needs its zero-P set when tracing, but `max(0, ·)` can give zero P at runtime (w = ∞). | The zero-P set is decided by prior type. Every other P is floored at 1e-12·max(P) (GN2). |
| 10 | major | e9 measured the exact-Hessian Newton, not IRLS-GN. Kernel steps cost n JVPs. Call counting is impossible inside `while_loop`. | Cost is wall time after compilation plus gradient-equivalents. e9 is rerun with IRLS rows on OzSTAR as GN4's gate, with an escalation rule. |
| 11 | major | The 200-coordinate default is unevidenced (n JVPs per step), and small images would lose `max_step_size`'s protection. | The default limit is 50; `method="lm"` goes up to 200. Images (array `ImproperUniform`) stay on L-BFGS. An OzSTAR size sweep decides any rise (D1). |
| 12 | major | Pipeline: the condition threshold depended on units; fitted noise values were dropped from `start`; the warm-up divergence check misfires during step-size search. | The condition number is the correlation matrix's. Noise values are kept. Simpler fallback: one `mcmc.run`, then a rerun with adapted mass if > 1 % of post-warm-up draws diverge. `chain_init_params(mass=)`. |
| 13 | major | `svi._start_scale` was not updated: no `noise=`, a dense-only index, and a dead except clause. Gains data change silently. | All four listed in GN2 item 6, with the behaviour change in the CHANGELOG. |
| 14 | major | Extra observable blocks' rows were underspecified. One block ignores `errors`, so vis_scale rows would add curvature its likelihood lacks. | Extra blocks get live rows and no normalizer rows. No interface change. |
| 15 | minor | `lm_default` must gate both rules; explicit `method="lm"` ignores it. | Done (GN3 method rule). |
| 16 | minor | A jvp-of-grad with a ones tangent gives Hessian row sums for non-elementwise priors and tied terms. | P is restricted to `event_shape == ()`. Tied and multivariate priors give 0 there (GN2 item 3). |
| 17 | minor | `marginal=` was underspecified: where `s_max` comes from, per-block f in the s_max loop, a too-strict noise ban. A wrapper is better than a flag. | `_MarginalObjective` wrapper in `epochs.py` (no `_Surface` move or new module), `s_max` from each surface, per-block dof in `score`, mean terms allowed, `flat_problem` defined in GN3. |
| 18 | minor | Is vis_amp live or frozen in the alias rows? | Live (variable projection, clipped). |
| 19 | minor | Tests stored at 1e-12 break across JAX versions; the evidence test pinned worse old optima; the fixture grep was too broad; the local benchmarks were too heavy for the laptop. | Path assertions plus fit-tolerance comparisons; evidence ≥ old − 0.05; about 6 curated fixtures; benchmarks moved to OzSTAR with unit-sized local smoke runs (D6). |
| 20 | minor | Missing tests: list noise, tied noise, flat coordinates, AngleVector with noise, float32 gains, vmap. | Added (GN1, GN2). The vmap test on a real objective is the orbit solver's `fit_multistart` test. |
| 21 | minor | §3 wording: adapted vs no mass, "Fisher" for gains, the eigenbasis scaling, the check timing, jargon, two different "D"s, "one number per observable", natural gradient. | §3 rewritten on each point. The prior diagonal is now P and the LM diagonal M. |
| 22 | minor | Holding only coordinates exactly on a bound causes clip–reject cycles. | ε-active set (Bertsekas 1982), with a coupled-bound test. |

The review confirmed these points, which the revision keeps:

- frozen-σ mean rows plus √2 log σ rows give the exact Fisher
  information for diagonal Gaussians, including blockdiag(JᵀJ, 2N_b) and
  the θ–noise block of `vis_error_rel`;
- the IRLS weight and its majorise–minimise argument, by concavity of
  ln χ² (the dropped term is O(2/ν_b));
- exact gradients with acceptance on Φ, so curvature errors cost speed,
  not correctness;
- the optimistix path is untouched, and the pipeline default stays as
  it is;
- the PSD check, and the diagonal size guard computed without forming an
  n×n matrix;
- P equals 1/σ² for Normal priors;
- the fix for `_newton_polish` being unable to leave a bound.
