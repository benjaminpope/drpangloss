# Fitting internals: rationale cut from docstrings

Notes moved out of the private helpers of `virgil.fitting` (0.4.0 docstring
diet). The user-facing account is in
[`docs/gauss_newton_mass.md`](../docs/gauss_newton_mass.md) and
[`docs/common_arguments.md`](../docs/common_arguments.md).

## `_tolerance`

The gradient is a sum of terms that cancel at the minimum, and rounding leaves
it a floor which, relative to the starting gradient, is about eps times the
data's signal-to-noise ratio over the starting residuals: up to ~1e-4 in
float32 (eps ≈ 1.2e-7), where a fixed `gtol = 1e-4` was out of reach and LM ran
all its steps. The √eps floor (3.5e-4 in float32, 1.5e-8 in float64) clears
it, and is the usual stopping limit for finite-precision optimizers. The
1/1000-of-the-starting-gradient part lets a warm start (e.g. along an
L-curve) still converge; neither part goes below `1e-6 * gtol`, so a fit
started exactly at a zero-residual optimum, whose gradient is rounding noise,
is converged at once.

## `_capped_lbfgs` and `max_step_size`

Without the cap, a softmax image can take a step of ~20 in its
log-brightnesses that sends most pixels to ~exp(-20); their gradient,
proportional to their brightness, then vanishes and the fit stalls at a
spurious stationary point far above the minimum. Scaling the whole step,
rather than clipping each coordinate, keeps the quasi-Newton direction, and
makes the zoom line search (whose step is then at most 1) evaluate the points
actually taken, as `optax.value_and_grad_from_state` needs.

## `_lbfgs` restarts and `lbfgs_memory`

A round that ends at its step limit restarts from its last point with fresh
curvature memory, so that a memory spoilt on the way (a parameter running into
a bound, say) cannot hold the fit for the rest of `max_steps`: on macOS a
limb-darkened star's fit with fitted error scales crawled for all 20 000 steps
in one round, where Linux converged in under 700. The default memory is 50
(optax's is 10): on regularized images 10 left fits short of their optimum at
many weights, and weakly regularized ones running to the step limit; 50 found
lower losses in fewer steps, at a higher cost per step.

## `_GradientStoppedLM`

optimistix's own test, on the change of the residuals, passes at once from a
warm start, and a test on the parameters never passes for log-brightness pixels
that should be dark (they drift towards -inf without changing the image).

## `_lm` chunks

optimistix's LM state (its damping, the last Jacobian) holds closures that
cannot be carried between jitted calls without recompiling, so a restart per
chunk resets the damping.

## `_covariance`

With fewer rows than coordinates (an image's latents) the covariance is
inverted by Woodbury, solving a system the size of the rows: with W = P^-½ and
K = JW, (P + JᵀJ)⁻¹ = W (I - Kᵀ(I + KKᵀ)⁻¹K) W. The coordinates with P = 0 by
type (`curvature_zero`) are given P = 1, then that is taken back off,
(A - EᵀE)⁻¹ = A⁻¹ + A⁻¹Eᵀ(I - EA⁻¹Eᵀ)⁻¹EA⁻¹ with E picking them out of A. Every
other P is floored at 1e-12 times the largest, so that W is finite, since
`max(0, ·)` can give zero at run time. Positive definiteness is checked on the
result, by a Cholesky factorization of the covariance itself: a cancellation in
float32 can give a non-positive-definite matrix from factors that look fine.

## `_Objective.curvature_diag`

The terms are separable, so a `jax.jvp` of `jax.grad` with a tangent of ones
gives the diagonal. A prior with a flat coordinate adds nothing to `fit`'s
loss, so without the Jacobian its P is 0. A Normal prior on the real line gives
1/σ².

## `_Objective.curvature_rows`

A likelihood term's `log_norm` has no rows and adds no curvature. Mean rows are
the whitened residuals with the diagonal errors frozen, so that gains, closure
offsets and extra observable blocks stay live.

## `likelihood._curvature_rows`

For Gaussian data, mean plus normalizer rows give the Fisher information:
`blockdiag(JᵀJ, 2 N)` in `ln s` for one scale `s` of `N` errors. Uncorrelated
phases use the Gaussian σ, not the von Mises effective error (which would
overstate the curvature in a weak phase's scale). Correlated closure phases use
the effective errors of the whitened rows (`ClosureNoise.whiten`), without the
periodic penalty rows, whose errors are constants. Extra blocks get none: their
`whiten` has its own error model. Gain and offset widths get curvature only
through the mean rows, a positive semi-definite heuristic.
