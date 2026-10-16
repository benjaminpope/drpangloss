# Analytic marginalization of linear parameters (`_linear.py`)

Moved from the `_linear` module docstring, which keeps a short summary.

The data are `d = m + A w + n`: a model m, a design A (n × k) times linear
parameters w, and noise `n ~ N(0, D)`, `D = diag(σ²)`. With a Gaussian
prior `w ~ N(μ, Λ)`, w integrates out in closed form (Luger, Foreman-Mackey
& Hogg 2017, *AJ* 153, 216):

    d ~ N(m + A μ, D + A Λ Aᵀ).

In whitened coordinates, `x = D^-½ (d - m - A μ)` and `U = D^-½ A Λ^½` (one
column per parameter), the covariance is `I + U Uᵀ`. Everything in
`_linear` works on `(x, U)`. It returns a whitened residual of the same
length as x, so the likelihood keeps one residual vector, and the
log-determinant `½ log det(I + UᵀU)`, kept as effective errors or a
log-normalization.

## Rules

- **Keep the log-determinant.** It depends on the model wherever A or σ
  does, e.g. a flux scale times the model's spectrum, or fitted jitter.
- **The prior is stated by the caller, finite, and never derived from the
  data.** A flat prior is only a limit, up to a constant that depends on
  its width: the width is part of the model.
- **Report the conditional posterior of w** (`posterior`).
- **A Gaussian on a scale-type parameter is a proposal only.** For a
  positive scale such as an OI_FLUX grey scale k, the Jeffreys prior is
  1/k on stated bounds. Reweight samples of the conditional posterior by
  `1 / (k N(k; μ, Λ))` (with bounds), and make no evidence claims from the
  Gaussian.

## Two ways to whiten, with the same result

- **Successive rank-one steps** (`whiten_rank_one`, `whiten_blocks`): a
  rank-one covariance `I + w wᵀ` is whitened by
  `R = I - w wᵀ / (q (q + 1))`, `q = √(1 + wᵀw)`, since
  `R (I + w wᵀ) Rᵀ = I`, with log-determinant `log(1 + wᵀw)`. Applying R to
  x and to the remaining columns, column by column, whitens `I + U Uᵀ`.
  Only scalar square roots appear, so the gradients stay smooth where
  columns are degenerate (equal visibilities on every baseline) or widths
  go to zero, where an eigendecomposition's are not. Use it for the
  calibration gains, closure offsets and grey scales, blocked by frame.
- **A dense Cholesky** (`whiten_cholesky`): `S = I + UᵀU = L Lᵀ` and
  `u = x - W (I + L⁻ᵀ)⁻¹ Wᵀ x`, with `W = U L⁻ᵀ`. This is an exact square
  root, again without an eigendecomposition. Use it for a few
  well-conditioned columns, such as RV zero points or a four-column
  Thiele–Innes basis. It is O(n k²).

`LinearMarginal` wraps either for one design with a stated prior. The
Laplace covariance of a fit's latent parameters
(`fitting._gauss_newton_covariance`) uses the same Woodbury identity, but
to invert a curvature, not to marginalize a likelihood, so it is not built
on this module.

In float32, χ² is off by ~7e-5 relative at a prior sd of 200σ and ~2e-5 at
2 × 10⁴σ against dense float64; the Cholesky and rank-one paths agree and
the log-determinants are exact (see `codebase_review_2026-10b.md`).
