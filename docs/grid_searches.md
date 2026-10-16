# Grid searches: methods and choices

[`virgil.grid_fit`](api/grid_fit.md) evaluates the likelihood, or a fitted
flux, at every point of a grid of positions. This page says what the linearized
flux map computes and how to read it; the other functions are described in
their docstrings, and [Binary search](binary_search.md) shows them in use.

## Linearized flux map {#linear-flux-grid}

[`linear_flux_grid`][virgil.grid_fit.linear_flux_grid] is a fast first pass for
companion searches, beside the iterative
[`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid]. It is the
equivalent of the linear contrast map (`lincmap`) of
[fouriever](https://github.com/kammerje/fouriever) (Kammerer; it has no paper).

For a faint companion (flux $f \ll 1$) the whitened residuals $r$ of the
likelihood are linear in $f$ at fixed position, $r(f) = r(0) + f g$, with
$g = \mathrm{d}r/\mathrm{d}f$ computed exactly by forward-mode automatic
differentiation of the model: no finite difference and no hand-derived
closure-phase derivative. The weighted least-squares solution is

$$
\hat f = -\frac{g\cdot r(0)}{g\cdot g}, \qquad \sigma_f = (g\cdot g)^{-1/2},
$$

which is $g^T C^{-1}(d - m_0)/(g^T C^{-1} g)$ with $C$ the data covariance,
because $r$ is already whitened. Correlated closure phases are handled exactly
as in the likelihood ([`whitened_residuals`][virgil.likelihood.whitened_residuals]),
and every observable in the data contributes. It costs one model evaluation
with its derivative per grid position, and no optimizer.

$\hat f$ is not constrained to be positive (as with `optimized_flux_grid` and
`lincmap`), so noise gives negative values and an SNR of either sign. Unlike
`lincmap`, which returns the *variance* $1/(g\cdot g)$, `flux_error` is the
standard deviation.

### Limits of the linearization, and `n_iter` {#n-iter}

With `n_iter=0` the linearization holds only for $f \ll 1$. The closure phase
of a binary scales as $f$ only to first order, with corrections of order
$f^2$ (and $f$ times the $|V|$ change for amplitudes), so for a bright
companion ($f \sim 0.3$) $\hat f$ is biased by tens of per cent and $\sigma_f$
is unreliable. Each of `n_iter` Gauss–Newton steps relinearizes at the current
$\hat f$ per pixel, $\hat f \leftarrow \hat f - g\cdot r(\hat f)/(g\cdot g)$,
and $\sigma_f$ comes from the final $g$. A few steps (3 at $f \sim 0.3$) reach
the optimizer's $\hat f$ and the Laplace $\sigma_f$, at `n_iter + 1` model
evaluations per pixel. The steps are not safeguarded, so for companions far
brighter than the primary, or strongly non-linear residuals, they can fail to
converge: use `optimized_flux_grid` to refine candidates. Where the phase
residuals are of order 1 rad, phase wrapping is not linear either. A companion
at a position where $g$ is nearly zero (a null of the baselines) has a large
$\sigma_f$ and so a small SNR.

### Priors and the Bayes factor {#linear-flux-prior}

With `prior=`, the result also holds the posterior mean and sd of $f$ and the
log Bayes factor against $f = 0$ (`log_bayes_factor`; $\log B > 0$ favours a
companion at that pixel). All of these hold in the linear model about the final
linearization point: exactly only where the residuals are linear in $f$ over the
posterior ($f \ll 1$, or after enough `n_iter` for the point to sit near the
posterior). The position is not marginalized. The likelihood in $f$ is
Gaussian with mean $\hat f$ and sd $\sigma_f$, so the Bayes factor is
$B = \int N(f;\hat f,\sigma_f^2)\,p(f)\,\mathrm{d}f \,/\, N(0;\hat f,\sigma_f^2)$.

**`dist.LogUniform(f_min, f_max)` (recommended).** The flux ratio is a scale
parameter spanning decades, so its prior is the invariant measure of the
scaling group (the Jeffreys prior under that group action),
$p(f) = 1/(f \ln(f_{\max}/f_{\min}))$, not the root-Fisher-information prior of
the linearized likelihood, which has constant Fisher information in $f$ and
would be flat. It is improper without bounds and the evidence needs a proper
prior, so both are required ($0 < f_{\min} < f_{\max}$). The Bayes factor
depends on them, as it must for a scale prior: for $\hat f$ well inside the
bounds, widening them changes $\log B$ by about
$-\Delta \ln\ln(f_{\max}/f_{\min})$ (the Occam factor). Choose them from the
physics or the data: $f_{\max}$ the brightest companion you would entertain,
$f_{\min}$ a little below the faintest contrast the data reach (for example
the smallest `flux_error` over the grid). The integral is a fixed 256-node
Gauss–Legendre quadrature in $\ln f$ over the part of the bounds where the
likelihood is not negligible; the posterior mean and sd come from the same
quadrature. It is `vmap`- and `jit`-compatible.

**`dist.Normal(mean, sd)`.** A Gaussian-prior evidence, a computational
approximation (the prior has support on negative $f$). With
$P = g\cdot g + 1/\mathrm{sd}^2$ the posterior is Gaussian with mean
$(g\cdot g\,\hat f + \text{mean}/\mathrm{sd}^2)/P$ and sd $P^{-1/2}$, and

$$
\log B = -\tfrac12\log(\mathrm{sd}^2 P)
+ \frac{(g\cdot g\,\hat f + \text{mean}/\mathrm{sd}^2)^2}{2P}
- \frac{\text{mean}^2}{2\,\mathrm{sd}^2}
$$

(Luger, Foreman-Mackey & Hogg 2017, arXiv:1710.11136).
