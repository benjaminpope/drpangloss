# A Gauss–Newton mass matrix for NUTS

[`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] turns a fit into a
mass matrix for NUTS, the No-U-Turn Sampler of [Hoffman & Gelman
(2014)](https://arxiv.org/abs/1111.4246) as implemented in numpyro (see
[Betancourt 2017](https://arxiv.org/abs/1701.02434) for an introduction). Use
it to sample any posterior with one peak: images, and orbits and binaries with
fitted error scales and gains. [Sampling the posterior](imaging_sampling.md)
shows it on an image.

## The matrix {#the-matrix}

Near the maximum a posteriori, a posterior with one peak is close to a
Gaussian whose precision, in the unconstrained coordinates $z$ that numpyro
samples, is $F = J^\top J + \mathrm{diag}(P)$. Here $J$ is the Jacobian, with
respect to $z$, of a vector of *curvature rows*: the whitened residuals of
every dataset (with the diagonal errors held fixed), the residuals of
likelihood terms, and of angle-vector priors. $J^\top J$ is never negative in
any direction, since $\delta^\top J^\top J\delta = |J\delta|^2$, so $F$ is
positive semi-definite. $P$ is the curvature of the elementwise priors and of
the coordinate transforms, each coordinate's own, with the Jacobian of
numpyro's bijection included, as NUTS's potential has it (a Normal prior on
the real line gives $1/\sigma^2$).

The covariance $F^{-1}$ is the inverse mass matrix: it whitens that Gaussian,
so that directions the data fix tightly take the same step size as those left
to the prior. Without it, the step size shrinks to suit the tightest direction.
On a 62² Gaussian-field image fitted to 588 AMI observables the matrix cut NUTS
from its full tree depth (1023 leapfrog steps) per draw to 63; on an
8-parameter orbit, from 87 steps to about 6.

The matrix is a preconditioner: errors in it cost efficiency, not correctness.

## Fitted error terms {#error-terms}

With `noise=` the matrix covers the error terms too, under their numpyro site
names (`"noise.vis_scale"`, `"noise[1].phi_scale"`). The likelihood then has a
second term, $\sum_k\log\sigma_k$, that is not a sum of squares, but its
expected curvature is known for Gaussian data: a scale $s$ multiplying $N$
errors adds $2N$ in $\ln s$, whatever the model, and the matrix is
$\mathrm{blockdiag}(J^\top J, 2N)$. virgil builds it with extra rows
$\sqrt2\log\sigma_k$ next to the residuals. Added errors (`vis_error_rel`) get
their cross terms with the model parameters the same way. This is the Fisher
information. Phases with errors above about 0.5 rad (a von Mises, not a
Gaussian, likelihood) have their scale's curvature overstated, by up to a few
times, which costs efficiency only.

## Gains {#gains}

Gains and closure-phase offsets that are correlated across channels are
integrated out analytically: their covariance whitens the residuals, and $F$
differentiates through that whitening. That is a good curvature for the model
parameters. For fitted gain widths it is a positive semi-definite heuristic
rather than the Fisher information.

## What is dropped {#dropped}

The exact Hessian also has $\sum r\,\nabla^2 r$, each residual times the
curvature of the model. It is small when the residuals are noise-like or the
model is nearly linear over the error bars. It matters when the fit is poor
($\chi^2/N \gg 1$) or the model bends on the scale of the errors, where $F$ is
still a safe direction but a poorer guide to the error bars. A likelihood
term's `log_norm` (a fitted RV jitter's, or the Jacobian of
[`position_angle_prior`][virgil.orbits.position_angle_prior]) has no rows, so
its curvature is left out.

## When not to use it {#when-not-to-use-it}

- Several modes (orbit aliases, the 180° flip of a symmetric binary): the
  matrix describes one mode. Sample each separately.
- Nearly circular orbits ($e \lesssim 0.05$): $\omega$ and the periastron time
  are degenerate, and NUTS reports divergences.
- Companions too faint to detect, whose posterior fills the prior.
- Closure phases that can pass through ±180°, where the likelihood has sharp
  walls.
- $\chi^2/N \gg 1$ (a misspecified model).

Use it at fixed field hyperparameters ($\sigma$ and $\ell$, chosen for example
by [`log_evidence`][virgil.imaging.log_evidence]): the curvature depends on
them, so a matrix computed at one $\sigma$ and $\ell$ is wrong when they move.
Every sampled parameter must be in `priors`: a tightly constrained one left out
(such as an image's flux) keeps its identity mass and its tiny step size.

## Size {#size}

The dense matrix has $n^2$ entries. With more than 8192 coordinates the result
is a diagonal mass, with a `UserWarning`: images of 128² and above get this. Its
entries are the diagonal of the covariance $F^{-1}$, computed without forming
the matrix, when there are fewer rows than coordinates (an image), and
$1/\mathrm{diag}(F)$ otherwise.
