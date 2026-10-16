# Common arguments

[`fit`][virgil.fitting.fit], [`numpyro_model`][virgil.likelihood.numpyro_model],
[`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] and the other fitting
entry points take the same arguments. They are documented once here; each
function's docstring says only what is specific to it.

## `model` {#model}

Either a template model whose leaves at the paths in `priors` are free (for
`fit` their values are the starting point), or a function called with the
parameter values as keyword arguments that returns a
[`SourceModel`][virgil.models.SourceModel]. A function lets you use
parameters that are not leaves of the model, such as a separation and position
angle, or one inclination shared by two components (see
[`build_model`][virgil.likelihood.build_model]).

The function may return a list of models, one per dataset in `data`, sharing
parameters: for example a scene and a [`Rotated`][virgil.models.Rotated] copy
of it, for two epochs between which it turns, or binaries at two epochs with
one flux ratio and a position each. Model `i` is compared with dataset `i`.
Regularizers then act on the first model only: an `Image` that appears only in
a later model is not regularized.

## `priors` {#priors}

A dict from parameter path (e.g. `"comp.flux"` or `"env.log_brightness"`; see
[`image_priors`][virgil.imaging.image_priors]), or function argument name, to
a numpyro distribution. In `numpyro_model` each key is also the sample-site
name. See [Which priors?](conventions.md#which-priors) for the choice.

- Priors on fluxes (keys named `flux` or ending in `.flux`) must have
  non-negative support.
- An angle (degrees) with an [`AngleVector`][virgil.angles.AngleVector] prior is
  a 2-D vector, with no wrap boundary at 0°/360°. `numpyro_model` samples it
  at the site `"<path>_vec"` and records the angle as the deterministic site
  `"<path>"`; `fit` gives its von Mises prior, if any, a least-squares form.
- A prior that is uniform in some coordinate of its parameter has a *flat
  coordinate* (see [`flat_coordinates`](#flat_coordinates)), in which `fit`
  optimizes it. Other priors (`Normal`, `Beta`, `HalfNormal`, ...) are
  evaluated in the model's own parameters, without the Jacobian of the
  bijection. The `values` that `fit` returns are always in the model's own
  parameters. The reasoning is in
  [Priors and the MAP](conventions.md#priors-and-the-map).

## `data` {#data}

An `OIData` or a sequence of them, fitted jointly. May be empty (`()`) when
`likelihoods` holds all the data. Their Gaussian log likelihood is added with
`numpyro.factor` in `numpyro_model`.

## `noise` {#noise}

Priors on error terms that are fitted with the parameters, as `{term: prior}`.
A dict applies to every dataset (numpyro sites and `fit` values
`"noise.<term>"`); a list of dicts gives each dataset its own
(`"noise[i].<term>"`).

| Term | Meaning | See |
|---|---|---|
| `vis_scale`, `phi_scale` | multiply the visibility and phase uncertainties | [`inflated_errors`][virgil.likelihood.inflated_errors] |
| `vis_error_rel`, `phi_error` | added in quadrature: a fraction of the model visibility; radians | [`inflated_errors`][virgil.likelihood.inflated_errors] |
| `vis_gain_<group>` | width of gains correlated across channels | [`OIData.with_gains`][virgil.oidata.OIData.with_gains] |
| `phi_offset_<group>` | width of closure-phase offsets | [`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets] |
| `wavel_scale`, `wavel_offset` | wavelength scale and offset | [`OIData.with_wavelength_scale`][virgil.oidata.OIData.with_wavelength_scale] |
| `north_angle` | the North angle, in degrees | [`OIData.with_north_angle`][virgil.oidata.OIData.with_north_angle] |

The priors of `wavel_scale`, `wavel_offset` and `north_angle` may be of either
sign (e.g. `Normal(1, 2e-4)`).

**Priors.** The error terms are scale parameters, so their default (Jeffreys)
prior is log-uniform on stated bounds; a `Uniform(0, ...)` favours large
values. The bounds must contain the plausible values: for the factors
`vis_scale` and `phi_scale`, whose neutral value is 1, e.g.
`dist.LogUniform(0.1, 10.0)`; for the added errors and widths (`vis_error_rel`,
`phi_error`, `vis_gain_<group>`, `phi_offset_<group>`), e.g.
`dist.LogUniform(1e-4, 0.3)`. `wavel_scale` is a scale too: log-uniform about 1
unless a calibration supplies a Gaussian.

**Tied terms.** An entry may be a function of the dict of fitted or sampled
values (keyed like `priors`) instead of a prior. It ties the term to
parameters in `priors`, as in a hierarchical model where each epoch's error
scale is drawn from a population with fitted hyperparameters (see
[`hierarchical_scales`][virgil.priors.hierarchical_scales]). In
`numpyro_model` it is recorded as a deterministic site, and the term's
`log_prior(values)`, if it has one, is added once (site `"noise_prior"`).

**Loss.** With fitted error terms the loss is the full Gaussian negative log
likelihood, including $\sum\log\sigma$, so the default `fit` method is L-BFGS;
so it is for data with gains, whose covariance depends on the model. Fitting
error terms together with an image is degenerate (a smoother image with larger
errors fits as well): estimate them with a parametric model first.

## `likelihoods` {#likelihoods}

Further Gaussian likelihood terms that are not visibilities: callables of the
parameter values (a dict keyed like `priors`) that return whitened residuals.
The built-in ones are [`PositionData.term`][virgil.orbits.PositionData.term]
and [`RVData.term`][virgil.orbits.RVData.term], for an orbit's positions and
radial velocities. They add their full normalized Gaussian log density, like
the `OIData` terms; a plain callable adds $-\tfrac12\sum r^2$ only. In
`numpyro_model`, term `i` is added with `numpyro.factor` as site
`"likelihood_<i>"`; in `fit` their χ² follow the datasets' in `info`.

## `regularizers` {#regularizers}

Penalties or log-prior terms on the model, e.g. from
[`virgil.imaging`][virgil.imaging]. `fit` takes any of them. `numpyro_model`
accepts only genuine prior densities (those with `probabilistic` true, such as
a [`Centroid`][virgil.imaging.Centroid] prior), added with `numpyro.factor`;
penalties such as maximum entropy are for `fit`.

## `flat_coordinates` {#flat_coordinates}

`fit` always optimizes a prior that is uniform in some coordinate in that
*flat coordinate*: $\log x$ for `LogUniform(a, b)` on $[\log a, \log b]$,
$\cos i$ for an
[`IsotropicInclination`][virgil.priors.IsotropicInclination], $\sin(\mathrm{lat})$
for an [`IsotropicLatitude`][virgil.priors.IsotropicLatitude] (any prior with a
`flat_coordinate()` method), and the parameter itself for `Uniform`.

For `numpyro_model` and `gauss_newton_mass`, `flat_coordinates` (default
`False`) chooses the coordinates NUTS moves in. `False` is numpyro's bijection
of each prior's support. `True` samples in the same flat coordinates as `fit`,
for parameters and error terms alike: a `LogUniform(a, b)` in the logit of $\log
x$ on $[\log a, \log b]$, where the prior's potential is exactly logistic. The
sites keep their names, their values are the model's parameters, and the
posterior is the same: only the coordinate NUTS moves in changes.

- **Pass the same value to both functions.** With `adapt_mass_matrix=False` a
  matrix in the wrong coordinates is never corrected, and sampling is silently
  slow. `numpyro_model` records the choice as its `flat_coordinates`
  attribute, and [`chain_init_params`][virgil.likelihood.chain_init_params]
  (given `mass=`) raises on a mismatch. Unconstrained `init_params` computed in
  numpyro's coordinates no longer apply with `True`.
- It does not cure hierarchical funnels, which come from the dependence
  between a group scale and its members: use the non-centred form or the mass
  matrix for those.
- It is off by default: simulation-based calibration (1000 replicates) showed
  no gain in effective sample size per second, nor in divergences, over
  numpyro's bijection.

## `dtype` {#dtype}

Precision of a fit: `"float64"` (the default) runs in a local
`jax.enable_x64` context, `"float32"` is faster and less precise. The returned
model and values are cast back to JAX's precision outside the fit (float32,
unless x64 is enabled), so that they work with the rest of your code. See
[Precision](conventions.md#precision).
