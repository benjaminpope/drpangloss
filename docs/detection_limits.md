# Detection limits: methods and choices

[`virgil.limits`](api/limits.md) turns a non-detection into a flux limit at
each position of a grid. This page says what each method computes and which
to use; [Contrast limits](contrast_limits.md) runs them on simulated data.
To calibrate a detection threshold, with the look-elsewhere effect of a grid
search, see [Detection ROC curves](detection_roc.md).

## Units {#units}

A companion is parameterized by its **flux** relative to the primary
(companion/primary, 0.01 for a companion 100 times fainter). Limits are
returned in these units. Results are usually reported as a **contrast**,
primary/companion (100 here), or a magnitude difference
$\Delta m = 2.5\log_{10}(\text{contrast})$ (5 mag here).
[`flux_to_contrast`][virgil.limits.flux_to_contrast],
[`flux_to_delta_mag`][virgil.limits.flux_to_delta_mag] and their inverses
convert; zero or negative fluxes map to a large finite contrast, not to
infinity.

## Which method? {#which-method}

| | [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] | [`absil_limits`][virgil.limits.absil_limits] | [`injection_limits`][virgil.limits.injection_limits] |
|---|---|---|---|
| Statistics | Bayesian, flux $\ge 0$ prior | frequentist $\chi^2$ test | frequentist $\chi^2$ test, as CANDID |
| Input | best-fit flux and its Laplace error on a grid | model, data, grid | model, data, grid |
| Answers | what flux is credible at each position, given a source there | what flux would have been detected at $n\sigma$ | the same, with the companion injected into the data |

Use the **Ruffio** limit when you want a posterior statement (a percentile of
the flux, with a median and credible interval if you like), conditioned on the
astrometry being right and a source being there. Use **Absil** or
**injection** for a detection-style limit, "a companion brighter than this
would have been detected at $n\sigma$". They differ only in the sign of the
cross term between the residuals of the data and the signal: Absil uses
$\chi^2(\text{data} - s(f))/\chi^2(\text{data})$, injection (Gallenne et al.
2015, section 3.2, as in CANDID's `detectionLimit(methods=["injection"])`)
uses $\chi^2(\text{data} + s(f))/\chi^2(\text{data})$ for the null model.

## Ruffio et al. (2018) {#ruffio}

The unconstrained (possibly negative) best-fit flux at a position, with its
Laplace uncertainty, is the Gaussian $N(\text{mean}, \sigma)$; the positivity
prior truncates it to $f \ge 0$ (eqn 8 of
[Ruffio et al. 2018](https://arxiv.org/abs/1809.08261)).
[`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] returns a quantile of
the truncated posterior, so `percentile=norm.cdf(2)` is a 2-sigma-equivalent
upper limit, and 0.16, 0.5, 0.84 give a median and credible interval. Take
`mean` from
[`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid] and `sigma` from
[`laplace_flux_uncertainty_grid`][virgil.grid_fit.laplace_flux_uncertainty_grid].

The flat $f \ge 0$ prior is the published convention, and deliberately not
the Jeffreys prior for a scale: a log-uniform prior on the flux makes the
posterior improper as $f \to 0$, so there would be no finite upper limit.

The quantile is computed from the upper tail, $\text{mean} + \sigma z$ with
$Q(z) = (1 - \text{percentile})\,Q(-\text{mean}/\sigma)$ and $Q$ the standard
normal survival function, in log space so that it stays accurate when the best
fit is many sigma below zero. Where the tail underflows, the large-deviation
limit $z = \sqrt{a^2 - 2\log(1 - \text{percentile})}$, with
$a = -\text{mean}/\sigma$, is the starting point, and two Newton steps on
$\log Q(z)$ polish it.

## Absil and injection limits {#absil-injection}

At each grid position both find the flux whose significance, from a
chi-squared ratio converted by [`nsigma`][virgil.limits.nsigma], first reaches
`sigma`.

- **Absil** ([Absil et al. 2011](https://arxiv.org/abs/1110.1178)): the
  ratio of the model with the companion to the no-companion model.
- **Injection**: a companion of flux $f$ is added to every observable
  (including extras: the observables of the model with the companion minus
  those of the no-companion model) and the no-companion model is fitted to the
  result. The companion model fits the injected data exactly as the null model
  fits the original data, so the ratio is
  $\chi^2(\text{data} + s(f))/\chi^2(\text{data})$ for the null model. With
  gains or OI_FLUX data, whose whitening depends on the model, the companion
  model's $\chi^2$ is computed in full.

The no-companion model sets every parameter in `grid` to zero. The number of
degrees of freedom is the number of data points; fitted parameters are not
subtracted.

**The search.** The flux is stepped upward from the lower bound in quarter
decades until the significance reaches `sigma`, and that step is bisected in
log flux. Quarter-decade steps mean that a significance that rises above
`sigma` and falls back within one decade is not stepped over. For a
normalized scene the significance falls again once the companion outshines
the primary (flux well above 1), and for injection it can dip at very faint
fluxes because of the cross term, so an unbounded search
(`flux_bounds=None`) should start below that, from the smallest positive
value of the flux axis. It goes at most 40 decades from its start. Limits
outside `flux_bounds` are returned at the nearer bound, with a
`RuntimeWarning` counting them.

**Range of `sigma`.** It must exceed the significance of a chi-squared ratio
of 1 (about 0.67 for many degrees of freedom), and be below the largest
significance the floating-point type can represent (about 12.9 in float32, 37
in float64; enable float64 with
`jax.config.update("jax_enable_x64", True)`).

**Differences from CANDID.** CANDID refits the primary's diameter to the
injected data before computing the null chi-squared (for V² and T3); virgil
does not, and its null model is exactly the one with every grid parameter set
to zero. CANDID brackets the limit in steps of 1.4 in flux and interpolates
linearly in significance; virgil solves the criterion by bisection. The
chi-squared and the number of degrees of freedom are virgil's.

## Significance {#significance}

[`nsigma`][virgil.limits.nsigma] converts a reduced-chi-squared ratio to a
Gaussian-equivalent significance, as in Absil et al. (2011): the statistic
$x = \text{ndof}\,\chi^2_{r,\text{test}}/\chi^2_{r,\text{true}}$ is compared
with a chi-squared distribution of `ndof` degrees of freedom, and its
upper-tail probability is expressed as the equivalent two-sided Gaussian
significance. The upper tail is computed directly (with the regularized
incomplete gamma function), not as $1 - \text{cdf}$, so significances stay
finite and accurate far beyond 8σ, including in float32 (up to about 13σ).

[`chi2ppf`][virgil.limits.chi2ppf] is $2\,\text{gammaincinv}(\text{df}/2, p)$,
inverting `jax.scipy.special.gammainc` by Halley's method from a
Wilson-Hilferty start, with the residual taken in the upper tail for
$p > 1/2$. It is accurate to about $10^{-14}$ relative in float64 for $p$ in
$[10^{-10}, 1 - 10^{-10}]$, needs no optional dependency, and is
differentiable and `jit`-able in both arguments. $p$ is clipped to
$[\epsilon, 1-\epsilon]$ of its own float type; near $p = 1$ precision is
lost, so use `nsigma` for small tail probabilities.

## Reading the detection plots {#detection-plots}

The three plots of a [`DetectionMC`][virgil.detection.DetectionMC] (simulated
searches from [`injection_recovery`][virgil.detection.injection_recovery]) are
described in [Detection ROC curves](detection_roc.md); the conventions are
here.

**Null distribution** ([`plot_null_distribution`][virgil.plotting.plot_null_distribution]).
The curve is the empirical exceedance (survival) function of the statistic
over the companion-free simulations: at each threshold, the fraction at or
above it, which is the threshold's false-alarm probability. It is drawn on a
log scale so the tail that sets detection thresholds is visible. The reference
curve is what the statistic would do at one position fixed in advance:
$\tfrac12 P(\chi^2_1 \ge x)$ for `delta_chi2` (Wilks's theorem with the flux
on its boundary at zero), the Gaussian tail for `max_snr`; `log_bayes_factor`
has none. The gap between the reference and the empirical curve is the
look-elsewhere effect of searching a grid. The legend sits outside the axes
so that it never covers the tail; leave room for it in subplot grids.

**ROC curves** ([`plot_roc`][virgil.plotting.plot_roc]). Each curve follows a
detection threshold from high to low: its false-positive rate (the fraction of
companion-free simulations above it, the false-alarm probability) on the
x-axis and its true-positive rate (the fraction of injected companions
recovered, the completeness) on the y-axis. A statistic that cannot tell
companions from noise lies on the chance curve TPR = FPR; a better one lies
above it. The x-axis is logarithmic by default, from half a null draw's worth
($0.5/n_\text{null}$) to 1, since detections are claimed at small
false-alarm probabilities; on it the chance curve is a curve, not a line.
`mark_wilks` marks local significances (default 3σ and 5σ): a vertical line at
the false-alarm probability Wilks's theorem gives for one position fixed in
advance, the one-sided Gaussian tail (0.135% at 3σ, $2.9\times10^{-7}$ at
5σ), and a circle on each `delta_chi2` or `max_snr` curve where the threshold
equals that local significance (`delta_chi2` = 9 or `max_snr` = 3 at 3σ). Its
horizontal distance from the line is the look-elsewhere effect. With several
statistics, line style tells the statistics apart and colour the flux or
separation selections.

**Completeness map** ([`plot_completeness`][virgil.plotting.plot_completeness]).
Each cell is the fraction of injected companions of that separation and flux
detected at the false-alarm probability `fap`
([`DetectionMC.completeness`][virgil.detection.DetectionMC.completeness]).
Cells span halfway to their neighbours, in separation and in log flux;
injections of zero flux have no Δmag or contrast and are left out. The
contour lines are the empirical contrast curves of
[`DetectionMC.contrast_curve`][virgil.detection.DetectionMC.contrast_curve]:
in each separation bin the completeness is made non-decreasing in flux and
interpolated linearly in log flux (linearly in Δmag), and they are absent
where the injected fluxes do not bracket the level. The map sets the axis
limits to its cells, so curves drawn afterwards, such as Absil limits from
`plot_contrast_curve`, do not widen them; `plot_contrast_curve` redraws the
legend, so place it again at the end.
