# Cross-validation of a hyperparameter

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

[`virgil.crossval`](api/crossval.md) chooses a hyperparameter, such as the
weight λ of a regularizer, by k-fold cross-validation: fit with the setting
on the training points of each fold and score the held-out points. It works
with any likelihood term that takes a boolean mask of the points to use;
[`ComplexVisData.term`](radio_interferometry.md) is one.

## The score

For each (setting, fold) the held-out score is the negative log likelihood of
the held-out points under the training fit, up to constants,
`½ Σ r² + log_norm`, with the fitted values, error scales included (`r` the
whitened residual and `log_norm` the term's normalization, `Σ 2 ln s_d` for
radio, over the held-out points only). The record also has the raw χ² at
`s = 1`, if you give `unit_scale`, and the same sums per group, if you give
`groups` (uv annuli, ddids). A smaller score is better.

## A radio λ sweep

```python
import numpyro.distributions as dist
from virgil import CalibrationNuisance, ComplexVisData, Image
from virgil.imaging import MaxEntropy
from virgil.crossval import cross_validate
from virgil.radio import dartboard_folds

data = ComplexVisData.read_ms("target.ms", stride=100)
nuis = CalibrationNuisance(data)
fold, annulus = dartboard_folds(data, k=5)

def scene(values):
    return Image(values["log_brightness"], pixel_scale_mas=2.0)

def model(**values):  # defined once: a new function recompiles the fit
    return scene(values)

def fit_kwargs(weight):
    priors = nuis.priors()
    priors["log_brightness"] = dist.Normal(0.0, 3.0)  # per pixel, as in the radio guide
    return dict(model=model, priors=priors, data=(), regularizers=[MaxEntropy(weight)])

def unit_scale(values):  # the error scales set to 1, for the raw chi^2
    return {**values, "cal.sigma": values["cal.sigma"] * 0 + 1}

out = cross_validate(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0], 5, fold,
    fit_kwargs=fit_kwargs,
    make_term=lambda mask: data.term(scene, nuis, mask=mask),
    init=init,                      # e.g. a fit to all the data
    groups=annulus,
    unit_scale=unit_scale,
    fit_ft={"eps_f64": 1e-6},       # fast transform for the fits
    score_ft={"eps_f64": 1e-12},    # accurate transform for the scores
    fit_options={"method": "lbfgs", "time_limit": 3600},
)
out["settings"][out["best"]], out["settings"][out["one_se"]], out["edge"]
```

The fits stay on the direct radio term (`chi2="auto"` is switched off for them).

`aggregate` returns arrays and dicts, with no plotting: `settings`,
`fold_scores` (settings × folds), `score` (the sum over folds), `mean` and
`sd` over folds, `se = sd/√K`, `best` (the smallest mean), `one_se` (the most
regularized setting whose mean is within one `se` of the best), `edge` (true
if either is at an end of the grid, so extend the grid), and per-group curves
`group_score`, `group_chi2`, `group_n` summed over folds, with `group_best`.

The sweep compiles once if every task uses the same `model`, `make_term`
and regularizer *structure*, with the setting only in array values (a
regularizer's weight, a prior's scale): the folds differ only in the mask's
values. `set_ft_defaults` is read when a function is traced, so keep the
same `fit_ft` and `score_ft` in a process; both are restored afterwards.

## Cluster array jobs

One (setting, fold) per array task: each calls `cv_task`, writes its record,
and a last step aggregates the files.

```python
import itertools, sys
from virgil.crossval import cv_task, save_record, aggregate_files

settings = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
pairs = list(itertools.product(settings, range(5)))
setting, k = pairs[int(sys.argv[1])]  # SLURM_ARRAY_TASK_ID
record = cv_task(setting, k, fold, fit_kwargs=fit_kwargs,
                 make_term=make_term, init=init, groups=annulus)
save_record(record, f"cv/task_{sys.argv[1]}.json")
```

```python
import glob
out = aggregate_files(sorted(glob.glob("cv/task_*.json")))
```

The records are JSON, so `aggregate_files` gives the same numbers as
`cross_validate` in one process; it raises if a (setting, fold) pair is
missing or repeated.

## Other data

`make_term(mask)` returns a likelihood term using the points where `mask`
(length `N`) is true, with array shapes independent of the mask. Calling it
on the values gives whitened residuals as `m` blocks of `N` (`m = 2` for
the real and imaginary parts of a visibility), zero at masked points; an
optional `has_log_norm` and `log_norm(values)` give the normalization of the
unmasked points. Nothing else is assumed.
