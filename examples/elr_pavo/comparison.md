# PAVO rapid-rotator re-analysis: virgil vs Dholakia

Ours: `virgil.GravityDarkenedStar` (grey, `n_lat=32`) fitted by
`fit_pavo.py` with his priors, likelihood and NUTS settings (see its
docstring). His: the posteriors in his own `_NUTS.h5` files in
[jax-interferometry](https://github.com/shashankdholakia/jax-interferometry)
at commit `70689ed3dba338d59c98d02e8126a07a3b4e86da`, converted to our
conventions in `reference_posteriors.json` (`inc = 90 deg - his inc`,
`pa = his obl`, `jitter = exp(logsig)`). Each cell is median +/- sd of the
posterior samples. `D/sigma` is (ours - his) / sqrt(sd_ours^2 + sd_his^2);
rows with |D/sigma| > 0.5 are flagged with `*`.

## upsUMa (HD_84999)

Not re-fitted for this comparison (no `upsUMa_summary.json` in the output directory).

## epsCep (HD_211336)

672 data points; 2 chains x 2500 samples after 2000 warmup; mean tree depth 3.9, 17 divergences.

| parameter | Dholakia | ours | D/sigma | r_hat | n_eff |
| --- | --- | --- | --- | --- | --- |
| diam_eq (mas) | 0.7962 +/- 0.017 | 0.7964 +/- 0.018 | +0.01 | 1.003 | 527 |
| omega | 0.6339 +/- 0.087 | 0.6351 +/- 0.09 | +0.01 | 1.001 | 767 |
| inc (deg) | 80.04 +/- 11 | 80.11 +/- 10 | +0.00 | 1.000 | 1398 |
| pa (deg) | 156.9 +/- 4.8 | 157 +/- 4.7 | +0.02 | 1.000 | 2340 |
| jitter (V^2) | 0.04539 +/- 0.0014 | 0.04531 +/- 0.0014 | -0.04 | 1.000 | 2605 |

## lamBoo (HD_125162)

Not re-fitted for this comparison (no `lamBoo_summary.json` in the output directory).

## upsTau (HD_28024)

295 data points; 2 chains x 2500 samples after 2000 warmup; mean tree depth 5.7, 0 divergences.

| parameter | Dholakia | ours | D/sigma | r_hat | n_eff |
| --- | --- | --- | --- | --- | --- |
| diam_eq (mas) | 0.86 +/- 0.038 | 0.8592 +/- 0.037 | -0.01 | 1.000 | 1160 |
| omega | 0.8518 +/- 0.1 | 0.848 +/- 0.1 | -0.03 | 1.001 | 1172 |
| inc (deg) | 72.49 +/- 12 | 72.2 +/- 12 | -0.02 | 1.000 | 841 |
| pa (deg) | 18.41 +/- 2.3 | 18.49 +/- 2.1 | +0.02 | 1.000 | 2326 |
| jitter (V^2) | 0.03197 +/- 0.0016 | 0.03198 +/- 0.0016 | +0.00 | 1.000 | 2787 |
