# `virgil.ensemble`

Ensembles of randomized image reconstructions in the manner of PYRA and
MYTHRA (Drevon et al. 2025): draw regularizer families, weights, pixel
sizes, fields and starting images; fit each group as an L-curve; select the
members that fit the data and average them into a mean image with a
per-pixel spread. A group can instead use a Gaussian-process prior whose
hyperparameters the evidence chooses (`evidence_grid`, `gp_members`); `combine`
averages both kinds of member. See the tutorial [Imaging, part 7](../imaging_ensemble.md).

::: virgil.ensemble
