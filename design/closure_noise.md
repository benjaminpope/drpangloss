# Closure-phase noise model (`_closure.py`)

Moved from the `_closure` module docstring, which keeps a short summary.

The closure phase of triangle (a, b, c) is φ_ab + φ_bc − φ_ac. Noise on the
baseline phases makes the closure phases of triangles that share a baseline
correlated, and with N telescopes only (N−1)(N−2)/2 of the triangles of one
frame and channel are independent: three of four, for four telescopes.
Treating all four as independent counts the closure phases 4/3 times.

## Covariance

The model is that of Kammerer et al. (2020, A&A 644, A110, §2.2): the
reported variances σ² on the diagonal, and a fixed correlation between
triangles from equal noise on every baseline,

    C = D^½ R D^½,   R = T Tᵀ / 3,   D = diag(σ²),

where T is the triangle-by-baseline matrix of +1, +1, −1 entries; R has
unit diagonal and ±1/3 between triangles sharing a baseline. C keeps every
reported error, is never indefinite, and scales with the errors, so
rescaled or inflated errors carry through.

## Approximation

This is an approximation whenever the σ within a group differ. True closure
phases are exact closures T b of baseline phases b, so their noise lies in
the column space of T, while C spans D^½ times it: the two agree only for
equal σ. With unequal σ the χ² of real closure-phase noise is slightly off
its nominal distribution (for four telescopes with one noisy baseline, a
mean of 2.87 against the 3 expected for three independent closure phases),
and `ClosureNoise.sample` draws noise from C that is not the closure of any
set of baseline phases. The exact model would be `C = T diag(s) Tᵀ` with
baseline variances s chosen to reproduce σ². The correlation is built from
the per-triangle errors σ, so it is exact only when the baseline phase
errors are equal; with unequal baseline errors the whitened χ² comes out
slightly low, by about 2% with one baseline three times noisier than the
others. OIFITS carries no per-baseline phase errors, so the data do not
allow better.

## Whitening

`ClosureNoise` groups the triangles that share baselines (one frame and
channel each). It whitens residuals r by dividing by σ, projecting onto an
orthonormal basis Q of the column space of T (the independent combinations)
and solving with the Cholesky factor of M = Q R Qᵀ, which is fixed. For r in
the column space of C this is exactly rᵀ C⁺ r, and the normalization is
½ log pdet C = ½ (log det M + log det Q D Qᵀ).

## Index dtype

The arrays are NumPy, built once and reused in either x64 mode. JAX (0.10
and later) caches the converted copy of a NumPy array by identity, whatever
the mode it was made in, so a float32 fit followed by a float64 one could be
handed int indices of the wrong width. The indices are therefore int32,
which is the same in both modes, and the floats are cast to the data's dtype
at use.
