# FlaredDisk* against PolarDisk (stage P6 report)

Status: comparison only (2026-10-14). `FlaredDisk*` is unchanged; the switch to the `PolarDisk` engine (design/radial_profiles.md §5) waits for Ben and the `pds70-disk` branch. Not linked from public docs. The tests are `tests/test_flared_vs_polar_disk.py`.

## Method

`FlaredDiskPowerLaw` (Blakely et al. 2024 geometry) with a 192 x 192 grid of 0.5 mas pixels, against a `PolarDisk` with `GaussianRing(r0 = 20, sigma = FWHM / 2.355)`, FWHM 8 mas, pa 30 deg, `n_r = 96`. Baselines: 60 radii from 0.002 to 0.06 cycles/mas, at 13 position angles. float64, so the numbers below are the model difference, not rounding. Metric: max |V_flared - V_polar| over baselines (both disks have V(0) = 1).

## Where they agree (to the numerical-transform error)

| Case | max abs dV |
|---|---|
| Flat ring (`aspect = 0`, `symmetric` large), inc 50, pixel 1.0, 0.5, 0.25 mas | 1.5e-9 at every pixel size |
| Same, with `n = 2` (phase function cos^2(theta/2), a first harmonic) against `Harmonics(c=0, s=0.5/(0.5 + symmetric), form="linear")`, symmetric 0.5, 1, 3 | 1.6e-9 |

So for a flat disk with any phase function that is a low-degree trigonometric polynomial, `FlaredDisk` is exactly a `PolarDisk` ring with harmonics (the azimuth of FlaredDisk, from the near-side minor axis, is PolarDisk's azimuth from the major axis minus 90 deg: cos(theta_F) = sin(theta_P)). The pixel grid is not the limit once the ring is resolved by about 8 pixels across its FWHM. HG and Gaussian phase functions are not finite harmonic series: they need a `VonMises`-like or truncated expansion, not tested here.

## Where they differ: the flared surface, explained (2026-10-14 follow-up)

What Blakely et al. (2024, arXiv:2404.13032) write, Eqs. 2-3: z = H100 (sqrt(x^2 + y^2) / 100 au)^beta, with x, y "the potentially rotated and/or inclined coordinates of the disk mid-plane", and the apparent radius r = sqrt(x^2 + (y + z sin i)^2 + z^2). The paper does not say how the sky is deprojected or whether a surface-brightness Jacobian is applied. So virgil's `FlaredDisk` follows the paper's Eq. 3 literally (sin i, z at the pixel's mid-plane radius, z^2 added); it is not a virgil transcription error. Whether the paper's construction is exact geometry is a different question, answered below: it is not.

The hypothesis was that the mismatch with `PolarDisk(height=PowerLawHeight(...))` is (a) a sign convention, (b) sin i against tan i, (c) z at the pixel's radius. A reference model, `_thin` in the test file, builds the exact thin-surface geometry on the same pixel grid with four switchable ingredients, and was compared with `PolarDisk(GaussianRing, height=PowerLawHeight(h0 = -aspect r0, r0, psi = beta))` (same baselines as above, float64):

1. **Sign (a): verified.** A surface lifted towards the observer moves the ring towards the far side, so the match is a *negative* `h0`, which is what the paper's (y + z sin i) does. `PowerLawHeight` is right (positive h shifts towards pa + 90, the near side); `FlaredDisk` is right; the naive `h0 = +z` is wrong.
2. **tan i (b): verified, but small on its own.** Exact geometry is Y_sky = y_mid cos i - z sin i, so y_mid = y + z tan i (the sky shift z sin i, as `PolarDisk` has it). `FlaredDisk` uses z sin i in y_mid, a sky shift z sin i cos i, smaller by cos i. This fully accounts for the fitted k of about -cos i only in combination with (d); alone it hardly helps (below).
3. **Consistent z (c): verified, second order in the aspect.** z must follow the radius of the surface point, not of the pixel's mid-plane projection (a fixed point, iterated 40 times in the reference). The z^2 added to r^2 is a separate, smaller effect (max |dV| 1.6e-3 at aspect 0.05, 6.4e-3 at 0.1, 2.5e-2 at 0.2, independent of inclination).
4. **Jacobian (d): not in the original hypothesis, and the dominant omission.** FlaredDisk assigns B = f(r) per *sky* pixel, but the map from the mid-plane to the sky is not area-preserving once z varies around the ring: dy/dy_mid = 1 - z'(R) (y_mid / R) tan i. A ring of the thin surface must keep its flux, so B = f(r) / (dy/dy_mid). Without it the ring is brighter on one side by a factor 1 / (1 - beta aspect tan i sin(phi)), an artificial first harmonic of amplitude about beta aspect tan i (0.27 at inc 70, aspect 0.1), which is what makes the residual large and phase-dominated at high inclination.

With all four, FlaredDisk's construction equals `PowerLawHeight(h0 = -aspect r0)` to **1.5e-9** (the transform error) at inc 20/50/70, aspect 0.05/0.1/0.2, psi = 1 and 1.25 (the one exception: inc 70, aspect 0.2, psi 1.25, where beta aspect tan i > 1 and the surface folds over; the map is no longer one-to-one).

Ablation, max |dV| against that `PowerLawHeight` (aspect 0.1, psi = 1; psi = 1.25 is within 20 per cent of these):

| inc | `FlaredDisk` as written | all four | all four, but sin i | but z at pixel radius | but z^2 kept | but no Jacobian |
|---|---|---|---|---|---|---|
| 20 | 1.8e-2 | 1.5e-9 | 2.8e-3 | 1.8e-3 | 6.4e-3 | 2.0e-2 |
| 50 | 5.9e-2 | 1.5e-9 | 5.3e-2 | 1.9e-2 | 6.4e-3 | 6.6e-2 |
| 70 | 1.5e-1 | 1.5e-9 | 1.4e-1 | 9.3e-2 | 6.4e-3 | 1.4e-1 |

(Aspect 0.05 and 0.2 scale as aspect to aspect^2; the full set is the same 1.5e-9 everywhere.) Reading it: the Jacobian and the consistent z dominate at high inclination, sin versus tan matters mostly at low-to-moderate i once the Jacobian is right (note that tan alone without the Jacobian makes the match *worse* at inc 50: 6.5e-2 against 5.9e-2 for the aspect 0.1 case), and z^2 is the smallest. No single ingredient explains the residual: all four are needed, and the previously fitted k about -cos i was a compensation between them, not a property of the disk.

Best single-parameter fits (kept for reference): k = h0/z about -1.1 / -0.75 / -0.40 at inc 20 / 50 / 70, with max |dV| still 0.02-0.09 (see the earlier version of this table in the git history): these are the best a rigid lift can do against the Jacobian-biased brightness.

## What this means for the switch

- Flat `FlaredDisk*` (aspect = 0) is exactly a `PolarDisk` ring with harmonics to 1e-9; it could be switched with no change.
- A flared `FlaredDisk*` is **explained**, but not **reproduced** by any `PolarDisk` parameters: it is not a thin surface (its radius and brightness mapping are an approximation, see 2-4), and only agrees with the exact geometry as aspect * tan(i) goes to zero (to ~0.4 aspect in |dV| at inc 50). The paper's Eq. 3 is itself what we reproduce, so this is a property of the published model.
- **Do not deprecate `FlaredDisk*`** until it is reproducible as a special case of the general one: the rule is that no class is deprecated unless the general class reproduces it. It is not (for aspect > 0). Two ways to get there, for Ben and the `pds70-disk` branch: (1) add a `PolarDisk` height mode that reproduces the pixel construction exactly (needs the pixel-evaluated brightness, so the FT is by DFT); or (2) deprecate only the *flat* use, keep `FlaredDisk*` for flared fits, and recommend `PolarDisk` + `PowerLawHeight(h0 < 0 for a top surface towards the observer)` for new work, noting the change of meaning of `aspect`.
- Not tested: HG and Gaussian phase functions, `skew != 0`, occultation of the far side (neither has it), edge-on.
