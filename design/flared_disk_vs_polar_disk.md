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

## Where they differ: the flared surface

`FlaredDisk` does not render a thin 3-D surface. It deprojects each pixel as if on the mid-plane, (x, y), computes z = aspect r0 (rho / r0)^beta from that pixel's mid-plane radius rho, and evaluates the ring at r^2 = x^2 + (y + z sin i)^2 + z^2. This (a) moves the ring towards the **far** side, by z sin(i) in deprojected y, hence z sin(i) cos(i) on the sky, whereas `PolarDisk` with `PowerLawHeight(h0 > 0)` moves the ring towards the near side (pa + 90) by h sin(i); (b) varies the shift around the ring, since z follows the pixel's rho rather than the ring's radius; (c) adds z^2 to the 3-D radius.

Best single-parameter match: `PowerLawHeight(h0 = k z, r0, psi = beta)` with k fitted to minimise max |dV| (psi = 1, so that z = aspect * rho):

| inc | aspect | best k = h0 / z | cos(inc) | max abs dV at best k | max abs dV, flat ring |
|---|---|---|---|---|---|
| 20 | 0.05 / 0.1 / 0.2 | -1.10 / -1.09 / -1.04 | 0.94 | 0.009 / 0.018 / 0.038 | 0.022 / 0.045 / 0.086 |
| 50 | 0.05 / 0.1 / 0.2 | -0.75 / -0.75 / -0.73 | 0.64 | 0.019 / 0.038 / 0.073 | 0.050 / 0.100 / 0.190 |
| 70 | 0.05 / 0.1 / 0.2 | -0.41 / -0.40 / -0.40 | 0.34 | 0.023 / 0.046 / 0.089 | 0.060 / 0.120 / 0.232 |

- The sign is opposite to `Cone`/`PowerLawHeight`'s convention and the size is about z cos(i) (somewhat more at low i, because the z-dependence on rho matters).
- Taking h0 = +z (the naive translation) is 0.24 at inc 50, aspect 0.1 (flat ring: 0.10), worse than ignoring the height.
- Refitting the ring radius as well (r0 within about 2 per cent) reduces max |dV| by under 10 per cent: the residual is structural, not a radius offset. For psi = 1.25 the picture is the same, with errors about 20 per cent larger (k = -0.78, -0.42 at inc 50, 70).
- The residual grows linearly with aspect: max |dV| is about 0.38 aspect at inc 50, and about 0.45 aspect at inc 70.
- It is almost all **phase**: at inc 50, aspect 0.1, the rms amplitude difference is 0.001 to 0.004, while the rms phase difference is 0.01 rad below q = 0.01 cycles/mas, 0.13 rad at 0.01 to 0.02, and 0.3 to 0.45 rad beyond 0.02 (where |V| is 0.2 or less). The visibilities agree where the disk is unresolved, and the ring's displacement differs where it is resolved. The worst position angles are along the minor axis.

## What this means for the switch

- Flat `FlaredDisk*` (aspect = 0) can be switched to the engine with no change beyond the numerical-FT error (1e-9 here); the engine is also much cheaper than the 192^2 pixel DFT.
- A flared `FlaredDisk` is **not** reproduced by `PowerLawHeight` to better than about 0.4 aspect in |dV|. Whether that matters depends on the data: Blakely et al. fit aspects of 0.05 to 0.2 (with PDS 70 AMI closure phases), so the difference is comparable to the signal from the flaring itself, and results fitted with one model cannot be taken over to the other with the same parameters. The `aspect` parameter would also have to be redefined (k = -cos(i) is not a constant of the disk).
- Options for Ben and the `pds70-disk` branch: (1) keep `FlaredDisk` as the pixel model, and use the engine only for new disks; (2) add a `Height` variant to `PolarDisk` that reproduces the Blakely construction (a pixel-evaluated brightness, not a ring lift), which would keep the FT by quadrature only if the pixel step were kept; (3) switch, and flag the change of meaning of `aspect`/`flaring` in the changelog, since the physically correct thin-surface geometry is the one `PolarDisk` uses.
- Not tested: the HG and Gaussian phase functions, `skew != 0`, the surface occultation of the far side (neither model has it), and edge-on inclinations (FlaredDisk's deprojection divides by cos i).
