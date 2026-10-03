# Fitting a Keplerian orbit jointly with the interferometric scene

Status: **design**, 2026-10-03. It is to be built in drpangloss, with the Kepler solver and orbital geometry from [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) (0.1.0). No orbit code exists yet.

This note expands the "Keplerian orbits" item of Stage 8 ([`imaging_plan.md`](imaging_plan.md), [`pmoired_parity.md`](pmoired_parity.md)) using what Apep taught us (`~/data/apep_gravity/notes/lessons_for_drpangloss.md`, §3). The spectral half is in [`spectro_interferometry_workflow.md`](spectro_interferometry_workflow.md). Apep is the worked example throughout. The runnable sketch is [`sketches/attached_cone.py`](sketches/attached_cone.py).

## 1. Why: what Apep needs

| Measured or known | Value | Consequence for the design |
|---|---|---|
| GRAVITY binary, 2023–25 (three epochs) | 28.05 ± 0.013 mas, PA 96.1° (WC relative to WN); flux ratio 0.34–0.39 in K | Positions are precise enough to measure orbital motion within 1.8 yr (§4.4) |
| Cone (resolved dust) | Vertex on the line of centres, 11.9 mas from the WC star; axis = line of centres + 0.4° skew; tilt \|β\| = 0.3 (+0.4/−0.2)° | The scene is defined **in the binary's frame** (R2), not on the sky |
| JWST plume orbit (White et al. 2025) | P = 193 yr, e = 0.82, i = 24°, ω = 10 ± 10°, Ω = 164 ± 15°, T_peri ≈ 1956 | External priors (R5); their convention is unknown (§2.5) |
| Orbit-predicted line of centres (their elements, taken as jaxoplanet's) | PA 164.3° → 164.9° (2023.7 → 2025.5) | 68° from the measured 96°: **conventions are the biggest risk** (§2) |
| NACO masking (Han et al. 2020) | 47 mas at PA 274–278° | A 180° flip, and a biased separation at ~λ/D (R7, R8, §5.3) |
| Distance | 2.4 kpc assumed; the dynamical parallax gives 5.5–7 kpc for 20–40 M☉ with 28 mas | Distance and mass are **derived** (R6) |

## 2. Conventions

### 2.1 Definitions (drpangloss)
All quantities below are verified numerically against jaxoplanet. The checks are §5.2's tests 1–6; the scratch scripts were run on 2026-10-03.

| Quantity | Definition |
|---|---|
| Sky axes | `dra` positive East, `ddec` positive North, both in mas, as in `AGENTS.md` |
| Position angle | North through East, from the primary to the secondary |
| Third axis | `dz` positive **away from the observer**. (dra, ddec, dz) is then right-handed, and d(dz)/dt has the sign of the secondary's radial velocity relative to the primary (positive = receding) |
| Primary (reference) | The star at the origin of the relative vector, which is the scene's reference component (`flux=1`). It need not be the more massive star. **Apep: the WN star** (brighter in K) |
| Secondary | The orbiting star; r = secondary − primary. **Apep: the WC star**, 28 mas East |
| Flux ratio | The scene's `flux` of the secondary, relative to the primary, per band or spectrum (`AGENTS.md` flux convention). It belongs to the scene, not the orbit |
| `inc` | 0 ≤ i < 180°. **i < 90° means the position angle increases with time** (counterclockwise on the sky, North through East) |
| `Omega` (Ω) | PA of the **ascending node**, defined as the node where the secondary *recedes* (dz increasing). Visual data alone fix Ω only modulo 180° (§2.3) |
| `omega` (ω) | The **secondary's** argument of periastron, measured from the ascending node in the direction of motion. This is the visual-binary convention; the spectroscopic ω of the primary is ω − 180° |
| `t_peri` | MJD of periastron |
| `period` | days (reported in years) |
| `a_mas` | Angular semimajor axis of the **relative** orbit, in mas |
| Tilt β | arcsin(dz/\|r\|), the line of centres' elevation out of the sky plane: positive when the secondary is the farther star |

### 2.2 Mapping to jaxoplanet
jaxoplanet's `OrbitalBody.relative_position(t)` returns (X, Y, Z) with X North, Y East and Z **toward** the observer. Its `relative_angles` returns PA = atan2(Y, X).

| drpangloss | jaxoplanet |
|---|---|
| (dra, ddec, dz) | (Y, X, −Z) × a_mas / `semimajor` |
| `omega` (secondary) | `omega_peri` + 180° (jaxoplanet's ω is the primary's, the radial-velocity convention: with ω = 0 the relative periastron lies at the descending node) |
| `Omega` | `asc_node` (the receding node: identical) |
| `inc` | `inclination` (identical; i < 90° turns the position angle forward) |
| `t_peri` − `t_ref` | `time_peri` (days) |
| `radial_velocity(t)` | `radial_velocity` (positive = redshift, for the primary) |

**Pitfalls found in jaxoplanet 0.1.0:**
1. **`parallax=` scales wrongly.** A 1 au orbit at a parallax of 0.1″ returns a separation of 4624, not 0.1. It multiplies by the au-to-R☉ factor instead of dividing (the code is marked `TODO`). **Never pass `parallax`.** Divide by `semimajor` and multiply by `a_mas` instead, as in the sketch. Not yet reported upstream.
2. **Units are R☉, M☉ and days, with G built in.** Fixing `period` and a dummy central mass, then dividing by `semimajor`, gives positions in units of a, independent of mass.
3. **Time precision.** jaxoplanet takes whatever times it is given. In float32, MJD ≈ 60000 resolves to only about 0.004 d. Always pass `mjd − t_ref`, with `t_ref` a static float64.

### 2.3 Symmetries, checked numerically
1. (Ω, ω) → (Ω + 180°, ω + 180°): identical sky positions, and dz changes sign. Visual orbits cannot tell these apart; radial velocities, or a 3-D scene with an *optically thick* or asymmetric component, can.
2. ω → ω + 180° alone: r → −r at all times. This is the same as swapping which star is the reference. So a "which star is primary" error and an "ω of which star" error are the same 180° flip.
3. i → 180° − i: reverses the sense of rotation.
4. For a nearly face-on orbit near apastron, PA ≈ Ω (mod 180°) whatever ω is. That is why Apep's PA tests Ω directly.

### 2.4 Thiele–Innes constants (in the conventions above, verified against jaxoplanet)
With X = cos E − e and Y = √(1 − e²) sin E:
- `ddec = A X + F Y`, `dra = B X + G Y`, `dz = C X + H Y`, where
- A = a(cos ω cos Ω − sin ω sin Ω cos i), B = a(cos ω sin Ω + sin ω cos Ω cos i),
- F = a(−sin ω cos Ω − cos ω sin Ω cos i), G = a(−sin ω sin Ω + cos ω cos Ω cos i),
- C = a sin ω sin i, H = a cos ω sin i.

### 2.5 Apep: 164° against 96°, and a hypothesis
Read in jaxoplanet's convention (ω = 10° for the primary), White et al.'s elements predict PA 164.3°, 164.6° and 164.9° for 2023.7, 2024.5 and 2025.5. Reading ω as the secondary's gives 344°. Neither reflection in §2.3 reaches 96°.
- **Hypothesis.** If the plume code measures Ω counterclockwise from the East (RA) axis rather than from North through East, then PA_node = 90° − Ω = −74° ≡ 106° (mod 180°). That is 10° from the measured 96°, inside their ±15°.
- That map is a **reflection**, so it also reverses the predicted sense of rotation. The direct convention predicts PA *increasing* by 0.33°/yr; the reflected one predicts it *decreasing*.
- GRAVITY can tell them apart without new code. The predicted change over 2023.7–2025.5 is 0.6°. The statistical PA error per epoch is about 0.013/28 rad ≈ 0.03°, which is still ≲ 0.2° even after the 2–7× error inflation.
- The orbit also predicts the separation growing by 0.22 mas over the same interval (the sketch: 27.96 → 28.19 mas), which is likewise measurable.
- **Action (no code):** refit the binary per epoch with the cone shared, and compare dPA/dt and d(sep)/dt with ±0.33°/yr and +0.12 mas/yr. A 0.1% wavelength-scale error per epoch (0.03 mas) does not mask the separation trend.
- **Status (2026-10-03).** White et al.'s Ω and ω conventions are unknown. Ben will ask the first author in person. The question was passed to the Apep workspace in `~/data/apep_gravity/notes/omega_convention_question.md`. Until it is answered, use the JWST elements only as axial priors on Ω (R5), never as a fixed Ω.

## 3. Engine: jaxoplanet, under a drpangloss orbit module
- **Use** `jaxoplanet.orbits.keplerian.Body` / `OrbitalBody` for positions, velocities and radial velocities. That way the Kepler solver (with its custom derivatives) and the radial-velocity sign live in one tested place.
- **Do not expose** jaxoplanet objects as fitted parameters. A drpangloss `KeplerOrbit` (an equinox/zodiax module) holds *our* parameters (§2.1), and builds a jaxoplanet `OrbitalBody` inside each call. This keeps the parameter paths ours (`"orbit.omega"`), confines the ω offset to one function, and insulates us from 0.x API changes.
- **Where it lives.** A new `src/drpangloss/orbits.py`, imported by `models` (where `Attached` goes, per `AGENTS.md`). The import order becomes `_utils` → `orbits` → `models`.
- **Dependency (decided 2026-10-03).** jaxoplanet is an **optional** dependency: the extra `drpangloss[orbits]`, imported lazily inside `orbits.py`, with an error naming the extra if it is missing. It needs only `jax` and `equinox`. Tests that use it skip when it is absent, and CI installs the extra.

## 4. Requirements and proposed interfaces

### R1. An orbit that exposes the 3-D relative vector
```python
orbit = KeplerOrbit(period=..., t_peri=..., ecc=..., inc=..., omega=...,
                    Omega=..., a_mas=..., t_ref=60500.0)  # t_ref static
dra, ddec, dz = orbit.relative(mjd)        # mas; mjd as float64 or offsets
vra, vdec, vz = orbit.relative_velocity(mjd)   # mas / day
orbit.position_angle(mjd), orbit.separation(mjd), orbit.tilt(mjd)
```
- Alternative parameterisations (R4) are classes with the same `relative` method, and each converts to `KeplerOrbit`.
- `to_jaxoplanet()` and `from_jaxoplanet(body, a_mas)` are the only converters. There are no converters to other orbit packages.

### R2. Components attached to the binary frame
```python
cone = Attached(TruncatedCone(...), orbit, anchor="secondary", dpa=0.4)
scene_at_t = cone.at(mjd)    # the component with dra, ddec, pa, tilt set
```
- **Anchor.** `"primary"` (origin), `"secondary"` (at r), `"barycentre"` (at r·q/(1+q), which needs a mass ratio), or a fraction along r. Apep's cone is anchored at the secondary. Its vertex is the cone's own `tip` along the axis, so no extra anchor parameter is needed.
- **Axis.** The line of centres, primary → secondary, in 3-D. `Attached` sets the component's `pa` to PA(r) + `dpa`, and its `tilt` to β(r) + `dtilt`.
  - Components declare which attributes are their axis (a class attribute, e.g. `axis_fields = ("pa", "tilt")`). Components with a PA but no tilt (`EllipticalGaussian`, `GaussianArc`) take only `pa`.
  - **Apep, in the sketch:** the attached cone's `pa` follows the orbit, from 96.28° to 96.88°. Its tilt comes out as +0.13° to +0.40° (the WC star the farther, in §2.1's convention). The fitted |β| = 0.3° could not determine that sign; the orbit does, given Ω and ω.
- **Orbital skew.** Default: a fitted `dpa` (and `dtilt`). Optionally, a physical aberration: the axis follows v_wind n̂ − Δv⊥, with Δv the stars' relative velocity. That needs physical velocities, and so the distance (R6). It lies in the orbital plane, so for a nearly face-on orbit it shows up as a PA offset. It is 0.03–0.2° on Apep now, but large near periastron (and dominant in WR 140). Its sign convention, which way the cone trails, needs its own test before it is offered (open question 7).
- **The scene's time dependence goes through a snapshot method.** Any `SourceModel` may implement `at(mjd) -> SourceModel`. The default returns `self`; `System.at` maps over its children. Model signatures do **not** gain an `mjd` argument, so `BinaryModel*` and every static scene keep the current fast path (`AGENTS.md`).

### R3. MJD per datum
- `OIData.mjd` and `OIData.frame`, per baseline sample, are specified in the spectro note §2.5. Closure phases inherit their time through `i_cps*`. Their three legs share one frame time, which is needed so that a moving model's closure phase is self-consistent.
- **Evaluation.** `OIData.model` checks whether the model is time-dependent (whether `at` is overridden anywhere in the tree). If it is, it evaluates `scene.at(t_f)` once per unique frame time and maps over frames: `jax.vmap` over padded per-frame index arrays. Otherwise nothing changes.
  - **Apep:** about 10–20 frames per epoch, so 30–60 snapshot evaluations, which is trivial.
  - **Short-period binaries:** the same code, with positions moving within a night.
- **Epochs.** `OIData.epochs()` groups frames for per-epoch nuisances (spectro note §2.5).

### R4. Parameterisations for short arcs
On Apep, sampling Campbell elements (a, e, i, ω, Ω, T, P) over a 1.8 yr arc of a 193 yr orbit is pathological: most elements are unconstrained, and correlated along curved ridges. Three alternatives, all converting to `KeplerOrbit`:

| Class | Parameters | When |
|---|---|---|
| `KeplerOrbit` | P, t_peri, e, i, ω, Ω, a_mas; samplers see √e cos ω, √e sin ω, cos i, and Ω ± ω for nearly face-on orbits | Well-covered orbits; external element priors |
| `ThieleInnesOrbit` | A, B, F, G (+ C, H, if dz is wanted), P, t_peri, e | Positions are **linear** in A, B, F, G at fixed (P, e, t_peri), so these sample well and the Ω/ω degeneracy at small i disappears. §2.4 gives the conversion |
| `StateVectorOrbit` | At `t_ref`: (dra, ddec) [mas], (vra, vdec) [mas/yr], and dz, vz; plus μ = 4π² a_mas³/P² [mas³ yr⁻²] | **Short arcs.** The measured quantities (position, and its rate from multi-epoch data) are parameters with near-Gaussian posteriors; dz, vz and μ carry the physical priors. Converted to elements analytically (the standard state-vector → elements relations, written from first principles) |

- Notes on `StateVectorOrbit`:
  - μ is distance-free, and couples to mass only through D (R6). A prior on P from the JWST plume is a prior on μ given a_mas.
  - The conversion has removable singularities at e = 0 and i = 0. Use the (h, k) = e(sin ϖ, cos ϖ) and (p, q) = tan(i/2)(sin Ω, cos Ω) forms internally, with tests near both.
  - Unbound states (positive energy) are outside the prior support and are rejected.
- **What Apep's GRAVITY data constrain.** Position, its 1.8 yr rate, and the sign of dz (only through the attached cone and an assumed Ω/ω). The rate of the position angle depends only on e, P and the phase, not on a, M or D. So even this short arc constrains the phase and the sense of rotation.

### R5. External priors and radial velocities
- **Element priors** are numpyro distributions on `KeplerOrbit` paths, as for any parameter. For other parameterisations, a prior on a *derived* element (e.g. the JWST Ω when sampling `StateVectorOrbit`) goes in through a model function plus a log-density term. Today that is a regulariser in `fit`, and it must also be accepted by `numpyro_model` as a genuine prior. Document that such a prior does not include the Jacobian of the reparameterisation.
- **Ambiguity modulo 180°.** When the external source only knows Ω modulo 180° (or its convention is in doubt; §2.5), the prior is axial: a von Mises on 2Ω. Provide a two-line `axial_von_mises(mean, kappa)` helper.
- **Radial velocities.** `RVData(mjd, rv, d_rv, star="primary" | "secondary")` adds a likelihood term from jaxoplanet's `radial_velocity`. It needs the mass ratio q, the systemic γ, and a physical scale (M_tot or D; R6). Its σ terms reuse `noise=`-style inflation (`rv_jitter`). This is the only data that fixes Ω absolutely (§2.3, symmetry 1).

### R6. Distance and mass as derived quantities (dynamical parallax)
- **Native parameters:** a_mas and P, which are distance-independent. Derived: M_tot = (a_mas · D)³ / P², with a in arcsec, D in pc, P in yr and M in M☉.
- **An optional `distance_pc` parameter with a prior.** Only distance-dependent terms use it: RV amplitudes, physical skew velocities, the dust speed from the plume's proper motion, luminosities. **Keep distance-free constraints separate:** the plume's P (from angular expansion and shell spacing) is distance-free, but its 1020 km/s dust speed is 90 mas/yr × 2.4 kpc.
- **Apep.** The orbit sketch has |r|/a = 1.71–1.72 at ν ≈ 170°, so a_mas = 16.4 mas for 28.05 mas. With M_tot = 20–40 M☉ that puts Apep at 5.5–7 kpc (NACO's 47 mas gives 3.3–4.2 kpc). At 5.5–7 kpc the dust speed is 2400–3000 km/s, about the wind speed. The 3-D orbit replaces `dynamical_parallax.py`'s assumption of a sky-plane separation with the fitted geometry.
- **Report:** a posterior on D (or a curve of D against M_tot), never a single number when M_tot is only bounded.

### R7. Fits across instruments
- **One scene and one orbit for all datasets.** Each dataset has its own:
  - **bandpass and resolution:** 6a's smearing;
  - **error model:** `noise=` terms, or the rank-one correlated nuisances (spectro note §2.4);
  - **wavelength scale:** `wavel_scale` (spectro note §2.6);
  - **closure-phase sign:** a per-instrument test (§5.3), and never a free sign parameter.
- **Interface:** this exists already. `fit(scene_fn, priors, [gravity_23, gravity_24, gravity_25, naco_16, naco_19], noise=[...])`, where `scene_fn` returns one time-dependent scene (or one per dataset) and each dataset's MJD drives `at`.
- **Fit visibilities, not derived positions.** Positions published from other instruments enter only through their visibilities. Positions are used only when the raw data are unavailable, through a `PositionData(mjd, sep, pa, cov)` term that **warns** that it ignores the scene.

### R8. `simulate(scene, template)`: bias tests and planning across instruments
```python
fake = simulate(scene, template, errors="template", key=key,
                mjd=None, noise=None)
```
- **Inputs.**
  - `template` is any `OIData`: real data from another instrument, or `coverage.vlti_oidata` / `nrm_oidata`.
  - The scene is evaluated at the template's own MJDs, through its bandpass and smearing.
  - **Errors:**
    - `"template"`: the template's σ;
    - a `noise` dict: inflated σ, with nuisances *drawn*, including correlated blocks and a wavelength scale;
    - `None`: noiseless.
  - `mjd=` shifts or replaces the epochs, for planning.
- **Built on** `OIData.with_model`, which already preserves sampling, closure indices and operators. The new parts are time-dependence (R3), drawing nuisances, and 6a's smearing.
- **Apep use.** The GRAVITY scene (orbit + cone + halo) is put through the NACO mask's coverage, then fitted with a binary, to quantify the 28-against-47 mas bias. A throwaway version gave 32–87 mas depending on band and model (`~/data/apep_naco/naco_view.py`).
- A companion helper, `bias_test(scene, template, fit_model, priors, n)`, reports the distribution of fitted parameters over noise draws.

### R9. Scenes that are static in the co-rotating frame, and scenes that change
| Hypothesis | Model |
|---|---|
| Static on the sky | Components not attached; the companion alone moves |
| Static in the co-rotating frame | Components `Attached` to the orbit, with shared shape parameters (the default for Apep near apastron, where gas refreshes the dust edge in ~0.4 yr) |
| Changing | `Attached`, with per-epoch shape parameters θ_e = θ + δ_e, δ_e ~ N(0, τ²) and τ fitted (per-dataset model function plus a hierarchical prior) |

- τ → 0 recovers co-rotation, so the posterior on τ (or the evidence) tests "static in the co-rotating frame" against "changing".
- **Apep's scale.** The sketch's co-rotating scene differs from a frozen 2024.5 scene by max |ΔV| ≈ 0.03 in 2023.7 and 2025.5 (on random 30–130 m baselines). That is dominated by the companion's 0.3 mas motion, and compares with 15–25% V² noise after inflation. The cone's own 0.6° turn is small next to that.

## 5. Test plan

### 5.1 Reference ephemerides
1. **An independent evaluator.** A NumPy-only implementation (SciPy's root finder for Kepler's equation, Thiele–Innes, §2.4) against `KeplerOrbit.relative` over a grid of (e, i, ω, Ω, phase), in float64 to 1e-10 relative. In float32 with `t_ref`, to 1e-5. This is how §2.4 was checked.
2. **A published visual binary with radial velocities, so that Ω is absolute.** Proposal: **α Cen AB**. Its orbit is in Pourbaix & Boffin (2016, A&A 586, A90) and Akeson et al. (2021, AJ 162, 14). Test the predicted (separation, PA) against their tabulated or plotted epochs, with each paper's stated ω (of B relative to A, or of A) and Ω conventions mapped to §2.1. **The values must be copied from the papers when the test is written, not from memory.**
3. **A second, interferometric visual binary** with a published orbit from VLTI or CHARA data, to check an interferometrist's convention against ours. **To be chosen; see the reminder in §7.**

### 5.2 Convention unit tests (fast)
1. i < 90° ⇒ PA increases. i → 180° − i reverses it.
2. (Ω + 180°, ω + 180°): the same (dra, ddec), and dz flips.
3. ω + 180°: r → −r. This equals swapping the primary and secondary.
4. dz increases (the secondary recedes) at the node with PA = Ω, and `relative_velocity`'s z component matches the finite difference of dz.
5. Round trips: `KeplerOrbit` ↔ `ThieleInnesOrbit` ↔ `StateVectorOrbit` ↔ `to_jaxoplanet` / `from_jaxoplanet`, including near e = 0 and i = 0.
6. jaxoplanet's ω offset and RV sign: our secondary ω = jaxoplanet ω + 180°. The primary's radial velocity is positive while the secondary approaches.
7. **The `Attached` orientation test** (in the spirit of `AGENTS.md`'s orientation tests): an attached elongated component at a known orbital phase renders with its major axis at the orbit's PA. Its `render()` Fourier-transforms to its `model()`.
8. **Apep regression** (from the sketch): with the sketch's elements, PA 95.89° → 96.49° and tilt +0.13° → +0.40° over 2023.7–2025.5.

### 5.3 OIFITS closure-phase signs: round trips to catch 180° flips
NACO's 274–278° is GRAVITY's 96° reversed. Closure-phase sign conventions (OI_T3 baseline order, the instrument's conjugation, the sign of u and v) and the choice of "brighter" star each flip a binary by 180°.
1. **Synthetic.** A binary at 28 mas, PA 96°, flux 0.36 (unequal fluxes, so that the flip is visible) is simulated on GRAVITY-like and 7-hole-mask coverage. It is then written through each available writer:
   - drpangloss `write_oifits`;
   - AMICAL's OIFITS writer, when AMICAL is installed (skip otherwise);
   - a GRAVITY-layout file. For this, the column layout and `STA_INDEX` ordering of a real GRAVITY product are copied, and its data replaced.

   Each file is read with `read_oifits`, refitted, and checked: PA within 1° of 96° (not 276°), and the flux ratio below 1.
2. **Real anchors, one per instrument path.** A binary with a well-known orbit observed with GRAVITY, and one with NACO/SPHERE masking (reduced by AMICAL, see the masking workflow). Each is fitted with drpangloss, and its PA compared with the orbit's prediction. Only a real anchor tests the *pipeline's* convention, as opposed to our writer's. **The targets are not chosen yet; see the reminder in §7.**
3. **The OIFITS v2 sign convention.** Check `read_oifits`'s phase sign against the OIFITS v2 standard (Duvert et al. 2017) explicitly. Do this once, in a docstring and a test, since every instrument path depends on it.

### 5.4 End to end
1. **Simulated Apep first:** orbit (in §2.1's convention) + attached cone + halo, on the three GRAVITY epochs' coverage and on NACO coverage, through `simulate`. The joint fit recovers the elements' identifiable combinations (position, rate, the sign of dz given Ω), and the NACO-only binary fit reproduces the bias of R8.
2. **Then real Apep:** GRAVITY 2023–25, plus NACO 2016 and 2019 once re-reduced, plus the JWST plume elements as axial or uncertain priors (R5).

## 6. Mapping and effort

| Item | Goes in | Effort (agent h) | Depends on |
|---|---|---|---|
| `orbits.py`: `KeplerOrbit`, `ThieleInnesOrbit`, converters to and from jaxoplanet; tests 5.1.1 and 5.2 | Stage 8 (Keplerian orbits), brought forward | 4–5 | `[orbits]` extra |
| `StateVectorOrbit` and its regular forms | Stage 8 | 3 | the above |
| `SourceModel.at` and time-dependent `OIData.model` | new | 3–4 | `OIData.mjd`/`frame` (spectro §2.5) |
| `Attached` (anchor, axis, `dpa`/`dtilt`); orientation test | new (binary-frame attachment) | 3 | `at`, `orbits.py` |
| Physical skew (aberration) | new, later | 2 | `Attached`, R6 |
| `RVData`, `PositionData`, axial priors | Stage 8 | 2–3 | `orbits.py` |
| `distance_pc` and derived mass; reporting | Stage 8 | 1 | — |
| `simulate`, `bias_test` | new | 2–3 | `at`; 6a smearing optional |
| `TruncatedCone` into the library (`axis_fields`; jaxbessel's `j0`; an elliptical cross-section option; render ↔ model test) | new | 3–4 | — |
| PA round-trip tests (5.3.1, 5.3.3); real anchors (5.3.2) | new | 2, then 2 per anchor | AMICAL optional |
| Per-epoch PA and separation rates on Apep (§2.5 action) | science, no code | 1 | — |

## 7. Decisions and open questions

### Decided (Ben, 2026-10-03)
1. **jaxoplanet is an optional dependency** (`drpangloss[orbits]`; §3).
2. **The user-facing conventions are those of §2.1:**
   - ω is the secondary's (the visual-binary convention, matching Thiele–Innes);
   - Ω is the PA of the receding node;
   - dz is positive away from the observer (right-handed with (dra, ddec), with the same sign as radial velocity);
   - i < 90° means the PA increases.

   jaxoplanet's conventions stay internal to `KeplerOrbit`.
3. **Orbits are built here,** in drpangloss, not in a separate package or by Toon.
4. **No orbitize! or orvara:** neither as converters nor as dependencies.

### Pending
1. **White et al. 2025's Ω and ω convention.** Ben will ask the first author (§2.5). Until then, the JWST elements enter only as axial priors.

### Reminder for Ben: choose the real anchor binaries
Before the tests in §5.1.2–3 and §5.3.2 are written, choose:
- a visual binary with radial velocities, so that Ω is absolute (α Cen AB is proposed);
- an interferometric visual binary with a published VLTI or CHARA orbit;
- a binary with a well-known orbit **observed with GRAVITY**;
- one **observed with NACO or SPHERE masking** (reduced with AMICAL).

The synthetic round trips in §5.3.1 and §5.3.3, and the unit tests in §5.2, do not wait for these.

### Still open (defaults in force until Ben says otherwise)
1. **Time dependence:** the snapshot method `at(mjd)` (default), not an `mjd` argument on every `model()`.
2. **Physical orbital skew:** a fitted `dpa` only (default), until a system near periastron needs the physical skew.
3. **End to end:** simulated Apep first, then real Apep (default).
4. **Filing the jaxoplanet `parallax=` bug upstream:** ask first.
