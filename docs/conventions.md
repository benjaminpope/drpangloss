# Conventions

This page collects the conventions virgil uses: which way the axes point, what sign a phase has, what a flux means, and so on. None of them is hard, but they are the places where two reasonable people disagree, and where a wrong guess flips a binary by 180° without any error message. The API pages document each function; this page explains the choices behind them. Every statement here has been checked against the code, and the tests named below fail if one of them changes.

1. [Sky coordinates and images](#sky-coordinates-and-images)
2. [Baselines and the Fourier sign](#baselines-and-the-fourier-sign)
3. [Observables](#observables)
4. [Closure phases with four or more telescopes](#closure-phases-with-four-or-more-telescopes)
5. [Fluxes](#fluxes)
6. [Times and frames](#times-and-frames)
7. [Orbits](#orbits)
8. [Which priors?](#which-priors)
9. [Priors and the MAP](#priors-and-the-map)
10. [Precision](#precision)

## Sky coordinates and images

Positions on the sky are offsets from a reference point, in milliarcseconds:

- `dra` is the right-ascension-like offset, **positive to the East**;
- `ddec` is the declination-like offset, **positive to the North**.

Because East is on the left of a sky image, `dra` increases to the *left*. This is the usual astronomer's picture of the sky, not the usual picture of a graph.

**Position angle** (PA) is measured from North through East, in degrees: PA 0° is North, 90° is East, 180° is South and 270° is West. It is counter-clockwise on a sky image with North up and East left. A companion at separation $s$ and position angle $\theta$ has

$$
\mathrm{dra} = s \sin\theta, \qquad \mathrm{ddec} = s \cos\theta,
$$

which is how [`BinaryModelAngular`][virgil.models.BinaryModelAngular] relates to [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]. Going the other way, $\theta = \operatorname{arctan2}(\mathrm{dra}, \mathrm{ddec})$ (note the order of the arguments), taken modulo 360°. All other angles that are position angles, such as the orientation of an ellipse's major axis or the phase of an azimuthal modulation, follow the same rule.

### Images

Rendered images and `(dra, ddec)` grids are ordinary 2D arrays: the first index is the row and runs from the top of the picture to the bottom, the second is the column and runs left to right. The sky coordinates attached to them are:

| Array index | Direction on the sky | Coordinate |
| --- | --- | --- |
| row 0 | top, North | largest `ddec` |
| row increases | downwards, towards South | `ddec` decreases |
| column 0 | left, East | largest `dra` |
| column increases | rightwards, towards West | `dra` decreases |

The origin `(0, 0)` is at the *centre of the centre pixel* (for an even number of pixels, the corner shared by the four central pixels), at index $(n-1)/2$ along each axis. A pixel's offset along one axis is $((n-1)/2 - \mathrm{index})\,h$ for pixel size $h$ in mas. Both `image_coordinates` and [`pixel_offsets`][virgil._geometry.pixel_offsets] follow this rule.

[`render`][virgil.models.SourceModel.render] draws a model on `npix` × `npix` pixels spanning `fov_mas`, so the pixel scale is `fov_mas / npix`. The pixel scale of an [`Image`][virgil.models.Image] component is set directly by `pixel_scale_mas`. Both use the orientation above. To check it, put a [`PointSource`][virgil.models.PointSource] at `dra=10` on a 64-pixel, 64 mas grid: the brightest pixel is in row 31, column 21, to the left of centre and level with it. With `ddec=10` instead it is in row 21, column 31, above the centre.

If you plot a sky image yourself, make sure East ends up on the left and North at the top. The plotting functions in [`virgil.plotting`][virgil.plotting] do this for you however the axes were built. The project's regression tests assert the position of the brightest pixel for a known offset or position angle (for example `test_gaussian_disk_render_uses_interferometric_image_orientation`), because a mirrored image looks perfectly sensible and passes any test that only checks the image is finite.

## Baselines and the Fourier sign

A baseline is the vector between two telescopes, projected onto the plane perpendicular to the line of sight. Its components are $u$ (East-West) and $v$ (North-South), in metres. Models take `u`, `v` and a wavelength in metres, and divide: the **spatial frequencies** $u/\lambda$ and $v/\lambda$ are then in cycles per radian. They are converted with $1\,\mathrm{mas} = \pi/(180 \times 3600 \times 1000)$ rad when they meet a position in milliarcseconds.

By the van Cittert–Zernike theorem, the complex visibility is the Fourier transform of the sky brightness $I(x, y)$, normalized to 1 at zero baseline. virgil's sign convention is

$$
V(u, v) = \frac{\iint I(x, y)\, e^{-2\pi i (u x + v y)}\,dx\,dy}{\iint I(x, y)\,dx\,dy},
$$

where $x$ is `dra` and $y$ is `ddec`, both converted to radians, and $u, v$ are in cycles per radian. This is the convention of `offset_phase` (in `virgil._geometry`), and every model and every pixel-image Fourier transform in the package follows it. A point source at $(\mathrm{dra}, \mathrm{ddec})$ therefore has

$$
V = \exp\left[-2\pi i\,(u\,\mathrm{dra} + v\,\mathrm{ddec})\right],
$$

with the offsets converted to radians.

**What a positive phase means.** The phase of a point source is *minus* $2\pi$ times the dot product of the baseline and the offset, in wavelengths. So a source displaced to the East ($\mathrm{dra} > 0$) seen on a baseline with $u > 0$ has a *negative* visibility phase. A source at $\mathrm{dra} = 10$ mas on $u = 5$ m at 1 µm gives $-1.523$ rad, as it should. A positive phase means the source is displaced the other way along the baseline: towards negative $u$ (West) for an East-West baseline.

**Where $u, v$ come from in OIFITS.** An OIFITS file lists, for each baseline, `UCOORD` and `VCOORD` in metres and the two telescopes in `STA_INDEX`. [`read_oifits`][virgil.oifits.read_oifits] takes `UCOORD` and `VCOORD` as they are and does not re-sign them, so virgil's convention only gives the right answer if the file's $(u, v)$ were produced with the same one: the vector runs from the first station of `STA_INDEX` to the second, $u$ points East and $v$ North, and the phase is $\arg V$ with the sign of the equation above. The test `tests/test_pa_round_trip.py` builds files this way, in the layout of several instruments, and checks that a binary comes back at its true PA and not 180° away. Real data tests the *pipeline*: if an instrument's reduction conjugates its phases, or uses the opposite baseline direction, a companion appears on the opposite side of the primary, and only an observation of a binary with a known orbit can tell. Treat the PA of a first fit of a new instrument's data as unverified until it has been compared with a known one.

Two further details. A closure triangle $(a, b, c)$ that needs the baseline $(a, b)$ when the file stores only $(b, a)$ uses the conjugate: the reader adds a sample at the triangle leg's own $(u, v)$, so the model's visibility there is evaluated directly. A baseline stored in neither orientation gets a flagged sample in the same way, at the leg's $(u, v)$ from the `OI_T3` row (`U1COORD`, `V1COORD` for $(a, b)$, `U2COORD`, `V2COORD` for $(b, c)$ and their sum for $(a, c)$), with a warning that the V² coverage is incomplete. Baselines are looked up in the visibility table of the same array (`ARRNAME`) and instrument (`INSNAME`), or else of another instrument of that array with identical wavelengths. And AMIGO's mixed-DISCO products are stored with the opposite sign of $(u, v)$, which [`load_oi_data`][virgil.amigo.load_oi_data] negates on reading.

## Observables

`OIData` holds two kinds of observable, a visibility channel and a phase channel, each with an uncertainty.

**Visibilities** come in three forms, selected by `vis_mode`:

| `vis_mode` | Model value | Notes |
| --- | --- | --- |
| `"v2"` | $\lvert V \rvert^2$ | squared visibility, the commonest OIFITS product |
| `"amp"` | $\lvert V \rvert$ | visibility amplitude |
| `"logamp"` | $\ln \lvert V \rvert$ | log-amplitude |

By default (`"auto"`) the data stay in the form they were supplied in. If you ask for a different one, the data and their errors are converted. The errors are propagated linearly, $\sigma_f = \lvert f'(V)\rvert\,\sigma_V$, but because the derivative of $\sqrt{V^2}$ or $\ln V^2$ blows up near zero, where noisy data can be negative, the derivative is evaluated with the data floored at their own uncertainty. For squared visibilities converted to amplitudes the error is $\sigma_{V^2} / (2\sqrt{\max(V^2, \sigma_{V^2})})$. Converting between forms is an approximation, and low-signal data are better fitted in the form they were measured in.

**Phases** are one of:

- **closure phases** (`cp_flag=True`), for a triangle of telescopes $(a, b, c)$:
  $$
  \varphi_{abc} = \varphi_{ab} + \varphi_{bc} - \varphi_{ac},
  $$
  where $\varphi_{ab}$ is the phase of the visibility on the baseline $(a, b)$. The three legs are stored as the sample indices `i_cps1`, `i_cps2` and `i_cps3`, so the closure phase is `phase[i_cps1] + phase[i_cps2] - phase[i_cps3]`, the third leg being subtracted. [`cp_indices`][virgil.oidata.cp_indices] builds the indices from station numbers, and [`closure_phases`][virgil.oidata.closure_phases] evaluates the sum. The result is wrapped into $[-\pi, \pi)$. Closure phases are unaffected by any telescope-dependent phase error, and by any shift of the whole source, which adds a phase linear in the baseline to every visibility and cancels in the sum;
- **absolute phases** (`cp_flag=False`), the phase $\arg V$ of each sample, which need a phase reference and so are rarer.

Phases are **radians inside virgil**, including every `phi` and `d_phi` attribute and the output of `closure_phases`. OIFITS stores phases in degrees, so [`read_oifits`][virgil.oifits.read_oifits] converts on reading (from the column's unit, assuming degrees if it is missing). [`write_oifits`][virgil.oifits.write_oifits] does **not** convert: its tables hold phases and phase errors in degrees, as OIFITS stores them, so convert `phi` and `d_phi` with `numpy.rad2deg` before writing (otherwise both come back about 57 times too small). A dictionary passed to `OIData` directly takes `phi_unit="deg"` for degrees.

**Wrapping and the likelihood.** A phase is an angle, so a model phase of $\pi - \epsilon$ and a data phase of $-\pi + \epsilon$ agree closely. For unprojected phases the residual $\Delta$ (model minus data) therefore enters the likelihood as the chord

$$
r = \frac{2\sin(\Delta/2)}{\sigma},
$$

which is $\Delta/\sigma$ for small $\Delta$. Its square, and so the likelihood, is unchanged when $\Delta$ changes by $2\pi$ and smooth where the phase wraps (the chord itself changes sign, so it is the squared residual, not the residual vector, that is periodic). This is a von Mises likelihood with concentration $1/\sigma^2$. `OIData.residuals` instead wraps the difference into $[-\pi, \pi)$, which is for display; fits and likelihoods never use it, but use [`whitened_residuals`][virgil.likelihood.whitened_residuals].

**Flags.** A sample is *flagged* when its `FLAG` is set in the file, or its value or uncertainty is not finite (the `vis_flag` and `phi_flag` masks mark bad samples with `True`). Flagged samples are dropped from the observables; the arrays `u`, `v` and `wavel` keep every sample. `vis_index` lists the visibility samples that were kept, and for absolute phases `phi_index` the phase samples (each is `None` when none was dropped). Flagged closure phases are instead removed from `phi`, `d_phi` and the `i_cps*` arrays themselves, so `phi_index` stays `None` for closure phases even when some were dropped.

## Closure phases with four or more telescopes

With three telescopes there is one triangle and one closure phase per frame and wavelength. With $N$ telescopes there are $N(N-1)(N-2)/6$ triangles, but only $(N-1)(N-2)/2$ of them are independent: 3 of the 4 triangles of four telescopes, 10 of the 20 for six. Triangles that share a baseline have noise in common, so treating them as independent counts the same information more than once, and makes the fits too confident.

virgil handles this by default. For each frame and wavelength it groups the triangles that share baselines, keeps only the independent combinations, and whitens them with a covariance following Kammerer et al. (2020, A&A 644, A110). That model assumes equal noise on every baseline phase, which gives a correlation of $\pm 1/3$ between two triangles that share a baseline, with the sign set by whether the shared baseline enters both triangles in the same sense, and keeps each triangle's own reported error on the diagonal. It is an approximation: when the errors on a group's triangles differ a lot, the true noise is not exactly of this form, and the $\chi^2$ is slightly off its nominal distribution. The details and the size of the effect are in the docstring of `virgil._closure`.

The practical consequences are that [`n_independent`][virgil.oidata.OIData.n_independent] counts the observables that remain, and so is what to use for degrees of freedom; and that [`whitened_residuals`][virgil.likelihood.whitened_residuals] returns one residual per independent combination, which for four or more telescopes are not the original triangles, followed by the penalty residuals described below. Their likelihood cannot use the chord $2\sin(\Delta/2)$, because a chord changes sign when $\Delta$ changes by $2\pi$, which is harmless in one square but not in the cross terms of a correlated combination (the $\chi^2$ would jump wherever a residual crosses $\pm\pi$, and gradient-based fits of high-S/N four-telescope data stall there). Instead the sines $s_i=\sin\Delta_i$, which are smooth and $2\pi$-periodic, are whitened with that covariance, giving the independent combinations, and each closure phase adds one uncorrelated periodic penalty residual $q_i/\sigma_i = 2\sin^2(\Delta_i/2)/\sigma_i$. For small residuals $\chi^2 = \Delta^\mathsf{T}C^{-1}\Delta$ up to $O(\Delta^3)$ corrections, the correlated Gaussian; at $\Delta_i=\pi$ the sines vanish, which alone would be a false minimum, and the penalty there is $(2/\sigma_i)^2$, which removes it. The $\chi^2$ is continuous and smooth everywhere. The penalty rows carry no normalization, so the Gaussian normalization of the whitened rows is kept. `whitened_residuals` therefore has [`n_residuals`][virgil.oidata.OIData.n_residuals] entries, `n_independent` plus one per closure phase, while `n_independent` is still the number to use for degrees of freedom. For the Gaussian-process image prior that goes alongside these likelihoods, see [Gaussian processes and information field theory](gp_and_ift.md).

## Likelihood normalization

[`model_loglike`][virgil.likelihood.model_loglike] is $-\tfrac12\sum r^2$ for the residuals of [`whitened_residuals`][virgil.likelihood.whitened_residuals], plus each density's normalization, which matters when the errors are fitted:

- Visibilities and projected (kernel or DISCO) phases are Gaussian: $-\log\sigma - \tfrac12\log 2\pi$ per residual.
- An uncorrelated unprojected phase is von Mises with concentration $\kappa = 1/\sigma^2$, with the exact normalizer $-\log 2\pi - \log i_{0e}(\kappa)$. This is the Gaussian one for $\sigma \ll 1$, but stays a normalized density on the circle when $\sigma$ is large (for example a fitted `phi_error`). virgil returns it as the effective error $\sqrt{2\pi}\,i_{0e}(\kappa)$.
- Correlated closure phases keep the Gaussian normalization, their small-$\sigma$ limit, since a correlated circular density has no closed-form normalizer. Their periodic penalty rows carry the effective error $1/\sqrt{2\pi}$, so they add nothing to it, and the whitened rows carry effective errors whose log-sum is $\tfrac12\log$ of the covariance's pseudo-determinant.
- With gains or closure-phase offsets, the visibility (or phase) covariance is whitened with the gains marginalized, and the normalization includes $\tfrac12\log\det$ of that covariance, which depends on the model. The widths are the error terms `vis_gain_<group>` and `phi_offset_<group>`; see [`virgil.gains`][virgil.gains].
- Extra observables follow the phases, each block whitened by itself (see [`virgil.observables`][virgil.observables]).

## Fluxes

Everywhere in virgil, `flux` is a **relative weight**, and it is never an absolute brightness. This is forced by the data: a visibility is normalized to 1 at zero baseline, so only ratios of fluxes can be measured.

- In a [`System`][virgil.models.System] the visibility is the flux-weighted mean $V = \sum_i f_i V_i / \sum_i f_i$. Each component is a shape normalized to unit flux, and `flux` is how much of the total it contributes.
- Keep one **reference component**, usually the star, at `flux=1`, and fit the others relative to it. If every flux were free, scaling them all by the same factor would change nothing, and the fit would have a degeneracy. A companion's `flux` is then its companion/star flux ratio: 0.01 for a companion 100 times fainter.
- The two binary models, [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian] and [`BinaryModelAngular`][virgil.models.BinaryModelAngular], keep a historical convention that agrees with the above: their `flux` *is* the companion/primary ratio, and the primary is implicitly 1. They are the only places where a `flux` is a ratio instead of a weight, and the two meanings coincide in a `System` where the primary has `flux=1` (`BinaryModelCartesian.to_system` gives the equivalent `System`).
- Fluxes are non-negative. [`Component`][virgil.models.Component]s and [`System`][virgil.models.System]s reject negative fluxes given as numbers when they are built; values changed later with `set`, or traced inside a fit, are not checked, and the two legacy binary models do not check at all, so priors and grid axes must not allow negative fluxes. The one deliberate exception is [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid], whose best-fitting flux at each position may be negative, as the Ruffio et al. upper limits need.

Reports and plots follow the astronomer's convention instead. **Contrast** is primary/companion, so 100 for a flux ratio of 0.01, and **Δmag** is $2.5\log_{10}(\text{contrast})$, 5 mag in that example. [`flux_to_contrast`][virgil.limits.flux_to_contrast], [`contrast_to_flux`][virgil.limits.contrast_to_flux], [`flux_to_delta_mag`][virgil.limits.flux_to_delta_mag] and [`delta_mag_to_flux`][virgil.limits.delta_mag_to_flux] convert between them, and the plotting functions take `units="flux"`, `"contrast"` or `"delta_mag"`. A model parameter is therefore never called `contrast`.

**Chromatic fluxes.** A component's `flux` may be a spectrum from [`virgil.spectra`][virgil.spectra] instead of a number, which gives its weight at each wavelength (SPARCO; Kluska et al. 2014). The `ratio` of [`PowerLaw`][virgil.spectra.PowerLaw] or [`BlackBody`][virgil.spectra.BlackBody] is the flux weight *at the reference wavelength* `wavel0`, in metres (default $1.65\,\mu$m, H band), and the other wavelengths follow from the spectrum's shape: $\mathrm{ratio}\,(\lambda/\lambda_0)^{\mathrm{index}}$ for the power law (index $-4$ for a Rayleigh–Jeans star in $F_\lambda$), and a Planck curve scaled to the same value at $\lambda_0$ for the black body. The reference flux is also what is used when a model is rendered, since an image has no wavelength. Choose `wavel0` in the middle of your data, so that `ratio` is something you can interpret.

**Continuum plus lines, and excesses.** [`Sum`][virgil.spectra.Sum] adds named spectra, so that a continuum and its emission or absorption lines are one flux, reached by name like the components of a `System` (`"star.flux.brg.amplitude"`):

```python
from virgil.spectra import GaussianLine, PowerLaw, Sum

flux = Sum(
    continuum=PowerLaw(0.3, index=-4.0, wavel0=2.2e-6),
    brg=GaussianLine(0.1, line_wavel=2.1661e-6, fwhm=2.0e-9),
)
```

A line's `amplitude` is its peak flux (negative for absorption), and `fwhm` is in metres. Every spectrum's reference flux is its value at `wavel0`, and only the *total* must be non-negative, so a part may be negative where the others cover it. For a spectrum measured channel by channel, [`Nodes`][virgil.spectra.Nodes] interpolates (linearly or by a natural cubic spline) between free fluxes at fixed wavelengths. `Nodes(excess, wavel, outside=0.0)` is an excess that vanishes outside its window, so it can sit on a continuum in a `Sum`. Smoothness is a prior on the node values, not a property of the class. `System.total_spectrum(wavel)` gives the summed flux of the scene (the model of an OI_FLUX spectrum, up to a grey scale), and `OIData.select` restricts data to wavelength windows (`wavel_min`, `wavel_max`, `ranges=`, `exclude=`) and to some observables (`observables=`), e.g. the continuum on either side of a line, or closure phases alone.

## Times and frames

Times are Modified Julian Dates in days. At MJD 60000, a float32 number can only change in steps of 0.0039 d (about 5.6 minutes), and a multi-year campaign cannot be resolved to better than that: a binary's orbital motion between nights would be quantized. virgil therefore never keeps an absolute MJD in a JAX array.

`OIData` stores each sample's time as a static float64 `t_ref` (the earliest time in the data) plus a float32 array `dt` of days since `t_ref`. A difference of up to 1000 days is resolved to about 5 seconds. [`OIData.mjd`][virgil.oidata.OIData.mjd] rebuilds the absolute float64 times, as a NumPy array, for inspection and plotting. Models of time should be written in terms of `dt`, never `mjd`.

The data also carry a `frame` number per sample. A **frame** is one exposure of one instrument, which is the unit within which closure phases make sense: the three baselines of a triangle must be measured together. [`read_oifits`][virgil.oifits.read_oifits] decides which rows belong to one exposure from their `MJD` (within about 9 s) and `INSNAME`, and the baselines tied together by closure phases. With `frame_mjd="mean"` (the default) every sample in a frame is given the mean time of the frame's rows, which is what you want when a pipeline stamps the closure-phase and visibility tables with slightly different times; `frame_mjd="row"` keeps each row's own time.

For analyses that treat nights separately, [`OIData.epochs`][virgil.oidata.OIData.epochs] labels each sample with an epoch number: a run of frames with no gap longer than `gap_days` (default 0.5 d), numbered from 0 in time order. A frame is never split between epochs. [`OIData.split_by_epoch`][virgil.oidata.OIData.split_by_epoch] returns one `OIData` per epoch, which is how you would fit a binary's position night by night. It does not work on projected (kernel or DISCO) observables.

For orbit fits across epochs, [`Epochs`][virgil.epochs.Epochs] groups datasets into named epochs and evaluates a moving scene once per dataset, at the mean time of its samples (a **snapshot**), instead of at every sample's own time. That is exact for data with one time per dataset and an excellent approximation whenever the scene moves by much less than the resolution $\lambda/B$ within a dataset, as a binary with a period of months or more does over a night; [`Epochs.spread_days`][virgil.epochs.Epochs] says how far each dataset's samples lie from its snapshot. Each snapshot is a static model, so a binary on its orbit ([`OrbitalBinary`][virgil.models.OrbitalBinary]) keeps the fast binary path.

An orbit's `t_ref` is the zero of its `dt_peri`. Give it a time near the data (e.g. `KeplerOrbit(..., t_ref=60500.0)`): with the default `t_ref=0` and MJD times, `dt_peri` is counted from MJD 0, and virgil warns.

## Orbits

The conventions below are implemented by [`virgil.orbits`](api/orbits.md). The tutorial [Orbits from interferometric data](orbit_fitting.md) uses them end to end, fitting an orbit jointly to every epoch's visibilities and closure phases.

An orbit gives the position of a **secondary** star relative to a **primary** (or reference) star, which sits at the origin and is the scene's reference component at `flux=1`. It need not be the more massive star. The relative position is $\mathbf{r} = (\mathrm{dra}, \mathrm{ddec}, dz)$ of the secondary minus the primary.

The primary is also the scene's phase reference: the visibilities of a scene with an orbit are those of an image centred on the primary, not on the photocentre or the barycentre. A component tied to the orbit by [`Attached`][virgil.models.Attached] sits on the secondary by default, on the primary with `anchor="primary"`, or at a fraction of the way from the primary to the secondary (e.g. `anchor=q / (1 + q)` for the barycentre).

- The first two components are the sky offsets above, East and North, in mas.
- The third axis, `dz`, is positive **away from the observer**. With East, North and away-from-us, the axes form a right-handed set. So $d(dz)/dt$ has the sign of the secondary's radial velocity relative to the primary: positive means it is receding.

The angles follow the usual visual-binary conventions, with one care for each:

| Symbol | Name in code | Definition |
| --- | --- | --- |
| $i$ | `inc` | 0° to 180°. $i < 90°$ means the position angle *increases* with time (counter-clockwise on a North-up sky image) |
| $\Omega$ | `Omega` | the PA of the **ascending node**, defined as the node where the secondary *recedes* ($dz$ increasing) |
| $\omega$ | `omega` | the **secondary's** argument of periastron, from the ascending node, in the direction of motion |

The $\omega$ here is the visual-binary one, for the secondary relative to the primary. The spectroscopic convention, which describes the *primary's* motion, differs by 180°: $\omega_{\rm spec} = \omega - 180°$. Orbits taken from a radial-velocity paper need this correction. Other symbols: `period` in days, `a_mas` the angular semimajor axis of the relative orbit in mas, and `dt_peri` the time of periastron minus `t_ref` in days, relative for the float32 reason in the previous section.

**What the data cannot tell apart.** These degeneracies are properties of the geometry, not bugs.

1. $(\Omega, \omega) \to (\Omega + 180°, \omega + 180°)$ gives the same sky positions and flips the sign of $dz$. A visual orbit cannot distinguish them. Radial velocities, or a scene component that is not front-back symmetric, can.
2. $\omega \to \omega + 180°$ alone sends $\mathbf{r} \to -\mathbf{r}$ at all times. This is the same as swapping which star is the reference: a "which star is the primary" error and an "$\omega$ of which star" error are the same 180° flip.
3. $i \to 180° - i$ reverses the sense of rotation on the sky.
4. For a nearly face-on orbit the positions depend on $\Omega$ and $\omega$ only through their sum, so astrometry measures the sum well and each separately poorly.

**Swapping primary and secondary** is the same thing seen from the data. A binary with the companion at $\mathbf{r}$ and flux ratio $f$ looks, to the visibilities, like one with the companion at $-\mathbf{r}$ and flux ratio $1/f$: only the origin of the image moves (from the primary to the other star), so every squared visibility and closure phase is identical, and the visibilities differ only by a phase linear in $u$ and $v$. This was checked with `BinaryModelCartesian`: `(dra, ddec, f)` and `(-dra, -ddec, 1/f)` give the same $V^2$ and closure phases to float32 rounding, and a visibility ratio of $\exp[+2\pi i(u\,\mathrm{dra} + v\,\mathrm{ddec})]$. So the choice of which star is the reference is a convention, to be fixed once, by requiring the reference to be the brighter star in the band, say, and every PA, flux ratio and $\omega$ read afterwards must follow it.

## Which priors?

virgil's examples use the Jeffreys prior under the group that acts on each parameter, unless there is strong information to the contrary (a measurement, a population model). The prior then does not depend on how the parameter is written: a log-uniform prior on a period is also log-uniform in the frequency, and an isotropic orientation stays isotropic however it is parametrized. Every prior also has finite, stated bounds, which should contain the plausible values with a margin, and a scale's lower bound is nonzero, since `LogUniform(0, ...)` is undefined.

| Parameter | Group | Prior |
|---|---|---|
| Position offsets `dra`, `ddec`; time of periastron; phase offsets; spectral index; log gains | translation | `Uniform` |
| Fluxes and flux ratios; amplitudes; angular sizes and widths; periods; semi-major axes; noise and gain widths | scaling | `LogUniform`, with a nonzero lower bound |
| Position angle, node, periastron, spin, longitude | rotation of the circle | `AngleVector()` (no edge at 0°/360°), or `Uniform` over exactly one period |
| Inclination of an orbit or spin axis | rotation of the sphere | [`IsotropicInclination`][virgil.priors.IsotropicInclination] (uniform in cos i); over (0, 90) when only \|cos i\| is identifiable |
| Latitude of a point on a sphere | rotation of the sphere | [`IsotropicLatitude`][virgil.priors.IsotropicLatitude] (uniform in sin lat) |
| Eccentricity | none | `Uniform`, as the interim prior |
| Kipping's limb-darkening $(q_1, q_2)$ | none (uniform over the physical triangle) | `Uniform(0, 1)` each |

Two pitfalls follow. A `Uniform` prior over more than one period of an angle, such as (−360°, 720°), counts the circle several times. And a `Uniform` prior on a flux or a size favours large values, so the posterior depends on the upper bound. The [binary search](binary_search.md) and [hierarchical inference](hierarchical_inference.md) tutorials are the templates: they sample the companion's flux in log space, and the second infers a population of fluxes.

For orbits, [`orientation_priors`][virgil.orbits.orientation_priors] gives the node and periastron as angle vectors.

### Passing priors to functions

Prefer numpyro distributions: `dist.LogUniform(low, high)` and `dist.Normal(mean, sd)` are accepted wherever a prior is a parameter (`fit`, `numpyro_model`, `noise=`, and `linear_flux_grid(prior=)`). The exception is the analytically marginalized Gaussians, where a plain `(mean, sd)` pair (scalars, or one per instrument or telescope) stands for $N(\text{mean}, \text{sd}^2)$: `RVData.term(marginalize_offsets=(mean, sd))` and `with_flux_scale(scale=(mean, sd))`. A marginalized prior is never defaulted, so `marginalize_offsets=True` is an error.

### Angle priors without a wrap {#angle-vectors}

A prior on an angle $\theta$ with support on an interval, such as `Uniform(0, 360)`, numpyro's `VonMises` or [`AxialVonMises`][virgil.orbits.AxialVonMises], reaches samplers and [`fit`][virgil.fitting.fit] through a bijection onto the real line, so an angle whose posterior straddles the end of the interval meets a wall there. [`AngleVector`][virgil.angles.AngleVector] removes the wall: it samples a vector $v = r(\cos\theta, \sin\theta)$ in $\mathbb{R}^2$ at the site `"<path>_vec"`, and the model receives $\theta = \operatorname{atan2}(v_2, v_1)$ in degrees in $[0°, 360°)$. [`numpyro_model`][virgil.likelihood.numpyro_model] records $\theta$ as a deterministic site `"<path>"`, and `fit` reports it under `"<path>"`, with the vector under `"<path>_vec"`.

The density of $v$ is

$$p(v) = \frac{e^{-(r-1)^2 / 2s^2}}{Z_s}\, p(\theta), \qquad Z_s = \int_0^\infty r\, e^{-(r-1)^2/2s^2}\,dr,$$

so $r$ and $\theta$ are independent and $\theta$ has exactly the density $p(\theta)$ (per radian): uniform, $1/2\pi$, by default; a von Mises, $\exp[\kappa(\cos(\theta - \mu) - 1)] / (2\pi\, \mathrm{i0e}(\kappa))$, given `mean` and `kappa`; or, with `axial=True`, the von Mises of $2\theta$, so that $\theta$ and $\theta + 180°$ are equally likely. The radius is a nuisance with a ring prior, whose mode is the unit circle, so a maximum a posteriori fit keeps away from the origin, where $\theta$ is undefined; the ring width $s$ (`ring_width`, default 0.25) does not change the prior on $\theta$.

Every term has a least-squares form, so Levenberg–Marquardt and [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] take it unchanged, and the density is normalized in $\mathbb{R}^2$, so evidences stay normalized. The ring's residual is $(r - 1)/s$. A von Mises enters as a chord, $\sqrt{\kappa}\,(\hat v - \hat m)$ with $\hat v = v / r$ and $\hat m = (\cos\mu, \sin\mu)$: half its square is $\kappa(1 - \cos(\theta - \mu))$, the von Mises exponent, just as virgil's phase residuals $2\sin(\Delta/2)/\sigma$ are chords with $\kappa = 1/\sigma^2$ (see [`whitened_residuals`][virgil.likelihood.whitened_residuals]). The axial chord is the same on the doubled direction, $(\cos 2\theta, \sin 2\theta) = (x^2 - y^2, 2xy)/r^2$.

The construction follows Octofitter's `UniformCircular` (Thompson et al. 2023, AJ 166, 164) and exoplanet's `Angle` (Foreman-Mackey et al. 2021, JOSS 6, 3285), which sample $v \sim N(0, I)$ and so peak at the origin; the ring and the chord priors are virgil's.

### Azimuthal harmonics {#azimuthal-harmonics}

Every azimuthal profile of a [`PolarDisk`][virgil.models.PolarDisk] is, at each radius, a real function of the azimuth $\theta$ (in the plane of the disk, from the projected major axis towards `pa + 90`) with the Fourier series

$$
\frac{I(\theta)}{I_0} = \sum_{m=-M}^{M} I_m e^{im\theta}, \qquad I_0 = 1, \quad I_{-m} = \bar I_m,
$$

the *image harmonics* $I_m$ (complex, $m = 0 \dots M$, normalized to the azimuthal mean). All the forms map to and from them, with [`to_image_harmonics`][virgil.harmonics.to_image_harmonics] and [`from_image_harmonics`][virgil.harmonics.from_image_harmonics] (or `Harmonics.convert`):

- **linear**: $1 + \sum_m c_m \cos m\theta + s_m \sin m\theta$, so $c_m = 2\,\mathrm{Re}\,I_m$ and $s_m = -2\,\mathrm{Im}\,I_m$ (exact);
- **positive**: $|h(e^{i\theta})|^2 / \sum_j |h_j|^2$ with $h(z) = 1 + h_1 z + \dots + h_M z^M$, whose roots are all outside the unit circle. It is built from complex reflection coefficients $k_j$ with $|k_j| < 1$ by the Levinson step-up recursion $h^{(j)}_i = h^{(j-1)}_i + k_j \overline{h^{(j-1)}_{j-i}}$, and $k = w \tanh|w| / |w|$ from unconstrained latents $w_j \in \mathbb{C}$. Then $I_m = \sum_j h_{j+m} \bar h_j / \sum_j |h_j|^2$ (exact), and the other way round is a spectral factorization (`from_image_harmonics`);
- **von Mises**, $\exp[\kappa\cos(\theta - \theta_0)]$, truncated at the degree of the model;
- the legacy amplitude and azimuth pairs `az_amps`/`az_pas` of [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim] and [`TruncatedCone`][virgil.models.TruncatedCone].

Use the conversions to report any model in one convention, to warm-start a positive-form fit from a linear or parametric one, and to check models against each other. The positive form's reflection coefficients are an internal parametrization: report the image harmonics.

References: Dumitrescu 2007, *Positive Trigonometric Polynomials and Signal Processing Applications* (Springer, doi:10.1007/978-1-4020-5125-8), for the Fejér–Riesz theorem, spectral factorization and the Schur–Levinson recursion; Barndorff-Nielsen & Schou 1973, J. Multivariate Anal. 3, 408 (doi:10.1016/0047-259X(73)90030-4), for the unconstrained parametrization of a minimum-phase polynomial by reflection (partial autocorrelation) coefficients, here with a complex tanh map.

### Hierarchical scales {#hierarchical-scales}

[`hierarchical_scales`][virgil.priors.hierarchical_scales] draws a set of scale factors, say one error scale per night, from a log-normal population: $\log s_k \sim N(\log \mu, \sigma^2)$, with the median $\mu$ and spread $\sigma$ sampled too under log-uniform priors. The population pulls poorly constrained members towards the typical value, and $\mu$ and $\sigma$ say how well the stated errors describe the instrument.

The **centred** form (the default) samples $u_k = \log s_k$ and adds the population density $N(u_k \mid \log\mu, \sigma)$. It suits members that the data measure well, such as the error scales of epochs with tens of closure phases or more: $u_k$ is then nearly independent of the hyperparameters. The **non-centred** form samples $z_k \sim N(0, 1)$ with $s_k = \mu\,e^{\sigma z_k}$, and suits members the data barely constrain. With well-measured members it makes a curved ridge, $z_k \propto 1/\sigma$, on which NUTS diverges (in a test with eight members measured to 7%, 10 divergences in 4000 draws against none centred). The [hierarchical inference](hierarchical_inference.md) tutorial uses the same idea for a population of fluxes.

## Priors and the MAP

virgil's default priors are the invariant (Jeffreys) measures of the groups acting on each parameter: uniform for locations, log-uniform for scales, and isotropic for orientations (uniform in $\cos i$ for an inclination). A maximum a posteriori point is not invariant under a change of variables, because a density picks up a Jacobian. The mode of `LogUniform`'s density $1/x$ in $x$ is at the lower bound, so a fit in $x$ would pull every scale down, although nothing in the prior prefers small scales.

[`fit`][virgil.fitting.fit] therefore optimizes each such parameter in its *flat coordinate*, the coordinate in which its prior is uniform: $\log x$ for `LogUniform(a, b)`, on $[\log a, \log b]$; $\cos i$ for an isotropic inclination; $\sin(\mathrm{lat})$ for an isotropic latitude (any prior with a `flat_coordinate()` method); and the parameter itself for `Uniform`. There the prior is constant and adds nothing to the loss, so the MAP is the maximum of the likelihood (times any other priors) inside the prior's range, and Levenberg–Marquardt applies. Other priors (`Normal`, `Beta`, `HalfNormal`, ...) have no flat coordinate and are evaluated in the model's own parameters. Fitted values are always reported in the model's own parameters.

A Gaussian approximation at the fit should be taken in the same flat coordinate, and carried to the model's parameters by the delta method, $\sigma_x = |{\rm d}x/{\rm d}u|\,\sigma_u$ (for a log-uniform scale, $\sigma_x = x\,\sigma_{\log x}$). [`laplace_cov`][virgil.inference.laplace_cov] and [`fisher`][virgil.inference.fisher] are curvatures of the likelihood alone, in the model's parameters, so they do not depend on this. By default [`numpyro_model`][virgil.likelihood.numpyro_model] lets numpyro map each prior's support to NUTS's unconstrained coordinates. With `flat_coordinates=True` it samples in the same flat coordinates as `fit` instead (NUTS moves in the logit of $u$ on its interval, where the prior is a logistic density), and [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] takes the same `flat_coordinates` to give its mass matrix in those coordinates. Its matrix includes the curvature of each coordinate's own prior and transform (the logistic transform of a bounded or flat-coordinate prior adds $2\sigma(1-\sigma)$ in the logit's coordinate), as the sampler's potential has it. Pass the same value to both: a mismatched matrix is never adapted, and [`chain_init_params`][virgil.likelihood.chain_init_params] (given `mass=`) raises on it. Samples are always the model's own parameters, under the same site names.

## Precision

virgil never switches on JAX's 64-bit mode globally. All library code runs in float32 by default, and is written to give correct results in float64 too. The fitting entry points, such as [`fit`][virgil.fitting.fit], instead run their optimization in a local float64 context: they cast the model, data and priors to float64 on the way in, and restore the setting on exit. Pass `dtype="float32"` to `fit` for the faster, less precise version. The helper that does this is `virgil._precision.run_in`, which you will see in the source.

Two things are worth knowing. First, float32 is the reason absolute times are stored as `dt`, as described above. Second, matrix products and Fourier transforms over pixels use the highest matmul precision, because on A100 and H100 GPUs the default silently uses TF32 and a relative error of about $10^{-3}$.
