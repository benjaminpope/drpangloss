<!-- AUTO-GENERATED FROM notebooks/polar_disk_rings.ipynb by scripts/sync_tutorial_docs.py. -->
# A planet in a gap between inclined rings

[`PolarDisk`](polar_disk.md) describes a disk by a radial profile $I_0(r)$ in the plane of the disk, tilted by `inc` and rotated by `pa`. Its visibility is the Hankel transform of the profile, normalized to one at zero baseline,

$$
V(q) = \frac{\int I_0(r)\, J_0(2\pi q' r)\, r\,\mathrm{d}r}{\int I_0(r)\, r\,\mathrm{d}r},
$$

with $q'$ the baseline's spatial frequency in the disk plane, so the profile can be a parametric ring or a free function. Two optional extras are used here: an azimuthal harmonic that makes the disk lopsided, and a spectral index, $I(r,\lambda) = I_0(r)(\lambda/\lambda_0)^{\alpha(r)}$. In the non-parametric form, $\ln I_0$ at $n$ radial nodes is $\sigma L z$ with $LL^\top$ a Matérn correlation matrix of length $\ell$ and $z \sim \mathcal N(0, I)$ the free parameters, so the intensity is positive for any $z$.

We simulate a star with three inclined rings, the outermost lopsided, and a planet in the gap between the second and third, observed with VLTI-like closure phases and squared visibilities at nine wavelengths. We analyze the data twice: with the family that made them (three Gaussian rings and a planet), and with one non-parametric `PolarDisk` whose profile, colour and lopsidedness are Gaussian processes. Both run at the maximum a posteriori (MAP) solution with a Laplace approximation, with no sampling. The reference page [`PolarDisk`](polar_disk.md) lists every option.

## The scene and the data

The star is a 6000 K blackbody point source; all other fluxes are relative to it at 3 microns. The rings are `PolarDisk`s with `GaussianRing` profiles at 3, 7 and 15 mas, with blackbodies of 1500, 900 and 600 K (so the inner ring is bluer), and the outer one has a degree-one harmonic. The planet is a 1000 K `PointSource`. Ring radii are in the disk plane, so we place the planet there: with position angle 30° the projected major axis points along $(\sin 30^\circ, \cos 30^\circ)$ in $(\Delta\alpha, \Delta\delta)$, where sky and plane distances agree, and we put the planet at 11 mas on that axis to the South-West, midway between the rings at 7 and 15 mas, so $(\Delta\alpha, \Delta\delta) = (-5.5, -9.53)$ mas. The first cell checks this by deprojecting. The rings are coplanar, so `scene` hands one `inc` and `pa` to all of them; the same function is fitted below.

The data mimic a low-resolution L-band run with the four unit telescopes: nine channels from 2.4 to 3.6 microns and seven hour angles, giving 378 squared visibilities (error 0.01) and 252 closure phases (error 0.5°), 567 independent data once the closure phases' correlations are accounted for. The beam, drawn in the corner of each image, is about 6 by 3 mas, so the 4 mas between the first two rings is about one resolution element. The longest baseline gives a half-fringe $\lambda/2B$ of 2 to 3 mas, and structure much finer than that is not measured. The image shows the truth at 3 microns, East to the left, with the star saturated to show the outer ring.

```python
import sys
from pathlib import Path

root = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()
)
sys.path.insert(0, str(root / "src"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from tqdm.auto import tqdm

from virgil import (
    BlackBody, GaussianRing, Harmonics, PointSource, PolarDisk,
    PolarKernel, RadialGP, System, fit, gp_priors,
)
from virgil.coverage import vlti_oidata
from virgil.imaging import beam, log_evidence
from virgil.inference import laplace_cov
from virgil.likelihood import whitened_residuals
from virgil.priors import IsotropicInclination
from virgil.plotting import (
    plot_data_model_correlation, plot_model, plot_residual_map,
)

W0 = 3.0e-6  # reference wavelength, m


def scene(inc, pa, r1, s1, f1, t1, r2, s2, f2, t2, r3, s3, f3, t3,
          h_re, h_im, dra, ddec, fp, tp):
    """Star, three coplanar rings and a planet; one inc and pa for the rings."""
    def ring(r0, sigma, flux, temp, **kw):
        return PolarDisk(GaussianRing(r0, sigma), inc=inc, pa=pa, n_r=24,
                         flux=BlackBody(flux, temp, W0), **kw)
    return System(
        star=PointSource(flux=BlackBody(1.0, 6000.0, W0)),
        ring1=ring(r1, s1, f1, t1),
        ring2=ring(r2, s2, f2, t2),
        ring3=ring(r3, s3, f3, t3, harmonics=Harmonics(re=h_re, im=h_im)),
        planet=PointSource(dra=dra, ddec=ddec, flux=BlackBody(fp, tp, W0)),
    )


truth = dict(inc=50.0, pa=30.0, r1=3.0, s1=0.8, f1=0.15, t1=1500.0,
             r2=7.0, s2=1.0, f2=0.25, t2=900.0, r3=15.0, s3=1.8, f3=0.5,
             t3=600.0, h_re=0.5, h_im=0.2, dra=-5.5, ddec=-9.526, fp=0.04,
             tp=1000.0)
truth_model = scene(**truth)


def plane_radius(dra, ddec, inc, pa):
    """Disk-plane radius of a sky offset: rotate to the major/minor axes, de-foreshorten the minor."""
    pa_r = np.radians(pa)
    major = dra * np.sin(pa_r) + ddec * np.cos(pa_r)
    minor = dra * np.cos(pa_r) - ddec * np.sin(pa_r)
    return float(np.hypot(major, minor / np.cos(np.radians(inc))))


print("planet: sky separation "
      f"{np.hypot(truth['dra'], truth['ddec']):.2f} mas, disk-plane radius "
      f"{plane_radius(truth['dra'], truth['ddec'], truth['inc'], truth['pa']):.2f} mas "
      f"(rings at {truth['r1']}, {truth['r2']}, {truth['r3']} mas)")
template = vlti_oidata(
    wavelengths_m=np.linspace(2.4e-6, 3.6e-6, 9),
    hour_angles_h=(-3, -2, -1, 0, 1, 2, 3),
    sigma_v2=0.01, sigma_cp_deg=0.5,
)
data = template.with_model(truth_model, key=jax.random.PRNGKey(0))
fig, ax = plt.subplots(figsize=(5, 4.6))
plot_model(truth_model, fov_mas=60, npix=256, ax=ax, saturate=0.99,
           beam=beam(data), title="Truth at 3 micron")
plt.show()
```

```text
planet: sky separation 11.00 mas, disk-plane radius 11.00 mas (rings at 3.0, 7.0, 15.0 mas)
```

![polar_disk_rings output 3.2](generated/polar_disk_rings_cell003_out02.png)

## The parametric fit

The truth family has 20 parameters: shared geometry, a radius, width, flux and temperature per ring, the outer harmonic's two components, and the planet's offset, flux and temperature. Their priors are invariant ones: `IsotropicInclination` (uniform in $\cos i$) for the inclination, uniform for angles and locations (position angle, harmonic components, planet offsets), and log-uniform for scales (ring radii, widths, fluxes, temperatures). We start 5 percent from the truth, as a local optimizer needs, run `fit` (Levenberg–Marquardt) to the MAP, and take the Laplace covariance with `laplace_cov`.

The fit reaches $\chi^2 = 583$ for 567 independent data on the quoted errors, $\chi^2/N = 1.03$ with no rescaling. Most parameters land within about one Laplace standard deviation of the truth, and the planet's position and flux are precise (0.03 mas, 3 percent). The inner ring is the exception: its radius and flux are 2.4 and 3 standard deviations off (3.19 against 3.0 mas, 0.126 against 0.15), and its width is unresolved, 0.15 mas with a Laplace standard deviation of 0.38 mas against a truth of 0.8 mas. A Gaussian approximation is poor for a scale pressed against its prior edge, which matters for the evidence below.

```python
priors = dict(inc=IsotropicInclination(0, 85), pa=dist.Uniform(-90, 90),
              h_re=dist.Uniform(-1, 1), h_im=dist.Uniform(-1, 1),
              dra=dist.Uniform(-20, 20), ddec=dist.Uniform(-20, 20),
              fp=dist.LogUniform(0.003, 0.3), tp=dist.LogUniform(300, 3000))
for i in (1, 2, 3):
    priors |= {f"r{i}": dist.LogUniform(0.5, 25), f"s{i}": dist.LogUniform(0.1, 4),
               f"f{i}": dist.LogUniform(0.01, 2),
               f"t{i}": dist.LogUniform(300, 3000)}
start = {k: v * (1 + 0.05 * (-1) ** i) for i, (k, v) in enumerate(truth.items())}
par = fit(scene, priors, data, init=start)
names = list(priors)
x_map = np.array([float(par.values[k]) for k in names])
sigma = np.sqrt(np.diag(np.asarray(laplace_cov(x_map, names, scene, data))))
print(f"chi2 = {par.info['chi2'][0]:.0f} for N = {par.info['ndata'][0]} "
      f"independent data (chi2/N = {par.info['chi2'][0] / par.info['ndata'][0]:.2f}), "
      f"converged: {par.info['converged']}")
print(f"{'':>6}{'truth':>9}{'MAP':>9}{'Laplace sd':>12}")
for k, x, s in zip(names, x_map, sigma):
    print(f"{k:>6}{truth[k]:9.3f}{x:9.3f}{s:12.3f}")
```

```text
chi2 = 583 for N = 567 independent data (chi2/N = 1.03), converged: True
          truth      MAP  Laplace sd
   inc   50.000   50.064       0.228
    pa   30.000   30.182       0.285
  h_re    0.500    0.489       0.017
  h_im    0.200    0.201       0.008
   dra   -5.500   -5.498       0.023
  ddec   -9.526   -9.512       0.028
    fp    0.040    0.040       0.001
    tp 1000.000  990.005      27.621
    r1    3.000    3.193       0.079
    s1    0.800    0.151       0.388
    f1    0.150    0.126       0.008
    t1 1500.000 1558.117      80.013
    r2    7.000    6.891       0.089
    s2    1.000    1.113       0.118
    f2    0.250    0.256       0.010
    t2  900.000  937.975      17.245
    r3   15.000   15.009       0.022
    s3    1.800    1.815       0.032
    f3    0.500    0.500       0.009
    t3  600.000  599.646       6.866
```

## The non-parametric fit

Now the model that does not know about rings. One `PolarDisk` has a `RadialGP` profile on 48 nodes between 1.5 and 24 mas, with Matérn-3/2 correlation length $\ell = 1.5$ mas and $\sigma = 3$ on $\ln I_0$. Its `Harmonics(degree=1, prior=PolarKernel(...))` has an amplitude and phase that are Gaussian processes in radius, and its `spectral_index` is another `RadialGP`. The low harmonic degree stops the disk absorbing the planet, a compact, strongly asymmetric source, which keeps its parametric form with the star. `gp_priors` writes the standard-normal priors of every latent. The hyperparameters ($\sigma$, $\ell$, `ell_r`, `ell_theta`) are fixed by hand, not fitted, and the geometry, total disk flux and planet start at the parametric MAP; the geometry is held at those values (uncertainties 0.2° and 0.3°).

The inner edge is at 1.5 mas, below the first ring and a little under the beam. With the edge at 0.5 mas an earlier attempt put about a quarter of the disk's light into an unresolved core that traded off against the star's flux, which the data cannot separate. At 1.5 mas the fit cannot hide light in the centre, though the profile still rises steeply at the innermost nodes and some trade with the star remains; the disk flux is about 7 percent above the truth. The fit reaches $\chi^2/N = 1.20$, worse than the parametric 1.03, and recovers the planet's position to within 0.05 mas and its flux (0.043 of the star) to 8 percent of the parametric 0.040. The disk flux is 0.96, against 0.88 for the fitted rings and a truth of 0.90.

```python
v = par.values
r_in, r_out = 1.5, 24.0
disk = PolarDisk(
    RadialGP(r_in, r_out, sigma=3.0, ell=1.5, n=48),
    harmonics=Harmonics(degree=1, prior=PolarKernel(
        ell_r=3.0, ell_theta=8.0, sigma=0.5, n=10, r_in=r_in, r_out=r_out)),
    spectral_index=RadialGP(r_in, r_out, sigma=3.0, ell=2.0, n=22),
    wavel0=W0, inc=v["inc"], pa=v["pa"], n_r=48,
    flux=v["f1"] + v["f2"] + v["f3"],
)
gp = System(
    star=PointSource(flux=BlackBody(1.0, 6000.0, W0)),
    disk=disk,
    planet=PointSource(dra=v["dra"], ddec=v["ddec"],
                       flux=BlackBody(v["fp"], v["tp"], W0)),
)
gp_priors_ = gp_priors(gp) | {
    "disk.flux": dist.LogUniform(0.05, 5.0),
    "planet.dra": dist.Uniform(-20, 20), "planet.ddec": dist.Uniform(-20, 20),
    "planet.flux.ratio": dist.LogUniform(0.003, 0.3),
    "planet.flux.temperature": dist.LogUniform(300, 3000),
}
gpfit = fit(gp, gp_priors_, data)
gv = gpfit.values
print(f"chi2 = {gpfit.info['chi2'][0]:.0f} for N = {gpfit.info['ndata'][0]} "
      f"independent data (chi2/N = {gpfit.info['chi2'][0] / gpfit.info['ndata'][0]:.2f}), "
      f"converged: {gpfit.info['converged']}, {gpfit.info['steps']} steps")
print(f"planet offset (dra, ddec) = ({gv['planet.dra']:.2f}, {gv['planet.ddec']:.2f}) mas "
      f"(parametric: {v['dra']:.2f}, {v['ddec']:.2f}; "
      f"truth {truth['dra']:.2f}, {truth['ddec']:.2f})")
print(f"planet flux at 3 micron = {gv['planet.flux.ratio']:.3f} of the star "
      f"(parametric: {v['fp']:.3f}; truth 0.040)")
print(f"disk flux at 3 micron = {gv['disk.flux']:.2f} "
      f"(parametric rings: {v['f1'] + v['f2'] + v['f3']:.2f}; truth 0.90)")
```

```text
chi2 = 680 for N = 567 independent data (chi2/N = 1.20), converged: True, 57 steps
planet offset (dra, ddec) = (-5.53, -9.51) mas (parametric: -5.50, -9.51; truth -5.50, -9.53)
planet flux at 3 micron = 0.043 of the star (parametric: 0.040; truth 0.040)
disk flux at 3 micron = 0.96 (parametric rings: 0.88; truth 0.90)
```

## The evidence, parametric against Gaussian process

For the parametric model the Laplace evidence is $\log Z \approx -\tfrac12\chi^2 + \log\pi(\hat\theta) + \tfrac d2\log 2\pi + \tfrac12\log\det\Sigma$, and `log_evidence` computes the equivalent for the Gaussian-process latents, $-\tfrac12\chi^2 - \tfrac12|z|^2 - \tfrac12\log\det(I + J^TJ)$. The two share a normalization, so only their difference means anything, and it is not a calibrated Bayes factor, for two reasons that pull opposite ways. The Gaussian-process value holds every non-latent parameter (disk flux, planet, and geometry, which was tuned on these data) at its MAP with no Occam factor, which favours the Gaussian process. The parametric family contains the truth, which no real analysis can assume, and it marginalizes all 20 parameters, which favours the parametric model. The Laplace approximation is also rough for the width pressed against its prior edge.

We also refit the Gaussian process with a profile length of 4 mas instead of 1.5. This compares two fits with the same held parameters, which is the cleaner use of the evidence, as in MacKay's framework for choosing a hyperparameter.

```python
gpm = gpfit.model
# Laplace evidence of the parametric fit: log L + log prior + (d/2) log 2 pi
# + (1/2) log det Sigma, in the same data normalization as log_evidence.
cov = np.asarray(laplace_cov(x_map, names, scene, data))
chi2_par = float(np.sum(np.asarray(whitened_residuals(par.model, data)) ** 2))
log_prior = sum(float(priors[k].log_prob(par.values[k])) for k in names)
logz_par = (-0.5 * chi2_par + log_prior + 0.5 * len(names) * np.log(2 * np.pi)
            + 0.5 * np.linalg.slogdet(cov)[1])
logz = {}
for ell in tqdm((1.5, 4.0), desc="profile length scale"):
    f = gpfit if ell == 1.5 else fit(gp.set("disk.profile.ell", ell), gp_priors_, data)
    logz[ell] = float(log_evidence(f, data, path="disk"))
print(f"parametric (20 parameters, Laplace-marginalized): log Z = {logz_par:8.1f}")
for ell, z in logz.items():
    print(f"GP (other parameters at MAP), profile length scale {ell:4.2f} mas: log Z = {z:8.1f}"
          + ("  (used below)" if ell == 1.5 else ""))
```

```text
profile length scale:   0%|          | 0/2 [00:00<?, ?it/s]
```

```text
parametric (20 parameters, Laplace-marginalized): log Z =   -386.1
GP (other parameters at MAP), profile length scale 1.50 mas: log Z =   -448.8  (used below)
GP (other parameters at MAP), profile length scale 4.00 mas: log Z =   -448.6
```

## What the data taught the Gaussian process

The parametric $\log Z$ is $-386$ and the Gaussian process's is $-449$ (about 63 nats lower), unsurprising for a model that is not told where the rings are. The two length scales, 1.5 and 4 mas, have the same evidence to within 0.2 nats, so **the data do not measure the profile's length scale**: it is a prior choice, not a result, and the fine structure of the posterior is set by it wherever the data are silent. Three points keep this honest.

- A Matérn length is not a feature width. A squared-exponential bump with the same curvature as Matérn-3/2 has width $\ell/\sqrt3$, so $\ell = 1.5$ mas matches the rings' Gaussian $\sigma$ of 0.8 to 1.8 mas, and 4 mas matches 2.3 mas. There is no sense in which the Gaussian process "finds a finer scale" than the rings.
- With $\sigma = 3$ the prior is rougher than this scene: its draws typically have about 18 local maxima of $\ln I_0$ between 1.5 and 24 mas (about 11 at 4 mas), against 3 for the truth. Neither length is the one that matches the truth, and the evidence cannot say which is better.
- The data cannot resolve scales below the half-fringe of 2 to 3 mas. The parametric ring-1 width is equally unresolved (0.15 ± 0.38 mas), and the ripples in the Gaussian-process band and its steep rise at the inner edge are the prior and the star-core degeneracy, not resolved structure.

The panels compare the prior of the profile and of the spectral index (16 to 84 percent of draws at the fixed hyperparameters) with draws from the Laplace posterior of the latents, $\mathcal N(z_0, (I + J^TJ)^{-1})$, the approximation the evidence uses. Only the latents vary; geometry, planet and disk flux stay at their MAP values, so the bands are narrower than the full posterior. The profile goes from a prior spanning four decades to a posterior with three bumps and a gap, plus the edge rise, and the spectral index from a wide band to a step, lower at the inner radii and higher in the outer ring.

```python
latents = ["disk.profile.z", "disk.harmonics.z", "disk.spectral_index.z"]
z0 = [np.asarray(gpm.get(p)) for p in latents]
flat0 = jnp.concatenate([jnp.ravel(z) for z in z0])


def residuals(flat):
    m, k = gpm, 0
    for p, z in zip(latents, z0):
        m = m.set(p, flat[k:k + z.size].reshape(z.shape))
        k += z.size
    return whitened_residuals(m, data)


# Laplace posterior of the latents, N(z0, (I + J^T J)^-1), as log_evidence uses
jac = np.asarray(jax.jacfwd(residuals)(flat0))
chol = np.linalg.cholesky(np.linalg.inv(np.eye(flat0.size) + jac.T @ jac))
rng = np.random.default_rng(0)
draws = np.asarray(flat0) + rng.standard_normal((200, flat0.size)) @ chol.T

rg = np.linspace(r_in, r_out, 200)


def profile_alpha(flat):
    """Surface brightness (unit integral times the disk flux), alpha(r), |I_1|."""
    m, k = gpm, 0
    for p, z in zip(latents, z0):
        m = m.set(p, flat[k:k + z.size].reshape(z.shape))
        k += z.size
    i0 = m.disk.profile.intensity(rg, 0.1)
    i0 = i0 / jnp.trapezoid(i0 * rg, rg) * m.disk.flux
    return (i0, m.disk.spectral_index(rg),
            jnp.abs(m.disk.harmonics.image_harmonics(rg)[1]))


post_i0, post_al, post_h1 = (
    np.asarray(x) for x in jax.vmap(profile_alpha)(jnp.asarray(draws))
)
key = jax.random.PRNGKey(1)
prior_i0 = np.asarray(gpm.disk.profile.draws(key, 200, r=rg, positive=True))
prior_i0 = prior_i0 / np.trapezoid(prior_i0 * rg, rg, axis=1)[:, None] * float(gpm.disk.flux)
prior_al = np.asarray(gpm.disk.spectral_index.draws(key, 200, r=rg))
band = lambda x: np.percentile(x, [16, 84], axis=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, pr, po, label in zip(axes, (prior_i0, prior_al), (post_i0, post_al),
                             (r"I$_0$(r) at 3 micron ($\int I_0\,r\,dr$ = flux)", r"$\alpha$(r)")):
    ax.fill_between(rg, *band(pr), color="0.8", label="prior, 16-84%")
    ax.fill_between(rg, *band(po), color="C0", alpha=0.5, label="posterior, 16-84%")
    ax.set_xlabel("radius in the disk plane (mas)")
    ax.set_ylabel(label)
axes[0].set_yscale("log")
axes[0].set_ylim(1e-5, 1e1)
axes[0].legend()
fig.suptitle("The data's effect on the GP profile and colour (MAP latents' Laplace posterior)")
plt.tight_layout()
plt.show()
```

![polar_disk_rings output 11.1](generated/polar_disk_rings_cell011_out01.png)

## Truth, parametric fit and deconvolution side by side

The top row shows the truth, the parametric fit and the Gaussian-process disk at 3 microns on one colour scale, with the beam in the corner; the second row shows the fits minus the truth as signed residuals (symmetric scales). The three middle panels compare the profile, colour and lopsidedness with the truth, with the blue bands from the Laplace posterior above. The truth profile is the sum of the three Gaussians, normalized so that $\int I_0\, r\,dr$ is the flux; the colour is the intensity-weighted logarithmic slope of the three blackbodies at 3 microns; the lopsidedness is the outer ring's first harmonic weighted by its share of the light. The spectra panel integrates the Gaussian-process disk over three radial windows (inside 5.3 mas, 5.3 to 10 mas, beyond 10 mas) against the corresponding blackbodies. The bottom row shows both fits' residuals over the quoted errors, with their root mean squares.

The Gaussian process puts bumps at about 3.5, 7 and 15 mas against the true 3, 7 and 15, within one resolution element, and cuts a gap between the second and third rings with a minimum near 10 mas. The planet at 11 mas sits a little nearer the third ring than that minimum. It follows the profile's contrast within its bands, including the fall beyond 20 mas where the data do not constrain it and the band widens to the prior. The spectral index follows the truth for the outer ring (red, slope near 3), and the band is wide for the middle ring, whose colour is weakly determined. The inner ring is where it fails. The index band at 3 to 4 mas contains the truth (about $-1.7$), but the index rises steeply at the innermost radii, an edge effect where the data have little leverage, so the flux-weighted spectrum of the innermost window rises with wavelength where the true ring 1 falls: the Gaussian process gets the sign of the inner ring's colour wrong. The lopsidedness is near 0.4 beyond 10 mas as in the truth, and near zero between 5 and 9 mas, with a wide band (up to about 0.3) inside. A power law in wavelength only approximates a blackbody over this band, as the dashed lines show. In the residual maps, dominated by the centre, the Gaussian-process residual is about three times the parametric one (the scales differ), and nothing is visible at the outer ring or the planet.

```python
par_model, bm = par.model, beam(data)
y, err = data.flatten_data()
n_vis = np.asarray(data.vis).size
lam = np.linspace(2.4e-6, 3.6e-6, 50)
alpha_of = lambda t: float(jax.grad(lambda l: jnp.log(BlackBody(1.0, t, W0)(l)))(W0) * W0)


def truth_profile(r, p):
    """I0(r) at 3 micron, the intensity-weighted alpha(r) and the harmonic |I_1|(r).

    Each ring is exp[-(r - r_i)^2 / 2 s_i^2], as GaussianRing has it, normalized by
    sqrt(2 pi) s_i r_i so that the integral of I0 r dr is the ring's flux (exact to
    <0.1 percent here, where r_i >> s_i).
    """
    g = [p[f"f{i}"] * np.exp(-0.5 * ((r - p[f"r{i}"]) / p[f"s{i}"]) ** 2)
         / (np.sqrt(2 * np.pi) * p[f"s{i}"] * p[f"r{i}"]) for i in (1, 2, 3)]
    a = [alpha_of(p[f"t{i}"]) for i in (1, 2, 3)]
    h3 = abs(np.asarray(Harmonics(re=p["h_re"], im=p["h_im"]).image_harmonics())[1])
    tot = sum(g) + 1e-30
    return sum(g), sum(gi * ai for gi, ai in zip(g, a)) / tot, h3 * g[2] / tot


fig = plt.figure(figsize=(16, 14))
grid = fig.add_gridspec(4, 6, hspace=0.45, wspace=1.5)
models = [(truth_model, "Truth"), (par_model, "Parametric fit"), (gpm, "GP deconvolution")]
for j, (m, name) in enumerate(models):
    plot_model(m, fov_mas=60, npix=256, ax=fig.add_subplot(grid[0, 2 * j:2 * j + 2]),
               saturate=0.99, beam=bm, title=name)
true_image = truth_model.render(npix=256, fov_mas=60)
for j, (m, name) in enumerate(models[1:]):
    plot_residual_map(np.asarray(m.render(npix=256, fov_mas=60) - true_image), 60,
                      ax=fig.add_subplot(grid[1, 2 * j:2 * j + 2]),
                      title=f"{name} minus truth")
ax = fig.add_subplot(grid[2, 0:2])
i_t, a_t, h_t = truth_profile(rg, truth)
i_p, a_p, h_p = truth_profile(rg, dict(par.values))
ax.fill_between(rg, *band(post_i0), color="C0", alpha=0.4, label="GP 16-84%")
ax.plot(rg, i_t, "k", label="truth")
ax.plot(rg, i_p, "C1--", label="parametric")
ax.set(yscale="log", ylim=(1e-5, 1e1), xlabel="radius (mas)", ylabel=r"I$_0$(r) ($\int I_0\,r\,dr$ = flux)")
ax.legend(fontsize=7)
ax = fig.add_subplot(grid[2, 2:4])
ax.fill_between(rg, *band(post_al), color="C0", alpha=0.4)
ax.plot(rg, np.where(i_t > 1e-3 * i_t.max(), a_t, np.nan), "k")
ax.plot(rg, np.where(i_p > 1e-3 * i_p.max(), a_p, np.nan), "C1--")
ax.set(xlabel="radius (mas)", ylabel=r"$\alpha$(r)", ylim=(-5, 6))
ax = fig.add_subplot(grid[2, 4:6])
ax.fill_between(rg, *band(post_h1), color="C0", alpha=0.4)
ax.plot(rg, np.where(i_t > 1e-3 * i_t.max(), h_t, np.nan), "k")
ax.plot(rg, np.where(i_p > 1e-3 * i_p.max(), h_p, np.nan), "C1--")
ax.set(xlabel="radius (mas)", ylabel=r"lopsidedness $|I_1|/I_0$", ylim=(0, 1))
ax = fig.add_subplot(grid[1, 4:6])
nodes, w = (np.asarray(x) for x in gpm.disk.profile.quadrature())
frac, al = w / w.sum(), np.asarray(gpm.disk.spectral_index(nodes))
for i, (lo, hi) in enumerate([(0, 5.3), (5.3, 10.0), (10.0, 99)]):
    inside = (nodes >= lo) & (nodes < hi)
    f_gp = float(gpm.disk.flux) * (frac[inside, None] * (lam / W0) ** al[inside, None]).sum(0)
    t_i = BlackBody(truth[f"f{i + 1}"], truth[f"t{i + 1}"], W0)(lam)
    ax.plot(lam * 1e6, t_i, f"C{i}", label=f"ring {i + 1}")
    ax.plot(lam * 1e6, f_gp, f"C{i}--")
ax.plot(lam * 1e6, BlackBody(truth["fp"], truth["tp"], W0)(lam), "C3", label="planet")
ax.plot(lam * 1e6, BlackBody(gv["planet.flux.ratio"], gv["planet.flux.temperature"], W0)(lam), "C3--")
ax.set(xlabel="wavelength (micron)", ylabel="flux / star at 3 micron", yscale="log")
ax.set_title("solid: truth, dashed: GP", fontsize=8)
ax.legend(fontsize=7)
for col, (sl, label) in enumerate([(slice(0, n_vis), r"V$^2$"), (slice(n_vis, None), "closure phase")]):
    ax = fig.add_subplot(grid[3, 3 * col:3 * col + 3])
    for m, name, c in [(par_model, "parametric", "C1"), (gpm, "GP", "C0")]:
        z = np.asarray(data.residuals(data.model(m)))[sl] / err[sl]
        ax.plot(z, ".", ms=3, color=c, label=f"{name}: rms z = {np.sqrt(np.mean(z**2)):.2f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="data index", ylabel=f"{label} residual / quoted error", ylim=(-5, 5))
    ax.legend(fontsize=7, loc="upper right")
plt.show()
```

![polar_disk_rings output 13.1](generated/polar_disk_rings_cell013_out01.png)

## Data against model

The model squared visibilities and closure phases against the data, with the one-to-one line. Both fits predict the same 378 and 252 numbers, so the points lie on top of each other and on the line, from the lowest visibilities near 5 percent to the highest near 65 percent. The residual plot above is the sharper test: the squared-visibility root mean square is 1.14 for the Gaussian process and 1.02 for the parametric model.

```python
def summary(m):
    pred = np.asarray(data.model(m))
    return {"vis_mean": pred[:n_vis], "vis_std": 0 * pred[:n_vis],
            "phi_mean": pred[n_vis:], "phi_std": 0 * pred[n_vis:]}


fig, _ = plot_data_model_correlation(
    data, {"parametric": summary(par_model), "GP": summary(gpm)}, figsize=(9, 4.5))
fig.suptitle("Both fits against the simulated data", y=1.03)
plt.show()
```

![polar_disk_rings output 15.1](generated/polar_disk_rings_cell015_out01.png)

## Summary

A non-parametric `PolarDisk` found the rings, the gap, the outer ring's lopsidedness and the planet from the same 630 numbers as a parametric fit of the true family, without being told there are three rings. The ring radii agree with the truth to about a resolution element, the outer and middle rings' colours agree with the blackbody slopes, and the planet's position and flux agree with the parametric fit. The inner ring's colour is spoiled by the edge of the profile.

The costs are visible. The fit is worse on the quoted errors ($\chi^2/N$ 1.20 against 1.03), the middle ring's colour is poorly determined, a power law in wavelength only approximates a blackbody, and the inner edge had to stay above the resolution. The profile's length scale is not measured: 1.5 and 4 mas have the same evidence, the prior is rougher than the scene, and structure finer than the 2 to 3 mas half-fringe, in either fit, is not constrained by the data. The Laplace $\log Z$ is 63 nats higher for the parametric model, but the comparison is only indicative (see above). For real data, use the two together: the Gaussian process to find structure and propose a parametric family, and the parametric fit and its evidence to confirm and measure it. Sampling the hyperparameters with NUTS and the whitened latents is the natural next step and is not done here.
