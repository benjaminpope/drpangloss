<!-- AUTO-GENERATED FROM notebooks/polar_disk_rings.ipynb by scripts/sync_tutorial_docs.py. -->
# A planet in a gap between inclined rings

This notebook shows what [`PolarDisk`](polar_disk.md) is for. We build a synthetic scene, a star with three concentric, inclined dust rings, the outermost of them lopsided, and a planet orbiting in the gap between the second and third rings, and we observe it with VLTI-like closure-phase and squared-visibility data at several wavelengths. We then analyze the same data in two ways. First we fit the family the data came from, three Gaussian rings and a planet, with invariant priors. Then we forget that there are rings at all and ask one non-parametric component, a `PolarDisk` whose radial profile, colour and lopsidedness are Gaussian processes, to find the rings, the gap, each ring's colour and the planet by itself. By the end you should know how to write both models, how to read the radial profile $I_0(r)$ and the spectral index $\alpha(r)$ of a Gaussian-process disk against the truth, and what the Laplace evidence says about the price of the extra freedom. Everything runs at the maximum a posteriori (MAP) solution with a Laplace approximation to the posterior, with no sampling.

## The scene and the data

The star is a point source with a 6000 K blackbody spectrum, and every other component's flux is relative to it at the reference wavelength of 3 microns. The three rings are `PolarDisk`s with `GaussianRing` profiles at radii of 3, 7 and 15 mas, each with its own blackbody (1500, 900 and 600 K), so that the inner ring is bluer than the outer one. The outer ring also has an azimuthal harmonic of degree one, which makes one side of it brighter, and the planet is a `PointSource` with a 1000 K blackbody that sits in the gap between the second and third rings. The rings are coplanar, so they must share an inclination and a position angle; we get that by building the whole scene in a function of the parameters that hands one `inc` and `pa` to every ring. This is the same function we will fit in the next step.

The data resemble a low-resolution L-band run with the four VLTI unit telescopes: nine channels from 2.4 to 3.6 microns, seven hour angles, so 378 squared visibilities with errors of 0.01 and 252 closure phases with errors of 0.5 degrees (567 independent data once the closure phases' correlations are accounted for). The beam of this coverage, drawn in the corner of every image below, is about 6 by 3 mas, so the 4 mas between the first two rings is of the order of one resolution element. The picture shows the truth at 3 microns with East to the left, the star saturated so that the faint outer ring can be seen, and the planet as the dot below and to the right of the centre.

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
             t3=600.0, h_re=0.5, h_im=0.2, dra=6.0, ddec=-9.0, fp=0.04,
             tp=1000.0)
truth_model = scene(**truth)
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

## The parametric fit

The truth family has 20 parameters: the shared geometry, a radius, width, flux and temperature for each ring, the two components of the outer ring's harmonic, and the planet's offset, flux and temperature. We give them invariant priors: uniform for angles, positions and radii, which are locations, and log-uniform for widths, fluxes and temperatures, which are scales. We start 5 percent away from the truth, which is what a local optimizer needs, and run `fit` to the MAP solution with Levenberg-Marquardt. A coarse search of the planet's position and the ring radii would normally come first for a real target. We then take the Laplace covariance of the likelihood at the MAP point, `laplace_cov`, to get an uncertainty on each parameter and, in a few cells, the evidence.

The fit reaches a raw $\chi^2$ of 580 for 567 independent data on the quoted errors, so $\chi^2/N = 1.02$, without rescaling. Most parameters are recovered to within about one Laplace standard deviation, and the planet's position and flux are precise (0.03 mas and 3 percent). The exception is the width of the innermost ring, which is not resolved: it lands on 0.15 mas, with a Laplace standard deviation of 0.38 mas that is larger than the value itself, and the truth of 0.8 mas lies within its range. A Gaussian approximation is poor for a scale parameter that is pressed against the edge of its prior, which will matter for the evidence below.

```python
priors = dict(inc=dist.Uniform(0, 85), pa=dist.Uniform(-90, 90),
              h_re=dist.Uniform(-1, 1), h_im=dist.Uniform(-1, 1),
              dra=dist.Uniform(-20, 20), ddec=dist.Uniform(-20, 20),
              fp=dist.LogUniform(0.003, 0.3), tp=dist.LogUniform(300, 3000))
for i in (1, 2, 3):
    priors |= {f"r{i}": dist.Uniform(0.5, 25), f"s{i}": dist.LogUniform(0.1, 4),
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

## The non-parametric fit

Now the model that does not know about rings. A single `PolarDisk` has a `RadialGP` profile on 48 nodes between 1.5 and 24 mas, the log of $I_0(r)$ being a Matérn Gaussian process with a correlation length of 1.5 mas. Its harmonics are `Harmonics(degree=1, prior=PolarKernel(...))`, a first-order lopsidedness whose amplitude and phase are themselves Gaussian processes in the radius, and its `spectral_index` is another `RadialGP`, so that the intensity at radius $r$ and wavelength $\lambda$ is $I_0(r)(\lambda/\lambda_0)^{\alpha(r)}$. The harmonic's low degree is what stops the disk from absorbing the planet, which is a compact, strongly asymmetric source, and the planet and star keep their parametric forms. Every Gaussian-process latent has a standard-normal prior, which `gp_priors` writes for us. The hyperparameters are fixed at values chosen by eye (a later cell compares two values of the profile's length scale by the evidence), and we start the geometry, the disk's total flux and the planet from the parametric fit of the previous step, which are held or started there rather than refitted from scratch. The geometry is not refitted: the inclination and position angle are the MAP values above, which have uncertainties of 0.2 and 0.3 degrees.

The inner edge of the Gaussian process is set at 1.5 mas, which is below the first ring but a little under the beam. An earlier try with the inner edge at 0.5 mas, below any resolution, let the disk put about a quarter of its light into an unresolved core that traded off against the star's flux, which the data cannot tell apart; the visibilities were fitted just as well. We keep the edge at 1.5 mas, so the fit cannot hide light in the centre. The fit reaches $\chi^2/N = 1.17$ on the quoted errors, a little worse than the parametric model's 1.02, and recovers the planet's position within 0.1 mas and its flux to within 5 percent of the parametric values.

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
      f"(parametric: {v['dra']:.2f}, {v['ddec']:.2f}; truth 6.00, -9.00)")
print(f"planet flux at 3 micron = {gv['planet.flux.ratio']:.3f} of the star "
      f"(parametric: {v['fp']:.3f}; truth 0.040)")
print(f"disk flux at 3 micron = {gv['disk.flux']:.2f} "
      f"(parametric rings: {v['f1'] + v['f2'] + v['f3']:.2f}; truth 0.90)")
```

## The evidence, parametric against Gaussian process

How much should we believe the extra freedom? The Laplace evidence is $\log Z \approx -\tfrac12\chi^2 + \log\pi(\hat\theta) + \tfrac d2\log 2\pi + \tfrac12\log\det\Sigma$ for the parametric model, with $\Sigma$ the Laplace covariance, and `log_evidence` computes the equivalent expression for the Gaussian-process latents, $-\tfrac12\chi^2 - \tfrac12|z|^2 - \tfrac12\log\det(I + J^TJ)$. Both are in the same normalization of the data's likelihood, so only differences between them mean anything. Two caveats apply and both favour the parametric model. The Gaussian-process evidence is conditional on its fixed hyperparameters and on the geometry, the planet and the disk flux held at their MAP values, whereas the parametric evidence marginalizes all 20 parameters; and the parametric family contains the truth, which is of course not something a real analysis can assume. We also compare the profile's correlation length of 1.5 mas with a longer one of 4 mas, which is how MacKay's evidence framework chooses a hyperparameter.

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
print(f"parametric (20 parameters, all marginalized): log Z = {logz_par:8.1f}")
for ell, z in logz.items():
    print(f"GP, profile length scale {ell:4.2f} mas: log Z = {z:8.1f}"
          + ("  (used below)" if ell == 1.5 else ""))
```

## What the data taught the Gaussian process

The evidence tells us that the data prefer the model that knows about the rings, by a wide margin, which is not surprising. What the Gaussian process buys is not evidence but not having to know. The two panels compare the prior of the profile and of the spectral index, a band of the 16 to 84 percent range of draws from the Gaussian-process prior at the fixed hyperparameters, with a band of draws from the Laplace posterior of the latents, $\mathcal N(z_0, (I + J^TJ)^{-1})$, the same approximation that the evidence uses. Only the latents vary in the posterior bands: the geometry, the planet and the disk flux are held at their MAP values, so the bands are narrower than the full posterior would be. The profile has gone from an unstructured prior that spans four decades to a posterior that has three bumps and a gap, and the spectral index from a wide band to a band that traces a step, with a lower value at the inner radii and a higher one in the outer ring.

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
                             ("I$_0$(r) (flux per mas$^2$ at 3 micron)", r"$\alpha$(r)")):
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

## Truth, parametric fit and deconvolution side by side

The top row shows the three images at 3 microns on the same colour scale, with the beam in the corner; the second row shows the two fits minus the truth as plain signed residuals, not z-scores, since the MAP images have no uncertainty, on a colour scale that is symmetric about zero. The three middle panels compare the radial profile, the colour and the lopsidedness with the truth. The truth for the profile is the sum of the three Gaussians in the same units as the Gaussian process, for the colour it is the intensity-weighted logarithmic slope of the three blackbodies at 3 microns, and for the lopsidedness it is the outer ring's first harmonic weighted by that ring's share of the light at each radius. The blue bands are the Laplace posterior of the latents from the previous cell. The spectra panel integrates the Gaussian-process disk's flux, $I_0(r)(\lambda/\lambda_0)^{\alpha(r)}$ over three radial windows (inside 5.3 mas, between 5.3 and 10 mas, and outside 10 mas) and sets each against the blackbody of the corresponding ring. The bottom row shows the residuals of both fits, divided by the quoted errors, for the squared visibilities and the closure phases, with the root mean square of each in the legend.

The Gaussian process puts bumps at about 3.7, 7 and 15 mas against the true 3, 7 and 15 mas, so the ring radii agree to within one resolution element, and it cuts a gap between about 9 and 11 mas, in which the planet sits. It reproduces the radial contrast of the profile to within its bands, including the steep fall beyond 20 mas, where the data have no constraint and the band widens to the prior. The spectral index follows the truth: the inner ring is blue, the outer ring is red with a slope near 3, and the band in between is wide, so the middle ring's colour is only weakly determined. The rise of the index at the innermost radii is an edge effect of the Gaussian process, where the data have little leverage. The third middle panel shows the lopsidedness, the first harmonic's amplitude relative to the azimuthal mean: it is near 0.4 beyond 10 mas, as in the truth, and consistent with zero in the inner rings. The recovered spectra are of the right order and ordering but, as the dashed lines show, a power law in wavelength over a band with this much curvature only approximates a blackbody, and the inner ring's spectrum is flatter than the truth. In the residual maps, which are dominated by the centre, the residual of the Gaussian-process image is about three times that of the parametric one (the scales differ), and both lie within the inner ring and on the star, with nothing visible at the outer ring or the planet.

```python
par_model, bm = par.model, beam(data)
y, err = data.flatten_data()
n_vis = np.asarray(data.vis).size
lam = np.linspace(2.4e-6, 3.6e-6, 50)
alpha_of = lambda t: float(jax.grad(lambda l: jnp.log(BlackBody(1.0, t, W0)(l)))(W0) * W0)


def truth_profile(r, p):
    """I0(r) at 3 micron, the intensity-weighted alpha(r) and the harmonic |I_1|(r)."""
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
ax.set(yscale="log", ylim=(1e-5, 1e1), xlabel="radius (mas)", ylabel="I$_0$(r)")
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

## Data against model

For completeness, the model squared visibilities and closure phases against the data, with the errors on the data and the one-to-one line, for both fits. Since the observables are the same 378 and 252 numbers whichever model made them, the two sets of points lie on top of each other and on the line, from the lowest visibilities at about 5 percent to the highest at about 65 percent; the model is in no part of the range systematically away from the data. The residual plot above is the more sensitive test, in which the Gaussian process has a squared-visibility root mean square of 1.14 against the parametric model's 1.02.

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

## Summary

A non-parametric `PolarDisk` found the rings, the gap, the outer ring's lopsidedness and the planet from the same 630 numbers that a parametric fit of the true family used, without being told that there are three rings. The ring radii agree with the truth to within about a resolution element, the colour of the inner and outer rings agrees with the blackbody slopes, and the planet's position and flux agree with the parametric fit. The costs are visible too: the parametric evidence is higher by about 57 nats, the fit is a little worse on the quoted errors ($\chi^2/N$ of 1.17 against 1.02), the colour of the middle ring is poorly determined, a power law in wavelength only approximates a blackbody, and the inner edge of the Gaussian process needed to stay above the resolution. For real data, the sensible use of the two is together: the Gaussian process to find the structure and propose a parametric family, and the parametric fit, with its evidence, to confirm and measure it. The Gaussian-process hyperparameters here were fixed; sampling them, with NUTS and the whitened latents, is the next step and is not done in this notebook.
