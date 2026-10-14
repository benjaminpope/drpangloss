"""Identifiability of composite scenes: a linear-Gaussian toy, NumPy only.

Companion to design/composite_scenes.md. Each component of a scene (a point
star, a pixel image on its own grid, a postage stamp at an offset) is a block
of a linear model ``V = A b`` with a Gaussian prior ``b ~ N(m, K)``: a
Gaussian process on that component's own grid and support, plus a broad
total-flux mode. With complex visibilities and Gaussian noise, the evidence
``N(d; A m, A K Aᵀ + N)`` and the posterior of every component's total flux
are exact, so the toy measures what the data can tell apart:

- the posterior correlation of the components' fluxes (near -1: they trade
  flux, so they are not separately identified);
- each component's posterior flux against the truth (absorption);
- the evidence of each decomposition, which includes every Occam factor
  (fluxes and pixels alike are marginalized).

Simplifications, deliberate: brightness (not log-brightness) is Gaussian, so
positivity is not enforced; the data are calibrated complex visibilities,
not V² and closure phases, so the problem is linear. Units: the beam
λ/B_max is 1. Run with ``python composite_identifiability.py``; about a
minute on a laptop.
"""

import itertools

import numpy as np

SIGMA_V = 0.01  # noise on each of Re V, Im V (total flux 1)
FLUX_SD = 1.0  # prior sd of each component's total flux (broad)
# Normalized visibilities fix the total flux: one zero-spacing datum,
# weighted to σ = SIGMA_V / NORM (a linear stand-in for V(0) = 1).
NORM = 10.0


# --- coverage and Fourier matrices ------------------------------------------


def coverage(seed=0, n_tel=6, n_ha=10, channels=(0.85, 0.95, 1.05, 1.15)):
    """Earth-rotation uv tracks of ``n_tel`` random stations, |uv| ≤ 1/beam
    at the longest baseline and shortest wavelength (VLTI-like: 15 baselines,
    10 hour angles, 4 channels)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, (n_tel, 2))
    pairs = list(itertools.combinations(range(n_tel), 2))
    b = np.array([xy[j] - xy[i] for i, j in pairs])
    uv = []
    for h in np.linspace(-1.0, 1.0, n_ha):  # hour angle, radians
        c, s = np.cos(h), np.sin(h)
        rot = np.stack([c * b[:, 0] - s * b[:, 1], 0.7 * (s * b[:, 0] + c * b[:, 1])], 1)
        for lam in channels:
            uv.append(rot / lam)
    uv = np.concatenate(uv)
    return uv / np.hypot(*uv.T).max()


def fourier(uv, x, y):
    """Complex visibility matrix of unit points at (x, y): (n_uv, n_points)."""
    return np.exp(-2j * np.pi * (uv[:, :1] * x[None] + uv[:, 1:] * y[None]))


def grid(npix, pixel, dra=0.0, ddec=0.0):
    """Pixel centres (x East, y North) of a square grid centred on (dra, ddec)."""
    c = (np.arange(npix) - (npix - 1) / 2) * pixel
    x, y = np.meshgrid(c[::-1] + dra, c[::-1] + ddec)
    return x.ravel(), y.ravel()


def realify(z):
    return np.concatenate([z.real, z.imag])


def design(uv, x, y):
    """Real design matrix: Re V, Im V, and the weighted total flux."""
    return np.vstack([realify(fourier(uv, x, y)), NORM * np.ones((1, x.size))])


# --- components ---------------------------------------------------------------


class Component:
    """A block of the linear model: positions, prior mean and covariance."""

    def __init__(self, name, x, y, mean, cov):
        self.name, self.x, self.y, self.mean, self.cov = name, x, y, mean, cov


def point(name, x=0.0, y=0.0, flux=0.5):
    return Component(name, np.array([x]), np.array([y]), np.array([flux]), np.array([[FLUX_SD**2]]))


def image(name, npix, pixel, ell, amp, dra=0.0, ddec=0.0, hole=None, radius=None, flux=0.5):
    """A GP image: squared-exponential kernel of length ``ell`` and pixel sd
    ``amp`` on the support (a disc of ``radius`` minus a ``hole``, both about
    the grid centre), with a broad total-flux mode and a flat mean of total
    ``flux``."""
    x, y = grid(npix, pixel, dra, ddec)
    r = np.hypot(x - dra, y - ddec)
    keep = np.ones_like(r, bool)
    if hole is not None:
        keep &= r >= hole
    if radius is not None:
        keep &= r <= radius
    x, y = x[keep], y[keep]
    n = x.size
    d2 = (x[:, None] - x[None]) ** 2 + (y[:, None] - y[None]) ** 2
    cov = amp**2 * np.exp(-0.5 * d2 / ell**2) + (FLUX_SD / n) ** 2 * np.ones((n, n))
    return Component(name, x, y, np.full(n, flux / n), cov + 1e-10 * np.eye(n))


# --- the exact linear-Gaussian posterior --------------------------------------


class Fit:
    def __init__(self, uv, data, comps):
        self.comps = comps
        self.blocks = [design(uv, c.x, c.y) for c in comps]
        A = np.concatenate(self.blocks, 1)
        m = np.concatenate([c.mean for c in comps])
        sizes = [c.mean.size for c in comps]
        K = np.zeros((sum(sizes),) * 2)
        o = np.cumsum([0] + sizes)
        for i, c in enumerate(comps):
            K[o[i] : o[i + 1], o[i] : o[i + 1]] = c.cov
        S = A @ K @ A.T + SIGMA_V**2 * np.eye(A.shape[0])
        L = np.linalg.cholesky(S)
        r = data - A @ m
        w = np.linalg.solve(L, r)
        self.log_z = float(-0.5 * w @ w - np.log(np.diag(L)).sum() - 0.5 * r.size * np.log(2 * np.pi))
        Sinv_r = np.linalg.solve(L.T, w)
        self.mean = m + K @ A.T @ Sinv_r
        self.offsets, self.A, self.K, self.L = o, A, K, L
        self.chi2_red = float(np.sum((data - A @ self.mean) ** 2) / SIGMA_V**2 / data.size)

    def flux_moments(self):
        """Posterior mean, sd and correlation of each component's total flux."""
        o = self.offsets
        W = np.zeros((self.K.shape[0], len(self.comps)))
        for i in range(len(self.comps)):
            W[o[i] : o[i + 1], i] = 1.0
        KW = self.K @ W
        V = np.linalg.solve(self.L, self.A @ KW)
        cov = W.T @ KW - V.T @ V
        sd = np.sqrt(np.diag(cov))
        return self.mean @ W, sd, cov / np.outer(sd, sd)

    def residual_covariance_solve(self, data):
        """S⁻¹ (d - A μ_prior) with S the marginal covariance, for searches."""
        r = data - self.A @ np.concatenate([c.mean for c in self.comps])
        return np.linalg.solve(self.L.T, np.linalg.solve(self.L, r))

    def image(self, npix, pixel):
        """The posterior-mean scene binned onto a common grid."""
        out = np.zeros((npix, npix))
        c = (npix - 1) / 2
        for comp, (a, b) in zip(self.comps, zip(self.offsets[:-1], self.offsets[1:])):
            col = np.clip(np.rint(c - comp.x / pixel).astype(int), 0, npix - 1)
            row = np.clip(np.rint(c - comp.y / pixel).astype(int), 0, npix - 1)
            np.add.at(out, (row, col), self.mean[a:b])
        return out


def search(uv, data, fit, npix=41, pixel=0.125, extent=None, noise_only=False, n_peaks=3, min_sep=0.5):
    """Matched-filter flux map of a point added to ``fit``'s scene, with the
    scene's components marginalized (the covariance S, not just the noise),
    so that flux the extended component would absorb still shows. Returns
    the ``n_peaks`` best local maxima (x, y, SNR, flux), at least
    ``min_sep`` apart, best first. ``noise_only``: the residual of the
    posterior mean against the noise alone, as ``linear_flux_grid`` on a
    fitted scene does."""
    x, y = grid(npix, pixel)
    if extent is not None:
        keep = np.hypot(x, y) <= extent
        x, y = x[keep], y[keep]
    G = design(uv, x, y)
    G[-1] = 0.0  # a point adds flux, not total flux (normalized data)
    if noise_only:  # linear_flux_grid on the posterior-mean residual
        Sinv_r, Sinv_G = (data - fit.A @ fit.mean) / SIGMA_V**2, G / SIGMA_V**2
    else:
        Sinv_r = fit.residual_covariance_solve(data)
        Sinv_G = np.linalg.solve(fit.L.T, np.linalg.solve(fit.L, G))
    f = G.T @ Sinv_r / np.einsum("ij,ij->j", G, Sinv_G)
    snr = f * np.sqrt(np.einsum("ij,ij->j", G, Sinv_G))
    peaks = []
    for k in np.argsort(-snr):
        if all(np.hypot(x[k] - a, y[k] - b) >= min_sep for a, b, _, _ in peaks):
            peaks.append((float(x[k]), float(y[k]), float(snr[k]), float(f[k])))
        if len(peaks) == n_peaks:
            break
    return peaks


# --- truths ---------------------------------------------------------------------


def truth_ring(planet_sep, planet_pa=60.0, star=0.5, planet=0.03, r_in=1.5, inc=45.0, pa=30.0,
               rim=0.5, outer=0.15):
    """Star + inclined ring (bright near side) + faint outer ring, gap
    between them, + a planet: the ``star_disk_planet`` phantom family."""
    x, y = grid(201, 0.05)
    p = np.radians(pa)
    along, across = x * np.sin(p) + y * np.cos(p), x * np.cos(p) - y * np.sin(p)
    ci = np.cos(np.radians(inc))
    r = np.hypot(along, across / ci)
    az = np.arctan2(across / ci, along)
    inner = np.exp(-0.5 * ((r - r_in) / (0.15 * r_in)) ** 2) * (1 + rim * np.sin(az))
    ring_out = np.exp(-0.5 * ((r - 3 * r_in) / (0.2 * 3 * r_in)) ** 2)
    disk = (1 - outer) * inner / inner.sum() + outer * ring_out / ring_out.sum()
    disk_flux = 1 - star - planet
    q = np.radians(planet_pa)
    px, py = planet_sep * np.sin(q), planet_sep * np.cos(q)
    xs = np.concatenate([[0.0, px], x])
    ys = np.concatenate([[0.0, py], y])
    fl = np.concatenate([[star, planet], disk_flux * disk])
    return xs, ys, fl, {"star": star, "planet": planet, "disk": disk_flux, "planet_xy": (px, py)}


def truth_envelope(radius, core=0.4, halo=0.25, seed=3):
    """Unresolved core + broad Gaussian halo + clumps within ``radius``:
    the ``envelope`` phantom family."""
    rng = np.random.default_rng(seed)
    x, y = grid(241, 0.06)
    g = lambda dx, dy, fwhm: np.exp(-0.5 * ((x - dx) ** 2 + (y - dy) ** 2) / (fwhm / 2.355) ** 2)  # noqa: E731
    h = g(0, 0, 2 * radius)
    clumps = sum(
        rng.uniform(0.2, 1) * g(*(radius * np.sqrt(rng.random()) * np.array([np.sin(a), np.cos(a)])),
                                 radius * rng.uniform(0.25, 0.75))
        for a in rng.uniform(0, 2 * np.pi, 5)
    )
    rest = 1 - core - halo
    env = halo * h / h.sum() + rest * clumps / clumps.sum()
    return (np.concatenate([[0.0], x]), np.concatenate([[0.0], y]), np.concatenate([[core], env]),
            {"star": core, "env": 1 - core})


def observe(uv, truth, seed=1):
    x, y, f, info = truth
    rng = np.random.default_rng(seed)
    v = design(uv, x, y) @ f
    return v + SIGMA_V * rng.standard_normal(v.size), info


# --- model families -------------------------------------------------------------


def best_over(make, grid_):
    """The best evidence over a hyperparameter grid (empirical Bayes)."""
    fits = [(make(*g), g) for g in grid_]
    return max(fits, key=lambda t: t[0].log_z)


def report(name, fit, hyper, truth_fluxes):
    mean, sd, corr = fit.flux_moments()
    names = [c.name for c in fit.comps]
    parts = ", ".join(f"{n} {m:.3f}±{s:.3f}" for n, m, s in zip(names, mean, sd))
    pairs = ", ".join(f"ρ({names[i]},{names[j]})={corr[i, j]:+.2f}"
                      for i, j in itertools.combinations(range(len(names)), 2))
    print(f"  {name:<34s} log Z {fit.log_z:9.1f}  χ²/N {fit.chi2_red:5.2f}  hyper {hyper}\n"
          f"      fluxes: {parts}\n      {pairs or '-'}   truth {truth_fluxes}")
    return {"name": name, "log_z": fit.log_z, "chi2_red": fit.chi2_red, "fluxes": dict(zip(names, mean)),
            "sd": dict(zip(names, sd)), "corr": corr.tolist(), "hyper": hyper}


ELLS = (0.25, 0.5, 1.0)
AMPS = (0.0005, 0.001, 0.002, 0.004, 0.008)


def case_ring(uv, planet_sep, label):
    truth = truth_ring(planet_sep)
    data, info = observe(uv, truth)
    tf = {k: round(v, 3) for k, v in info.items() if k != "planet_xy"}
    print(f"\nCase A ({label}): star + ring with gap + planet at {planet_sep} beams; truth {tf}")
    npix, pix = 49, 0.25  # 12 beams across, holds the outer ring (4.5 beams)
    hp = list(itertools.product(ELLS, AMPS))
    out = []
    f1, h1 = best_over(lambda l, a: Fit(uv, data, [image("img", npix, pix, l, a, flux=1.0, radius=6)]), hp)
    out.append(report("single GP image", f1, h1, tf))
    f2, h2 = best_over(lambda l, a: Fit(uv, data, [point("star"), image("disk", npix, pix, l, a, radius=6)]), hp)
    out.append(report("star + GP", f2, h2, tf))
    f3, h3 = best_over(lambda l, a: Fit(uv, data, [point("star"), image("disk", npix, pix, l, a, hole=0.5,
                                                                          radius=6)]), hp)
    out.append(report("star + GP (hole 0.5 beam)", f3, h3, tf))
    # Stamps, one at a time: candidates from a search on the current scene
    # with its components marginalized; each candidate's offset refined by
    # the evidence over a sub-beam grid (dra, ddec fitted); the best kept if
    # it raises log Z by at least ln 100.
    tx, ty = info["planet_xy"]
    l3, a3 = h3
    base = [point("star"), image("disk", npix, pix, l3, a3, hole=0.5, radius=6)]
    noise = search(uv, data, f3, extent=5.5, noise_only=True, n_peaks=1)[0]
    print(f"      noise-only search on the posterior-mean residual: SNR {noise[2]:.1f} flux {noise[3]:.3f}")
    stamps, current = [], f3
    for k in range(2):
        peaks = search(uv, data, current, extent=5.5)
        print(f"      search {k + 1} peaks (x, y, SNR, flux): "
              + "; ".join(f"({a:.2f}, {b:.2f}) {c:.1f} {d:.3f}" for a, b, c, d in peaks)
              + f"; truth ({tx:.2f}, {ty:.2f})")
        best = None
        for px, py, _, _ in peaks:
            offs = [(px + dx, py + dy) for dx in (-0.1, 0, 0.1) for dy in (-0.1, 0, 0.1)]
            cand = best_over(lambda ox, oy: Fit(uv, data, base + stamps + [
                image(f"stamp{k + 1}", 5, 0.1, 0.1, 0.01, dra=ox, ddec=oy, flux=0.0)]), offs)
            print(f"        stamp at ({cand[1][0]:.2f}, {cand[1][1]:.2f}): Δlog Z {cand[0].log_z - current.log_z:+.1f}")
            if best is None or cand[0].log_z > best[0].log_z:
                best = cand
        if best[0].log_z - current.log_z < np.log(100):
            break
        ox, oy = best[1]
        stamps.append(image(f"stamp{k + 1}", 5, 0.1, 0.1, 0.01, dra=ox, ddec=oy, flux=0.0))
        current = best[0]
        err = float(np.hypot(ox - tx, oy - ty))
        out.append(report(f"star + GP(hole) + {k + 1} stamp(s)", current, (l3, a3, round(float(ox), 2),
                                                                         round(float(oy), 2)), tf))
        out[-1]["offset_error"] = err
        print(f"      stamp {k + 1} is {err:.2f} beams from the planet")
    at_truth = Fit(uv, data, base + [image("planet", 5, 0.1, 0.1, 0.01, dra=tx, ddec=ty, flux=0.0)])
    print(f"      (reference: one stamp at the true position, Δlog Z {at_truth.log_z - f3.log_z:+.1f})")
    return out


def case_envelope(uv, radius):
    truth = truth_envelope(radius)
    data, info = observe(uv, truth)
    tf = {k: round(v, 3) for k, v in info.items()}
    print(f"\nCase B: core + envelope of radius {radius} beams; truth {tf}")
    out = []
    big = max(4 * radius, 4.0)
    npix_c, pix_c = 49, 0.25
    pix_e = big / 24
    ells = tuple(sorted({0.25, 0.5, 1.0, radius / 2, radius}))
    hp = list(itertools.product(ells, AMPS))
    f1, h1 = best_over(lambda l, a: Fit(uv, data, [image("img", npix_c, pix_c, l, a, flux=1.0)]), hp)
    out.append(report("single GP image (12 beams)", f1, h1, tf))
    f2, h2 = best_over(lambda l, a: Fit(uv, data, [point("star"), image("env", 49, pix_e, l, a,
                                                                          radius=big)]), hp)
    out.append(report("star + GP", f2, h2, tf))
    f3, h3 = best_over(lambda l, a: Fit(uv, data, [point("star"), image("env", 49, pix_e, l, a, hole=0.5,
                                                                          radius=big)]), hp)
    out.append(report("star + GP (hole 0.5 beam)", f3, h3, tf))
    split = abs(out[-1]["fluxes"]["star"] - out[-2]["fluxes"]["star"])
    print(f"      star flux moves by {split:.3f} between hole and no hole (posterior sd "
          f"{out[-2]['sd']['star']:.3f}): the split is set by the model, not the data, where this exceeds the sd")
    # Two GPs: compact (2 beams, fine pixels, short ℓ) and extended (coarse).
    f5, h5 = best_over(lambda l, a: Fit(uv, data, [image("core", 9, 0.25, 0.25, 0.05, radius=1.0),
                                                   image("env", 49, pix_e, l, a, radius=big)]), hp)
    out.append(report("compact GP + extended GP", f5, h5, tf))
    # Two GPs on one grid, different ℓ: identical to one GP with K1 + K2.
    l5, a5 = h5
    f6 = Fit(uv, data, [image("short", 49, pix_e, 0.25, a5, radius=big),
                        image("long", 49, pix_e, l5, a5, radius=big)])
    out.append(report("two GPs, one grid (ℓ 0.25 + ℓ*)", f6, (0.25, l5), tf))
    return out


def main():
    uv = coverage()
    print(f"{len(uv)} uv points, σ(V) = {SIGMA_V}")
    results = {"A_off_ring": case_ring(uv, 2.6, "planet in the gap"),
               "A_on_ring": case_ring(uv, 1.3, "planet on the inner ring")}
    for radius in (0.5, 1.0, 2.0, 4.0):
        results[f"B_R{radius}"] = case_envelope(uv, radius)
    return results


if __name__ == "__main__":
    main()
