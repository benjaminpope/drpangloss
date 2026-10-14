"""RadialGP, PolarKernel, spectral_index and freeze_geometry (stage P4).

Everything is tiny: a few dozen visibilities, at most 48 nodes, and NUTS
runs of a couple of hundred draws.
"""

import time
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS
from scipy.special import j0 as scipy_j0

from virgil import models
from virgil._geometry import image_coordinates
from virgil.coverage import vlti_oidata
from virgil.fitting import fit
from virgil.imaging import log_evidence
from virgil.likelihood import numpyro_model
from virgil.models import (
    Cone,
    GaussianRing,
    Harmonics,
    PointSource,
    PolarDisk,
    PolarKernel,
    RadialGP,
    RadialLinear,
    RadialSum,
    System,
    ThinRing,
    gp_priors,
)
from virgil.simulate import simulate

from ._compiles import count_compiles

_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0
WAVEL = 3.5e-6

DATA = vlti_oidata(
    hour_angles_h=(-3.0, -1.5, 0.0, 1.5, 3.0),
    wavelengths_m=[3.2e-6, 3.6e-6, 4.0e-6],
    sigma_v2=0.01,
    sigma_cp_deg=0.5,
)
SMALL = vlti_oidata(
    hour_angles_h=(-1.5, 0.0, 1.5),
    wavelengths_m=[3.4e-6, 3.8e-6],
    sigma_v2=0.02,
    sigma_cp_deg=1.0,
)


def _matern15(x, ell):
    a = onp.sqrt(3.0) * onp.abs(x[:, None] - x[None, :]) / ell
    return (1.0 + a) * onp.exp(-a)


def _uv(n=40, seed=0, qmax=0.25):
    """Baselines (metres) out to ``qmax`` cycles/mas at WAVEL."""
    rng = onp.random.default_rng(seed)
    scale = qmax / _MAS2RAD_REF * WAVEL / 1.0
    return (
        rng.uniform(-scale, scale, n),
        rng.uniform(-scale, scale, n),
    )


def _cvis(model, u, v, wavel=WAVEL):
    return onp.asarray(model.model(np.asarray(u), np.asarray(v), wavel))


# --- RadialGP: whitening, prior, positivity, gradients ------------------------


def test_whitened_latents_give_the_matern_prior():
    gp = RadialGP(0.0, 10.0, sigma=2.0, ell=3.0, mu=0.5, n=8)
    nodes = onp.asarray(gp.nodes())
    # The defining map: g = mu + sigma L z, linear in z, g(z = 0) = mu.
    assert onp.allclose(gp(nodes), 0.5)
    z = onp.random.default_rng(0).standard_normal(8)
    g1 = onp.asarray(gp.set("z", z)(nodes))
    g2 = onp.asarray(gp.set("z", 2.0 * z)(nodes))
    assert onp.allclose(g2 - 0.5, 2.0 * (g1 - 0.5), atol=1e-5)
    # z ~ N(0, I) gives covariance sigma^2 K of the Matern kernel.
    draws = onp.asarray(gp.draws(jax.random.PRNGKey(1), 40000)) - 0.5
    cov = draws.T @ draws / len(draws)
    expected = 4.0 * _matern15(nodes, 3.0)
    assert onp.max(onp.abs(cov - expected)) < 0.15
    assert onp.allclose(onp.diag(cov), 4.0, rtol=0.05)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, float("inf")])
def test_every_matern_smoothness_builds_a_valid_factor(nu):
    gp = RadialGP(0.0, 10.0, ell=6.0, nu=nu, n=20)  # ell >> spacing
    g = onp.asarray(gp.set("z", np.ones(20))(np.linspace(0.0, 10.0, 5)))
    assert onp.all(onp.isfinite(g))
    assert bool(gp.is_physical())
    with pytest.raises(ValueError, match="nu"):
        RadialGP(0.0, 10.0, nu=3.0)


def test_profile_is_positive_and_the_disk_is_normalized():
    rng = onp.random.default_rng(2)
    z = rng.standard_normal(32) * 2.0
    disk = PolarDisk(
        RadialGP(0.5, 12.0, sigma=1.5, ell=2.0, z=z), inc=40.0, pa=10.0, n_r=32
    )
    r, w = disk.profile.quadrature()
    assert bool(np.all(w > 0.0))
    assert bool(np.all(disk.profile.intensity(r, 0.0) > 0.0))
    assert onp.isclose(
        complex(disk.model(np.zeros(1), np.zeros(1), WAVEL)[0]), 1.0
    )
    assert bool(disk.is_physical())
    # No flux outside the range, in the render.
    image = onp.asarray(disk.render(npix=96, fov_mas=40.0))
    assert onp.all(image >= 0.0)


def test_a_profile_needs_one_latent_per_node_and_a_range():
    with pytest.raises(ValueError, match="one latent per quadrature node"):
        PolarDisk(RadialGP(0.5, 5.0, z=np.zeros(7)), n_r=16)
    with pytest.raises(ValueError, match="radial range"):
        PolarDisk(RadialGP(), n_r=16).model(np.ones(2), np.ones(2), WAVEL)


def test_gradients_are_finite_in_every_parameter():
    disk = PolarDisk(
        RadialGP(0.5, 10.0, sigma=1.0, ell=2.0, z=0.3 * onp.arange(24) / 24.0),
        harmonics=Harmonics(degree=2, prior=PolarKernel(3.0, 4.0, 0.4)),
        spectral_index=RadialGP(sigma=0.5, ell=3.0),
        wavel0=WAVEL,
        inc=35.0,
        n_r=24,
    )
    u, v = _uv(20)
    wavel = np.linspace(3.2e-6, 3.9e-6, 20)

    def power(model):
        return np.sum(
            np.abs(model.model(np.asarray(u), np.asarray(v), wavel)) ** 2
        )

    grads = jax.grad(power)(disk)
    for path in (
        "profile.z",
        "profile.sigma",
        "profile.ell",
        "profile.r_in",
        "profile.r_out",
        "harmonics.z",
        "harmonics.prior.sigma",
        "harmonics.prior.ell_r",
        "harmonics.prior.ell_theta",
        "spectral_index.z",
        "spectral_index.sigma",
        "spectral_index.ell",
    ):
        assert onp.all(onp.isfinite(onp.asarray(grads.get(path)))), path
    assert onp.any(onp.asarray(grads.get("profile.z")) != 0.0)


def test_a_gp_profile_recovers_a_known_ring_in_a_noiseless_map_problem():
    truth = PolarDisk(GaussianRing(5.0, 0.8), n_r=48)
    data = simulate(truth, DATA)
    start = PolarDisk(RadialGP(0.5, 10.0, sigma=1.5, ell=1.5), n_r=24)
    result = fit(start, gp_priors(start), data)
    assert result.info["converged"]
    model = result.model
    r = onp.linspace(0.5, 10.0, 96)
    profile = onp.array(model.profile.intensity(r, 0.0))
    profile /= profile.max()
    expected = onp.exp(-0.5 * ((r - 5.0) / 0.8) ** 2)
    assert abs(r[profile.argmax()] - 5.0) < 0.5
    # The shape within a resolution element: the half-maximum edges.
    half = r[profile > 0.5]
    assert abs(half.min() - 3.9) < 0.6 and abs(half.max() - 6.1) < 0.6
    assert onp.sqrt(onp.mean((profile - expected) ** 2)) < 0.15
    chi2 = result.info["chi2"][0]
    assert chi2 / result.info["ndata"][0] < 0.1


def test_the_dft_of_the_render_matches_the_node_profile_visibility():
    rng = onp.random.default_rng(3)
    disk = PolarDisk(
        RadialGP(
            0.5,
            11.0,
            sigma=1.0,
            ell=3.0,
            z=rng.standard_normal(48),
        ),
        harmonics=Harmonics(
            degree=2,
            prior=PolarKernel(3.0, 5.0, 0.5, n=12),
        ).set("z", rng.standard_normal((2, 2, 12))),
        inc=40.0,
        pa=25.0,
        blur_fwhm=0.5,
        n_r=48,
    )
    npix, fov = 384, 60.0
    image = onp.asarray(disk.render(npix=npix, fov_mas=fov))
    xx, yy = (onp.asarray(a).ravel() for a in image_coordinates(npix, fov))
    u, v = _uv(30, seed=4, qmax=0.2)
    phase = onp.exp(
        -2j
        * onp.pi
        * _MAS2RAD_REF
        * (onp.outer(u, xx) + onp.outer(v, yy))
        / WAVEL
    )
    dft = phase @ image.ravel() / image.sum()
    assert onp.max(onp.abs(dft - _cvis(disk, u, v))) < 5e-3


def test_a_gp_works_in_a_radial_sum_and_a_system():
    profile = RadialSum(
        ring=GaussianRing(3.0, 0.5),
        halo=RadialGP(1.0, 9.0, ell=2.0, weight=0.2),
    )
    scene = System(
        star=PointSource(), disk=PolarDisk(profile, n_r=16, flux=0.5)
    )
    assert list(scene.disk.gp_sites()) == ["profile.halo"]
    priors = gp_priors(scene, sigma=(0.1, 3.0))
    assert set(priors) == {"disk.profile.halo.z", "disk.profile.halo.sigma"}
    vis = _cvis(scene, *_uv(5))
    assert onp.all(onp.isfinite(vis))


# --- PolarKernel -----------------------------------------------------------------


def test_the_polar_kernel_variance_falls_with_the_order_as_the_spectral_density():
    kernel = PolarKernel(
        ell_r=3.0, ell_theta=4.0, sigma=0.5, r_in=2.0, r_out=10.0
    )
    degree = 5
    values = onp.asarray(kernel.draws(jax.random.PRNGKey(0), degree, n=20000))
    # shape (n, 2, degree, n_r): the variance of every coefficient (c and s,
    # re and im alike) at every radius.
    var = values.var(axis=0).mean(axis=(0, 2))
    assert var.shape == (degree,)
    assert onp.all(onp.diff(var) < 0.0)
    expected = 0.25 * onp.asarray(kernel.spectrum(degree))
    assert onp.allclose(var, expected, rtol=0.1)
    # Closed form: S(m)/S(1) = ((1 + a^2) / (1 + (m a)^2))^(nu + 1/2).
    a = 4.0 / 6.0
    m = onp.arange(1, degree + 1)
    assert onp.allclose(
        onp.asarray(kernel.spectrum(degree)),
        ((1 + a**2) / (1 + (m * a) ** 2)) ** 2.0,
        rtol=1e-5,
    )
    # Diagonal in m, and a Matern correlation in r with the length ell_r.
    flat = values.transpose(1, 2, 0, 3)  # (kind, m, draw, r)
    corr = onp.corrcoef(flat[0, 0, :, 0], flat[0, 1, :, 0])[0, 1]
    assert abs(corr) < 0.05
    nodes = onp.asarray(kernel.nodes())
    cov = onp.cov(flat[1, 0].T)
    expected_corr = _matern15(nodes, 3.0)
    got = cov / onp.sqrt(onp.outer(onp.diag(cov), onp.diag(cov)))
    assert onp.max(onp.abs(got - expected_corr)) < 0.06


def test_a_longer_azimuthal_length_suppresses_high_orders():
    short = PolarKernel(2.0, 0.5, r_in=1.0, r_out=9.0).spectrum(4)
    long = PolarKernel(2.0, 20.0, r_in=1.0, r_out=9.0).spectrum(4)
    assert onp.all(onp.asarray(long)[1:] < onp.asarray(short)[1:])
    assert onp.isclose(float(short[0]), 1.0)


@pytest.mark.parametrize("form", ["positive", "linear"])
def test_polar_kernel_harmonics_in_a_disk(form):
    kernel = PolarKernel(ell_r=3.0, ell_theta=4.0, sigma=0.15, n=10)
    harmonics = Harmonics(degree=3, prior=kernel, form=form)
    assert harmonics.z.shape == (2, 3, 10) and harmonics.form == form
    z = onp.random.default_rng(5).standard_normal((2, 3, 10))
    disk = PolarDisk(
        GaussianRing(5.0, 1.0),
        harmonics=harmonics.set("z", z),
        inc=30.0,
        n_r=24,
    )
    # The range is the profile's; the coefficients are the kernel's values.
    assert onp.isclose(float(disk.harmonics.prior.r_in), 0.0)
    assert float(disk.harmonics.prior.r_out) > 5.0
    r, _ = disk.profile.quadrature()
    values = onp.asarray(disk.harmonics.prior.values(disk.harmonics.z, r))
    pair = disk.harmonics._pair(r)
    assert onp.allclose(values[0], pair[0]) and onp.allclose(
        values[1], pair[1]
    )
    if form == "positive":
        assert float(disk.min_intensity()) >= 0.0
    assert onp.all(onp.isfinite(_cvis(disk, *_uv(8))))
    # A centred disk with odd harmonics has non-trivial closure phase signal.
    assert onp.max(onp.abs(_cvis(disk, *_uv(8)).imag)) > 0.0
    with pytest.raises(ValueError, match="needs its degree"):
        Harmonics(prior=kernel)
    with pytest.raises(ValueError, match="do not give"):
        Harmonics(c=0.1, degree=2, prior=kernel)


# --- spectral_index ------------------------------------------------------------


def _ring_profile(r, wavel, wavel0, alpha):
    return onp.exp(-0.5 * ((r - 4.0) / 0.7) ** 2) * (wavel / wavel0) ** alpha(
        r
    )


def test_spectral_index_matches_a_brute_force_hankel_transform():
    wavel0 = 3.5e-6
    alpha = RadialLinear(-1.0, 0.5, 4.0)
    disk = PolarDisk(
        GaussianRing(4.0, 0.7),
        spectral_index=alpha,
        wavel0=wavel0,
        flux=0.7,
        n_r=48,
    )
    u, v = _uv(25, seed=7, qmax=0.3)
    rr = onp.linspace(0.0, 8.0, 40001)
    for wavel in (2.5e-6, 3.5e-6, 4.5e-6):
        q = onp.hypot(u, v) / wavel * _MAS2RAD_REF
        intensity = _ring_profile(rr, wavel, wavel0, alpha) * rr
        bessel = scipy_j0(2.0 * onp.pi * onp.outer(q, rr))
        expected = (bessel * intensity).sum(1) / intensity.sum()
        assert onp.max(onp.abs(_cvis(disk, u, v, wavel) - expected)) < 3e-5
        # V(0) = 1 at every wavelength.
        assert onp.isclose(
            complex(disk.model(np.zeros(1), np.zeros(1), wavel)[0]), 1.0
        )
        # The weight is the light at that wavelength relative to wavel0.
        light = (
            intensity.sum()
            / (_ring_profile(rr, wavel0, wavel0, alpha) * rr).sum()
        )
        assert onp.isclose(float(disk._weight(wavel)), 0.7 * light, rtol=1e-4)
    assert onp.isclose(float(disk._weight(wavel0)), 0.7)
    assert onp.isclose(float(disk._weight()), 0.7)


def test_chromatic_model_equals_the_per_wavelength_model():
    rng = onp.random.default_rng(8)
    disk = PolarDisk(
        RadialGP(0.5, 9.0, sigma=1.0, ell=2.0, z=rng.standard_normal(24)),
        harmonics=Harmonics(w=(0.3 + 0.1j, -0.2j)),
        spectral_index=RadialGP(sigma=0.8, ell=3.0, z=rng.standard_normal(16)),
        wavel0=3.5e-6,
        inc=35.0,
        pa=20.0,
        blur_fwhm=0.4,
        n_r=24,
    )
    scene = System(star=PointSource(flux=1.0), disk=disk.set("flux", 0.6))
    u, v = _uv(30, seed=9)
    wavel = onp.linspace(2.8e-6, 4.4e-6, 30)
    together = onp.asarray(
        scene.model(np.asarray(u), np.asarray(v), np.asarray(wavel))
    )
    each = onp.array(
        [
            complex(
                scene.model(
                    np.asarray(u[i : i + 1]),
                    np.asarray(v[i : i + 1]),
                    float(w),
                )[0]
            )
            for i, w in enumerate(wavel)
        ]
    )
    assert onp.max(onp.abs(together - each)) < 2e-6
    # The disk's weight inside the system follows its light at each wavelength.
    weights = onp.asarray(scene.disk._weight(np.asarray(wavel)))
    assert weights.shape == wavel.shape and onp.all(weights > 0.0)
    assert not onp.allclose(weights, weights[0])
    # Harmonics are grey: the image harmonics do not depend on wavelength.
    assert onp.allclose(
        onp.asarray(disk.harmonics.image_harmonics(np.ones(3))),
        onp.asarray(disk.harmonics.image_harmonics(np.ones(3))),
    )


def test_spectral_index_makes_the_inner_edge_redder_than_the_outer():
    rings = RadialSum(
        inner=GaussianRing(2.0, 0.4), outer=GaussianRing(6.0, 0.4)
    )
    disk = PolarDisk(
        rings,
        spectral_index=RadialLinear(0.0, 1.0, 4.0),
        wavel0=3.5e-6,
        n_r=48,
    )
    # Brighter at long wavelengths where alpha > 0 (the outer ring): the
    # outer ring's share grows with wavelength, so the visibility changes.
    u = np.array([15.0])
    short = complex(disk.model(u, 0.0 * u, 2.5e-6)[0])
    long = complex(disk.model(u, 0.0 * u, 4.5e-6)[0])
    assert abs(short - long) > 1e-3


def test_spectral_index_validation():
    from virgil.spectra import PowerLaw

    with pytest.raises(ValueError, match="scalar flux"):
        PolarDisk(
            GaussianRing(4.0, 1.0),
            spectral_index=0.5,
            wavel0=3.5e-6,
            flux=PowerLaw(1.0, index=0.0, wavel0=3.5e-6),
        )
    with pytest.raises(ValueError, match="wavel0"):
        PolarDisk(GaussianRing(4.0, 1.0), spectral_index=0.5)
    with pytest.raises(TypeError, match="not a profile"):
        PolarDisk(
            GaussianRing(4.0, 1.0),
            spectral_index=GaussianRing(1.0, 1.0),
            wavel0=3.5e-6,
        )
    # A disk without a spectral index still takes a spectrum.
    PolarDisk(
        GaussianRing(4.0, 1.0), flux=PowerLaw(1.0, index=0.0, wavel0=3.5e-6)
    )


# --- freeze_geometry ------------------------------------------------------------


def _frozen_pair(x64=True):
    rng = onp.random.default_rng(10)
    disk = PolarDisk(
        RadialGP(0.5, 9.0, sigma=1.0, ell=2.0, z=rng.standard_normal(24)),
        harmonics=Harmonics(
            degree=2, prior=PolarKernel(3.0, 4.0, 0.3, n=8)
        ).set("z", rng.standard_normal((2, 2, 8))),
        height=Cone(55.0, 0.5),
        spectral_index=RadialGP(sigma=0.5, ell=3.0, z=rng.standard_normal(16)),
        wavel0=3.5e-6,
        inc=50.0,
        pa=30.0,
        blur_fwhm=0.3,
        n_r=24,
    )
    return disk, disk.freeze_geometry(DATA)


def test_a_frozen_disk_equals_the_unfrozen_model_exactly():
    with jax.enable_x64(True):
        disk, frozen = _frozen_pair()
        assert disk.frozen is None and frozen.frozen is not None
        got = frozen.model(DATA.u, DATA.v, DATA.wavel)
        expected = disk.model(DATA.u, DATA.v, DATA.wavel)
        assert onp.max(onp.abs(onp.asarray(got - expected))) < 1e-12
        # Refitting the profile keeps the cache valid and exact.
        z = onp.random.default_rng(11).standard_normal(24)
        refit_f = frozen.set("profile.z", z)
        refit_u = disk.set("profile.z", z)
        diff = refit_f.model(DATA.u, DATA.v, DATA.wavel) - refit_u.model(
            DATA.u, DATA.v, DATA.wavel
        )
        assert onp.max(onp.abs(onp.asarray(diff))) < 1e-12


def test_a_frozen_disk_with_no_harmonics_is_a_matrix_vector_product():
    disk = PolarDisk(
        RadialGP(0.5, 9.0, z=np.linspace(-1.0, 1.0, 24)), inc=30.0, n_r=24
    )
    frozen = disk.freeze_geometry(DATA)
    assert frozen.frozen.cos == () and frozen.frozen.p0.shape == (
        DATA.u.shape[0],
        24,
    )
    got = onp.asarray(frozen.model(DATA.u, DATA.v, DATA.wavel))
    expected = onp.asarray(disk.model(DATA.u, DATA.v, DATA.wavel))
    assert onp.max(onp.abs(got - expected)) < 1e-6


def test_a_frozen_disk_falls_back_when_the_geometry_changes():
    with jax.enable_x64(True):
        disk, frozen = _frozen_pair()
        for path, value in (
            ("inc", 20.0),
            ("pa", 80.0),
            ("profile.r_out", 8.0),
        ):
            value = np.asarray(value, dtype=float)
            moved_f, moved_u = frozen.set(path, value), disk.set(path, value)
            diff = moved_f.model(DATA.u, DATA.v, DATA.wavel) - moved_u.model(
                DATA.u, DATA.v, DATA.wavel
            )
            assert onp.max(onp.abs(onp.asarray(diff))) < 1e-12, path
        # Other baselines of the same shape: still correct.
        other = frozen.model(1.1 * DATA.u, DATA.v, DATA.wavel)
        expected = disk.model(1.1 * DATA.u, DATA.v, DATA.wavel)
        assert onp.max(onp.abs(onp.asarray(other - expected))) < 1e-12
        assert (
            float(
                onp.max(
                    onp.abs(
                        onp.asarray(
                            other - frozen.model(DATA.u, DATA.v, DATA.wavel)
                        )
                    )
                )
            )
            > 1e-3
        )


def test_a_frozen_profile_refit_is_faster_and_does_not_recompile():
    n_r = 64
    big = vlti_oidata(
        hour_angles_h=tuple(onp.linspace(-4, 4, 20)),
        wavelengths_m=onp.linspace(3.0e-6, 4.0e-6, 20),
    )
    disk = PolarDisk(
        RadialGP(0.5, 12.0, sigma=1.0, ell=2.0), inc=40.0, pa=10.0, n_r=n_r
    )
    frozen = disk.freeze_geometry(big)

    @eqx.filter_jit
    def evaluate(model, data):
        return model.model(data.u, data.v, data.wavel)

    def best(model, repeats=15):
        evaluate(model, big).block_until_ready()
        times = []
        for i in range(repeats):
            m = model.set("profile.z", 0.01 * (i + 1) * np.ones(n_r))
            start = time.perf_counter()
            evaluate(m, big).block_until_ready()
            times.append(time.perf_counter() - start)
        return min(times)

    assert best(frozen) < best(disk)
    # Changing values never recompiles, frozen or not.
    for model in (disk, frozen):
        evaluate(model, big).block_until_ready()
        with count_compiles() as compiles:
            for i in range(3):
                m = model.set("profile.z", 0.1 * i * np.ones(n_r))
                m = m.set("profile.sigma", np.asarray(1.0 + i, dtype=float))
                m = m.set("profile.ell", np.asarray(1.0 + i, dtype=float))
                evaluate(m, big).block_until_ready()
        assert not compiles


# --- fit, NUTS, log_evidence ------------------------------------------------------


def _small_problem():
    truth = PolarDisk(GaussianRing(4.0, 0.7), n_r=32)
    data = simulate(truth, SMALL, key=jax.random.PRNGKey(0))
    start = PolarDisk(RadialGP(0.5, 9.0, sigma=1.0, ell=2.0), n_r=12)
    return start, data


def test_nuts_samples_the_whitened_latents_and_the_hyperparameters():
    start, data = _small_problem()
    priors = gp_priors(start, sigma=(0.3, 3.0), ell=(1.0, 6.0))
    assert set(priors) == {"profile.z", "profile.sigma", "profile.ell"}
    result = fit(start, {"profile.z": priors["profile.z"]}, data)
    init = {**result.values, "profile.sigma": 1.0, "profile.ell": 2.0}
    for flat in (False, True):
        model = numpyro_model(start, priors, data, flat_coordinates=flat)
        mcmc = MCMC(
            NUTS(
                model,
                init_strategy=numpyro_init(init),
            ),
            num_warmup=150,
            num_samples=150,
            progress_bar=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mcmc.run(jax.random.PRNGKey(1))
        samples = mcmc.get_samples()
        assert samples["profile.z"].shape == (150, 12)
        assert onp.all(onp.isfinite(onp.asarray(samples["profile.z"])))
        sigma = onp.asarray(samples["profile.sigma"])
        ell = onp.asarray(samples["profile.ell"])
        assert sigma.min() >= 0.3 and sigma.max() <= 3.0
        assert ell.min() >= 1.0 and ell.max() <= 6.0
        assert onp.std(onp.asarray(samples["profile.z"]), axis=0).min() > 0.0


def numpyro_init(values):
    from numpyro.infer.initialization import init_to_value

    return init_to_value(values=values)


def test_log_evidence_runs_for_every_gaussian_process_part():
    truth = System(
        star=PointSource(),
        disk=PolarDisk(
            GaussianRing(4.0, 0.7),
            harmonics=Harmonics(c=0.3, s=0.1),
            spectral_index=0.5,
            wavel0=3.6e-6,
            flux=0.5,
            n_r=32,
        ),
    )
    data = simulate(truth, SMALL, key=jax.random.PRNGKey(2))
    start = System(
        star=PointSource(),
        disk=PolarDisk(
            RadialGP(0.5, 9.0, sigma=1.5, ell=2.0),
            harmonics=Harmonics(
                degree=1, prior=PolarKernel(3.0, 4.0, 0.3, n=8)
            ),
            spectral_index=RadialGP(sigma=0.5, ell=4.0),
            wavel0=3.6e-6,
            flux=0.5,
            n_r=12,
        ),
    )
    priors = gp_priors(start)
    assert set(priors) == {
        "disk.profile.z",
        "disk.harmonics.z",
        "disk.spectral_index.z",
    }
    result = fit(start, priors, data)
    everything = log_evidence(result, data, path=None)
    assert onp.isfinite(everything)
    assert onp.isclose(everything, log_evidence(result, data, path="disk"))
    # A subset of the latents gives a different, finite evidence.
    assert models.PolarDisk.gp_sites  # public
    from virgil import imaging

    assert imaging._evidence_latents(result.model, "disk", "x") == (
        "disk.profile.z",
        "disk.harmonics.z",
        "disk.spectral_index.z",
    )
    with pytest.raises(TypeError, match="RadialGP or a PolarKernel"):
        log_evidence(
            System(star=PointSource(), disk=PolarDisk(GaussianRing(4.0, 1.0))),
            data,
            path="disk",
        )


def test_the_evidence_prefers_the_correlation_length_the_data_came_from():
    # Truth drawn from a GP profile of length 2; evidence at that length
    # beats a far-too-short and a far-too-long one, as MacKay's framework.
    gp = RadialGP(0.5, 9.0, sigma=1.5, ell=2.0, n=16)
    z = jax.random.normal(jax.random.PRNGKey(4), (16,))
    truth = PolarDisk(gp.set("z", z), n_r=16)
    data = simulate(truth, DATA, key=jax.random.PRNGKey(5))
    evidence = {}
    for ell in (0.2, 2.0, 12.0):
        start = PolarDisk(gp.set("ell", ell), n_r=16)
        result = fit(start, gp_priors(start), data)
        evidence[ell] = log_evidence(result, data, path=None)
    assert evidence[2.0] > evidence[0.2] and evidence[2.0] > evidence[12.0]


def test_fitting_a_gp_hyperparameter_by_map_warns():
    start, data = _small_problem()
    priors = gp_priors(start, sigma=(0.3, 3.0))
    with pytest.warns(UserWarning, match="biased at the MAP"):
        fit(start, priors, data, max_steps=2)


def test_gp_priors_for_a_polar_kernel():
    disk = PolarDisk(
        GaussianRing(4.0, 1.0),
        harmonics=Harmonics(degree=2, prior=PolarKernel(2.0, 3.0, n=6)),
        n_r=16,
    )
    priors = gp_priors(
        disk, sigma=(0.05, 1.0), ell=(0.5, 5.0), ell_theta=(0.5, 20.0)
    )
    assert set(priors) == {
        "harmonics.z",
        "harmonics.prior.sigma",
        "harmonics.prior.ell_r",
        "harmonics.prior.ell_theta",
    }
    assert priors["harmonics.z"].event_shape == (2, 2, 6)
    assert isinstance(priors["harmonics.prior.sigma"], dist.LogUniform)
    with pytest.raises(ValueError, match="no RadialGP"):
        gp_priors(PolarDisk(ThinRing(3.0)))
    with pytest.raises(ValueError, match="bounds"):
        gp_priors(disk, sigma=(2.0, 1.0))
