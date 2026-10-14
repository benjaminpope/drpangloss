import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil._geometry import pixel_offsets
from virgil._utils import mas2rad
from virgil.coverage import vlti_oidata
from virgil.fields import PolarField, polar_field_priors, polar_interpolate
from virgil.fitting import fit
from virgil.imaging import image_priors, log_evidence
from virgil.models import (
    GaussianDisk,
    GaussianRing,
    Image,
    PointSource,
    PolarDisk,
    System,
)

from ._compiles import count_compiles

N_R, N_T, SHAPE, R_MAX = 8, 12, (24, 24), 12.0


def _field(seed=0, shape=SHAPE, **kwargs):
    latent = onp.random.default_rng(seed).normal(size=(N_R, N_T))
    options = (
        dict(sigma=1.0, length_r=3.0, length_theta=1.0, inc=40.0, pa=25.0)
        | kwargs
    )
    return PolarField(latent, shape, R_MAX, **options)


def _matern(d, ell, nu):
    x = d / ell
    if nu == 0.5:
        return onp.exp(-x)
    if nu == 1.5:
        return (1 + onp.sqrt(3) * x) * onp.exp(-onp.sqrt(3) * x)
    a = onp.sqrt(5) * x
    return (1 + a + a * a / 3) * onp.exp(-a)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_covariance_is_the_separable_polar_matern_kernel(nu):
    sigma, ell_r, ell_t = 1.7, 3.0, 0.8
    field = _field(sigma=sigma, length_r=ell_r, length_theta=ell_t, nu=nu)
    jac = jax.jacfwd(lambda z: field.set("latent", z).polar_values().ravel())(
        field.latent
    )
    jac = onp.asarray(jac).reshape(N_R * N_T, -1)
    r = onp.arange(N_R) * field.spacing
    th = 2 * onp.pi * onp.arange(N_T) / N_T
    k_r = _matern(onp.abs(r[:, None] - r[None]), ell_r, nu)
    k_t = _matern(
        2 * onp.abs(onp.sin(0.5 * (th[:, None] - th[None]))), ell_t, nu
    )
    expected = sigma**2 * onp.kron(k_r, k_t)
    assert onp.allclose(jac @ jac.T, expected, atol=2e-5 * sigma**2)
    # sigma is the standard deviation of the field.
    assert onp.mean(onp.diag(jac @ jac.T)) == pytest.approx(sigma**2, rel=1e-4)


def test_latents_are_whitened_and_zero_latent_is_the_template():
    # An empirical check that standard-normal latents give variance sigma^2.
    sigma = 2.0
    field = _field(sigma=sigma)
    draws = onp.random.default_rng(1).normal(size=(400, N_R, N_T))
    values = jax.vmap(lambda z: field.set("latent", z).polar_values())(draws)
    assert float(np.std(values)) == pytest.approx(sigma, rel=0.1)
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    flat = _field(mean=template).set("latent", np.zeros((N_R, N_T)))
    floor = 1e-3
    expected = onp.log(template / template.max() + floor)
    assert onp.allclose(flat.evaluate(1.0), expected)


def test_interpolation_is_second_order_and_exact_at_nodes():
    def smooth(r, th):
        return onp.cos(2 * th) * onp.exp(
            -0.5 * (r / 5.0) ** 2
        ) + 0.3 * onp.sin(th)

    rng = onp.random.default_rng(2)
    r = rng.uniform(1.0, 9.0, 400)
    th = rng.uniform(0.0, 2 * onp.pi, 400)

    def error(n_r, n_t):
        spacing = 10.0 / (n_r - 1)
        grid_r = onp.arange(n_r) * spacing
        grid_t = 2 * onp.pi * onp.arange(n_t) / n_t
        values = smooth(grid_r[:, None], grid_t[None, :])
        got = polar_interpolate(values, 0.0, spacing, r, th)
        return onp.max(onp.abs(onp.asarray(got) - smooth(r, th)))

    coarse, fine = error(21, 32), error(41, 64)
    assert fine < 0.3 * coarse and fine < 2e-2
    # Exact at the nodes, periodic in azimuth, zero well beyond the grid.
    values = onp.random.default_rng(3).normal(size=(5, 8))
    node = polar_interpolate(
        values, 1.0, 2.0, np.array(3.0), np.array(2 * onp.pi * 3 / 8)
    )
    assert float(node) == pytest.approx(values[1, 3], abs=1e-5)
    wrap = polar_interpolate(
        values, 1.0, 2.0, np.array([4.0] * 2), np.array([0.0, 2 * onp.pi])
    )
    assert float(wrap[0]) == pytest.approx(float(wrap[1]), abs=1e-5)
    beyond = polar_interpolate(
        values, 1.0, 2.0, np.array([11.5, 99.0]), np.array([1.0, 1.0])
    )
    assert onp.allclose(beyond, 0.0)


def test_pixels_follow_the_disk_frame():
    field = _field(shape=(25, 25), inc=60.0, pa=0.0)
    h = 1.0
    eta = onp.asarray(field.evaluate(h))
    offsets = onp.asarray(pixel_offsets(25, h))  # (centre - index) * h
    index = lambda offset: int(onp.argmin(onp.abs(offsets - offset)))
    # An inclined circle of radius a: the major axis is North (pa = 0), the
    # minor axis East-West, foreshortened by cos(inc) = 1/2.
    a = 8.0
    north = eta[index(a), index(0.0)]
    east = eta[index(0.0), index(a * 0.5)]
    west = eta[index(0.0), index(-a * 0.5)]
    # (East is x > 0, which is column index below the centre.)
    # The deprojected East point has r = a and azimuth 90 degrees.
    expected = polar_interpolate(
        field.polar_values(),
        0.0,
        field.spacing,
        np.array(a),
        np.array(onp.pi / 2),
    )
    assert float(east) == pytest.approx(float(expected), abs=1e-5)
    assert float(north) == pytest.approx(
        float(
            polar_interpolate(
                field.polar_values(),
                0.0,
                field.spacing,
                np.array(a),
                np.array(0.0),
            )
        ),
        abs=1e-5,
    )
    # West is at azimuth 270 degrees.
    assert float(west) == pytest.approx(
        float(
            polar_interpolate(
                field.polar_values(),
                0.0,
                field.spacing,
                np.array(a),
                np.array(1.5 * onp.pi),
            )
        ),
        abs=1e-5,
    )


def test_a_long_azimuthal_length_gives_an_axisymmetric_inclined_field():
    field = _field(shape=(25, 25), inc=60.0, pa=0.0, length_theta=1e4)
    offsets = onp.asarray(pixel_offsets(25, 1.0))
    eta = onp.asarray(field.evaluate(1.0))
    index = lambda offset: int(onp.argmin(onp.abs(offsets - offset)))
    assert eta[index(8.0), index(0.0)] == pytest.approx(
        eta[index(0.0), index(4.0)], abs=1e-3
    )


def test_visibilities_match_a_direct_dft_of_the_rendered_image():
    ps, wavel = 1.0, 2.0e-6
    image = Image(_field(), ps, flux=1.0)
    u = onp.array([5.0, -12.0, 20.0, 31.0])
    v = onp.array([8.0, 3.0, -17.0, 9.0])
    got = onp.asarray(image.model(u, v, wavel))
    brightness = onp.asarray(image.brightness)
    assert brightness.sum() == pytest.approx(1.0)
    # The rendering reproduces the pixels at their centres.
    rendered = onp.asarray(image.render(SHAPE[0], SHAPE[0] * ps))
    assert onp.allclose(rendered / rendered.sum(), brightness, atol=1e-6)
    x = onp.asarray(pixel_offsets(SHAPE[1], ps)) * mas2rad
    y = onp.asarray(pixel_offsets(SHAPE[0], ps)) * mas2rad
    phase = (
        -2j
        * onp.pi
        * (
            (u / wavel)[:, None, None] * x[None, None, :]
            + (v / wavel)[:, None, None] * y[None, :, None]
        )
    )
    expected = onp.sum(brightness * onp.exp(phase), axis=(1, 2))
    assert onp.allclose(got, expected, atol=1e-6)
    # The same field as plain pixels gives the same visibilities.
    plain = Image(image.eta, ps)
    assert onp.allclose(plain.model(u, v, wavel), got, atol=1e-6)


def test_gradients_are_finite_in_every_parameter():
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    base = _field(mean=template)
    u, v = onp.array([20.0, -35.0]), onp.array([15.0, 40.0])
    names = [
        "latent", "sigma", "length_r", "length_theta",
        "inc", "pa", "dra", "ddec",
    ]  # fmt: skip

    def power(values):
        field = base.set(names, values)
        return np.sum(np.abs(Image(field, 1.0).model(u, v, 1.6e-6)) ** 2)

    values = [getattr(base, n) for n in names]
    grads = jax.grad(power)(values)
    for name, g in zip(names, grads):
        assert onp.all(onp.isfinite(onp.asarray(g))), name
        assert onp.any(onp.asarray(g) != 0.0), name


def test_changing_values_does_not_recompile():
    u, v = onp.array([20.0, -35.0]), onp.array([15.0, 40.0])

    @eqx.filter_jit
    def power(field):
        return np.sum(np.abs(Image(field, 1.0).model(u, v, 1.6e-6)) ** 2)

    base = _field()
    power(base)  # compile
    with count_compiles() as compiles:
        for k in range(3):
            changed = base.set(
                ["sigma", "length_r", "length_theta", "inc", "pa", "dra"],
                [
                    np.asarray(x, dtype=float)
                    for x in (
                        1.0 + k,
                        3.0 + k,
                        0.5 + k,
                        30.0 + k,
                        10.0 * k,
                        0.1 * k,
                    )
                ],
            )
            power(changed)
    assert compiles == []


def test_from_disk_copies_the_geometry():
    disk = PolarDisk(
        GaussianRing(r0=6.0, sigma=1.0), inc=35.0, pa=70.0, dra=0.5, ddec=-0.25
    )
    field = PolarField.from_disk(disk, onp.zeros((6, 8)), SHAPE)
    assert float(field.inc) == 35.0 and float(field.pa) == 70.0
    assert float(field.dra) == 0.5 and float(field.ddec) == -0.25
    assert field.r_max_mas == pytest.approx(1.2 * disk.profile.extent()[1])


def test_hyperpriors_are_bounded_log_uniform():
    priors = polar_field_priors("env.log_brightness", sigma=(0.1, 2.0))
    assert set(priors) == {
        "env.log_brightness.sigma",
        "env.log_brightness.length_r",
        "env.log_brightness.length_theta",
    }
    assert all(isinstance(p, dist.LogUniform) for p in priors.values())
    assert float(priors["env.log_brightness.sigma"].high) == 2.0
    with pytest.raises(ValueError):
        polar_field_priors(sigma=(2.0, 1.0))


def test_a_polar_field_image_fits_and_has_a_finite_evidence():
    data = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])
    truth = System(
        star=PointSource(), env=GaussianDisk(3.0, dra=2.0, flux=0.4)
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(0))
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    field = _field(mean=template, inc=30.0, pa=10.0)
    start = System(star=PointSource(), env=Image(field, 1.0, flux=0.4))
    priors = image_priors(start)
    assert list(priors) == ["env.log_brightness.latent"]
    assert priors["env.log_brightness.latent"].event_shape == (N_R, N_T)
    result = fit(start, priors, data)
    first = fit(start, priors, data, max_steps=1)
    assert result.info["chi2_red"] < first.info["chi2_red"]
    assert onp.isfinite(log_evidence(result, data))


def test_fitting_polar_field_scales_by_map_warns():
    start = System(star=PointSource(), env=Image(_field(), 1.0, flux=0.4))
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    priors = image_priors(start) | polar_field_priors()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(start, priors, data, max_steps=1)
    assert any(
        "hyperparameters of a Gaussian field" in str(w.message) for w in caught
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"sigma": 0.0},
        {"length_r": -1.0},
        {"length_theta": 0.0},
        {"nu": 2.0},
        {"r_max_mas": 0.0},
        {"r_min_mas": 20.0},
        {"mean": onp.zeros((4, 4))},
        {"mean": -onp.ones(SHAPE)},
        {"image_shape": (24,)},
    ],
)
def test_bad_polar_field_parameters_are_rejected(bad):
    kwargs = dict(
        latent=onp.zeros((N_R, N_T)), image_shape=SHAPE, r_max_mas=R_MAX
    )
    with pytest.raises(ValueError):
        PolarField(**(kwargs | bad))
