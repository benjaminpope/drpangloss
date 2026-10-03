import jax
import numpy as onp
import pytest
import jax.numpy as np
from jax.test_util import check_grads
from scipy.special import jv, jvp

from drpangloss._geometry import (
    apply_elliptical_transf_coord,
    apply_elliptical_transf_spat_freq,
    check_az_prof_nonnegative,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)
from drpangloss.bessel import bessel_jn, j0, j1


@pytest.fixture(autouse=True)
def _float64_by_default():
    """Run these tests in float64, without changing the rest of the suite.

    Setting ``jax_enable_x64`` globally at import time made the precision of
    every other test module depend on collection order.
    """
    with jax.enable_x64(True):
        yield


# (jax_enable_x64, expected dtype, absolute tolerance vs scipy)
BESSEL_PRECISIONS = [
    pytest.param(False, "float32", 2e-6, id="float32"),
    pytest.param(True, "float64", 1e-14, id="float64"),
]


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_bessel_jn_low_orders_match_scipy(order, x64, dtype, atol):
    # Covers x = 0, negative x, the small-x regime where the upward
    # recurrence alone is unstable, and the switch to the recurrence.
    xs = onp.linspace(-30.0, 30.0, 1201)
    expected = onp.array([jv(m, xs) for m in range(order + 1)])
    with jax.enable_x64(x64):
        result = bessel_jn(order, np.asarray(xs))
    assert result.dtype == dtype
    assert result.shape == (order + 1, xs.size)
    assert onp.allclose(onp.asarray(result), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_bessel_jn_low_order_gradients_match_scipy(order, x64, dtype, atol):
    # d/dx J_n = (J_{n-1} - J_{n+1}) / 2, with J_{-1} = -J_1.
    xs = onp.linspace(-30.0, 30.0, 1201)
    expected = 0.5 * (jv(order - 1, xs) - jv(order + 1, xs))
    with jax.enable_x64(x64):
        grad = jax.vmap(jax.grad(lambda z: bessel_jn(order, z)[order]))(
            np.asarray(xs)
        )
    assert grad.dtype == dtype
    assert onp.allclose(onp.asarray(grad), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", [5, 8, 16])
def test_bessel_jn_high_orders_match_scipy(order):
    xs = onp.linspace(-60.0, 60.0, 2401)
    expected = onp.array([jv(m, xs) for m in range(order + 1)])
    result = bessel_jn(order, np.asarray(xs))
    assert onp.allclose(onp.asarray(result), expected, rtol=0.0, atol=1e-14)


def test_bessel_jn_accepts_scalars():
    assert bessel_jn(3, 2.5).shape == (4,)
    assert onp.allclose(bessel_jn(3, 2.5), [jv(m, 2.5) for m in range(4)])


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
def test_bessel_integer_inputs_are_promoted_to_float(x64, dtype, atol):
    # The coefficient tables are cast to the argument's dtype, so integer
    # arguments must reach them already promoted to float.
    xs = onp.arange(-12, 13)
    with jax.enable_x64(x64):
        x = np.asarray(xs)
        assert onp.allclose(j0(x), jv(0, xs), rtol=0, atol=atol)
        assert onp.allclose(j1(x), jv(1, xs), rtol=0, atol=atol)
        result = bessel_jn(4, x)
    assert result.dtype == dtype
    expected = [jv(m, xs) for m in range(5)]
    assert onp.allclose(result, expected, rtol=0, atol=atol)


def test_bessel_jn_float32_after_float64_call():
    # A fit runs in float64 inside jax.enable_x64 and the model is then
    # evaluated in float32. On JAX 0.10 this reused a float64 polyval
    # lowering for the float32 call and failed MLIR verification. The odd
    # size keeps the traces fresh whatever other tests ran first.
    xs = onp.linspace(-12.0, 12.0, 37)
    expected = onp.array([jv(m, xs) for m in range(5)])
    expected_grad = 0.5 * (jv(0, xs) - jv(2, xs))
    grad_j1 = jax.vmap(jax.grad(lambda z: bessel_jn(4, z)[1]))
    for x64, dtype, atol in [
        (True, "float64", 1e-14),
        (False, "float32", 2e-6),
    ]:
        with jax.enable_x64(x64):
            x = np.asarray(xs, dtype=dtype)
            result = bessel_jn(4, x)
            grad = grad_j1(x)
        assert result.dtype == grad.dtype == dtype
        assert onp.allclose(result, expected, rtol=0, atol=atol)
        assert onp.allclose(grad, expected_grad, rtol=0, atol=atol)


# Include x = 0, the CEPHES switch at |x| = 5 and the trig/recurrence switch at
# |x| = n + 2 for the orders tested.
DERIV_XS = onp.unique(
    onp.concatenate(
        [
            onp.linspace(-30.0, 30.0, 1201),
            [0.0, 4.999, 5.001, -5.001, 5.999, 6.001, 9.999, 10.001],
        ]
    )
)


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 8])
def test_bessel_jn_second_derivatives_match_scipy(order, x64, dtype, atol):
    with jax.enable_x64(x64):
        second = jax.vmap(
            jax.grad(jax.grad(lambda z: bessel_jn(order, z)[order]))
        )(np.asarray(DERIV_XS))
    expected = jvp(order, DERIV_XS, 2)
    assert second.dtype == dtype
    assert onp.allclose(onp.asarray(second), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", [1, 4, 8])
def test_bessel_jn_jacobians_all_orders_match_scipy(order):
    xs = np.asarray(DERIV_XS)
    expected = onp.array([jvp(m, DERIV_XS, 1) for m in range(order + 1)])
    for jac in (jax.jacfwd, jax.jacrev):
        # Elementwise in x, so vmapping scalar Jacobians avoids materializing
        # the dense diagonal Jacobian over the x axes.
        diag = onp.asarray(jax.vmap(jac(lambda z: bessel_jn(order, z)))(xs)).T
        assert onp.allclose(diag, expected, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("x", [0.0, 0.3, 4.999, 5.001, -7.5, 9.999, 10.001])
def test_bessel_custom_rules_pass_check_grads(x):
    x = np.asarray(x)
    check_grads(j0, (x,), order=2, modes=("fwd", "rev"))
    check_grads(j1, (x,), order=2, modes=("fwd", "rev"))
    check_grads(lambda z: bessel_jn(8, z), (x,), order=2, modes=("fwd", "rev"))


def test_bessel_derivatives_at_zero_are_exact():
    zero = np.asarray(0.0)
    assert float(jax.grad(j1)(zero)) == pytest.approx(0.5, abs=1e-15)
    assert float(jax.grad(jax.grad(j0))(zero)) == pytest.approx(
        -0.5, abs=1e-15
    )
    assert float(jax.grad(jax.grad(j1))(zero)) == pytest.approx(0.0, abs=1e-15)
    hess = jax.jacfwd(jax.jacrev(lambda z: bessel_jn(4, z)))(zero)
    assert onp.all(onp.isfinite(onp.asarray(hess)))
    assert onp.allclose(onp.asarray(hess), [jvp(m, 0.0, 2) for m in range(5)])


def test_undo_elliptical_transf_coord_is_identity_for_pa0_stretch1():
    x, y = np.array([3.0, -2.0]), np.array([-1.0, 4.0])
    xt, yt = undo_elliptical_transf_coord(x, y, pa=0.0, stretch=1.0)
    assert np.allclose(xt, x)
    assert np.allclose(yt, y)


def test_undo_elliptical_transf_coord_pa90_rotates_east_onto_major_axis():
    """A point due East (x=r, y=0) has PA=90deg from North. Undoing a
    PA=90deg rim rotation should place it on the transformed y-axis (i.e.
    aligned with the de-rotated major axis), verifying the North-to-East
    position angle convention.
    """
    r = 5.0
    xt, yt = undo_elliptical_transf_coord(r, 0.0, pa=90.0, stretch=1.0)
    assert xt == pytest.approx(0.0, abs=1e-10)
    assert yt == pytest.approx(r, abs=1e-10)


def test_undo_elliptical_transf_coord_nontrivial_pa_and_stretch():
    x, y, pa, stretch = 4.0, -6.0, 37.0, 0.6
    pa_rad = onp.deg2rad(pa)
    expected_xt = (x * onp.cos(pa_rad) - y * onp.sin(pa_rad)) / stretch
    expected_yt = x * onp.sin(pa_rad) + y * onp.cos(pa_rad)

    xt, yt = undo_elliptical_transf_coord(x, y, pa=pa, stretch=stretch)

    assert xt == pytest.approx(expected_xt)
    assert yt == pytest.approx(expected_yt)


@pytest.mark.parametrize("pa", [0.0, 37.0, 90.0, 143.0, -25.0])
@pytest.mark.parametrize("stretch", [1.0, 0.6, 0.15])
def test_apply_undoes_elliptical_transf_coord_round_trip(pa, stretch):
    x, y = np.array([3.5, -8.0, 0.0]), np.array([-2.5, 6.0, 4.2])
    xt, yt = undo_elliptical_transf_coord(x, y, pa=pa, stretch=stretch)
    x_rt, y_rt = apply_elliptical_transf_coord(xt, yt, pa=pa, stretch=stretch)
    assert np.allclose(x_rt, x, atol=1e-8)
    assert np.allclose(y_rt, y, atol=1e-8)


@pytest.mark.parametrize("pa", [0.0, 37.0, 90.0, 143.0, -25.0])
@pytest.mark.parametrize("stretch", [1.0, 0.6, 0.15])
def test_apply_undoes_elliptical_transf_spat_freq_round_trip(pa, stretch):
    u, v = np.array([12.0, -30.0, 5.0]), np.array([-8.0, 20.0, 0.0])
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa=pa, stretch=stretch)
    u_rt, v_rt = apply_elliptical_transf_spat_freq(
        ut, vt, pa=pa, stretch=stretch
    )
    assert np.allclose(u_rt, u, atol=1e-8)
    assert np.allclose(v_rt, v, atol=1e-8)


def test_check_az_prof_nonnegative_detects_valid_and_invalid_profiles():
    # 1 + 0.5*cos(theta) never drops below 0.5 -> valid.
    assert bool(
        check_az_prof_nonnegative(
            az_amps=np.array([0.5]), az_pas=np.array([0.0])
        )
    )
    # 1 + 1.5*cos(theta) dips to -0.5 at theta=180deg -> invalid.
    assert not bool(
        check_az_prof_nonnegative(
            az_amps=np.array([1.5]), az_pas=np.array([0.0])
        )
    )


def test_check_az_prof_nonnegative_accepts_no_modulation():
    assert check_az_prof_nonnegative(az_amps=np.array([]), az_pas=np.array([]))
