import jax
import numpy as onp
import pytest
import jax.numpy as np
from scipy.special import jv

from drpangloss._utils import (
    apply_elliptical_transf_coord,
    apply_elliptical_transf_spat_freq,
    bessel_jn,
    check_az_prof_nonnegative,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_bessel_jn_matches_scipy(order):
    # The upward recurrence used by `bessel_jn` loses accuracy once the
    # order approaches the argument (small x, higher order), so the
    # comparison range stays away from that regime; this is the same
    # numerical caveat that applies to the ported CEPHES-derived recursion.
    x = np.linspace(3.0, 30.0, 37)
    result = bessel_jn(order, x)[order]
    expected = jv(order, onp.asarray(x))
    assert onp.allclose(onp.asarray(result), expected, atol=1e-6)


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
