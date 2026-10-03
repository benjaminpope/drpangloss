"""Independent, whitened closure phases (``drpangloss._closure``)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from drpangloss.coverage import VLTI_UTS, vlti_oidata
from drpangloss.likelihood import model_loglike, whitened_residuals
from drpangloss.models import BinaryModelCartesian
from drpangloss.oidata import OIData

TRUTH = BinaryModelCartesian(dra=5.0, ddec=3.0, flux=0.1)
MODEL = BinaryModelCartesian(dra=4.0, ddec=3.5, flux=0.08)


def _four_telescopes(**kw):
    template = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    )
    return template.with_model(TRUTH, key=jax.random.PRNGKey(0), **kw)


def _incidence(data):
    """The dense triangle-by-baseline matrix T, +1 +1 -1 per closure phase."""
    i1, i2, i3 = (
        onp.asarray(i) for i in (data.i_cps1, data.i_cps2, data.i_cps3)
    )
    t = onp.zeros((i1.size, onp.asarray(data.u).size))
    for row, (a, b, c) in enumerate(zip(i1, i2, i3)):
        t[row, a] += 1.0
        t[row, b] += 1.0
        t[row, c] -= 1.0
    return t


def test_four_telescopes_keep_three_closure_phases_per_frame():
    data = _four_telescopes()
    assert data.cp_noise is not None
    n_vis, n_cp = data.vis.size, data.phi.size  # 3 frames x 4 triangles
    assert n_cp == 12
    assert data.n_independent == n_vis + 9
    assert whitened_residuals(MODEL, data).size == data.n_independent


def test_whitening_matches_the_dense_pseudo_inverse():
    # Equal errors give baseline variances σ²/3, so C = (σ²/3) T Tᵀ, of
    # rank 3 per frame; χ² is Δᵀ C⁺ Δ and the normalisation its pseudo-det.
    data = _four_telescopes()
    t = _incidence(data)
    sigma = onp.asarray(data.d_phi)
    assert onp.allclose(sigma, sigma[0])
    cov = sigma[0] ** 2 / 3.0 * t @ t.T
    prediction = onp.asarray(data.model(MODEL))
    n_vis = data.vis.size
    delta = prediction[n_vis:] - onp.asarray(data.phi)
    chord = 2.0 * onp.sin(0.5 * delta)
    chi2_dense = chord @ onp.linalg.pinv(cov) @ chord
    eig = onp.linalg.eigvalsh(cov)
    logdet_dense = onp.sum(onp.log(eig[eig > 1e-9 * eig.max()]))

    whitened = onp.asarray(whitened_residuals(MODEL, data))
    chi2_phase = onp.sum(whitened[n_vis:] ** 2)
    assert chi2_phase == pytest.approx(chi2_dense, rel=1e-5)
    _, errors = data.cp_noise.whiten(np.asarray(chord), data.d_phi)
    assert 2.0 * onp.sum(onp.log(errors)) == pytest.approx(
        logdet_dense, rel=1e-5
    )


def test_three_telescopes_are_unchanged():
    # One triangle per frame and channel: nothing correlates, and the
    # likelihood is the per-triangle chord one, as before.
    data = vlti_oidata(
        hour_angles_h=(0.0,),
        wavelengths_m=[3.5e-6],
        stations=onp.asarray(VLTI_UTS)[:3],
    ).with_model(TRUTH)
    assert data.cp_noise is None
    assert data.n_independent == data.vis.size + data.phi.size


def test_simulated_noise_comes_from_baseline_phases():
    # Closure-phase noise drawn from baseline-phase noise satisfies the one
    # closure relation of four telescopes in every frame: the alternating
    # sum of the four closure phases' noise is zero.
    noiseless = vlti_oidata(
        hour_angles_h=(0.0,), wavelengths_m=[3.5e-6]
    ).with_model(TRUTH)
    noisy = vlti_oidata(
        hour_angles_h=(0.0,), wavelengths_m=[3.5e-6]
    ).with_model(TRUTH, key=jax.random.PRNGKey(3))
    noise = onp.asarray(noisy.phi) - onp.asarray(noiseless.phi)
    null = onp.linalg.svd(_incidence(noisy).T)[2][-1]  # left null vector of T
    assert abs(null @ noise) < 1e-5 * onp.linalg.norm(noise)
    assert onp.linalg.norm(noise) > 0.0


def test_rescaled_errors_carry_through_the_whitening():
    data = _four_telescopes()
    n_vis = data.vis.size
    once = onp.asarray(whitened_residuals(MODEL, data))[n_vis:]
    twice = onp.asarray(whitened_residuals(MODEL, data.with_error_scale(2.0)))[
        n_vis:
    ]
    assert onp.allclose(twice, 0.5 * once, rtol=1e-5)
    plain = model_loglike(MODEL, data)
    inflated = model_loglike(MODEL, data, phi_error=0.05)
    assert onp.isfinite(plain) and onp.isfinite(inflated)


def test_a_flagged_closure_phase_leaves_the_rest_independent():
    data = _four_telescopes()
    record = {
        "u": data.u,
        "v": data.v,
        "wavel": data.wavel,
        "vis": data.vis,
        "d_vis": data.d_vis,
        "phi": data.phi,
        "d_phi": data.d_phi,
        "i_cps1": data.i_cps1,
        "i_cps2": data.i_cps2,
        "i_cps3": data.i_cps3,
        "phi_flag": onp.r_[True, onp.zeros(data.phi.size - 1, dtype=bool)],
    }
    flagged = OIData(record)
    # 11 closure phases remain: 3 in the first frame (all independent), 4
    # in each of the others (3 independent each).
    assert flagged.phi.size == 11
    assert flagged.n_independent == flagged.vis.size + 9
    assert onp.all(onp.isfinite(whitened_residuals(MODEL, flagged)))


def test_gradients_are_finite():
    data = _four_telescopes()
    grad = jax.grad(lambda f: model_loglike(MODEL.set("flux", f), data))(0.08)
    assert onp.isfinite(grad)


def test_a_correlating_operator_is_rotated_to_independent_outputs():
    # Two overlapping differences of three phases share one input, so their
    # outputs correlate. Keeping only the diagonal would count that input
    # twice; the operator is rotated so the outputs are independent and
    # the χ² equals the dense one.
    sigma = onp.array([0.1, 0.2, 0.3])
    operator = onp.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    phi = onp.array([0.3, -0.1, 0.2])
    data = OIData(
        {
            "u": onp.ones(3),
            "v": onp.zeros(3),
            "wavel": 1e-6,
            "vis": onp.ones(3),
            "d_vis": onp.full(3, 0.1),
            "phi": phi,
            "d_phi": sigma,
            "phi_mat": operator,
            "cp_flag": False,
        }
    )
    projected = onp.asarray(data.phi_mat) * sigma[None, :]
    assert onp.allclose(
        projected @ projected.T, onp.diag(onp.asarray(data.d_phi) ** 2)
    )
    cov = (operator * sigma) @ (operator * sigma).T
    y = operator @ phi
    chi2_dense = y @ onp.linalg.solve(cov, y)
    chi2 = onp.sum((onp.asarray(data.phi) / onp.asarray(data.d_phi)) ** 2)
    assert chi2 == pytest.approx(chi2_dense, rel=1e-6)
