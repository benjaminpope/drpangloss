from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from drpangloss.amigo import load_oi_data
from drpangloss.likelihood import model_loglike, whitened_residuals
from drpangloss.models import BinaryModelCartesian

from ._test_data import oidata_sim, true_values

PRODUCT = (
    Path(__file__).resolve().parents[1] / "data" / "calibrated_visibility.npy"
)
TRUTH = BinaryModelCartesian(*true_values)


def _shift_phases(data, shift):
    """Closure-phase data equal to the TRUTH model plus ``shift`` radians."""
    exact = data.model(TRUTH)[data.vis.size :]
    return eqx.tree_at(lambda d: d.phi, data, exact + shift)


def test_whitened_residuals_are_residuals_over_sigma_for_small_phases():
    model = BinaryModelCartesian(240.0, 160.0, 6e-4)
    data, errors = oidata_sim.flatten_data()
    # These residuals are far below π, so no wrapping is needed (wrapping
    # via mod(Δ + π, 2π) - π would itself cost float32 precision).
    delta = oidata_sim.model(model) - data
    assert np.max(np.abs(delta)) < 0.1
    assert np.allclose(
        whitened_residuals(model, oidata_sim), delta / errors, rtol=1e-5
    )


def test_phase_term_is_von_mises_and_smooth_across_pi():
    n_vis = oidata_sim.vis.size
    sigma = oidata_sim.d_phi
    for shift in (0.3, 3.0, np.pi, 3.3, 2.0 * np.pi - 0.3):
        data = _shift_phases(oidata_sim, shift)
        phase = whitened_residuals(TRUTH, data)[n_vis:]
        assert np.allclose(
            (phase * sigma) ** 2, 2.0 * (1.0 - np.cos(shift)), atol=1e-5
        )

    def loglike(shift):
        return model_loglike(TRUTH, _shift_phases(oidata_sim, shift))

    # The old wrapped-Δ² term had a kink at π (slopes of opposite sign on
    # either side); the chord term has zero slope there.
    below, above = (
        jax.grad(loglike)(np.pi - 1e-3),
        jax.grad(loglike)(np.pi + 1e-3),
    )
    scale = np.sum(1.0 / sigma**2)
    assert abs(below) < 2e-3 * scale and abs(above) < 2e-3 * scale


def test_model_loglike_is_gaussian_in_whitened_residuals():
    model = BinaryModelCartesian(240.0, 160.0, 6e-4)
    _, errors = oidata_sim.flatten_data()
    r = whitened_residuals(model, oidata_sim)
    expected = (
        -0.5 * np.sum(r**2)
        - np.sum(np.log(errors))
        - 0.5 * r.size * np.log(2.0 * np.pi)
    )
    assert np.allclose(model_loglike(model, oidata_sim), expected, rtol=1e-6)


def test_projected_phases_are_plain_residuals_over_sigma():
    data = load_oi_data(PRODUCT)["F430M"]
    model = BinaryModelCartesian(150.0, 100.0, 1e-2)
    observed, errors = data.flatten_data()
    expected = (data.model(model) - observed) / errors
    assert onp.allclose(whitened_residuals(model, data), expected)
