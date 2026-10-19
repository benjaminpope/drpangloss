"""The scale-marginalized surface of data with gains (R1: ln s by quadrature)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from tests.test_gains import TRUTH, _data, _jacobian
from virgil._precision import run_in
from virgil.epochs import _Surface, marginal_loglike
from virgil.likelihood import _prediction_and_errors

S_MAX = 30.0


def _noisy(seed=0):
    """Data drawn from ``TRUTH`` with errors understated by a factor 3."""
    data = _data(n_frame=2, n_wave=4, seed=seed)
    rng = onp.random.default_rng(seed + 1)
    mean = onp.asarray(data.model(TRUTH))
    import equinox as eqx

    vis = mean + rng.normal(0, 0.03, mean.shape)
    return eqx.tree_at(
        lambda d: d.vis, data, np.asarray(vis).reshape(data.vis.shape)
    )


def test_zero_width_gains_reproduce_the_plain_surface():
    with run_in("float64"):
        plain = _noisy()
        gained = plain.with_gains(telescope=0.0, baseline=0.0)
        a = marginal_loglike(TRUTH, plain, s_max=S_MAX)
        b = marginal_loglike(TRUTH, gained, s_max=S_MAX)
        assert _Surface(gained, S_MAX).numeric
        assert float(b) == pytest.approx(float(a), abs=1e-6)


def test_quadrature_matches_a_dense_integral_and_cholesky_logdet():
    with run_in("float64"):
        data = _noisy().with_gains(telescope=0.05, baseline=0.03)
        score = float(marginal_loglike(TRUTH, data, s_max=S_MAX))
        prediction, errors, values, _, _ = _prediction_and_errors(
            TRUTH, data, {}
        )
        prediction, err = onp.asarray(prediction), onp.asarray(errors)
        resid = prediction - onp.asarray(values)
        gains = data.gains
        jac = _jacobian(data, prediction)
        bound = onp.log(S_MAX)
        grid = onp.linspace(-bound, bound, 4001)
        log_f = []
        for t in grid:
            cov = onp.asarray(
                gains.covariance(err * onp.exp(t), jac, gains.widths), float
            )
            chol = onp.linalg.cholesky(cov)
            z = onp.linalg.solve(chol, resid)
            logdet = 2.0 * onp.sum(onp.log(onp.diag(chol)))
            log_f.append(-0.5 * z @ z - 0.5 * logdet + onp.sum(onp.log(err)))
        log_f = onp.array(log_f)
        peak = log_f.max()
        dense = peak + onp.log(onp.trapezoid(onp.exp(log_f - peak), grid))
        assert score == pytest.approx(dense, abs=1e-4)


def test_gradients_are_finite():
    with run_in("float64"):
        data = _noisy().with_gains(telescope=0.05, baseline=0.03)

        def score(dra):
            model = TRUTH.__class__(dra=dra, ddec=-2.0, flux=0.3)
            return marginal_loglike(model, data, s_max=S_MAX)

        g = jax.grad(score)(np.asarray(3.0))
        assert onp.isfinite(float(g))


def test_vis_amp_is_refused_with_gains_and_defaults_off():
    from virgil.aliases import (
        _default_vis_amp,
        _epoch_residuals,
        _refuse_amp_with_gains,
    )

    with run_in("float64"):
        plain = _noisy()
        gained = plain.with_gains(telescope=0.05)
        surface = _Surface(gained, S_MAX)
        assert _default_vis_amp([plain]) is not None
        assert _default_vis_amp([gained]) is None
        _refuse_amp_with_gains(plain)
        with pytest.raises(ValueError, match="vis_amp.*gains"):
            _epoch_residuals(surface, TRUTH, gained, (0.5, 1.0))
