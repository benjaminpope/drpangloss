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


def _vlti(hours, n_wave, key=0, scale=3.0):
    """VLTI-like data (visibilities and correlated closure phases) drawn from
    ``TRUTH`` with errors understated by ``scale``."""
    from virgil.coverage import vlti_oidata

    data = vlti_oidata(
        hour_angles_h=hours,
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, n_wave),
        sigma_v2=0.01 * scale,
        sigma_cp_deg=0.5 * scale,
    )
    return data.with_model(TRUTH, jax.random.PRNGKey(key))


def _dense_marginal(model, data, bound, n=2001):
    """log ∫∫ exp(ll(s_vis, s_phi)) d ln s_vis d ln s_phi, up to a constant
    that does not depend on the model: the blocks are independent, so it is
    a sum of two 1-D integrals of ``model_loglike`` with the error scales."""
    from virgil.likelihood import model_loglike

    grid = np.linspace(-bound, bound, n)

    def one(name):
        def ll(t):
            return model_loglike(model, data, **{name: np.exp(t)})

        values = jax.vmap(ll)(grid)
        peak = values.max()
        return peak + np.log(np.trapezoid(np.exp(values - peak), grid))

    base = model_loglike(model, data)
    return float(one("vis_scale") + one("phi_scale") - base)


def _model(dra):
    return TRUTH.__class__(dra=dra, ddec=-4.0, flux=0.3)


@pytest.mark.slow  # compile-dominated: 30-40 s per case on CI
@pytest.mark.parametrize("nodes", [16, 32])
@pytest.mark.parametrize(
    "hours, n_wave, widths",
    [
        ((-2.0, 0.0, 2.0), 4, dict(baseline=0.02, triangle=0.01)),
        # Gain-dominated: the rank of the gains is half the observables.
        (
            tuple(onp.linspace(-4, 4, 24)),
            2,
            dict(baseline=0.02, triangle=0.01),
        ),
    ],
)
def test_two_blocks_with_gains_and_offsets_match_dense_integrals(
    hours, n_wave, widths, nodes, monkeypatch
):
    """Differences of the surface between two models equal those of the dense
    per-block integrals of ``model_loglike`` (so a gains norm on the wrong
    block, or a wrong window, shows)."""
    import virgil.epochs as epochs_module

    monkeypatch.setattr(epochs_module, "_N_NODES", nodes)
    with run_in("float64"):
        data = _vlti(hours, n_wave)
        data = data.with_gains(baseline=widths["baseline"])
        data = data.with_closure_offsets(triangle=widths["triangle"])
        surface = _Surface(data, S_MAX)
        assert surface.names == ("vis", "phi") and surface.numeric
        bound = onp.log(S_MAX)
        models = [_model(6.0), _model(6.4)]
        score = [float(marginal_loglike(m, data, s_max=S_MAX)) for m in models]
        dense = [_dense_marginal(m, data, bound) for m in models]
        # The quadrature's error falls as nodes are added: about 1e-3 at
        # 16 nodes and 1e-9 at 32 (the dense reference, 2001 trapezoid
        # points, is good to about 1e-4). 16 nodes are held to 0.01, the
        # default 32 to the tighter 1e-3.
        assert score[0] - score[1] == pytest.approx(
            dense[0] - dense[1], abs=0.01 if nodes == 16 else 1e-3
        )
        assert abs(score[0] - score[1]) > 1.0  # the models differ


def test_zero_width_gains_reproduce_the_plain_surface_when_tempered():
    with run_in("float64"):
        plain = _noisy()
        gained = plain.with_gains(telescope=0.0, baseline=0.0)
        for dof in (1.0, 1 / 41, 1 / 100):
            a = marginal_loglike(TRUTH, plain, s_max=S_MAX, dof=dof)
            b = marginal_loglike(TRUTH, gained, s_max=S_MAX, dof=dof)
            assert float(b) == pytest.approx(float(a), abs=1e-6)


@pytest.mark.slow  # compiles the gradient through the whole scale integral
def test_gradient_matches_a_finite_difference():
    with run_in("float64"):
        data = _vlti((-2.0, 0.0, 2.0), 4)
        data = data.with_gains(baseline=0.02).with_closure_offsets(
            triangle=0.01
        )

        def score(dra):
            return marginal_loglike(_model(dra), data, s_max=S_MAX)

        g = float(jax.grad(score)(np.asarray(6.0)))
        h = 1e-5
        fd = (float(score(6.0 + h)) - float(score(6.0 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-4)


def test_epoch_chi2_of_gained_data_reports_plain_block_chi2():
    from virgil.aliases import _epoch_chi2

    with run_in("float64"):
        data = _vlti((-2.0, 0.0, 2.0), 4).with_gains(baseline=0.02)
        surface = _Surface(data, S_MAX)
        table, amp = _epoch_chi2(surface, TRUTH, data, None)
        chi2 = surface.plain_chi2(table)
        assert chi2.shape == surface.nu.shape  # what per_epoch divides by nu
        assert onp.all(onp.isfinite(onp.asarray(chi2) / surface.nu))


def test_gained_data_without_s_max_take_the_numeric_default():
    with run_in("float64"):
        data = _vlti((-2.0, 0.0, 2.0), 4).with_gains(baseline=0.02)
        surface = _Surface(data)
        assert surface.numeric and surface.s_max == 10.0
