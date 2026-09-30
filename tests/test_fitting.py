import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from drpangloss._precision import cast_tree, run_in
from drpangloss.amigo import simulated_disco_record
from drpangloss.fitting import Problem, fit
from drpangloss.imaging import TSV, Centroid, MaxEntropy, image_priors
from drpangloss.likelihood import model_loglike
from drpangloss.models import BinaryModelCartesian, Image, PointSource, System
from drpangloss.oidata import OIData
from drpangloss.scenes import gaussian_blob

from ._test_data import oidata

TRUTH = BinaryModelCartesian(150.0, -80.0, 0.02)
DATA = oidata.with_model(TRUTH, key=jax.random.PRNGKey(0))
PRIORS = {
    "dra": dist.Uniform(-400.0, 400.0),
    "ddec": dist.Uniform(-400.0, 400.0),
    "flux": dist.Uniform(0.0, 0.5),
}
BINARY = Problem(BinaryModelCartesian(140.0, -70.0, 0.01), DATA, PRIORS)


def _image_problem(regularisers, npix=16):
    data = OIData(simulated_disco_record(max_baseline_m=4.0))
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(npix, 12.0, 25.0, dra=20.0), 12.0, flux=0.1
        ),
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(1))
    start = System(
        star=PointSource(),
        env=Image(np.zeros((npix, npix)), 12.0, flux=0.1),
    )
    return Problem(start, data, image_priors(start), regularisers)


@pytest.mark.parametrize("method", ["lm", "lbfgs", "adam"])
def test_fit_recovers_a_binary(method):
    options = {"max_steps": 3000} if method == "adam" else {}
    result = fit(BINARY, method, **options)
    assert abs(result.values["dra"] - 150.0) < 3.0
    assert abs(result.values["ddec"] + 80.0) < 3.0
    assert abs(result.values["flux"] - 0.02) < 2e-3
    assert result.info["chi2_red"] < 1.5
    assert result.info["converged"] in (True, None)


def test_optimisers_agree_on_a_binary():
    lm, lbfgs = fit(BINARY, "lm"), fit(BINARY, "lbfgs")
    for path in PRIORS:
        assert np.allclose(lm.values[path], lbfgs.values[path], rtol=1e-4)


def test_float32_and_float64_fits_agree():
    x64 = fit(BINARY, dtype="float64").values
    x32 = fit(BINARY, dtype="float32").values
    for path in PRIORS:
        # Results are cast back to the precision outside the fit.
        ambient = np.float64 if jax.config.jax_enable_x64 else np.float32
        assert x64[path].dtype == ambient
        assert np.allclose(x32[path], x64[path], rtol=1e-3)


def test_lm_and_lbfgs_agree_on_a_tsv_image():
    problem = _image_problem([TSV(1e3, path="env")])
    lm = fit(problem, "lm").model.env.brightness
    lbfgs = fit(problem, "lbfgs", max_steps=20_000).model.env.brightness
    assert np.max(np.abs(lm - lbfgs)) < 0.05 * np.max(lm)


def test_bijections_keep_parameters_in_their_support():
    values = BINARY.constrain(BINARY.init())
    assert np.allclose(values["flux"], 0.01) and np.allclose(
        values["dra"], 140.0
    )
    far = {path: np.asarray(50.0) for path in PRIORS}
    stretched = BINARY.constrain(far)
    assert 0.0 < stretched["flux"] < 0.5
    assert -400.0 < stretched["dra"] < 400.0


def test_residuals_loss_and_logdensity_are_consistent():
    z0 = BINARY.init()
    z1 = {k: v + 0.1 for k, v in z0.items()}
    half_sum = [0.5 * np.sum(BINARY.residuals(z) ** 2) for z in (z0, z1)]
    loss = [BINARY.loss(z) for z in (z0, z1)]
    # Uniform priors are flat, so loss = 0.5 Σ r² + const.
    assert np.allclose(half_sum[1] - half_sum[0], loss[1] - loss[0], rtol=1e-4)
    # logdensity = log L + log p + log |J|, with the Jacobian of each bijection.
    model = BINARY.build(z1)
    values = BINARY.constrain(z1)
    expected = model_loglike(model, DATA)
    for path, prior in PRIORS.items():
        expected += prior.log_prob(values[path])
        transform = dist.transforms.biject_to(prior.support)
        expected += transform.log_abs_det_jacobian(z1[path], values[path])
    # model_loglike has the Gaussian normalisation; logdensity drops it.
    normalisation = model_loglike(model, DATA) + 0.5 * np.sum(
        BINARY.residuals(z1) ** 2
    )
    assert np.allclose(
        BINARY.logdensity(z1), expected - normalisation, rtol=1e-5
    )


def test_normal_priors_are_least_squares_terms():
    priors = dict(PRIORS, dra=dist.Normal(150.0, 2.0))
    problem = Problem(BinaryModelCartesian(140.0, -70.0, 0.01), DATA, priors)
    r = problem.residuals(problem.init())
    assert r.size == DATA.flatten_data()[0].size + 1
    assert np.isclose(r[-1], (140.0 - 150.0) / 2.0)


def test_non_least_squares_objectives_default_to_lbfgs():
    problem = _image_problem([MaxEntropy(1.0, path="env")])
    assert not problem.has_residuals
    with pytest.raises(TypeError, match="lbfgs"):
        problem.residuals(problem.init())
    assert fit(problem, max_steps=50).info["method"] == "lbfgs"
    priors = dict(PRIORS, flux=dist.LogUniform(1e-4, 0.5))
    problem = Problem(BinaryModelCartesian(140.0, -70.0, 0.01), DATA, priors)
    assert not problem.has_residuals


def test_logdensity_rejects_penalties_but_accepts_centroid_priors():
    with pytest.raises(ValueError, match="TSV"):
        problem = _image_problem([TSV(1.0, path="env")])
        problem.logdensity(problem.init())
    problem = _image_problem([Centroid(10.0, path="env")])
    assert np.isfinite(problem.logdensity(problem.init()))


def test_problem_rejects_bad_paths_and_flux_priors():
    with pytest.raises(Exception):
        Problem(TRUTH, DATA, {"nonsense": dist.Uniform(0.0, 1.0)})
    with pytest.raises(ValueError, match="negative"):
        Problem(TRUTH, DATA, {"flux": dist.Normal(0.0, 1.0)})
    with pytest.raises(ValueError, match="method"):
        fit(BINARY, "newton")


def test_precision_helpers():
    tree = {"a": np.ones(3), "b": np.ones(2, dtype=np.complex64), "n": 3}
    before = jax.config.jax_enable_x64
    with run_in("float64"):
        cast = cast_tree(tree, "float64")
        assert cast["a"].dtype == np.float64
        assert cast["b"].dtype == np.complex128
        assert cast["n"] == 3
    assert jax.config.jax_enable_x64 == before
    with pytest.raises(ValueError):
        with run_in("float16"):
            pass
    assert onp.asarray(cast_tree(tree, "float32")["a"]).dtype == onp.float32
