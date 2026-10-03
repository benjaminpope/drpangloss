import jax
import jax.numpy as np
import numpy as onp
import numpyro
import numpyro.distributions as dist
import pytest

from virgil._precision import cast_tree, run_in
from virgil.coverage import ami_grid_record
from virgil.fitting import _Objective, fit
from virgil.imaging import TSV, Centroid, MaxEntropy, image_priors
from virgil.likelihood import numpyro_model, whitened_residuals
from virgil.models import BinaryModelCartesian, Image, PointSource, System
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

from ._compiles import count_compiles
from ._test_data import oidata

TRUTH = BinaryModelCartesian(150.0, -80.0, 0.02)
DATA = oidata.with_model(TRUTH, key=jax.random.PRNGKey(0))
PRIORS = {
    "dra": dist.Uniform(-400.0, 400.0),
    "ddec": dist.Uniform(-400.0, 400.0),
    "flux": dist.Uniform(0.0, 0.5),
}
START = BinaryModelCartesian(140.0, -70.0, 0.01)


def _image_fit(npix=16):
    """A star + Image scene's data, and a flat starting model with priors."""
    data = OIData(ami_grid_record(pitch_m=0.5))
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(npix, 12.0, 25.0, dra=20.0), 12.0, flux=0.1
        ),
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(1))
    start = System(
        star=PointSource(), env=Image(np.zeros((npix, npix)), 12.0, flux=0.1)
    )
    return start, image_priors(start), data


@pytest.mark.parametrize("method", ["lm", "lbfgs", "adam"])
def test_fit_recovers_a_binary(method):
    options = {"max_steps": 3000} if method == "adam" else {}
    result = fit(START, PRIORS, DATA, method=method, **options)
    assert abs(result.values["dra"] - 150.0) < 3.0
    assert abs(result.values["ddec"] + 80.0) < 3.0
    assert abs(result.values["flux"] - 0.02) < 2e-3
    assert result.info["chi2_red"] < 1.5
    assert result.info["converged"] in (True, None)


def test_optimisers_agree_on_a_binary():
    lm = fit(START, PRIORS, DATA, method="lm")
    lbfgs = fit(START, PRIORS, DATA, method="lbfgs")
    for path in PRIORS:
        assert np.allclose(lm.values[path], lbfgs.values[path], rtol=1e-4)


def test_a_function_model_needs_starting_values():
    def binary(dra, ddec, flux):
        return BinaryModelCartesian(dra, ddec, flux)

    with pytest.raises(ValueError, match="init"):
        fit(binary, PRIORS, DATA)
    init = {"dra": 140.0, "ddec": -70.0, "flux": 0.01}
    assert abs(fit(binary, PRIORS, DATA, init=init).values["dra"] - 150) < 3


def test_float32_and_float64_fits_agree():
    x64 = fit(START, PRIORS, DATA, dtype="float64")
    x32 = fit(START, PRIORS, DATA, dtype="float32")
    ambient = np.float64 if jax.config.jax_enable_x64 else np.float32
    for path in PRIORS:
        assert x64.values[path].dtype == ambient  # cast back after the fit
        assert np.allclose(x32.values[path], x64.values[path], rtol=1e-3)
    # Both converge. With a fixed gtol = 1e-4, below what rounding lets the
    # float32 gradient reach, the float32 fit ran all 1000 LM steps (where
    # float64 takes 7).
    for result in (x64, x32):
        assert result.info["converged"] is True
        assert result.info["steps"] < 50


def test_lm_and_lbfgs_agree_on_a_tsv_image():
    start, priors, data = _image_fit()
    regularisers = [TSV(1e3, path="env")]
    # A tight tolerance, so that this tests the solution, not the stopping
    # rule (at the default tolerance their losses can differ by ~2%).
    options = {"gtol": 1e-7, "max_steps": 20_000}
    lm = fit(start, priors, data, regularisers, method="lm", **options)
    lbfgs = fit(start, priors, data, regularisers, method="lbfgs", **options)
    assert np.isclose(lm.info["loss"], lbfgs.info["loss"], rtol=2e-3)
    a, b = lm.model.env.brightness, lbfgs.model.env.brightness
    a, b = a - a.mean(), b - b.mean()
    assert np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)) > 0.99


def test_bijections_keep_parameters_in_their_support():
    objective = _Objective(START, PRIORS, DATA)
    values = objective.constrain(objective.init())
    assert np.allclose(values["flux"], 0.01)
    assert np.allclose(values["dra"], 140.0)
    stretched = objective.constrain({p: np.asarray(50.0) for p in PRIORS})
    assert 0.0 < stretched["flux"] < 0.5
    assert -400.0 < stretched["dra"] < 400.0


def test_residuals_and_loss_are_consistent():
    objective = _Objective(START, PRIORS, DATA)
    z0 = objective.init()
    z1 = {k: v + 0.1 for k, v in z0.items()}
    half = [0.5 * np.sum(objective.residuals(z) ** 2) for z in (z0, z1)]
    loss = [objective.loss(z) for z in (z0, z1)]
    # Uniform priors are flat, so loss = 0.5 Σ r² + const.
    assert np.allclose(half[1] - half[0], loss[1] - loss[0], rtol=1e-4)


def test_normal_priors_are_least_squares_terms():
    priors = dict(PRIORS, dra=dist.Normal(150.0, 2.0))
    objective = _Objective(START, priors, DATA)
    r = objective.residuals(objective.init())
    assert r.size == DATA.n_independent + 1
    assert np.isclose(r[-1], (140.0 - 150.0) / 2.0)


def test_non_least_squares_objectives_default_to_lbfgs():
    start, priors, data = _image_fit()
    regularisers = [MaxEntropy(1.0, path="env")]
    objective = _Objective(start, priors, data, regularisers)
    with pytest.raises(TypeError, match="lbfgs"):
        objective.residuals(objective.init())
    result = fit(start, priors, data, regularisers, max_steps=50)
    assert result.info["method"] == "lbfgs"
    log_uniform = dict(PRIORS, flux=dist.LogUniform(1e-4, 0.5))
    assert fit(START, log_uniform, DATA).info["method"] == "lbfgs"


def test_numpyro_model_accepts_prior_regularisers_only():
    start, _, data = _image_fit()
    pixels = dist.Normal(np.zeros((16, 16)), 3.0).to_event(2)
    priors = {"env.log_brightness": pixels}
    with pytest.raises(ValueError, match="TSV"):
        numpyro_model(start, priors, data, [TSV(1.0, path="env")])
    model = numpyro_model(start, priors, data, [Centroid(10.0, path="env")])
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(0))
    ).get_trace()
    assert {"loglike", "regulariser_0"} <= set(trace)


def test_fit_rejects_bad_paths_flux_priors_and_methods():
    with pytest.raises(ValueError, match="nonsense"):
        fit(TRUTH, {"nonsense": dist.Uniform(0.0, 1.0)}, DATA)
    with pytest.raises(ValueError, match="negative"):
        fit(TRUTH, {"flux": dist.Normal(0.0, 1.0)}, DATA)
    with pytest.raises(ValueError, match="method"):
        fit(START, PRIORS, DATA, method="newton")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_step_size"):
            fit(START, PRIORS, DATA, method="lbfgs", max_step_size=bad)


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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_warm_start_still_converges(dtype):
    # Starting from the solution at a nearby weight, the gradient is small
    # from the outset; the fit must still move to the new solution rather
    # than stopping at once.
    start, priors, data = _image_fit()
    strong = fit(
        start, priors, data, [MaxEntropy(10.0, path="env")], dtype=dtype
    )
    weak = fit(
        start,
        priors,
        data,
        [MaxEntropy(1.0, path="env")],
        init=strong.values,
        dtype=dtype,
    )
    cold = fit(start, priors, data, [MaxEntropy(1.0, path="env")], dtype=dtype)
    assert weak.info["steps"] > 4  # stalled fits took 1-4 steps
    assert weak.info["loss"] <= cold.info["loss"] * (1 + 1e-2)
    # Both reach the minimum, not a stationary point with most pixels dark
    # (χ²_red ~ 3-6, at a loss that varied by 2x across platforms).
    assert weak.info["chi2_red"] < 1.5
    assert cold.info["chi2_red"] < 1.5


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_cold_maxent_fit_does_not_collapse(dtype):
    # From a flat image, a weakly regularised fit once took a first
    # quasi-Newton step of ~10 in log-brightness, switching most pixels
    # off for good (their gradients vanish with their flux): it stopped on
    # a few bright pixels with χ² ≈ 1330, where a strongly regularised fit
    # reaches ≈ 245. A weaker penalty must fit the data at least as well.
    # (In float32 the line search runs out of precision near the minimum,
    # so that fit may stop unconverged.)
    start, priors, data = _image_fit()
    weak, strong = (
        fit(start, priors, data, [MaxEntropy(w, path="env")], dtype=dtype)
        for w in (1.0, 100.0)
    )
    assert weak.info["chi2"][0] <= strong.info["chi2"][0]
    b = weak.model.env.brightness
    assert np.mean(b > 1e-3 * np.max(b)) > 0.1  # collapsed fits: 2%


@pytest.mark.parametrize("method", ["lm", "lbfgs", "adam"])
def test_repeated_fits_do_not_recompile(method):
    # The solvers are jitted once at module level, so a second fit of a
    # problem with the same structure (another start, regulariser weight
    # or dataset of the same size) reuses the compilation. When they were
    # defined inside each call, every fit recompiled, which took most of
    # its time.
    start, priors, data = _image_fit()
    options = {"max_steps": 20} if method == "adam" else {}
    again = data.with_model(start, key=jax.random.PRNGKey(2))

    def fit_tsv(data, weight):
        regs = [TSV(weight, path="env")]
        return fit(start, priors, data, regs, method=method, **options)

    fit_tsv(data, 10.0)
    with count_compiles() as compiles:
        fit_tsv(again, 3.0)
    assert not compiles


def test_fit_recovers_error_scales():
    # Noise twice the stated errors: the fitted scales should be near 2.
    data = oidata.with_model(TRUTH, key=jax.random.PRNGKey(3), noise_scale=2.0)
    noise = {
        "vis_scale": dist.Uniform(0.0, 10.0),
        "phi_scale": dist.Uniform(0.0, 10.0),
    }
    result = fit(START, PRIORS, data, noise=noise)
    assert result.info["method"] == "lbfgs"
    # The maximum-likelihood scale is the rms of the residuals at the truth,
    # whitened by the stated errors. There are only 15 independent closure
    # phases, so one draw scatters by ~20% about the injected 2: compare
    # with this draw's own rms rather than with 2.
    whitened = onp.asarray(whitened_residuals(TRUTH, data))
    n_vis = onp.size(data.vis)
    rms = {
        "vis_scale": onp.sqrt(onp.mean(whitened[:n_vis] ** 2)),
        "phi_scale": onp.sqrt(onp.mean(whitened[n_vis:] ** 2)),
    }
    for term, expected in rms.items():
        assert result.values[f"noise.{term}"] == pytest.approx(
            expected, rel=0.15
        )
    assert result.values["noise.vis_scale"] > 1.4  # 2x noise, well sampled
    # χ² is computed with the inflated errors.
    assert abs(result.info["chi2_red"] - 1.0) < 0.05
    assert abs(result.values["dra"] - 150.0) < 5.0


def test_fit_noise_per_dataset_and_validation():
    noise = [{"phi_error": dist.Uniform(0.0, 1.0)}, {}]
    result = fit(START, PRIORS, [DATA, DATA], noise=noise)
    assert set(result.values) == set(PRIORS) | {"noise[0].phi_error"}
    with pytest.raises(ValueError, match="2 datasets"):
        fit(START, PRIORS, [DATA, DATA], noise=[{}])
    with pytest.raises(ValueError, match="Unknown noise term"):
        fit(START, PRIORS, DATA, noise={"jitter": dist.Uniform(0.0, 1.0)})
    with pytest.raises(ValueError, match="non-negative"):
        fit(START, PRIORS, DATA, noise={"vis_scale": dist.Normal(1.0, 1.0)})
    with pytest.raises(TypeError, match="least-squares"):
        fit(
            START,
            PRIORS,
            DATA,
            method="lm",
            noise={"vis_scale": dist.Uniform(0.0, 5.0)},
        )


def test_numpyro_model_samples_noise_terms():
    noise = {"vis_scale": dist.Uniform(0.0, 5.0)}
    model = numpyro_model(START, PRIORS, DATA, noise=noise)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    assert "noise.vis_scale" in trace
