"""Robustness of the shared pipeline machinery: ``_core``, ``_common`` and
the orbit pipeline's use of it."""

import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.pipeline import (
    BinaryPipeline,
    DiskPipeline,
    OrbitPipeline,
    StarPipeline,
)
from virgil.pipeline import _checks as checks
from virgil.pipeline import orbit
from virgil.pipeline._common import (
    _integer,
    _pick,
    _positive,
    _prior_repr,
    _validate_sampler,
)


@pytest.mark.parametrize(
    "pipeline", [BinaryPipeline, StarPipeline, DiskPipeline, OrbitPipeline]
)
def test_from_oifits_refuses_an_empty_list_in_every_pipeline(
    pipeline, tmp_path
):
    with pytest.raises(ValueError):
        pipeline.from_oifits([], output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "pipeline", [BinaryPipeline, StarPipeline, DiskPipeline]
)
def test_from_oifits_names_the_pipeline_in_the_empty_list_error(
    pipeline, tmp_path
):
    with pytest.raises(ValueError, match=pipeline.__name__):
        pipeline.from_oifits([], output=tmp_path / "run")


# --- prior descriptions are stable ------------------------------------------


@pytest.mark.parametrize(
    "prior, text",
    [
        (dist.Uniform(0.0, 2.0), "Uniform(low=0.0, high=2.0)"),
        (dist.Normal(1.0, 3.0), "Normal(loc=1.0, scale=3.0)"),
        (dist.LogUniform(1e-5, 0.5), "LogUniform(low=1e-05, high=0.5)"),
    ],
)
def test_prior_repr_names_the_class_and_parameters(prior, text):
    assert _prior_repr(prior) == text


def test_prior_repr_has_no_memory_address_and_is_reproducible():
    import jax.numpy as np

    a = _prior_repr(dist.LogUniform(np.asarray(1e-3), np.asarray(1.0)))
    b = _prior_repr(dist.LogUniform(np.asarray(1e-3), np.asarray(1.0)))
    assert a == b and "0x" not in a and "object at" not in a


def test_prior_repr_of_a_prior_without_parameters_is_its_class():
    class Bare:
        pass

    assert _prior_repr(Bare()) == "Bare()"


# --- shared validation ------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_a_bool_is_not_an_integer_or_a_positive_number(value):
    with pytest.raises(ValueError, match="num_samples"):
        _integer({"num_samples": value}, "num_samples", 0)
    with pytest.raises(ValueError, match="batch_size"):
        _positive({"batch_size": value}, "batch_size")


def test_validate_sampler_checks_the_nuts_settings_only():
    good = {
        "num_samples": 5,
        "num_chains": 2,
        "num_warmup": 0,
        "chain_method": "vectorized",
    }
    _validate_sampler(good)  # no wavel_range, error_floor or batch_size
    for bad, match in (
        ({"num_samples": 0}, "num_samples"),
        ({"num_chains": 1.5}, "num_chains"),
        ({"num_warmup": -1}, "num_warmup"),
        ({"chain_method": "x"}, "chain_method"),
    ):
        with pytest.raises(ValueError, match=match):
            _validate_sampler({**good, **bad})


def test_pick_is_strict_by_default_and_lenient_on_request():
    report = {"a": 1, "b": 2}
    assert _pick(report, ("b", "a")) == {"b": 2, "a": 1}
    with pytest.raises(KeyError):
        _pick(report, ("a", "c"))
    assert _pick(report, ("a", "c"), strict=False) == {"a": 1}


# --- the orbit pipeline on the shared helpers -------------------------------


@pytest.mark.parametrize("name", ["n_peaks", "max_grid", "num_chains"])
def test_orbit_settings_reject_a_bool_for_an_integer(name):
    s = {**OrbitPipeline.defaults(), name: True}
    with pytest.raises(ValueError, match=name):
        OrbitPipeline._validate(s)


def test_orbit_sampler_settings_are_validated_with_the_shared_checks():
    for bad, match in (
        ({"num_samples": 0}, "num_samples"),
        ({"num_warmup": -1}, "num_warmup"),
        ({"chain_method": "x"}, "chain_method"),
    ):
        with pytest.raises(ValueError, match=match):
            OrbitPipeline._validate({**OrbitPipeline.defaults(), **bad})


def test_the_optimum_gap_is_nan_when_no_point_can_be_scored():
    class Problem:
        def batch(self, x):
            return onp.full(len(x), onp.nan)

    gap = orbit._optimum_gap(Problem(), onp.zeros((10, 2)), onp.zeros(2))
    assert onp.isnan(gap)
    assert checks.mode_converged(gap).status == "warn"


def test_the_optimum_gap_is_never_negative():
    class Problem:
        def batch(self, x):
            return onp.arange(len(x), dtype=float)

    x = onp.zeros((4, 2))
    # The start scores 0, the best draw 3.
    assert orbit._optimum_gap(Problem(), x, onp.zeros(2)) == pytest.approx(3.0)

    class Flat:
        def batch(self, x):
            return onp.zeros(len(x))

    assert orbit._optimum_gap(Flat(), x, onp.zeros(2)) == 0.0


def _band(drawn):
    """``_band_summary`` of a band of two sites with the given draws."""
    physical = {"period": onp.exp(drawn["log_period"]), "ecc": drawn["ecc"]}
    priors = {
        "log_period": dist.Uniform(-5.0, 5.0),
        "ecc": dist.Uniform(0.0, 0.9),
    }

    class Bounds:
        lower = onp.zeros(9)
        upper = onp.ones(9)

    return orbit._band_summary(
        1,
        physical,
        drawn,
        {"diverging": onp.zeros((2, 50), bool)},
        priors,
        Bounds(),
    )


@pytest.mark.parametrize("stuck", ["log_period", "ecc"])
def test_a_band_with_a_stuck_site_has_a_nan_r_hat_whatever_the_site_order(
    stuck,
):
    rng = onp.random.default_rng(1)
    drawn = {
        "log_period": rng.normal(size=(2, 50)),
        "ecc": rng.uniform(0.1, 0.5, size=(2, 50)),
    }
    summary = _band(drawn)
    assert onp.isfinite(summary["r_hat_max"])
    assert onp.isfinite(summary["ess_bulk_min"])
    assert summary["divergence_fraction"] == 0.0
    assert set(summary["params"]["period"]) >= {"r_hat", "ess_bulk"}
    assert "std" not in summary["params"]["period"]
    drawn[stuck] = onp.full((2, 50), 0.3)
    summary = _band(drawn)
    assert onp.isnan(summary["r_hat_max"])
    assert checks.r_hat(summary["r_hat_max"]).status == "fail"
