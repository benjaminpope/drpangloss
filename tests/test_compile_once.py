"""A repeated call on same-shaped inputs compiles its kernels only once.

Guards the jit-recompile audit: `fit(progress=True)` (the chunked loop) and
`imaging.diagnose` must not rebuild a kernel per call.
"""

import jax
import numpyro.distributions as dist
import pytest

import virgil.fitting
from virgil.coverage import ami_grid_record
from virgil.fitting import fit
from virgil.imaging import diagnose
from virgil.models import BinaryModelCartesian, Image, PointSource, System
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

from ._compiles import count_compiles

PRIORS = {
    "dra": dist.Uniform(-400.0, 400.0),
    "ddec": dist.Uniform(-400.0, 400.0),
    "flux": dist.Uniform(0.0, 0.5),
}


@pytest.mark.parametrize("method", ["lm", "lbfgs", "adam"])
def test_fit_with_progress_compiles_once(method, monkeypatch):
    # Tiny chunks so that each method runs several chunk-to-chunk rounds.
    monkeypatch.setattr(virgil.fitting, "_LM_CHUNK", 2)
    monkeypatch.setattr(virgil.fitting, "_LBFGS_CHUNK", 3)
    monkeypatch.setattr(virgil.fitting, "_ADAM_CHUNK", 3)
    data = OIData(ami_grid_record(pitch_m=0.5)).with_model(
        BinaryModelCartesian(150.0, -80.0, 0.02), key=jax.random.PRNGKey(0)
    )
    start = BinaryModelCartesian(60.0, -20.0, 0.005)
    reports = []
    options = {"max_steps": 10} if method == "adam" else {}
    # new numbers (limits, rates, budget) must not recompile
    later = (
        {
            "max_steps": 7,
            "learning_rate": 0.5,
            "adam_eps": 1e-6,
            "time_limit": 1e6,
        }
        if method == "adam"
        else {}
    )
    fit(start, PRIORS, data, method=method, progress=reports.append, **options)
    assert len(reports) > 1  # several chunks ran
    with count_compiles() as compiles:
        fit(
            start,
            PRIORS,
            data,
            method=method,
            progress=reports.append,
            **later,
        )
    assert compiles == []


def test_diagnose_compiles_once():
    blob = gaussian_blob(16, 60.0, 100.0, 150.0, -100.0)
    model = System(
        star=PointSource(), env=Image.from_brightness(blob, 60.0, flux=0.1)
    )
    data = OIData(ami_grid_record(pitch_m=0.5)).with_model(
        model, key=jax.random.PRNGKey(3)
    )
    diagnose(model, data)  # warm up
    with count_compiles() as compiles:
        diagnose(model, data)
    assert compiles == []
