"""A repeated call on same-shaped inputs compiles its kernels only once.

Guards the jit-recompile audit: `fit(progress=True)` (the chunked loop) and
`imaging.diagnose` must not rebuild a kernel per call.
"""

import jax
import numpyro.distributions as dist

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


def test_fit_with_progress_compiles_once(capsys):
    record = ami_grid_record(pitch_m=0.5)
    data = OIData(record).with_model(
        BinaryModelCartesian(150.0, -80.0, 0.02), key=jax.random.PRNGKey(0)
    )
    start = BinaryModelCartesian(140.0, -70.0, 0.01)
    fit(start, PRIORS, data, progress=True)  # warm up
    with count_compiles() as compiles:
        fit(start, PRIORS, data, progress=True)
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
