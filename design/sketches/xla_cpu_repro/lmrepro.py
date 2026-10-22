import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import numpy as onp
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)
import virgil

print("virgil from", virgil.__file__, file=sys.stderr)
from virgil.coverage import vlti_oidata
from virgil.fitting import fit
from virgil.models import PointSource, QuadraticLimbDarkenedDisk, System

start_values = {
    "star.diam": 5.7,
    "star.q1": 0.5,
    "star.q2": 0.5,
    "comp.flux": 0.022,
    "comp.dra": -13.5,
    "comp.ddec": 8.7,
}
priors = {
    "star.diam": dist.Uniform(0.0, 18.0),
    "star.q1": dist.Uniform(0.0, 1.0),
    "star.q2": dist.Uniform(0.0, 1.0),
    "comp.flux": dist.Uniform(0.0, 1.0),
    "comp.dra": dist.Uniform(-166.0, 166.0),
    "comp.ddec": dist.Uniform(-166.0, 166.0),
}
truth = System(
    star=QuadraticLimbDarkenedDisk(6.0, q1=0.36, q2=0.29),
    comp=PointSource(0.03, -14.0, 9.0),
)
data = vlti_oidata(
    hour_angles_h=onp.linspace(-3.0, 3.0, 7),
    wavelengths_m=onp.linspace(1.5e-6, 2.4e-6, 6),
    sigma_v2=0.01,
    sigma_cp_deg=0.5,
).with_model(truth)
start = truth
for p, v in start_values.items():
    start = start.set(p, v)
n = int(sys.argv[1])
res = []
for i in range(n):
    r = fit(start, priors, data)
    res.append(
        (
            float(r.info["loss"]),
            tuple(float(r.values[k]) for k in start_values),
        )
    )
bad = sum(l > 1.0 for l, _ in res)
print(
    f"n={n} bad(loss>1)={bad} distinct_results={len(set(res))} losses_first5={[f'{l:.3g}' for l, _ in res[:5]]}"
)
