import json, sys, tempfile, pathlib
import jax, numpy as onp
sys.path.insert(0, "tests")
import test_pipeline_star as t
from virgil.pipeline import StarPipeline, star
print("jax", jax.__version__, jax.default_backend(), jax.config.jax_enable_x64)
tmpl = t.vlti_oidata(wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 3), hour_angles_h=(-2.5,0.0,2.5), sigma_v2=0.01, sigma_cp_deg=15.0)
data = tmpl.with_model(t.TRUTH, key=jax.random.PRNGKey(3))
orig = star._fit_from_starts
from virgil.fitting import fit as realfit
import virgil.fitting as vf
def spy(model, priors, data, *a, **k):
    k = dict(k)
    k["max_steps"] = 2000
    r = realfit(model, priors, data, *a, **k)
    print("ROUND0", r.info["converged"], r.info["steps"], r.info["loss"], r.info["grad_norm"], r.info["chi2"], {kk: float(onp.asarray(v)) for kk, v in r.values.items()})
    for i in range(1, 10):
        if r.info["converged"]:
            break
        k2 = {kk: vv for kk, vv in k.items() if kk != "init"}
        r = realfit(model, priors, data, *a, init=r.values, **k2)
        print("ROUND", i, r.info["converged"], r.info["steps"], r.info["loss"], r.info["grad_norm"], r.info["chi2"], {kk: float(onp.asarray(v)) for kk, v in r.values.items()})
    return r
vf.fit = spy
d = pathlib.Path(tempfile.mkdtemp())
res = StarPipeline(data, output=d/"run", error_scale="fit", **t.TINY).run(through="fit")
for name, m in res.summary["fit"]["models"].items():
    print(name, {k: m.get(k) for k in ("chi2","chi2_fitted","m2loglike")})
print("n", res.summary["chi2"]["n_independent"])
