import sys, warnings; warnings.filterwarnings("ignore")
exec(open(sys.argv[1]).read().split("for name, (truth, sv, priors) in scenes.items():")[0].replace("R = int(sys.argv[1])","R=1"))
import numpy as onp
truth, sv, priors = scenes["gaussian+point"]
data = vlti_oidata(hour_angles_h=onp.linspace(-3.0, 3.0, 7), wavelengths_m=onp.linspace(1.5e-6, 2.4e-6, 6), sigma_v2=0.01, sigma_cp_deg=0.5).with_model(truth)
start = truth
for p, v in sv.items(): start = start.set(p, v)
problem = _Objective(start, priors, data, (), None, ()); z0 = problem.init()
flat0, unravel = jax.flatten_util.ravel_pytree(z0)
@jax.jit
def run_flat(x):
    sol = optx.least_squares(res, solver, unravel(x), args=problem, max_steps=100, throw=False)
    return jax.flatten_util.ravel_pytree(sol.value)[0], sol.stats["num_steps"]
from jax import export
exp = export.export(run_flat, disabled_checks=[export.DisabledSafetyCheck.custom_call(t) for t in ("lapack_dgeqrf_ffi","lapack_dormqr_ffi","lapack_dtrsm_ffi")])(jax.ShapeDtypeStruct(flat0.shape, flat0.dtype))
open(sys.argv[2], "wb").write(exp.serialize())
onp.save(sys.argv[3], onp.asarray(flat0))
print("exported", len(exp.serialize()), "bytes; x0 =", onp.asarray(flat0))
