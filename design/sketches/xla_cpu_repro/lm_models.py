import sys, warnings; warnings.filterwarnings("ignore")
import jax, jax.flatten_util, jax.numpy as jnp, numpy as onp, equinox as eqx, optimistix as optx, lineax as lx
import numpyro.distributions as dist
jax.config.update("jax_enable_x64", True)
from virgil.coverage import vlti_oidata
from virgil.fitting import _Objective
from virgil.models import PointSource, QuadraticLimbDarkenedDisk, UniformDisk, GaussianDisk, System
R = int(sys.argv[1])
U = lambda a, b: dist.Uniform(a, b)
scenes = {
 "uniformdisk+point": (System(star=UniformDisk(6.0), comp=PointSource(0.03, -14.0, 9.0)), {"star.diam": 5.7, "comp.flux": 0.022, "comp.dra": -13.5, "comp.ddec": 8.7},
    {"star.diam": U(0., 18.), "comp.flux": U(0., 1.), "comp.dra": U(-166., 166.), "comp.ddec": U(-166., 166.)}),
 "gaussian+point": (System(star=GaussianDisk(6.0), comp=PointSource(0.03, -14.0, 9.0)), {"star.sigma": 5.7, "comp.flux": 0.022, "comp.dra": -13.5, "comp.ddec": 8.7},
    {"star.sigma": U(0., 18.), "comp.flux": U(0., 1.), "comp.dra": U(-166., 166.), "comp.ddec": U(-166., 166.)}),
 "ldstar_only": (System(star=QuadraticLimbDarkenedDisk(6.0, q1=0.36, q2=0.29)), {"star.diam": 5.7, "star.q1": 0.5, "star.q2": 0.5},
    {"star.diam": U(0., 18.), "star.q1": U(0., 1.), "star.q2": U(0., 1.)}),
 "point_only": (System(comp=PointSource(0.03, -14.0, 9.0), comp2=PointSource(0.5, 3.0, 2.0)), {"comp.flux": 0.022, "comp.dra": -13.5, "comp.ddec": 8.7},
    {"comp.flux": U(0., 1.), "comp.dra": U(-166., 166.), "comp.ddec": U(-166., 166.)}),
}
class GS(optx.LevenbergMarquardt):
    def terminate(self, fn, y, args, options, state, tags):
        g = jax.grad(lambda z: 0.5 * jnp.sum(args.residuals(z) ** 2))(y)
        return (jnp.max(jnp.abs(jax.flatten_util.ravel_pytree(g)[0])) <= 1e-9) | (state.num_steps >= 100), optx.RESULTS.successful
solver = GS(rtol=0.0, atol=0.0, linear_solver=lx.QR())
def res(z, args): return args.residuals(z)
@eqx.filter_jit
def run(z, p): return optx.least_squares(res, solver, z, args=p, max_steps=100, throw=False).value
for name, (truth, sv, priors) in scenes.items():
    data = vlti_oidata(hour_angles_h=onp.linspace(-3.0, 3.0, 7), wavelengths_m=onp.linspace(1.5e-6, 2.4e-6, 6), sigma_v2=0.01, sigma_cp_deg=0.5).with_model(truth)
    start = truth
    for p, v in sv.items(): start = start.set(p, v)
    problem = _Objective(start, priors, data, (), None, ()); z0 = problem.init()
    outs = [jax.tree.map(onp.asarray, run(z0, problem)) for _ in range(R)]
    key = lambda o: b"".join(x.tobytes() for x in jax.tree.leaves(o))
    print(f"{name:18s} runs={R} distinct_bitwise={len({key(o) for o in outs})}", flush=True)
