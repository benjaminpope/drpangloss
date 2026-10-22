import argparse, sys, jax, numpy as np
jax.config.update("jax_enable_x64", True)
from jax import export
ap = argparse.ArgumentParser(); ap.add_argument("--calls", type=int, default=100); a = ap.parse_args()
fn = export.deserialize(bytearray(open("lm_step.jaxexport", "rb").read())); x0 = np.load("x0.npy")
outs = [tuple(np.asarray(o) for o in fn.call(x0)) for _ in range(a.calls)]
print(f"jax {jax.__version__}  calls={a.calls}  distinct bitwise={len({b''.join(o.tobytes() for o in r) for r in outs})}  steps={outs[0][1]}")
