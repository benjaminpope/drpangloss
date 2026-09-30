"""Benchmark the image Fourier transforms: exact DFT against jax-finufft.

Times a jitted value-and-gradient of sum |V|^2 for each backend, image size
and number of frequencies, in float32 and float64, and reports the NUFFT's
error against the DFT. With --gpu it also checks that the DFT is not
silently computed in TF32 (which A100/H100 GPUs use for float32 matmuls by
default), and measures FINUFFT's accuracy on the GPU.

Examples
--------
    python scripts/bench_ft.py                      # laptop CPU
    python scripts/bench_ft.py --gpu --out gpu.json # on a GPU node

Needs the optional extra: pip install 'drpangloss[nufft]'.
"""

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as onp

from drpangloss._geometry import image_visibilities, pixel_offsets
from drpangloss._utils import mas2rad

PIXEL_SCALE_MAS = 10.0


def _problem(npix, npoints, dtype, seed=0):
    """A random positive image and frequencies inside its Nyquist band."""
    rng = onp.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, (npix, npix))
    image /= image.sum()
    # |u * pixel_scale| < 0.4 cycles per pixel, in baseline/wavelength units.
    fmax = 0.4 / (PIXEL_SCALE_MAS * mas2rad)
    uu, vv = rng.uniform(-fmax, fmax, (2, npoints))
    return (jnp.asarray(a, dtype=dtype) for a in (image, uu, vv))


def _time(fn, *args, repeats):
    jax.block_until_ready(fn(*args))  # compile
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return 1e3 * float(onp.median(times))


def _value_and_grad(backend):
    def loss(image, uu, vv):
        vis = image_visibilities(image, uu, vv, PIXEL_SCALE_MAS, backend)
        return jnp.sum(jnp.abs(vis) ** 2)

    return jax.jit(jax.value_and_grad(loss))


def speed_and_accuracy(npixes, npoints_list, dtypes, repeats):
    rows = []
    for dtype in dtypes:
        with jax.enable_x64(dtype == "float64"):
            funcs = {b: _value_and_grad(b) for b in ("dft", "nufft")}
            for npix in npixes:
                for npoints in npoints_list:
                    image, uu, vv = _problem(npix, npoints, dtype)
                    row = {"dtype": dtype, "npix": npix, "npoints": npoints}
                    for backend, fn in funcs.items():
                        row[f"{backend}_ms"] = _time(
                            fn, image, uu, vv, repeats=repeats
                        )
                    dft = image_visibilities(image, uu, vv, PIXEL_SCALE_MAS)
                    nufft = image_visibilities(
                        image, uu, vv, PIXEL_SCALE_MAS, "nufft"
                    )
                    row["nufft_max_err"] = float(jnp.max(jnp.abs(nufft - dft)))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows


def _dft_at_precision(image, uu, vv, precision):
    """The library's DFT, but with a chosen matmul precision."""
    x = pixel_offsets(image.shape[0], PIXEL_SCALE_MAS)
    cols = jnp.exp(-2j * jnp.pi * jnp.outer(mas2rad * uu, x))
    rows = jnp.exp(-2j * jnp.pi * jnp.outer(mas2rad * vv, x))
    partial = jnp.matmul(rows, image.astype(rows.dtype), precision=precision)
    return jnp.sum(partial * cols, axis=-1)


def gpu_checks(npix=256, npoints=10_000):
    """TF32 check for the DFT, and FINUFFT accuracy on the GPU."""
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(True):
        image64, uu64, vv64 = (
            jax.device_put(a, cpu) for a in _problem(npix, npoints, "float64")
        )
        reference = onp.asarray(
            image_visibilities(image64, uu64, vv64, PIXEL_SCALE_MAS)
        )
        nufft64 = image_visibilities(
            *_problem(npix, npoints, "float64"), PIXEL_SCALE_MAS, "nufft"
        )
        report = {
            "reference": "float64 DFT on CPU",
            "nufft_float64_max_err": float(
                onp.max(onp.abs(onp.asarray(nufft64) - reference))
            ),
        }
    image, uu, vv = _problem(npix, npoints, "float32")
    for name, precision in [
        ("dft_float32_highest_max_err", jax.lax.Precision.HIGHEST),
        ("dft_float32_default_max_err", jax.lax.Precision.DEFAULT),
    ]:
        vis = _dft_at_precision(image, uu, vv, precision)
        report[name] = float(onp.max(onp.abs(onp.asarray(vis) - reference)))
    nufft32 = image_visibilities(image, uu, vv, PIXEL_SCALE_MAS, "nufft")
    report["nufft_float32_max_err"] = float(
        onp.max(onp.abs(onp.asarray(nufft32) - reference))
    )
    # HIGHEST must give float32 accuracy (~1e-6), not TF32 (~1e-3).
    report["tf32_check_passed"] = report["dft_float32_highest_max_err"] < 1e-5
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--npix", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--npoints", type=int, nargs="+", default=[1_000, 10_000, 100_000]
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", help="write the results to this JSON file")
    args = parser.parse_args()

    results = {
        "platform": platform.platform(),
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "timings": speed_and_accuracy(
            args.npix, args.npoints, args.dtypes, args.repeats
        ),
    }
    if args.gpu:
        if jax.default_backend() != "gpu":
            raise SystemExit("--gpu given, but JAX sees no GPU.")
        results["gpu_checks"] = gpu_checks()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
