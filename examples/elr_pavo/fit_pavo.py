"""Re-analysis of CHARA/PAVO rapid-rotator V^2 fits with ``GravityDarkenedStar``.

This reproduces the analysis of Shashank Dholakia (``core/elr_fit.py`` in
https://github.com/shashankdholakia/jax-interferometry, commit 70689ed): NUTS
fits of the Espinosa Lara & Rieutord (2011) gravity-darkened star to PAVO
squared visibilities of upsilon UMa, epsilon Cep, lambda Boo and upsilon Tau.
The forward model is drpangloss's port of his ``ELR_Model``,
``drpangloss.GravityDarkenedStar`` (grey mode, ``n_lat=32``). Data cuts,
priors, likelihood and sampler settings follow his script; the data (his
``pavlist_l1l2.csv`` files, not committed) are read from ``data/pavo/<star>/``
and downloaded from his repository at the pinned commit if missing.

Conventions (his -> ours): ``inc_his = 90 deg - inc`` and ``obl = pa``. His
inclination prior ``U(0, pi/2)`` plus ``factor(log cos inc_his)`` is isotropic,
which in our convention is exactly ``cos(inc) ~ U(0, 1)``.

Run (all four stars with his settings, ~2000 warmup + 2500 samples x 2 chains):

    .venv/bin/python examples/elr_pavo/fit_pavo.py
    .venv/bin/python examples/elr_pavo/fit_pavo.py --stars upsTau --quick
    .venv/bin/python examples/elr_pavo/compare.py   # writes comparison.md

On OzSTAR/NT, one star per array task: see ``fit_pavo.sbatch``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "data" / "pavo"
COMMIT = "70689ed3dba338d59c98d02e8126a07a3b4e86da"
RAW_URL = (
    "https://raw.githubusercontent.com/shashankdholakia/jax-interferometry/"
    f"{COMMIT}/data/{{star}}/pavlist_l1l2.csv"
)
# directory name -> (CSV Star column, plot label)
STARS = {
    "upsUMa": ("HD_84999", r"$\upsilon$ UMa"),
    "epsCep": ("HD_211336", r"$\epsilon$ Cep"),
    "lamBoo": ("HD_125162", r"$\lambda$ Boo"),
    "upsTau": ("HD_28024", r"$\upsilon$ Tau"),
}
MAX_DIAM = 2.0  # mas; his cap, to keep the fit on the first visibility lobe
N_LAT = 32
VARS = ["diam", "omega", "inc", "pa", "logsig", "jitter"]


def load_pavo(name):
    """Baselines (m), wavelengths (m), V^2 and its error, with his cuts."""
    import pandas as pd

    path = DATA_DIR / name / "pavlist_l1l2.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} data from the pinned commit")
        urllib.request.urlretrieve(RAW_URL.format(star=name), path)
    df = pd.read_csv(path)
    df = df[(df.cal_v2sig > 0.0) & (df.cal_v2 < 1.0) & (df.cal_v2 > 0.0)]
    df = df[df["Star"] == STARS[name][0]]
    return dict(
        u=df["u"].values * 1e-6,  # table is in micro-metres
        v=df["v"].values * 1e-6,
        wavel=df["wl"].values * 1e-6,  # micro-metres -> metres
        v2=df["cal_v2"].values,
        v2_err=df["cal_v2sig"].values,
    )


def make_model(data):
    """His numpyro model, with ``GravityDarkenedStar`` as the forward model.

    Once drpangloss has an absolute ``vis_error`` noise term, this could be
    ``numpyro_model(..., noise=...)``; today its likelihood only inflates the
    errors relatively, so the jitter is written out here.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from drpangloss import GravityDarkenedStar

    u, v, wavel = (jnp.asarray(data[k]) for k in ("u", "v", "wavel"))
    v2, v2_err = jnp.asarray(data["v2"]), jnp.asarray(data["v2_err"])

    def model():
        diam = numpyro.sample("diam", dist.Uniform(1e-4, MAX_DIAM))
        omega = numpyro.sample("omega", dist.Uniform(0.0, 0.99))
        # isotropic orientation: cos(i) uniform (his U(0, pi/2) prior with a
        # log cos factor); i = 90 deg is equator-on
        cos_i = numpyro.sample("cos_i", dist.Uniform(0.0, 1.0))
        inc = numpyro.deterministic("inc", jnp.degrees(jnp.arccos(cos_i)))
        # pa and pa + 180 deg are degenerate in V^2 alone
        pa = numpyro.sample("pa", dist.Uniform(0.0, 180.0))
        logsig = numpyro.sample("logsig", dist.Normal(jnp.log(1e-3), 3.0))
        numpyro.deterministic("jitter", jnp.exp(logsig))
        star = GravityDarkenedStar(diam, omega, inc, pa, n_lat=N_LAT)
        vis2 = jnp.abs(star.model(u, v, wavel)) ** 2
        sigma = jnp.sqrt(v2_err**2 + jnp.exp(logsig) ** 2)
        numpyro.sample("v2", dist.Normal(vis2, sigma), obs=v2)

    return model


def summarise(samples, extra=None):
    """Median, 16/84 percentiles, mean and sd of each variable."""
    out = {}
    for k in VARS:
        x = np.asarray(samples[k]).ravel()
        p16, med, p84 = np.percentile(x, [16, 50, 84])
        out[k] = dict(
            median=med, p16=p16, p84=p84, mean=x.mean(), sd=x.std(ddof=1)
        )
        out[k].update((extra or {}).get(k, {}))
    return {k: {a: float(b) for a, b in v.items()} for k, v in out.items()}


def fit_star(name, args):
    import jax
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    data = load_pavo(name)
    print(f"{name}: {len(data['v2'])} samples")
    sampler = MCMC(
        NUTS(make_model(data), dense_mass=args.dense_mass),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.chains,
        chain_method=args.chain_method,
        progress_bar=args.progress_bar,
    )
    t0 = time.time()
    sampler.run(jax.random.PRNGKey(0), extra_fields=("num_steps",))
    runtime = time.time() - t0
    grouped = sampler.get_samples(group_by_chain=True)
    samples = {k: np.asarray(v).reshape(-1) for k, v in grouped.items()}
    diag = summary(
        {k: grouped[k] for k in VARS}, group_by_chain=True
    )  # r_hat, n_eff per variable
    extra = {
        k: dict(r_hat=float(diag[k]["r_hat"]), n_eff=float(diag[k]["n_eff"]))
        for k in VARS
    }
    steps = np.asarray(sampler.get_extra_fields()["num_steps"])
    result = dict(
        star=name,
        n_data=len(data["v2"]),
        runtime_s=runtime,
        mean_leapfrog_steps=float(steps.mean()),
        mean_tree_depth=float(np.log2(steps + 1).mean()),
        n_divergences=int(sampler.get_extra_fields()["diverging"].sum())
        if "diverging" in sampler.get_extra_fields()
        else None,
        settings=dict(
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            chains=args.chains,
            chain_method=args.chain_method,
            dense_mass=args.dense_mass,
        ),
        posterior=summarise(samples, extra),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}_samples.npz", **samples)
    (out / f"{name}_summary.json").write_text(json.dumps(result, indent=2))
    make_plots(name, data, samples, result["posterior"], out)
    return result


def make_plots(name, data, samples, post, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    from drpangloss import GravityDarkenedStar
    from drpangloss.plotting import plot_model

    label = STARS[name][1]
    med = {k: post[k]["median"] for k in VARS}
    star = GravityDarkenedStar(
        med["diam"], med["omega"], med["inc"], med["pa"], n_lat=N_LAT
    )
    vis2 = np.asarray(
        jnp.abs(star.model(data["u"], data["v"], data["wavel"])) ** 2
    )
    sigma = np.hypot(data["v2_err"], med["jitter"])
    # V^2 against baseline / wavelength, coloured by baseline angle
    bl = np.hypot(data["u"], data["v"]) / data["wavel"]
    theta = np.degrees(np.arctan2(data["v"], data["u"])) % 180
    fig, (a0, a1) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1]
    )
    a0.errorbar(bl, data["v2"], sigma, fmt="none", ecolor="0.7", lw=0.5)
    sc = a0.scatter(bl, data["v2"], c=theta, s=8, cmap="twilight_shifted")
    a0.plot(bl, vis2, "k.", ms=2, label="median model")
    a0.set(ylabel=r"$V^2$", title=label)
    a0.legend()
    fig.colorbar(sc, ax=[a0, a1], label="baseline angle (deg)")
    a1.errorbar(bl, data["v2"] - vis2, sigma, fmt=".", ms=3, lw=0.5)
    a1.axhline(0, c="k", lw=0.5)
    a1.set(xlabel=r"baseline / $\lambda$", ylabel="residual")
    fig.savefig(out / f"{name}_v2.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    star.plot_surface(ax=ax[0])
    ax[0].set_title(f"{label}: surface")
    plot_model(star, fov_mas=2.5 * med["diam"], ax=ax[1], title="render")
    fig.savefig(out / f"{name}_star.png", dpi=150)
    plt.close(fig)

    # pairs plot (a plain matplotlib stand-in for a corner plot)
    names = ["diam", "omega", "inc", "pa", "jitter"]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    for i, yi in enumerate(names):
        for j, xj in enumerate(names):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
            elif i == j:
                ax.hist(samples[xj], bins=40, color="0.4")
                ax.set_yticks([])
            else:
                ax.hist2d(samples[xj], samples[yi], bins=40, cmap="Greys")
            if i == n - 1:
                ax.set_xlabel(xj)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(yi)
            elif j > 0:
                ax.set_yticklabels([])
    fig.suptitle(label)
    fig.savefig(out / f"{name}_corner.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stars", nargs="+", default=list(STARS), choices=STARS)
    p.add_argument("--num-warmup", type=int, default=2000)
    p.add_argument("--num-samples", type=int, default=2500)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--out", default=str(HERE / "output"))
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: 50 warmup, 50 samples, 1 chain",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="fetch any missing data files and exit (e.g. on a login node)",
    )
    p.add_argument(
        "--dense-mass",
        action="store_true",
        help="learn a dense mass matrix in warmup (his runs used diagonal); "
        "much faster on the strongly correlated diameter-omega posterior",
    )
    p.add_argument(
        "--chain-method",
        default="parallel",
        choices=("parallel", "vectorized", "sequential"),
        help="how numpyro runs the chains; 'vectorized' avoids pmap",
    )
    p.add_argument(
        "--no-progress-bar",
        dest="progress_bar",
        action="store_false",
        help="no progress bar (batch jobs)",
    )
    args = p.parse_args()
    if args.quick:
        args.num_warmup, args.num_samples, args.chains = 50, 50, 1
    if args.download_only:
        for name in args.stars:
            load_pavo(name)
        return

    import numpyro

    # must precede any JAX computation so that chains run in parallel
    numpyro.set_host_device_count(args.chains)
    import jax

    # float64, as in his script. Set for the whole (script) process: inside
    # a jax.enable_x64 context numpyro's progress-bar callback fails.
    jax.config.update("jax_enable_x64", True)
    for name in args.stars:
        res = fit_star(name, args)
        print(
            f"{name}: {res['runtime_s']:.0f} s, mean tree depth "
            f"{res['mean_tree_depth']:.1f}"
        )
        for k, s in res["posterior"].items():
            print(f"  {k:7s} {s['median']:9.4f} +/- {s['sd']:.4f}")


if __name__ == "__main__":
    main()
