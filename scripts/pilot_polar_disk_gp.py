#!/usr/bin/env python3
"""Pilot for notebooks/polar_disk_rings.ipynb: choose the Gaussian-process settings cheaply.

Builds the notebook's scene and data and its parametric fit by running the
notebook's own code cells (so the pilot cannot drift from the tutorial), then
fits the non-parametric disk at a few (sigma, ell) settings of the profile
prior and prints, for each, chi2/N on the quoted errors, the planet's position
and flux against the truth and the parametric fit, the disk flux, the Laplace
log evidence, and the rms error of ln I0 against the true profile (over the
radii where the truth is within e^-6 of its peak, and over the whole range).

    python scripts/pilot_polar_disk_gp.py --out pilot.json
    python scripts/pilot_polar_disk_gp.py --settings "2,5;3,1.5" --out pilot.json

CPU only; about one fit per 100 s plus the parametric fit and compilation.
Run from the repository root. Not part of the package or its tests.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent.parent

# (sigma, ell) of the profile prior: the old tutorial's setting as a control,
# then the new default's neighbours (sigma 2 is about the truth's std of ln I0,
# ell 5 the half-fringe-scale equivalent Gaussian width, 6 a stress test).
DEFAULT = "3,1.5;2,3;2,5;2,7;3,5;6,5"


def cell(nb, i):
    return "".join(nb["cells"][i]["source"])


def before(text, marker):
    """The part of a cell's source before a marker line (checked)."""
    assert marker in text, marker
    return text[: text.index(marker)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--settings", default=DEFAULT, help='"sigma,ell;sigma,ell;..."'
    )
    ap.add_argument("--out", default="pilot_polar_disk_gp.json")
    ap.add_argument(
        "--dry", action="store_true", help="only check the notebook markers"
    )
    args = ap.parse_args()
    settings = [
        tuple(float(x) for x in s.split(",")) for s in args.settings.split(";")
    ]

    nb = json.load(open(ROOT / "notebooks" / "polar_disk_rings.ipynb"))
    scene_src = before(cell(nb, 2), "fig, ax = plt.subplots")
    gp_src = before(cell(nb, 6), "gpfit = fit(")
    evid_src = before(
        cell(nb, 8).split("gpm = gpfit.model\n", 1)[1], "ell_alt = 2.0"
    )
    if args.dry:
        print(
            "markers ok", len(scene_src), len(gp_src), len(evid_src), settings
        )
        return

    t0 = time.time()
    ns = {"__name__": "pilot"}
    os.chdir(ROOT)
    exec(
        scene_src, ns
    )  # scene, truth, data (the notebook's cell 2 without its figure)
    exec(cell(nb, 4), ns)  # the parametric fit (prints its table)
    exec(gp_src, ns)  # gp, gp_priors_, r_in, r_out
    exec(evid_src, ns)  # logz_par
    print(f"setup done at {time.time() - t0:.0f} s", flush=True)

    import numpy as onp

    fit, log_evidence = ns["fit"], ns["log_evidence"]
    truth, par, data, gp, gp_priors_ = (
        ns[k] for k in ("truth", "par", "data", "gp", "gp_priors_")
    )
    v = par.values
    rg = onp.linspace(ns["r_in"], ns["r_out"], 200)

    def ring_profile(p):
        """Sum of the three normalized Gaussians (the notebook's truth profile)."""
        return sum(
            p[f"f{i}"]
            * onp.exp(-0.5 * ((rg - p[f"r{i}"]) / p[f"s{i}"]) ** 2)
            / (onp.sqrt(2 * onp.pi) * p[f"s{i}"] * p[f"r{i}"])
            for i in (1, 2, 3)
        )

    i_true = ring_profile(truth)
    i_par = ring_profile({k: float(x) for k, x in par.values.items()})
    near = i_true > i_true.max() * onp.exp(-6)
    rms = lambda i, m: float(
        onp.sqrt(onp.mean((onp.log(i[m]) - onp.log(i_true[m])) ** 2))
    )
    true_disk_flux = truth["f1"] + truth["f2"] + truth["f3"]
    n = int(par.info["ndata"][0])
    rows = [
        dict(
            model="parametric",
            chi2_over_n=float(par.info["chi2"][0]) / n,
            planet_err_mas=float(
                onp.hypot(v["dra"] - truth["dra"], v["ddec"] - truth["ddec"])
            ),
            planet_flux_rel_err=float(v["fp"] / truth["fp"] - 1),
            disk_flux=float(v["f1"] + v["f2"] + v["f3"]),
            log_z=float(ns["logz_par"]),
            rms_ln_i0_near=rms(i_par, near),
            rms_ln_i0_all=rms(i_par, onp.ones_like(near)),
        )
    ]
    print(
        f"{'sigma':>6}{'ell':>6}{'chi2/N':>8}{'conv':>6}{'dplanet':>9}{'dFp/Fp':>8}"
        f"{'Fdisk':>7}{'logZ':>9}{'rms near':>9}{'rms all':>8}{'time':>6}"
    )
    for sigma, ell in settings:
        t1 = time.time()
        m = gp.set("disk.profile.sigma", sigma).set("disk.profile.ell", ell)
        f = fit(m, gp_priors_, data)
        g = f.values
        i0 = onp.asarray(f.model.disk.profile.intensity(rg, 0.1))
        i0 = i0 / onp.trapezoid(i0 * rg, rg) * float(g["disk.flux"])
        row = dict(
            model="gp",
            sigma=sigma,
            ell=ell,
            chi2_over_n=float(f.info["chi2"][0]) / n,
            converged=bool(f.info["converged"]),
            steps=int(f.info["steps"]),
            planet_err_mas=float(
                onp.hypot(
                    g["planet.dra"] - truth["dra"],
                    g["planet.ddec"] - truth["ddec"],
                )
            ),
            planet_flux_rel_err=float(
                g["planet.flux.ratio"] / truth["fp"] - 1
            ),
            disk_flux=float(g["disk.flux"]),
            disk_flux_rel_err=float(g["disk.flux"] / true_disk_flux - 1),
            log_z=float(log_evidence(f, data, path="disk")),
            rms_ln_i0_near=rms(i0, near),
            rms_ln_i0_all=rms(i0, onp.ones_like(near)),
            seconds=time.time() - t1,
        )
        rows.append(row)
        print(
            f"{sigma:6g}{ell:6g}{row['chi2_over_n']:8.2f}{str(row['converged']):>6}"
            f"{row['planet_err_mas']:9.3f}{row['planet_flux_rel_err']:8.2f}"
            f"{row['disk_flux']:7.2f}{row['log_z']:9.1f}{row['rms_ln_i0_near']:9.2f}"
            f"{row['rms_ln_i0_all']:8.2f}{row['seconds']:6.0f}",
            flush=True,
        )
    print(
        f"parametric: chi2/N {rows[0]['chi2_over_n']:.2f}, planet error {rows[0]['planet_err_mas']:.3f} mas, "
        f"flux error {rows[0]['planet_flux_rel_err']:+.2f}, disk flux {rows[0]['disk_flux']:.2f}, "
        f"log Z {rows[0]['log_z']:.1f}, rms ln I0 {rows[0]['rms_ln_i0_near']:.2f}; truth disk flux {true_disk_flux:.2f}"
    )
    json.dump(
        dict(rows=rows, truth=truth, seconds=time.time() - t0),
        open(args.out, "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
