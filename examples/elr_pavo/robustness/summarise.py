"""Collate the robustness runs (multistart, profile, injection) into figures
and ``robustness.md``; run locally after fetching ``output/`` from the
cluster. Stars or tests with no results are skipped.

    python summarise.py [--out output] [--stars upsTau ...]

``coverage_density`` is the density of the data's baseline position angles
at a given PA, relative to its mean over all PAs (1 = no preference). If the
fitted orientation were being pulled towards (or away from) the baseline
directions, the fits' density would be systematically above (below) 1, in
injection-recovery as well as in the real fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as c  # noqa: E402
import injection  # noqa: E402
import multistart  # noqa: E402
import profile_like as profile  # noqa: E402

PA_BINS = np.arange(0, 181, 5)
LEVELS = (2.3, 6.2)  # delta chi2 for 68% and 95% of two parameters


def pa_samples(star, data):
    """His-style NUTS posterior pa (mod 180) from our re-fit, else his median."""
    f = c.PAVO / "output" / f"{star}_samples.npz"
    if f.exists():
        return c.wrap180(np.load(f)["pa"]), "our NUTS posterior"
    return np.array([c.ref_median(star)["pa"]]), "his posterior median"


def density_at(pa, data):
    return float(np.mean(c.coverage_density(np.atleast_1d(pa), data.bl_pa)))


def draw_baseline_hist(ax, data, **kw):
    ax.hist(data.bl_pa, bins=PA_BINS, color="0.8", **kw)
    ax.set_yticks([])


# ------------------------------------------------------------- alignment
def alignment(star, data, out):
    pa, src = pa_samples(star, data)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    draw_baseline_hist(ax, data)
    ax.set(
        xlabel="position angle, East of North, mod 180 (deg)", xlim=(0, 180)
    )
    ax.set_ylabel("baseline samples")
    ax2 = ax.twinx()
    # the pole (red) and the long axis, the equator at pole + 90 (blue)
    for angles, color in ((pa, "C3"), (np.mod(pa + 90.0, 180.0), "C0")):
        if len(angles) > 1:
            ax2.hist(
                angles, bins=PA_BINS, color=color, alpha=0.6, density=True
            )
        else:
            ax2.axvline(angles[0], color=color)
    ax2.set_yticks([])
    ax.set_title(
        f"{star}: baseline PAs (grey), pole PA (red), long axis (blue)"
    )
    fig.tight_layout()
    fig.savefig(out / f"{star}_alignment.png", dpi=130)
    plt.close(fig)
    med = float(np.median(circ_unwrap(pa)))
    return src, med, density_at(pa, data), density_at(pa + 90, data)


def circ_unwrap(pa):
    """Shift PAs (mod 180) to be contiguous around their circular mean."""
    m = 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(pa)).mean()))
    return m + c.circ_diff(pa, m)


# ------------------------------------------------------------ multistart
def do_multistart(star, data, out, md):
    r = multistart.load(out, star)
    if r is None:
        return
    opt = multistart.optima(r["end"], r["chi2"], r["converged"], star)
    n = len(r["chi2"])
    md.append(f"### Multi-start ML ({n} starts)\n")
    md.append(
        f"{int(r['converged'].sum())}/{n} fits reported convergence; "
        f"median {np.median(r['seconds']):.0f} s and "
        f"{np.median(r['steps']):.0f} LM steps per fit. "
        f"{len(opt)} distinct optima (tolerances diam {multistart.TOL['diam_eq']}"
        f" mas, omega {multistart.TOL['omega']}, inc {multistart.TOL['inc']} "
        f"deg, pa {multistart.TOL['pa']} deg mod 180). z is the offset from "
        "his posterior median in units of his posterior sd (`z_all` in "
        "quadrature).\n"
    )
    md.append(
        "| diam (mas) | omega | inc | pa | chi2 | dchi2 | starts | "
        "converged | z diam/omega/inc/pa | z_all |"
    )
    md.append("| --- " * 10 + "|")
    for o in opt[:12]:
        z = "/".join(f"{o['z'][k]:+.1f}" for k in c.PARAMS)
        md.append(
            f"| {o['diam_eq']:.3f} | {o['omega']:.3f} | {o['inc']:.1f} | "
            f"{o['pa']:.1f} | {o['chi2']:.1f} | {o['dchi2']:.1f} | {o['n']} | "
            f"{o['n_converged']} | {z} | {o['z_all']:.1f} |"
        )
    if len(opt) > 12:
        md.append(f"\n({len(opt) - 12} further optima omitted.)")
    md.append("")
    ref = c.ref_median(star)
    best = r["chi2"].min()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, k, j in zip(axes, ("omega", "inc", "pa"), (1, 2, 3)):
        ax.scatter(
            r["end"][:, j],
            r["chi2"] - best + 0.1,
            c=r["converged"],
            cmap="coolwarm_r",
            s=14,
        )
        ax.axvline(ref[k], color="k", ls=":")
        ax.set(xlabel=k, yscale="log")
    axes[0].set_ylabel(r"$\Delta\chi^2$ + 0.1 from best")
    fig.suptitle(
        f"{star}: multi-start end points (red = converged; dotted: his median)"
    )
    fig.tight_layout()
    fig.savefig(out / f"{star}_multistart.png", dpi=130)
    plt.close(fig)
    md.append(f"![multistart]({star}_multistart.png)\n")


# --------------------------------------------------------------- profile
def do_profile(star, data, out, md):
    ref = c.ref_median(star)
    maps = {}
    for pair, names in profile.PAIRS.items():
        m = profile.assemble(out, star, pair)
        if m is not None:
            maps[pair] = m
    if not maps:
        return
    md.append("### Profile likelihood\n")
    fig, axes = plt.subplots(
        1, len(maps), figsize=(5 * len(maps), 4), squeeze=False
    )
    for ax, (pair, (a0, a1, chi2, conv)) in zip(axes[0], maps.items()):
        names = profile.PAIRS[pair]
        d = chi2 - np.nanmin(chi2)
        norm = matplotlib.colors.LogNorm(0.1, max(np.nanmax(d), 1e3))
        mesh = ax.pcolormesh(
            a0,
            a1,
            np.maximum(d, 0.1).T,
            shading="nearest",
            norm=norm,
            cmap="viridis_r",
        )
        if min(d.shape) > 2:
            ax.contour(
                a0,
                a1,
                np.nan_to_num(d, nan=1e9).T,
                levels=LEVELS,
                colors=["w", "r"],
                linewidths=1,
            )
        ax.plot(
            ref[names[0]],
            ref[names[1]],
            "k*",
            ms=12,
            mfc="w",
            label="his median",
        )
        ax.set(xlabel=names[0], ylabel=names[1])
        fig.colorbar(mesh, ax=ax, label=r"$\Delta\chi^2$")
        i, j = np.unravel_index(np.nanargmin(chi2), chi2.shape)
        md.append(
            f"- `{pair}`: grid {len(a0)}x{len(a1)}, min chi2 "
            f"{np.nanmin(chi2):.1f} at {names[0]}={a0[i]:.3g}, "
            f"{names[1]}={a1[j]:.3g}; {int(np.isnan(chi2).sum())} cells "
            f"missing, {int((~conv & ~np.isnan(chi2)).sum())} unconverged."
        )
        # local minima of the map: grid cells below all 8 neighbours
        loc = local_minima(d)
        md.append(
            f"  {len(loc)} grid-local minima with dchi2 <= 6.2: "
            + (
                ", ".join(
                    f"({names[0]}={a0[i]:.3g}, {names[1]}={a1[j]:.3g}, "
                    f"dchi2={d[i, j]:.1f})"
                    for i, j in loc
                    if d[i, j] <= 6.2
                )
                or "none"
            )
        )
    axes[0][0].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"{star}: profile dchi2 (contours 2.3 white, 6.2 red)")
    fig.tight_layout()
    fig.savefig(out / f"{star}_profile.png", dpi=130)
    plt.close(fig)
    md.append(f"\n![profile]({star}_profile.png)\n")


def local_minima(d):
    """Cells of a 2-d map that are no larger than all 8 neighbours."""
    out = []
    for i, j in np.ndindex(d.shape):
        if np.isnan(d[i, j]):
            continue
        win = d[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
        if d[i, j] <= np.nanmin(win) and np.sum(win == d[i, j]) == 1:
            out.append((i, j))
    return out


# ------------------------------------------------------------- injection
def pa_stats(r, data):
    err = c.circ_diff(r["fit_pa"], r["true_pa"])
    d_fit = c.coverage_density(r["fit_pa"], data.bl_pa)
    d_true = c.coverage_density(r["true_pa"], data.bl_pa)
    return err, d_fit, d_true


def do_injection(star, data, out, md):
    real = injection.load(out, star, False)
    null = injection.load(out, star, True)
    if real is not None:
        err, d_fit, d_true = pa_stats(real, data)
        n = len(err)
        trapped = real["chi2"] > real["chi2_true"] + 1.0
        # fits above the truth's chi2 are search failures; the rest are
        # genuine (noise-driven) optima that fit at least as well as the truth
        good = ~trapped
        md.append(f"### Injection-recovery, real coverage ({n} injections)\n")
        md.append(
            f"- pa error (fit - true, mod 180): median |err| "
            f"{np.median(abs(err)):.1f} deg, rms {np.sqrt(np.mean(err**2)):.1f}"
            f" deg, {np.mean(abs(err) < 10):.0%} within 10 deg."
        )
        md.append(
            f"- omega error (fit - true): median {np.median(real['fit_omega'] - real['true_omega']):+.3f}, "
            f"rms {np.std(real['fit_omega'] - real['true_omega']):.3f}; "
            f"{np.mean(real['fit_omega'] > 0.98):.0%} of fits at the omega = 0.99 bound."
        )
        md.append(
            f"- search failures (best chi2 more than 1 above chi2 at the "
            f"truth): {int(trapped.sum())}/{n}. Median chi2_fit - chi2_true = "
            f"{np.median(real['chi2'] - real['chi2_true']):+.1f}."
        )
        md.append(
            f"- coverage density at the fitted pa: mean {d_fit.mean():.2f} "
            f"+/- {d_fit.std() / np.sqrt(n):.2f}, at the injected pa "
            f"{d_true.mean():.2f} +/- {d_true.std() / np.sqrt(n):.2f} "
            "(a fit pulled towards the baselines would exceed the latter)."
        )
        md.append(
            f"- mean fitted-minus-injected density {np.mean(d_fit - d_true):+.2f} "
            f"+/- {np.std(d_fit - d_true) / np.sqrt(n):.2f}; for the "
            f"{int(good.sum())} well-searched fits "
            f"{np.mean((d_fit - d_true)[good]) if good.any() else np.nan:+.2f}.\n"
        )
        plot_real(star, data, real, err, out)
        md.append(f"![injection]({star}_injection.png)\n")
    if null is not None:
        d_fit = c.coverage_density(null["fit_pa"], data.bl_pa)
        d_perp = c.coverage_density(null["fit_pa"] + 90, data.bl_pa)
        n = len(d_fit)
        w = null["fit_omega"]
        # circular mean resultant length of 2*pa: 0 = uniform
        rbar = abs(np.exp(2j * np.radians(null["fit_pa"])).mean())
        md.append(f"### Null test: sphere injected ({n} injections)\n")
        md.append(
            f"- spurious omega: median {np.median(w):.2f}, 16-84% "
            f"{np.percentile(w, 16):.2f}-{np.percentile(w, 84):.2f}; "
            f"{np.mean(w > 0.3):.0%} above 0.3, {np.mean(w > 0.98):.0%} at the bound."
        )
        md.append(
            f"- spurious pa vs the baselines: coverage density at the fitted "
            f"pa {d_fit.mean():.2f} +/- {d_fit.std() / np.sqrt(n):.2f}, at "
            f"pa+90 {d_perp.mean():.2f} +/- {d_perp.std() / np.sqrt(n):.2f} "
            "(1 = no preference); mean resultant length of 2 pa "
            f"{rbar:.2f} (expected {1 / np.sqrt(n):.2f} for uniform)."
        )
        md.append(
            "- among fits with omega > 0.3 only: density at the fitted pa "
            f"{d_fit[w > 0.3].mean() if (w > 0.3).any() else np.nan:.2f}, at pa+90 "
            f"{d_perp[w > 0.3].mean() if (w > 0.3).any() else np.nan:.2f}.\n"
        )
        plot_null(star, data, null, out)
        md.append(f"![null]({star}_null.png)\n")


def plot_real(star, data, r, err, out):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    a = ax[0, 0]
    a.scatter(r["true_pa"], c.wrap180(r["fit_pa"]), c=r["true_omega"], s=14)
    for off in (-180, 0, 180):
        a.plot([0, 180], [off, 180 + off], "k", lw=0.5)
    a.set(
        xlim=(0, 180),
        ylim=(0, 180),
        xlabel="injected pa (deg)",
        ylabel="recovered pa (deg, mod 180)",
    )
    a2 = a.twinx()  # baseline PA histogram behind the points
    a2.hist(data.bl_pa, bins=PA_BINS, color="0.8", zorder=0)
    a2.set_yticks([])
    a.set_zorder(a2.get_zorder() + 1)
    a.patch.set_visible(False)
    b = ax[0, 1]
    b.scatter(r["true_pa"], err, s=14)
    b.axhline(0, color="k", lw=0.5)
    b.set(
        xlim=(0, 180),
        xlabel="injected pa (deg)",
        ylabel="pa error, fit - true (deg)",
    )
    b2 = b.twinx()
    b2.hist(data.bl_pa, bins=PA_BINS, color="0.8", zorder=0)
    b2.set_yticks([])
    b.set_zorder(b2.get_zorder() + 1)
    b.patch.set_visible(False)
    d = ax[1, 0]
    d.scatter(r["true_omega"], r["fit_omega"], c=r["true_inc"], s=14)
    d.plot([0.5, 0.95], [0.5, 0.95], "k", lw=0.5)
    d.set(xlabel="injected omega", ylabel="recovered omega")
    e = ax[1, 1]
    e.hist(r["chi2"] - r["chi2_true"], bins=30)
    e.axvline(0, color="k", lw=0.5)
    e.set(xlabel=r"$\chi^2$(best fit) $-$ $\chi^2$(truth)", ylabel="count")
    fig.suptitle(
        f"{star}: injection-recovery on the real coverage "
        "(grey: baseline PA histogram)"
    )
    fig.tight_layout()
    fig.savefig(out / f"{star}_injection.png", dpi=130)
    plt.close(fig)


def plot_null(star, data, r, out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].hist(r["fit_omega"], bins=np.linspace(0, 0.99, 34))
    ax[0].set(xlabel="spurious omega (sphere injected)", ylabel="count")
    ax[1].hist(c.wrap180(r["fit_pa"]), bins=PA_BINS, color="C3", alpha=0.7)
    ax[1].set(xlabel="spurious pa (deg)", ylabel="count", xlim=(0, 180))
    a2 = ax[1].twinx()
    a2.hist(data.bl_pa, bins=PA_BINS, color="0.8", zorder=0)
    a2.set_yticks([])
    ax[1].set_zorder(a2.get_zorder() + 1)
    ax[1].patch.set_visible(False)
    ax[2].scatter(c.wrap180(r["fit_pa"]), r["fit_omega"], s=14)
    ax[2].set(
        xlabel="spurious pa (deg)", ylabel="spurious omega", xlim=(0, 180)
    )
    fig.suptitle(f"{star}: null test (grey: baseline PA histogram)")
    fig.tight_layout()
    fig.savefig(out / f"{star}_null.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(c.OUT))
    p.add_argument(
        "--stars", nargs="+", default=list(c.STARS), choices=list(c.STARS)
    )
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [
        "# PAVO robustness: local minima and uv-coverage alignment\n",
        "Maximum-likelihood fits of `virgil.GravityDarkenedStar` "
        "(`n_lat=32`, float64) to the PAVO V^2, jitter fixed at "
        "Dholakia's posterior median. PAs are East of North, mod 180; "
        "baseline PA = atan2(u, v). See `examples/elr_pavo/robustness/`.\n",
    ]
    for star in args.stars:
        data = c.Dataset(star)
        md.append(f"## {star}\n")
        src, med, d0, d90 = alignment(star, data, out)
        md.append(
            f"### Alignment of the fitted PA with the baselines\n\n"
            f"{data.n} V^2 samples. Baseline PAs span "
            f"{np.percentile(data.bl_pa, 5):.0f}-"
            f"{np.percentile(data.bl_pa, 95):.0f} deg (5-95%). Fitted pole "
            f"PA ({src}): {med % 180:.1f} deg; coverage density at it "
            f"{d0:.2f}, at pa+90 {d90:.2f} (1 = no preference).\n\n"
            f"![alignment]({star}_alignment.png)\n"
        )
        do_multistart(star, data, out, md)
        do_profile(star, data, out, md)
        do_injection(star, data, out, md)
    (out / "robustness.md").write_text("\n".join(md))
    print(f"wrote {out / 'robustness.md'}")


if __name__ == "__main__":
    main()
