"""Multi-start maximum likelihood: how many local optima does the PAVO fit
of ``GravityDarkenedStar`` have, and does the best one match Dholakia's
NUTS posterior?

Starts are a Latin-hypercube design over the prior box (diam 0.3-2 mas,
omega 0-0.99, cos i 0-1, pa 0-180). Each start is fitted for (diam_eq,
omega, inc, pa) with the jitter fixed at his posterior median, using
``virgil.fitting.fit`` (Levenberg-Marquardt through optimistix; bounds via
the priors' bijections). Optima are then clustered (``optima``).

    python multistart.py --star upsTau --n-starts 64 [--task-index i --n-tasks k]

Each task writes ``output/multistart_<star>_<task>.npz`` and ``.json``.
"""

from __future__ import annotations

import argparse

import numpy as np

import common as c

TOL = dict(diam_eq=0.01, omega=0.02, inc=2.0, pa=2.0)  # cluster tolerances


def close(a, b):
    """True if two (diam, omega, inc, pa) points agree within TOL."""
    d = [abs(a[0] - b[0]) < TOL["diam_eq"], abs(a[1] - b[1]) < TOL["omega"]]
    d.append(abs(a[2] - b[2]) < TOL["inc"])
    # pa is meaningless for a pole-on star
    d.append(abs(c.circ_diff(a[3], b[3])) < TOL["pa"] or min(a[2], b[2]) < 1)
    return all(d)


def optima(end, chi2, converged, star):
    """Cluster end points; returns a list of dicts, best chi2 first.

    Each has the best member's parameters and chi2, ``dchi2`` from the
    overall best, ``n`` (starts reaching it), ``n_converged``, and
    ``z`` = its offset from his posterior median in units of his posterior
    sd (``pa`` circular), plus ``z_all`` (their quadrature sum).
    """
    ref = c.reference(star)
    order = np.argsort(chi2)
    clusters = []
    for i in order:
        for cl in clusters:
            if close(end[i], end[cl["members"][0]]):
                cl["members"].append(i)
                break
        else:
            clusters.append(dict(members=[i]))
    best = chi2[order[0]]
    out = []
    for cl in clusters:
        i = cl["members"][0]
        p = dict(zip(c.PARAMS, end[i]))
        z = {}
        for k in c.PARAMS:
            r = ref[k]
            d = (
                c.circ_diff(p[k], r["median"])
                if k == "pa"
                else p[k] - r["median"]
            )
            z[k] = float(d / r["sd"])
        out.append(
            dict(
                **{k: float(v) for k, v in p.items()},
                chi2=float(chi2[i]),
                dchi2=float(chi2[i] - best),
                n=len(cl["members"]),
                n_converged=int(converged[cl["members"]].sum()),
                z=z,
                z_all=float(np.sqrt(sum(v**2 for v in z.values()))),
            )
        )
    return out


def load(out, star):
    """Collated records of all tasks, or None."""
    r = c.collect(out, f"multistart_{star}_*.npz")
    if r is None:
        return None
    r["end"] = np.column_stack([r[f"end_{k}"] for k in c.PARAMS])
    r["start"] = np.column_stack([r[f"start_{k}"] for k in c.PARAMS])
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    c.add_task_args(p)
    p.add_argument("--n-starts", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data = c.Dataset(args.star)
    # the whole design is drawn in every task, so splitting does not change it
    starts = c.start_design(args.n_starts, args.seed)
    mine = c.my_items(args.n_starts, args)
    print(
        f"{args.star}: {data.n} samples, jitter {data.jitter:.4f}; "
        f"task {args.task_index}/{args.n_tasks}: {len(mine)} starts"
    )

    rec = {f"{w}_{k}": [] for w in ("start", "end") for k in c.PARAMS}
    rec.update(index=[], chi2=[], converged=[], steps=[], seconds=[])
    for i in mine:
        init = c.as_init(starts[i])
        r = c.fit_ml(data, data.v2, init)
        for k in c.PARAMS:
            rec[f"start_{k}"].append(init[k])
            rec[f"end_{k}"].append(r[k])
        for k in ("chi2", "converged", "steps", "seconds"):
            rec[k].append(r[k])
        rec["index"].append(i)
        print(
            f"start {i:3d}: chi2 {r['chi2']:9.2f}  diam {r['diam_eq']:.3f} "
            f"omega {r['omega']:.3f} inc {r['inc']:.1f} pa {r['pa']:.1f} "
            f"conv {r['converged']} steps {r['steps']} {r['seconds']:.0f}s",
            flush=True,
        )
    meta = dict(
        star=args.star,
        n_data=data.n,
        jitter=data.jitter,
        seed=args.seed,
        n_starts=args.n_starts,
        task_index=args.task_index,
        n_tasks=args.n_tasks,
        method="virgil.fitting.fit, LM",
        max_steps=c.MAX_STEPS,
    )
    c.save(
        args.out,
        f"multistart_{args.star}_{args.task_index}",
        {k: np.asarray(v) for k, v in rec.items()},
        meta,
    )


if __name__ == "__main__":
    main()
