"""Profile-likelihood maps of the PAVO fit: chi2 minimised over the other
parameters on a grid of two of them.

    python profile.py --star upsTau --pair pa,inc [--task-index i --n-tasks k]

Pairs and grids: ``pa,inc`` (5 deg steps, pa 0-175, inc 0-90), ``omega,inc``
(0.05 steps in omega up to 0.99, inc as before) and ``diam,omega`` (0.02 mas
steps over his median +/- 5 sd, clipped to (0.02, 2) mas). At each grid
point the other two parameters are fitted from ``--n-starts`` starting
points (his posterior median plus random draws from the prior box) and the
best chi2 kept, so a single trapped fit does not make a hole in the map.
The jitter is fixed at his posterior median.

Each task writes ``output/profile_<star>_<pair>_<task>.npz`` and ``.json``.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

import common as c

PAIRS = {
    "pa,inc": ("pa", "inc"),
    "omega,inc": ("omega", "inc"),
    "diam,omega": ("diam_eq", "omega"),
}


def grid(star, pair):
    """The two 1-d grids (in the order of ``pair``) for this star."""
    inc = np.arange(0.0, 90.01, 5.0)
    omega = np.append(np.arange(0.0, 0.99, 0.05), 0.99)
    if pair == "pa,inc":
        return np.arange(0.0, 180.0, 5.0), inc
    if pair == "omega,inc":
        return omega, inc
    r = c.reference(star)["diam_eq"]
    lo = max(r["median"] - 5 * r["sd"], 0.02)
    hi = min(r["median"] + 5 * r["sd"], c.MAX_DIAM)
    return np.arange(lo, hi + 1e-9, 0.02), omega


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    c.add_task_args(p)
    p.add_argument("--pair", required=True, choices=list(PAIRS))
    p.add_argument("--n-starts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-grid",
        type=int,
        default=None,
        help="smoke test: coarsen each axis to this many points",
    )
    args = p.parse_args()

    data = c.Dataset(args.star)
    names = PAIRS[args.pair]
    axes = grid(args.star, args.pair)
    if args.n_grid:  # coarse grid for a quick test
        axes = tuple(
            a[np.linspace(0, len(a) - 1, args.n_grid).round().astype(int)]
            for a in axes
        )
    points = list(itertools.product(range(len(axes[0])), range(len(axes[1]))))
    mine = c.my_items(len(points), args)
    free = [k for k in c.PARAMS if k not in names]
    med = c.ref_median(args.star)
    print(
        f"{args.star} {args.pair}: grid {len(axes[0])}x{len(axes[1])}, "
        f"task {args.task_index}/{args.n_tasks}: {len(mine)} points"
    )

    rec = {k: [] for k in ("i", "j", "chi2", "converged", "seconds")}
    rec.update({f"best_{k}": [] for k in c.PARAMS})
    for n in mine:
        i, j = points[n]
        fixed = {names[0]: axes[0][i], names[1]: axes[1][j]}
        # start 0: his median; the rest: random draws (seeded by grid point)
        rand = c.start_design(max(args.n_starts - 1, 1), args.seed + 1000 + n)
        starts = [med] + [c.as_init(r) for r in rand[: args.n_starts - 1]]
        best, total = None, 0.0
        for s in starts:
            r = c.fit_ml(data, data.v2, {**s, **fixed}, free=free)
            total += r["seconds"]
            if best is None or r["chi2"] < best["chi2"]:
                best = r
        best["seconds"] = total  # for all the starts at this grid point
        rec["i"].append(i)
        rec["j"].append(j)
        for k in ("chi2", "converged", "seconds"):
            rec[k].append(best[k])
        for k in c.PARAMS:
            rec[f"best_{k}"].append(best[k])
        print(
            f"({i},{j}) {names[0]}={fixed[names[0]]:.3f} "
            f"{names[1]}={fixed[names[1]]:.3f}: chi2 {best['chi2']:.2f} "
            f"conv {best['converged']}",
            flush=True,
        )
    meta = dict(
        star=args.star,
        pair=args.pair,
        names=names,
        n_data=data.n,
        jitter=data.jitter,
        n_starts=args.n_starts,
        seed=args.seed,
        task_index=args.task_index,
        n_tasks=args.n_tasks,
    )
    tag = args.pair.replace(",", "-")
    c.save(
        args.out,
        f"profile_{args.star}_{tag}_{args.task_index}",
        {
            **{k: np.asarray(v) for k, v in rec.items()},
            "axis0": axes[0],
            "axis1": axes[1],
        },
        meta,
    )


def assemble(out, star, pair):
    """Collate tasks into (axis0, axis1, chi2 map [i, j], converged map)."""
    tag = pair.replace(",", "-")
    files = sorted(c.Path(out).glob(f"profile_{star}_{tag}_*.npz"))
    if not files:
        return None
    parts = [np.load(f) for f in files]
    a0, a1 = parts[0]["axis0"], parts[0]["axis1"]
    chi2 = np.full((len(a0), len(a1)), np.nan)
    conv = np.zeros((len(a0), len(a1)), bool)
    for q in parts:
        chi2[q["i"], q["j"]] = q["chi2"]
        conv[q["i"], q["j"]] = q["converged"]
    return a0, a1, chi2, conv


if __name__ == "__main__":
    main()
