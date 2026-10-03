"""Injection-recovery on a star's real uv coverage.

Fake V^2 are simulated at the real data's u, v, per-sample wavelengths and
errors (his median jitter added in quadrature), with seeded Gaussian noise,
and fitted by multi-start maximum likelihood (best of ``--n-starts`` Latin-
hypercube starts; jitter fixed at its true value). Truths: diam = his median,
pa ~ U[0, 180), inc isotropic on [20, 90] deg (cos i ~ U(0, cos 20)),
omega ~ U[0.5, 0.95]. With ``--null`` the star is a sphere (omega = 0; pa and
inc are drawn the same way but meaningless), to see which orientation the
coverage alone pulls a spurious fit towards.

    python injection.py --star upsTau [--n-inject 100] [--seed 0] [--null]
        [--task-index i --n-tasks k]

Each task writes ``output/injection_<star>_<null|real>_<task>.npz`` and
``.json``. Injection k always uses ``default_rng([seed, k, null])``, so
splitting across tasks does not change the draws.
"""

from __future__ import annotations

import argparse

import numpy as np

import common as c

INC_MIN = 20.0


def truth(k, seed, null, diam):
    """The injected parameters of injection ``k`` and its noise stream."""
    rng = np.random.default_rng([seed, k, int(null)])
    pa = rng.uniform(0.0, 180.0)
    cos_i = rng.uniform(0.0, np.cos(np.radians(INC_MIN)))
    omega = 0.0 if null else rng.uniform(0.5, 0.95)
    p = dict(
        diam_eq=diam,
        omega=omega,
        inc=float(np.degrees(np.arccos(cos_i))),
        pa=pa,
    )
    return p, rng


def tag(null):
    return "null" if null else "real"


def load(out, star, null):
    return c.collect(out, f"injection_{star}_{tag(null)}_*.npz")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    c.add_task_args(p)
    p.add_argument("--n-inject", type=int, default=100)
    p.add_argument("--n-starts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--null", action="store_true", help="inject omega = 0")
    args = p.parse_args()

    data = c.Dataset(args.star)
    diam = c.ref_median(args.star)["diam_eq"]
    mine = c.my_items(args.n_inject, args)
    print(
        f"{args.star} {tag(args.null)}: {data.n} samples; task "
        f"{args.task_index}/{args.n_tasks}: {len(mine)} injections"
    )

    keys = ["index", "chi2", "chi2_true", "converged", "steps", "seconds"]
    rec = {k: [] for k in keys}
    rec.update({f"true_{k}": [] for k in c.PARAMS})
    rec.update({f"fit_{k}": [] for k in c.PARAMS})
    rec["chi2_starts"] = []
    for k in mine:
        true, rng = truth(k, args.seed, args.null, diam)
        # The null is an exact uniform disk, not GravityDarkenedStar at
        # omega = 0: its mesh is slightly anisotropic, so the drawn inc and
        # pa could imprint an orientation on the fake data.
        v2_true = (
            data.v2_disk(true["diam_eq"])
            if args.null
            else data.v2_model(**true)
        )
        v2 = v2_true + rng.normal(0.0, data.sigma)
        starts = c.start_design(args.n_starts, args.seed * 100003 + int(k))
        fits = [c.fit_ml(data, v2, c.as_init(s)) for s in starts]
        best = min(fits, key=lambda f: f["chi2"])
        rec["index"].append(k)
        rec["chi2_true"].append(
            float((((v2 - v2_true) / data.sigma) ** 2).sum())
        )
        rec["chi2_starts"].append([f["chi2"] for f in fits])
        rec["seconds"].append(sum(f["seconds"] for f in fits))
        for key in ("chi2", "converged", "steps"):
            rec[key].append(best[key])
        for q in c.PARAMS:
            rec[f"true_{q}"].append(true[q])
            rec[f"fit_{q}"].append(best[q])
        print(
            f"inj {k:3d}: true pa {true['pa']:6.1f} inc {true['inc']:5.1f} "
            f"omega {true['omega']:.2f} -> fit pa {best['pa']:6.1f} "
            f"inc {best['inc']:5.1f} omega {best['omega']:.2f}; chi2 "
            f"{best['chi2']:.1f} (truth {rec['chi2_true'][-1]:.1f})",
            flush=True,
        )
    meta = dict(
        star=args.star,
        null=args.null,
        n_data=data.n,
        jitter=data.jitter,
        seed=args.seed,
        n_inject=args.n_inject,
        n_starts=args.n_starts,
        task_index=args.task_index,
        n_tasks=args.n_tasks,
    )
    c.save(
        args.out,
        f"injection_{args.star}_{tag(args.null)}_{args.task_index}",
        {
            **{k: np.asarray(v) for k, v in rec.items()},
            # an empty task keeps the (0, n_starts) shape so tasks concatenate
            "chi2_starts": np.asarray(rec["chi2_starts"]).reshape(
                -1, args.n_starts
            ),
        },
        meta,
    )


if __name__ == "__main__":
    main()
