"""Compare our PAVO posteriors with Dholakia's; writes ``comparison.md``.

Reads ``output/<star>_summary.json`` (from ``fit_pavo.py``) and the
``"virgil"`` block of ``reference_posteriors.json``, which was extracted
from his own ``_NUTS.h5`` files at commit 70689ed.

    .venv/bin/python examples/elr_pavo/compare.py [--out DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STARS = {
    "upsUMa": "HD_84999",
    "epsCep": "HD_211336",
    "lamBoo": "HD_125162",
    "upsTau": "HD_28024",
}
# (reference key, our key, label)
PARAMS = [
    ("diam_eq", "diam", "diam_eq (mas)"),
    ("omega", "omega", "omega"),
    ("inc_deg", "inc", "inc (deg)"),
    ("pa_deg", "pa", "pa (deg)"),
    ("jitter_v2", "jitter", "jitter (V^2)"),
]
HEADER = """\
# PAVO rapid-rotator re-analysis: virgil vs Dholakia

Ours: `virgil.GravityDarkenedStar` (grey, `n_lat=32`) fitted by
`fit_pavo.py` with his priors, likelihood and NUTS settings (see its
docstring). His: the posteriors in his own `_NUTS.h5` files in
[jax-interferometry](https://github.com/shashankdholakia/jax-interferometry)
at commit `70689ed3dba338d59c98d02e8126a07a3b4e86da`, converted to our
conventions in `reference_posteriors.json` (`inc = 90 deg - his inc`,
`pa = his obl`, `jitter = exp(logsig)`). Each cell is median +/- sd of the
posterior samples. `D/sigma` is (ours - his) / sqrt(sd_ours^2 + sd_his^2);
rows with |D/sigma| > 0.5 are flagged with `*`.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "output"))
    args = p.parse_args()
    ref = json.loads((HERE / "reference_posteriors.json").read_text())
    lines = [HEADER]
    for name, hd in STARS.items():
        path = Path(args.out) / f"{name}_summary.json"
        if not path.exists():
            lines.append(f"## {name} ({hd})\n\nNo summary found.\n")
            continue
        ours = json.loads(path.read_text())
        s = ours["settings"]
        lines += [
            f"## {name} ({hd})\n",
            f"{ours['n_data']} data points; {s['chains']} chains x "
            f"{s['num_samples']} samples after {s['num_warmup']} warmup; "
            f"mean tree depth {ours['mean_tree_depth']:.1f}, "
            f"{ours['n_divergences']} divergences.\n",
            "| parameter | Dholakia | ours | D/sigma | r_hat | n_eff |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for rk, ok, label in PARAMS:
            a = ref["stars"][hd]["virgil"][rk]
            b = ours["posterior"][ok]
            z = (b["median"] - a["median"]) / np.hypot(a["sd"], b["sd"])
            flag = " *" if abs(z) > 0.5 else ""
            lines.append(
                f"| {label} | {a['median']:.4g} +/- {a['sd']:.2g} "
                f"| {b['median']:.4g} +/- {b['sd']:.2g} | {z:+.2f}{flag} "
                f"| {b.get('r_hat', float('nan')):.3f} "
                f"| {b.get('n_eff', float('nan')):.0f} |"
            )
        lines.append("")
    (HERE / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
