"""Plot the output of scripts/bench_ft.py: time against number of points.

python design/figures/plot_bench_ft.py design/figures/bench_ft_cpu.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

path = Path(sys.argv[1])
rows = json.loads(path.read_text())["timings"]
dtypes = sorted({r["dtype"] for r in rows})
fig, axes = plt.subplots(
    1, len(dtypes), figsize=(5 * len(dtypes), 3.8), squeeze=False
)
axes = axes[0]
for ax, dtype in zip(axes, dtypes):
    for i, npix in enumerate(sorted({r["npix"] for r in rows})):
        sel = [r for r in rows if r["dtype"] == dtype and r["npix"] == npix]
        m = [r["npoints"] for r in sel]
        ax.loglog(
            m,
            [r["dft_ms"] for r in sel],
            "-o",
            c=f"C{i}",
            label=f"DFT {npix}²",
        )
        ax.loglog(
            m,
            [r["nufft_ms"] for r in sel],
            "--s",
            c=f"C{i}",
            label=f"NUFFT {npix}²",
        )
    ax.set_title(f"{dtype}: jitted value + gradient")
    ax.set_xlabel("number of (u, v) points")
    ax.set_ylabel("time (ms)")
axes[0].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(path.with_suffix(".png"), dpi=130)
