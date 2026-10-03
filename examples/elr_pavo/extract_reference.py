"""Extract and convert PAVO reference posteriors from Shashank Dholakia's NUTS samples.

Downloads the posterior .h5 files from the jax-interferometry repository,
extracts summary statistics, converts to drpangloss conventions, and writes
a JSON file with medians, percentiles, means, and standard deviations.

Run with:
    uv run --no-project --with arviz --with h5netcdf --with netcdf4 python examples/elr_pavo/extract_reference.py
"""

import json
import tempfile
import urllib.request
from pathlib import Path
import numpy as np
import arviz as az


REPO_URL = (
    "https://raw.githubusercontent.com/shashankdholakia/jax-interferometry"
)
COMMIT_SHA = "70689ed3dba338d59c98d02e8126a07a3b4e86da"

STARS = {
    "HD_84999": "upsUma/HD_84999_NUTS.h5",
    "HD_211336": "epsCep/HD_211336_NUTS.h5",
    "HD_125162": "lamBoo/HD_125162_NUTS.h5",
    "HD_28024": "upsTau/HD_28024_NUTS.h5",
}

CONVENTIONS = (
    "Raw variables from Dholakia's NUTS samples: diam (mas), omega, inc (rad; 0=equator-on), "
    "obl (rad), logsig. Converted to drpangloss: diam_eq=diam, omega=omega, "
    "inc_deg=90-degrees(inc), pa_deg=degrees(obl), jitter_v2=exp(logsig)."
)


def download_file(url: str, cache_dir: Path) -> Path:
    """Download file with caching in a temp directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    local_path = cache_dir / filename

    if local_path.exists():
        print(f"Using cached {filename}")
        return local_path

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def extract_stats(idata, var_names):
    """Extract median, p16, p84, mean, sd, r_hat, ess_bulk for variables."""
    stats = {}
    for var in var_names:
        if var not in idata.posterior:
            continue

        samples = idata.posterior[var].values.flatten()

        stats[var] = {
            "median": float(np.median(samples)),
            "p16": float(np.percentile(samples, 16)),
            "p84": float(np.percentile(samples, 84)),
            "mean": float(np.mean(samples)),
            "sd": float(np.std(samples, ddof=1)),
        }

        # Add arviz diagnostics
        if hasattr(idata, "posterior"):
            if (
                hasattr(idata.posterior, "attrs")
                and "r_hat" in idata.posterior.attrs
            ):
                stats[var]["r_hat"] = float(
                    idata.posterior.attrs["r_hat"].get(var, np.nan)
                )
            if (
                hasattr(idata.posterior, "attrs")
                and "ess_bulk" in idata.posterior.attrs
            ):
                stats[var]["ess_bulk"] = float(
                    idata.posterior.attrs["ess_bulk"].get(var, np.nan)
                )

    # Get chain and draw counts
    n_chains = idata.posterior.dims.get("chain", 1)
    n_draws = idata.posterior.dims.get("draw", 1)
    stats["n_chains"] = int(n_chains)
    stats["n_draws"] = int(n_draws)

    return stats


def convert_samples(idata, raw_stats):
    """Convert samples from Dholakia conventions to drpangloss."""
    converted = {}

    # diam_eq = diam (no change)
    if "diam" in idata.posterior:
        samples = idata.posterior["diam"].values.flatten()
        converted["diam_eq"] = {
            "median": float(np.median(samples)),
            "p16": float(np.percentile(samples, 16)),
            "p84": float(np.percentile(samples, 84)),
            "mean": float(np.mean(samples)),
            "sd": float(np.std(samples, ddof=1)),
        }

    # omega = omega (no change)
    if "omega" in idata.posterior:
        samples = idata.posterior["omega"].values.flatten()
        converted["omega"] = {
            "median": float(np.median(samples)),
            "p16": float(np.percentile(samples, 16)),
            "p84": float(np.percentile(samples, 84)),
            "mean": float(np.mean(samples)),
            "sd": float(np.std(samples, ddof=1)),
        }

    # inc_deg = 90 - degrees(inc)
    if "inc" in idata.posterior:
        samples_rad = idata.posterior["inc"].values.flatten()
        samples_deg = 90.0 - np.degrees(samples_rad)
        converted["inc_deg"] = {
            "median": float(np.median(samples_deg)),
            "p16": float(np.percentile(samples_deg, 16)),
            "p84": float(np.percentile(samples_deg, 84)),
            "mean": float(np.mean(samples_deg)),
            "sd": float(np.std(samples_deg, ddof=1)),
        }

    # pa_deg = degrees(obl)
    if "obl" in idata.posterior:
        samples_rad = idata.posterior["obl"].values.flatten()
        samples_deg = np.degrees(samples_rad)
        converted["pa_deg"] = {
            "median": float(np.median(samples_deg)),
            "p16": float(np.percentile(samples_deg, 16)),
            "p84": float(np.percentile(samples_deg, 84)),
            "mean": float(np.mean(samples_deg)),
            "sd": float(np.std(samples_deg, ddof=1)),
        }

    # jitter_v2 = exp(logsig)
    if "logsig" in idata.posterior:
        samples_log = idata.posterior["logsig"].values.flatten()
        samples_lin = np.exp(samples_log)
        converted["jitter_v2"] = {
            "median": float(np.median(samples_lin)),
            "p16": float(np.percentile(samples_lin, 16)),
            "p84": float(np.percentile(samples_lin, 84)),
            "mean": float(np.mean(samples_lin)),
            "sd": float(np.std(samples_lin, ddof=1)),
        }

    # Chain and draw counts
    converted["n_chains"] = raw_stats.get("n_chains", 1)
    converted["n_draws"] = raw_stats.get("n_draws", 1)

    return converted


def round_dict(d, sig_figs=6):
    """Round numeric values to significant figures."""

    def round_val(x):
        if isinstance(x, (int, float)):
            if x == 0:
                return 0.0
            return float(f"{x:.{sig_figs}g}")
        return x

    return {
        k: round_val(v) if not isinstance(v, dict) else round_dict(v, sig_figs)
        for k, v in d.items()
    }


def main():
    output_path = Path("examples/elr_pavo/reference_posteriors.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "pavo_posteriors"

        data = {
            "source": {
                "repo": "https://github.com/shashankdholakia/jax-interferometry",
                "commit": COMMIT_SHA,
                "author": "Shashank Dholakia",
            },
            "conventions": CONVENTIONS,
            "stars": {},
        }

        for hd, rel_path in STARS.items():
            print(f"\nProcessing {hd}...")

            url = f"{REPO_URL}/{COMMIT_SHA}/{rel_path}"
            h5_path = download_file(url, cache_dir)

            # Load with arviz
            idata = az.from_netcdf(str(h5_path))

            # Extract raw statistics
            raw_stats = extract_stats(
                idata, ["diam", "omega", "inc", "obl", "logsig"]
            )

            # Convert to drpangloss conventions
            converted_stats = convert_samples(idata, raw_stats)

            data["stars"][hd] = {
                "raw": round_dict(raw_stats),
                "drpangloss": round_dict(converted_stats),
            }

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
